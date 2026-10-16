"""Cache real decoded OT DEV candidates on Kaggle; sender selection replays later."""
import argparse
import configparser
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from ..analyzers import DetectionAnalyzer
from ..v32.tracking import read_sequence
from ..v32vc.train import tensor
from ..v32vc.checkpoint import state_hash
from ..v33ap.tasks import prediction
from ..v33ap.sources import serial
from ..v34.checkpoint import load_checkpoint
from ..v34.protocol import codec_factory
from ..v34.preprocessing import region_maps
from ..v34.runner import input_pair,condition,infer
from ..v34.learning import domain_calibration
from ..v34r1.preprocessing import PROFILES,filter_clip
from ..v34r1.policy import shortlist

ROOT=Path(__file__).resolve().parents[2]


def mot17_plan(root,name,frames):
    allowed={'MOT17-'+n+'-FRCNN' for n in ('02','05','04','11','09','10','13')}
    if name not in allowed or frames<16 or frames%16: raise ValueError('frozen MOT17 sequence/whole windows required')
    path=Path(root)/name; ini=configparser.ConfigParser()
    if not ini.read(path/'seqinfo.ini'): raise FileNotFoundError('original seqinfo missing')
    seq=ini['Sequence']; fps=Fraction(seq['frameRate'])
    total,width,height=(int(seq[k]) for k in ('seqLength','imWidth','imHeight'))
    if (fps.denominator!=1 or fps.numerator not in (14,25,30) or min(total,width,height)<1 or frames>total
        or seq['name']!=name or seq['imDir']!='img1' or seq['imExt']!='.jpg'):
        raise ValueError('MOT17 original geometry/cadence drift')
    images=[path/'img1'/f'{i:06d}.jpg' for i in range(1,frames+1)]
    if any(not p.is_file() for p in images): raise FileNotFoundError('declared MOT17 source frame missing')
    return dict(name=name,path=path,images=images,frames=frames,full_frames=total,width=width,height=height,fps=fps)


@torch.no_grad()
def detect(detector,rgb):
    x=tensor(rgb,'cuda'); out=[]
    for start in range(0,len(rgb),2):
        batch=detector.model([x[0,:,i] for i in range(start,min(start+2,len(rgb)))])
        out.extend(prediction(v) for v in batch)
    return out


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--worker',type=int,choices=(0,1),required=True)
    p.add_argument('--out',type=Path,required=True); args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing') or not torch.cuda.is_available(): raise ValueError('Kaggle GPU only')
    if args.out.exists(): raise ValueError('immutable fresh worker output required')
    args.out.mkdir(parents=True); torch.set_num_threads(2); torch.manual_seed(303401); torch.backends.cudnn.benchmark=False
    import cv2
    cv2.setNumThreads(1)
    cfg=json.loads((ROOT/'configs/v34r1-kaggle.json').read_text()); protocol=json.loads((ROOT/'configs/v34r1-a3-ot-probe.json').read_text())
    manifest_raw=(args.root/'source-manifest.json').read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest()!=protocol['source_manifest_sha256']: raise ValueError('frozen source manifest drift')
    manifest=json.loads(manifest_raw)
    sequences=[r for i,r in enumerate(manifest['sequences']) if r['split']!='reserved_test' and i%2==args.worker]
    if len(sequences)!=2 or {r['split'] for r in sequences}!={'fit','validation'}: raise ValueError('two image-disjoint DEV sequences per worker required')
    detector=DetectionAnalyzer('resnet50','cuda'); detector.model.transform.min_size=(960,); detector.model.transform.max_size=960
    detector.model.roi_heads.detections_per_img=300
    expected_model='c4616e10536fb767f901700c621bb851a10259b6ed90f3af0e4596d5a6886533'
    if state_hash(detector.model.state_dict())!=expected_model: raise ValueError('frozen detector changed')
    model=load_checkpoint(ROOT/'checkpoints/v34_r1/controller_303401.pth','cuda')
    if model.checkpoint_sha256!=cfg['controller_sha256_by_task']['ot']: raise ValueError('A2 OT controller drift')
    (args.out/'protocol.json').write_text(json.dumps(dict(config=protocol,sequences=[r['name'] for r in sequences],
        checkpoint_sha256=model.checkpoint_sha256,detector_sha256=expected_model,source_manifest_sha256=protocol['source_manifest_sha256'],
        gpu=torch.cuda.get_device_name(0),scope='candidate cache only; source-only policy replays before GT scoring'),indent=2))
    rows=[]; sources={}; started=time.perf_counter(); torch.cuda.reset_peak_memory_stats(); actual_encodes=decoded_frames=0
    for sequence in sequences:
        name=sequence['name']; plan=mot17_plan(args.root,name,sequence['frames'])
        for member,sha in sequence['files'].items():
            if member.startswith('img1/') or member=='seqinfo.ini':
                if hashlib.sha256((args.root/name/member).read_bytes()).hexdigest()!=sha: raise ValueError('original MOT17 bytes drift')
        rgb,geometry=read_sequence(plan,960); clean=detect(detector,rgb)
        sources[name]=dict(split=sequence['split'],frames=sequence['frames'],fps=int(plan['fps']),width=plan['width'],height=plan['height'],geometry=geometry,
            source_predictions=clean,canonical_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
        local_cfg=dict(cfg,fps=int(plan['fps']))
        for offset in range(0,len(rgb),16):
            clip=rgb[offset:offset+16]; maps=region_maps(clip,clean[offset:offset+16]); pair=input_pair(clip,maps); filters={}
            for codec_name in cfg['codecs']:
                for qp in cfg['qps']:
                    cnn=infer(model,pair,condition(codec_name,qp,[True,False,True],{**cfg,'domain_condition':model.conditions==9},domain='ot'))
                    actions=shortlist(cnn,[True,False,True],cfg['damage_tolerance'],calibration=domain_calibration(model.metadata['calibration'],'ot'),
                        max_learned=cfg['source_shortlist'],extras=cfg['extra_profiles_by_task']['ot'])
                    codec=codec_factory('ot',codec_name,qp,local_cfg); codec.standard.ffmpeg=str(ROOT/cfg['ffmpeg_executable'])
                    packet_cache={}; pixel_cache={}
                    for action in actions:
                        start=time.perf_counter()
                        if action not in filters: filters[action]=filter_clip(clip,maps,PROFILES[action])
                        pixels=filters[action]; pixel_hash=hashlib.sha256(pixels.tobytes()).hexdigest(); reused=pixel_hash in packet_cache
                        if not reused:
                            packet_cache[pixel_hash]=codec.roundtrip(pixels,source_shape=clip.shape,task='ot'); actual_encodes+=1
                        packet=packet_cache[pixel_hash]; decoded_sha=hashlib.sha256(packet.decoded.tobytes()).hexdigest()
                        inferred=decoded_sha not in pixel_cache
                        if inferred: pixel_cache[decoded_sha]=detect(detector,packet.decoded); decoded_frames+=16
                        rows.append(dict(sequence=name,split=sequence['split'],start=offset+1,codec=codec_name,qp=qp,action=action,actions=actions,
                            bytes=packet.coded_bytes,elementary_bytes=packet.elementary_bytes,metadata_bytes=packet.metadata_bytes,
                            packet_sha256=packet.packet_sha256,decoded_sha256=decoded_sha,packet_reused=reused,decoded_reused=not inferred,
                            detections=pixel_cache[decoded_sha],seconds=time.perf_counter()-start,
                            codec_times=dict(codec.last_times) if not reused else {}))
                    print(f'OT_CACHE worker{args.worker} {name} start{offset+1} {codec_name} QP{qp}; {len(rows)} rows; {time.perf_counter()-started:.1f}s',flush=True)
            subset=[r for r in rows if r['sequence']==name and r['start']==offset+1]
            (args.out/f'{name}-{offset+1:06d}.json.gz').write_bytes(gzip.compress(json.dumps(serial(subset),allow_nan=False).encode(),mtime=0))
    if state_hash(detector.model.state_dict())!=expected_model: raise ValueError('frozen detector mutated')
    (args.out/'rows.json.gz').write_bytes(gzip.compress(json.dumps(serial(rows),allow_nan=False).encode(),mtime=0))
    (args.out/'sources.json.gz').write_bytes(gzip.compress(json.dumps(serial(sources),allow_nan=False).encode(),mtime=0))
    (args.out/'complete.json').write_text(json.dumps(dict(rows=len(rows),wall_seconds=time.perf_counter()-started,
        actual_encodes=actual_encodes,decoded_frames=decoded_frames,source_frames=sum(s['frames'] for s in sources.values()),
        peak_cuda_bytes=torch.cuda.max_memory_allocated()),indent=2))


if __name__=='__main__': main()

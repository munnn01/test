"""A8 frozen all21 OD source-only candidate measurements, streamed by image."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
from ..analyzers import DetectionAnalyzer
from ..data import read_image
from ..v32vc.train import tensor
from ..v32vc.checkpoint import state_hash
from ..v33ap.tasks import od_forward,prediction,distance
from ..v33ap.sources import serial
from ..v34.checkpoint import load_checkpoint
from ..v34.protocol import codec_factory
from ..v34.preprocessing import region_maps
from ..v34.runner import input_pair,condition,infer
from ..v34.learning import domain_calibration
from ..v34.tasks import detection_distance
from ..v34.evaluation import coco_predictions
from ..v34r1.preprocessing import PROFILES,filter_clip
from ..v34r1.policy import shortlist
from ..v34r1_a3.core import audit_grid
from ..v34r1_a3.pseudo import pseudo_damage


ROOT=Path(__file__).resolve().parents[2]


def write(path,obj):
    path.write_text(json.dumps(serial(obj),indent=2,allow_nan=False)+'\n',encoding='utf-8')


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--worker',type=int,choices=(0,1),required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing') or not torch.cuda.is_available():
        raise ValueError('Kaggle GPU execution required; no laptop/CPU fallback')
    if args.out.exists(): raise ValueError('immutable fresh worker directory required')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2); torch.manual_seed(303401); torch.backends.cudnn.benchmark=False
    import cv2
    cv2.setNumThreads(1)
    cfg=json.loads((ROOT/'configs/v34r1-kaggle.json').read_text())
    protocol=json.loads((ROOT/'configs/v34r1-a3-od-probe.json').read_text())
    manifest=json.loads(args.manifest.read_text())
    if manifest['source_id_protocol_sha256']!=hashlib.sha256((ROOT/'configs/v34r1-a8-expansion.json').read_bytes()).hexdigest():
        raise ValueError('frozen source ID protocol changed')
    # Only DEV records enter this process. TEST has hashes only, outside detector access.
    sources=[r for i,r in enumerate(manifest['images']) if i%2==args.worker]
    expected=[r['id'] for r in sources]
    count=250 if manifest.get('role')=='a8_shard250' else None
    if count is None or len(sources)!=count//2: raise ValueError('complete prespecified image cohort per worker required')
    if cfg['qps']!=[30,35,40,45] or cfg['codecs']!=['h264','h265']: raise ValueError('exact frozen four-QP/codec grid required')
    detector=DetectionAnalyzer('resnet50','cuda')
    detector.model.transform.min_size=(320,); detector.model.transform.max_size=640
    detector.model.roi_heads.detections_per_img=300
    checkpoint=ROOT/'checkpoints/v34_od_coco/controller_303401.pth'
    model=load_checkpoint(checkpoint,'cuda')
    if model.checkpoint_sha256!=cfg['controller_sha256_by_task']['od']: raise ValueError('A2 controller drift')
    if state_hash(detector.model.state_dict())!='c4616e10536fb767f901700c621bb851a10259b6ed90f3af0e4596d5a6886533':
        raise ValueError('frozen OD evaluator weights drift')
    write(args.out/'protocol.json',dict(worker=args.worker,expected_ids=expected,config=protocol,baseline=cfg,
          manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
          checkpoint_sha256=model.checkpoint_sha256,gpu=torch.cuda.get_device_name(0),
          scope='all21 fixed profiles; exact4QP; source-only candidate cache; no GT policy inputs'))
    rows=[]; clean=[]; reproducibility=None; actual_encodes=decoded_calls=logical_rows=0; started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    source_calls=controller_calls=0; timers=dict(source_detector_seconds=0.,a2_controller_seconds=0.,decoded_detector_seconds=0.,source_proxy_seconds=0.,repeat_detector_seconds=0.)
    for index,item in enumerate(sources):
        rows=[]; begin=time.perf_counter(); path=Path(item['path'])
        wire_archive=zipfile.ZipFile(args.out/f"image-{item['id']:012d}-packets.zip",'w',compression=zipfile.ZIP_STORED)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['file_sha256']: raise ValueError('DEV source hash drift')
        rgb,geometry=read_image(item,320)
        if hashlib.sha256(rgb.tobytes()).hexdigest()!=item['canonical_sha256']: raise ValueError('canonical source pixel drift')
        tick=time.perf_counter(); out,feature=od_forward(detector,tensor(rgb,'cuda'),(0,))
        source=prediction(out[0]); source_feature=feature.cpu()
        timers['source_detector_seconds']+=time.perf_counter()-tick; source_calls+=1
        maps=region_maps(rgb,[source]); pair=input_pair(rgb,maps); filtered={}; filtered_sha={}
        for action,profile in enumerate(PROFILES):
            filtered[action]=filter_clip(rgb,maps,profile)
            filtered_sha[action]=hashlib.sha256(filtered[action].tobytes()).hexdigest()
        clean.append(dict(image_id=item['id'],split=item['split'],source_predictions=source,
                          source_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                          preparation_seconds=time.perf_counter()-begin))
        for codec_name in cfg['codecs']:
            for qp in cfg['qps']:
                codec=codec_factory('od',codec_name,qp,cfg); codec.standard.ffmpeg=str(ROOT/cfg['ffmpeg_executable'])
                tick=time.perf_counter(); prediction_cnn=infer(model,pair,condition(codec_name,qp,[False,True,False],
                    {**cfg,'domain_condition':model.conditions==9},domain='od'))
                a2=shortlist(prediction_cnn,[False,True,False],cfg['damage_tolerance'],
                    calibration=domain_calibration(model.metadata['calibration'],'od'),
                    max_learned=cfg['source_shortlist'],extras=cfg['extra_profiles_by_task']['od'])
                timers['a2_controller_seconds']+=time.perf_counter()-tick; controller_calls+=1
                packet_cache={}; decoded_cache={}
                for action in range(len(PROFILES)):
                    start=time.perf_counter(); key=filtered_sha[action]; reused=key in packet_cache
                    if not reused:
                        packet_cache[key]=codec.roundtrip(filtered[action],source_shape=rgb.shape,task='od')
                    packet=packet_cache[key]; encode_times={} if reused else dict(codec.last_times)
                    pixel_sha=hashlib.sha256(packet.decoded.tobytes()).hexdigest(); decoded_reused=pixel_sha in decoded_cache
                    if not decoded_reused:
                        tick=time.perf_counter(); predicted,f=od_forward(detector,tensor(packet.decoded,'cuda'),(0,))
                        pred=prediction(predicted[0]); feature_cost=float(distance(f,source_feature))
                        timers['decoded_detector_seconds']+=time.perf_counter()-tick; tick=time.perf_counter()
                        current_cost=detection_distance([source],[pred])+.1*feature_cost
                        pseudo_ap=pseudo_damage(source,pred,threshold=protocol['pseudo_threshold'],topk=protocol['pseudo_max_targets'])
                        timers['source_proxy_seconds']+=time.perf_counter()-tick
                        decoded_cache[pixel_sha]=dict(prediction=pred,feature_cost=feature_cost,
                            current_cost=current_cost,pseudo_ap=pseudo_ap)
                    cached=decoded_cache[pixel_sha]
                    if index==0 and codec_name=='h264' and qp==30 and action==0:
                        repeat=codec.roundtrip(filtered[action],source_shape=rgb.shape,task='od')
                        repeat_times=dict(codec.last_times)
                        if repeat.packet_sha256!=packet.packet_sha256 or not np.array_equal(repeat.decoded,packet.decoded):
                            raise ValueError('repeated identity bytes/pixels differ')
                        tick=time.perf_counter(); again,_=od_forward(detector,tensor(repeat.decoded,'cuda'),(0,)); again=prediction(again[0])
                        timers['repeat_detector_seconds']+=time.perf_counter()-tick
                        for name in ('boxes','labels','scores'):
                            if not np.allclose(again[name],cached['prediction'][name],rtol=0,atol=1e-6):
                                raise ValueError('repeated detector output differs: '+name)
                        reproducibility=dict(codec='h264',qp=30,profile=0,image_id=item['id'],extra_actual_encodes=1,
                            extra_actual_decodes=1,repeat_codec_times=repeat_times,exact_packet_and_pixels=True,prediction_absolute_tolerance=1e-6)
                    wire_member=f'{codec_name}_{qp}_{action}.bin';wire_archive.writestr(wire_member,packet.data)
                    rows.append(dict(image_id=item['id'],split=item['split'],codec=codec_name,qp=qp,profile=action,wire_member=wire_member,
                        bytes=packet.coded_bytes,elementary_bytes=packet.elementary_bytes,metadata_bytes=packet.metadata_bytes,
                        bpp=packet.coded_bytes*8/(320*320),packet_sha256=packet.packet_sha256,decoded_sha256=pixel_sha,
                        filtered_sha256=key,packet_reused=reused,decoded_reused=decoded_reused,
                        a2_shortlist=a2,coco=coco_predictions(cached['prediction'],dict(item,image_id=item['id']),geometry),
                        prediction=cached['prediction'],feature_cost=cached['feature_cost'],current_cost=cached['current_cost'],
                        pseudo_ap=cached['pseudo_ap'],
                        seconds=time.perf_counter()-start,codec_times=encode_times))
        wire_archive.close()
        audit_grid(rows,[item['id']]); logical_rows+=len(rows)
        actual_encodes+=sum(not r['packet_reused'] for r in rows);decoded_calls+=sum(not r['decoded_reused'] for r in rows)
        # One image at a time keeps full-bank memory bounded.
        payload=[r for r in rows if r['image_id']==item['id']]
        (args.out/f"image-{item['id']:012d}.json.gz").write_bytes(gzip.compress(json.dumps(serial(payload),allow_nan=False).encode(),mtime=0))
        print(f'A8 OD worker{args.worker}: {index+1}/{len(sources)} images; {logical_rows} logical rows; {time.perf_counter()-started:.1f}s',flush=True)
    audit=dict(source_ids=expected,images=len(expected),logical_rows=logical_rows,complete_per_image_grid=True)
    if logical_rows!=len(expected)*168:raise ValueError('complete all21 image cohort required')
    write(args.out/'sources.json',clean)
    if state_hash(detector.model.state_dict())!='c4616e10536fb767f901700c621bb851a10259b6ed90f3af0e4596d5a6886533':
        raise ValueError('fresh evaluator state changed')
    counts=dict(source_detector_calls=source_calls,controller_calls=controller_calls,actual_encodes=actual_encodes+1,actual_decodes=actual_encodes+1,decoded_detector_calls=decoded_calls+1,independent_repeat_calls=1)
    write(args.out/'complete.json',dict(grid=audit,wall_seconds=time.perf_counter()-started,reproducibility=reproducibility,
        actual_work=counts,stage_seconds=timers,
        source_preparation_seconds=sum(r['preparation_seconds'] for r in clean),timing_scope='nested stage timers; source preparation includes source detector and filtering, do not add all fields',
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),actual_encodes=actual_encodes+1,
        decoded_inferences=decoded_calls+1,
        pseudo_loss_forwards=0,source_inferences=len(sources)))


if __name__=='__main__': main()

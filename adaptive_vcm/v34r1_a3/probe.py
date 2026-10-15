"""Kaggle-only exhaustive OD DEV measurements. No TEST image is inferred."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
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
from .core import audit_grid
from .pseudo import pseudo_damage

ROOT=Path(__file__).resolve().parents[2]


def write(path,obj):
    path.write_text(json.dumps(serial(obj),indent=2,allow_nan=False)+'\n',encoding='utf-8')


@torch.no_grad()
def losses(detector,rgb,targets):
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(303401)
        terms,_=od_forward(detector,tensor(rgb,'cuda'),(0,),targets)
    values={k:float(v) for k,v in terms.items()}
    if not all(np.isfinite(v) for v in values.values()): raise ValueError('nonfinite source pseudo loss')
    return values


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
    if manifest['source_id_protocol_sha256']!=hashlib.sha256((ROOT/'configs/v34r1-a3-od-probe.json').read_bytes()).hexdigest():
        raise ValueError('frozen source ID protocol changed')
    # Only DEV records enter this process. TEST has hashes only, outside detector access.
    sources=[r for i,r in enumerate(manifest['dev']) if i%2==args.worker]
    expected=[r['id'] for r in sources]
    if len(sources)!=64: raise ValueError('A3 worker must have exactly64 disjoint DEV images')
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
          scope='all21 fixed profiles; exact4QP; source-only proxies; no TEST scoring'))
    rows=[]; clean=[]; reproducibility=None; started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for index,item in enumerate(sources):
        begin=time.perf_counter(); path=Path(item['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['file_sha256']: raise ValueError('DEV source hash drift')
        rgb,geometry=read_image(item,320)
        if hashlib.sha256(rgb.tobytes()).hexdigest()!=item['canonical_sha256']: raise ValueError('canonical source pixel drift')
        out,feature=od_forward(detector,tensor(rgb,'cuda'),(0,))
        source=prediction(out[0]); source_feature=feature.cpu()
        targets=[]
        kept=np.flatnonzero(source['scores']>=protocol['pseudo_threshold'])
        kept=sorted(kept,key=lambda i:(-source['scores'][i],i))[:protocol['pseudo_max_targets']]
        targets.append(dict(boxes=torch.as_tensor(source['boxes'][kept],device='cuda',dtype=torch.float32).reshape(-1,4),
                            labels=torch.as_tensor(source['labels'][kept],device='cuda',dtype=torch.int64)))
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
                prediction_cnn=infer(model,pair,condition(codec_name,qp,[False,True,False],
                    {**cfg,'domain_condition':model.conditions==9},domain='od'))
                a2=shortlist(prediction_cnn,[False,True,False],cfg['damage_tolerance'],
                    calibration=domain_calibration(model.metadata['calibration'],'od'),
                    max_learned=cfg['source_shortlist'],extras=cfg['extra_profiles_by_task']['od'])
                packet_cache={}; decoded_cache={}
                for action in range(len(PROFILES)):
                    start=time.perf_counter(); key=filtered_sha[action]; reused=key in packet_cache
                    if not reused:
                        packet_cache[key]=codec.roundtrip(filtered[action],source_shape=rgb.shape,task='od')
                    packet=packet_cache[key]; encode_times={} if reused else dict(codec.last_times)
                    pixel_sha=hashlib.sha256(packet.decoded.tobytes()).hexdigest(); decoded_reused=pixel_sha in decoded_cache
                    if not decoded_reused:
                        predicted,f=od_forward(detector,tensor(packet.decoded,'cuda'),(0,))
                        pred=prediction(predicted[0]); feature_cost=float(distance(f,source_feature))
                        current_cost=detection_distance([source],[pred])+.1*feature_cost
                        pseudo_ap=pseudo_damage(source,pred,threshold=protocol['pseudo_threshold'],topk=protocol['pseudo_max_targets'])
                        loss=losses(detector,packet.decoded,targets)
                        decoded_cache[pixel_sha]=dict(prediction=pred,feature_cost=feature_cost,
                            current_cost=current_cost,pseudo_ap=pseudo_ap,pseudo_loss=sum(loss.values()),loss_components=loss)
                    cached=decoded_cache[pixel_sha]
                    if index==0 and codec_name=='h264' and qp==30 and action==0:
                        repeat=codec.roundtrip(filtered[action],source_shape=rgb.shape,task='od')
                        if repeat.packet_sha256!=packet.packet_sha256 or not np.array_equal(repeat.decoded,packet.decoded):
                            raise ValueError('repeated identity bytes/pixels differ')
                        again,_=od_forward(detector,tensor(repeat.decoded,'cuda'),(0,)); again=prediction(again[0])
                        for name in ('boxes','labels','scores'):
                            if not np.allclose(again[name],cached['prediction'][name],rtol=0,atol=1e-6):
                                raise ValueError('repeated detector output differs: '+name)
                        repeated_loss=losses(detector,repeat.decoded,targets)
                        if any(abs(repeated_loss[k]-cached['loss_components'][k])>1e-6 for k in repeated_loss):
                            raise ValueError('seeded pseudo detector loss is not repeatable')
                        reproducibility=dict(codec='h264',qp=30,profile=0,image_id=item['id'],extra_actual_encodes=1,
                            exact_packet_and_pixels=True,prediction_absolute_tolerance=1e-6,
                            max_loss_delta=max(abs(repeated_loss[k]-cached['loss_components'][k]) for k in repeated_loss),
                            note='common RNG seeds for sampled loss; image-dependent proposals are not fixed across different candidates')
                    rows.append(dict(image_id=item['id'],split=item['split'],codec=codec_name,qp=qp,profile=action,
                        bytes=packet.coded_bytes,elementary_bytes=packet.elementary_bytes,metadata_bytes=packet.metadata_bytes,
                        bpp=packet.coded_bytes*8/(320*320),packet_sha256=packet.packet_sha256,decoded_sha256=pixel_sha,
                        filtered_sha256=key,packet_reused=reused,decoded_reused=decoded_reused,
                        a2_shortlist=a2,coco=coco_predictions(cached['prediction'],dict(item,image_id=item['id']),geometry),
                        prediction=cached['prediction'],feature_cost=cached['feature_cost'],current_cost=cached['current_cost'],
                        pseudo_ap=cached['pseudo_ap'],pseudo_loss=cached['pseudo_loss'],loss_components=cached['loss_components'],
                        seconds=time.perf_counter()-start,codec_times=encode_times))
        # Gzip per-image checkpoints survive a later interrupted worker.
        payload=[r for r in rows if r['image_id']==item['id']]
        (args.out/f"image-{item['id']:012d}.json.gz").write_bytes(gzip.compress(json.dumps(serial(payload),allow_nan=False).encode(),mtime=0))
        print(f'A3 worker{args.worker}: {index+1}/{len(sources)} DEV images; {len(rows)} rows; {time.perf_counter()-started:.1f}s',flush=True)
    audit=audit_grid(rows,expected)
    (args.out/'rows.json.gz').write_bytes(gzip.compress(json.dumps(serial(rows),allow_nan=False).encode(),mtime=0))
    write(args.out/'sources.json',clean)
    if state_hash(detector.model.state_dict())!='c4616e10536fb767f901700c621bb851a10259b6ed90f3af0e4596d5a6886533':
        raise ValueError('pseudo-loss evaluation modified frozen detector state')
    write(args.out/'complete.json',dict(grid=audit,wall_seconds=time.perf_counter()-started,reproducibility=reproducibility,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),actual_encodes=sum(not r['packet_reused'] for r in rows)+1,
        decoded_inferences=sum(not r['decoded_reused'] for r in rows)+1,
        pseudo_loss_forwards=sum(not r['decoded_reused'] for r in rows)+1,source_inferences=len(sources)))


if __name__=='__main__': main()

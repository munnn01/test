"""One physical DEV video, all21 actions; each GPU owns one complete codec."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .contracts import audit_ot21
from ..analyzers import DetectionAnalyzer
from ..v32.tracking import read_sequence
from ..v32vc.checkpoint import state_hash
from ..v33ap.sources import serial
from ..v34.protocol import codec_factory
from ..v34.preprocessing import region_maps
from ..v34r1.preprocessing import PROFILES,filter_clip
from ..v34r1_a3.ot_cache import mot17_plan,detect

ROOT=Path(__file__).resolve().parents[2]
MODEL_SHA='c4616e10536fb767f901700c621bb851a10259b6ed90f3af0e4596d5a6886533'


def same_predictions(actual,expected):
    if len(actual)!=len(expected):raise ValueError('prediction frame count drift')
    error=0.
    for a,b in zip(actual,expected):
        for k in ('boxes','scores','labels'):
            x,y=np.asarray(a[k]),np.asarray(b[k])
            if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('prediction shape/finite drift')
            diff=float(np.max(np.abs(x-y))) if x.size else 0.;error=max(error,diff)
            if (k=='labels' and diff!=0) or diff>1e-6:raise ValueError('frozen detector prediction parity failed')
    return error


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--sequence',required=True)
    p.add_argument('--codec',choices=('h264','h265'),required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing') or not torch.cuda.is_available():raise ValueError('Kaggle GPU only')
    if args.out.exists():raise ValueError('immutable fresh worker output required')
    args.out.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(303401);torch.backends.cudnn.benchmark=False
    import cv2
    cv2.setNumThreads(1)
    cfg=json.loads((ROOT/'configs/v34r1-kaggle.json').read_text());protocol=json.loads((ROOT/'configs/v34r1-a6-ot21.json').read_text())
    declared={r['name']:r['split'] for r in protocol['pilot_sequences']};split=declared[args.sequence]
    manifest=json.loads((args.root/'source-manifest.json').read_text());frozen=next(r for r in manifest['sequences'] if r['name']==args.sequence)
    if manifest['gt_extracted'] or len(frozen['files'])!=65:raise ValueError('source64 without GT required')
    for member,sha in frozen['files'].items():
        if hashlib.sha256((args.root/args.sequence/member).read_bytes()).hexdigest()!=sha:raise ValueError('original source SHA drift')
    old_rows=[];old_source=None
    for w in (0,1):
        folder=ROOT/'results/v34r1_a3/ot-dev'/f'worker{w}'
        old_rows.extend(r for r in json.loads(gzip.decompress((folder/'rows.json.gz').read_bytes())) if r['sequence']==args.sequence and r['codec']==args.codec)
        s=json.loads(gzip.decompress((folder/'sources.json.gz').read_bytes()))
        if args.sequence in s:
            if old_source is not None:raise ValueError('overlapping old source')
            old_source=s[args.sequence]
    if old_source is None or not old_rows:raise ValueError('authenticated old source/candidate controls required')
    old={(r['start'],r['qp'],r['action']):r for r in old_rows}
    if len(old)!=len(old_rows):raise ValueError('duplicate old controls')
    plan=mot17_plan(args.root,args.sequence,64);rgb,geometry=read_sequence(plan,960)
    if hashlib.sha256(rgb.tobytes()).hexdigest()!=old_source['canonical_sha256'] or list(geometry)!=old_source['geometry']:raise ValueError('exact canonical source/geometry parity failed')
    detector=DetectionAnalyzer('resnet50','cuda');detector.model.transform.min_size=(960,);detector.model.transform.max_size=960
    detector.model.roi_heads.detections_per_img=300
    if state_hash(detector.model.state_dict())!=MODEL_SHA:raise ValueError('frozen detector state drift')
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats();clean=detect(detector,rgb)
    source_error=same_predictions(clean,old_source['source_predictions'])
    source=dict(split=split,frames=64,fps=int(plan['fps']),width=plan['width'],height=plan['height'],geometry=geometry,
        source_predictions=clean,canonical_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
    rows=[];actual_encodes=decoded_frames=0;old_checks=[];repeat_checks=[];filter_seconds=0.;model_seconds=0.;local_cfg=dict(cfg,fps=int(plan['fps']))
    for offset in range(0,64,16):
        clip=rgb[offset:offset+16];maps=region_maps(clip,clean[offset:offset+16]);filters={}
        for qp in protocol['qps']:
            codec=codec_factory('ot',args.codec,qp,local_cfg);codec.standard.ffmpeg=str(ROOT/cfg['ffmpeg_executable']);packets={};predictions={}
            old_actions=next(r['actions'] for r in old_rows if r['start']==offset+1 and r['qp']==qp)
            for action in protocol['profiles']:
                began=time.perf_counter()
                if action not in filters:
                    start=time.perf_counter();filters[action]=filter_clip(clip,maps,PROFILES[action]);filter_seconds+=time.perf_counter()-start
                pixels=filters[action];pixel_sha=hashlib.sha256(pixels.tobytes()).hexdigest();reused=pixel_sha in packets
                if not reused:packets[pixel_sha]=codec.roundtrip(pixels,source_shape=clip.shape,task='ot');actual_encodes+=1
                packet=packets[pixel_sha];decoded_sha=hashlib.sha256(packet.decoded.tobytes()).hexdigest();inferred=decoded_sha not in predictions
                if inferred:
                    start=time.perf_counter();predictions[decoded_sha]=detect(detector,packet.decoded);model_seconds+=time.perf_counter()-start;decoded_frames+=16
                row=dict(sequence=args.sequence,split=split,start=offset+1,codec=args.codec,qp=qp,action=action,actions=protocol['profiles'],old_actions=old_actions,
                    bytes=packet.coded_bytes,elementary_bytes=packet.elementary_bytes,metadata_bytes=packet.metadata_bytes,packet_sha256=packet.packet_sha256,
                    decoded_sha256=decoded_sha,packet_reused=reused,decoded_reused=not inferred,detections=predictions[decoded_sha],seconds=time.perf_counter()-began,
                    codec_times=dict(codec.last_times) if not reused else {})
                rows.append(row)
                if action in old_actions:
                    prior=old[(offset+1,qp,action)]
                    if any(row[k]!=prior[k] for k in ('bytes','elementary_bytes','metadata_bytes','packet_sha256','decoded_sha256')):raise ValueError('exact old candidate codec/paid bytes parity failed')
                    error=same_predictions(row['detections'],prior['detections']);old_checks.append(dict(start=offset+1,qp=qp,action=action,max_abs_error=error))
                if offset==0 and qp==30 and action==0:
                    again=codec.roundtrip(pixels,source_shape=clip.shape,task='ot');actual_encodes+=1
                    if again.packet_sha256!=packet.packet_sha256 or hashlib.sha256(again.decoded.tobytes()).hexdigest()!=decoded_sha or again.coded_bytes!=packet.coded_bytes:raise ValueError('independent anchor codec repeat failed')
                    start=time.perf_counter();again_pred=detect(detector,again.decoded);model_seconds+=time.perf_counter()-start;decoded_frames+=16
                    repeat_checks.append(dict(packet_identical=True,pixels_identical=True,max_abs_error=same_predictions(again_pred,row['detections'])))
            print('A6_OT_CACHE',args.sequence,args.codec,offset+1,qp,len(rows),round(time.perf_counter()-started,2),flush=True)
        subset=[r for r in rows if r['start']==offset+1]
        (args.out/f'window-{offset+1:06d}.json.gz').write_bytes(gzip.compress(json.dumps(serial(subset),allow_nan=False).encode(),mtime=0))
    audit_ot21(rows,{args.sequence:split},codec=args.codec)
    if len(old_checks)!=len(old_rows) or len(repeat_checks)!=1:raise ValueError('complete old control/repeat checks required')
    if state_hash(detector.model.state_dict())!=MODEL_SHA:raise ValueError('frozen detector mutated')
    (args.out/'rows.json.gz').write_bytes(gzip.compress(json.dumps(serial(rows),allow_nan=False).encode(),mtime=0))
    (args.out/'sources.json.gz').write_bytes(gzip.compress(json.dumps(serial({args.sequence:source}),allow_nan=False).encode(),mtime=0))
    report=dict(sequence=args.sequence,codec=args.codec,rows=len(rows),wall_seconds=time.perf_counter()-started,actual_encodes=actual_encodes,actual_decodes=actual_encodes,
        decoded_task_frames=decoded_frames,source_task_frames=64,source_unique_frames=64,source_prediction_max_abs_error=source_error,old_candidate_checks=old_checks,
        independent_anchor_repeat=repeat_checks,filter_seconds=filter_seconds,decoded_model_seconds=model_seconds,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        model_sha256=MODEL_SHA,gpu=torch.cuda.get_device_name(0),controller_calls=0,old_shortlist_reused_from_authenticated_parent=True,
        gt_extracted=False,tracker_calls=0,test_scored=False)
    (args.out/'complete.json').write_text(json.dumps(report,indent=2)+'\n');print('A6_OT_WORKER_COMPLETE',args.sequence,args.codec,flush=True)


if __name__=='__main__':main()

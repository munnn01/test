"""Kaggle-only frozen OT policies on new native-cadence256-frame videos."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
from ..analyzers import DetectionAnalyzer
from ..v32.tracking import load_trackeval,read_sequence,write_mot,stage_ground_truth
from ..v32vc.checkpoint import state_hash
from ..v33ap.sources import serial
from ..v34.protocol import codec_factory
from ..v34.preprocessing import region_maps
from ..v34r1.preprocessing import PROFILES,filter_clip
from ..v34r1.metrics import assess_curve
from ..v34r1_a3.tracking import CausalProxyGuard
from ..v34r1_a3.ot_cache import detect
from ..v34r1_a3.ot_analyze import official
from ..v34r1_a5.ot_choice import choose_lagrangian
from ..v34r1_a6.ot_cache import same_predictions
from ..v34r1_a7.shared import choose_shared
from .contracts import QPS,CODECS,shard,audit_selected
from .evidence import packet_file

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from prepare_v34r1_a8_sources import restore_ot
MODEL_SHA='c4616e10536fb767f901700c621bb851a10259b6ed90f3af0e4596d5a6886533'


def save(path,value):path.write_text(json.dumps(serial(value),indent=2,allow_nan=False)+'\n')


@torch.no_grad()
def one_video(record,frozen,cfg,baseline,out,package,detector):
    if out.exists():raise ValueError('immutable new-video output required')
    out.mkdir();source=ROOT/'data/v34r1_a8/ot'/record['name'];restored=restore_ot(record,cfg,source,expected=frozen)
    if (source/'gt/gt.txt').exists():raise ValueError('GPU sender source directory must not contain GT')
    native=frozen['native'];digits=6 if record['dataset']=='MOT17' else 8
    plan=dict(native,path=source,images=[source/'img1'/f'{i:0{digits}d}.jpg' for i in range(1,257)])
    choices=[];selected_rows=[];frames=defaultdict(list);rates=defaultdict(list);bank_rows=0;encodes=decodes=decoded_model_frames=0;repeats=[];window_hashes=[]
    states={(c,q):CausalProxyGuard(package.metrics.HOTA(),fps=native['fps'],lookback=32) for c in CODECS for q in QPS}
    anchors={(c,q):CausalProxyGuard(None,fps=native['fps'],lookback=32) for c in CODECS for q in QPS}
    started=time.perf_counter();bank=out/'bank';bank.mkdir();local_cfg=dict(baseline,fps=native['fps'])
    torch.cuda.reset_peak_memory_stats()
    for offset in range(0,256,16):
        window_plan=dict(plan,images=plan['images'][offset:offset+16],frames=16)
        rgb,geometry=read_sequence(window_plan,960);clean=detect(detector,rgb);maps=region_maps(rgb,clean);filters={};window_rows=[]
        window_hashes.append(dict(start=offset+1,canonical_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),geometry=list(geometry)))
        for c in CODECS:
            groups=[];all_trials=[];wires={}
            for q in QPS:
                state=states[(c,q)];state.begin(record['name'],clean);anchor=anchors[(c,q)];anchor.begin(record['name'],clean)
                codec=codec_factory('ot',c,q,local_cfg);codec.standard.ffmpeg=str(ROOT/baseline['ffmpeg_executable']);packets={};predictions={};group=[];trials=[]
                for action in range(21):
                    if action not in filters:filters[action]=filter_clip(rgb,maps,PROFILES[action])
                    pixels=filters[action];h=hashlib.sha256(pixels.tobytes()).hexdigest();reused=h in packets
                    if not reused:packets[h]=codec.roundtrip(pixels,source_shape=rgb.shape,task='ot');encodes+=1;decodes+=1
                    packet=packets[h];dh=hashlib.sha256(packet.decoded.tobytes()).hexdigest();inferred=dh not in predictions
                    if inferred:predictions[dh]=detect(detector,packet.decoded);decoded_model_frames+=16
                    trial=state.trial(predictions[dh]);trials.append(trial)
                    row=dict(id=record['name']+'/'+str(offset+1),sequence=record['name'],dataset=record['dataset'],start=offset+1,codec=c,qp=q,action=action,
                        bytes=packet.coded_bytes,elementary_bytes=packet.elementary_bytes,metadata_bytes=packet.metadata_bytes,packet_sha256=packet.packet_sha256,decoded_sha256=dh,
                        source_hota=trial.source_metrics,source_cost=trial.proxy_costs[0] if trial.proxy_costs is not None else None,packet_reused=reused,decoded_reused=not inferred)
                    if row['metadata_bytes']!=40 or row['bytes']!=row['elementary_bytes']+40:raise ValueError('paid OT header drift')
                    group.append(row);window_rows.append(row);bank_rows+=1
                    wires[(q,action)]=packet.data
                    if action==0:
                        anchor_trial=anchor.trial(predictions[dh]);anchor.commit(anchor_trial);frames[(c,q,'anchor')].extend(anchor_trial.frames)
                    if offset==0 and q==30 and action==0:
                        again=codec.roundtrip(pixels,source_shape=rgb.shape,task='ot');encodes+=1;decodes+=1
                        if again.packet_sha256!=packet.packet_sha256 or hashlib.sha256(again.decoded.tobytes()).hexdigest()!=dh or again.coded_bytes!=packet.coded_bytes:raise ValueError('independent OT anchor encode/decode failed')
                        repeat_pred=detect(detector,again.decoded);decoded_model_frames+=16
                        repeats.append(dict(codec=c,qp=q,packet_identical=True,pixels_identical=True,max_abs_error=same_predictions(repeat_pred,predictions[dh])))
                groups.append(group);all_trials.append(trials)
            if c=='h264':
                action,audit=choose_shared([[r['bytes']/g[0]['bytes'] for r in g] for g in groups],[[r['source_cost'] for r in g] for g in groups],8,list(range(21)))
                selected=[action]*4;audits=[audit]*4
            else:
                selected=[];audits=[]
                for g in groups:
                    i,audit=choose_lagrangian([r['bytes']/g[0]['bytes'] for r in g],[r['source_cost'] for r in g],32,list(range(21)));selected.append(i);audits.append(audit)
            for q,g,trials,action,audit in zip(QPS,groups,all_trials,selected,audits):
                state=states[(c,q)];state.commit(trials[action]);frames[(c,q,'adaptive')].extend(trials[action].frames)
                picked=g[action]
                if picked['bytes']>g[0]['bytes']:raise ValueError('paid<=identity cap failed')
                choices.append(dict(sequence=record['name'],start=offset+1,codec=c,qp=q,action=action,audit=audit,bytes=picked['bytes'],source_hota=picked['source_hota'],identity_source_hota=g[0]['source_hota']))
                for arm,row in [('anchor',g[0]),('adaptive',picked)]:
                    path=packet_file(out,wires[(q,row['action'])],row['packet_sha256'],row['bytes'])
                    selected_rows.append({k:row[k] for k in ('id','codec','qp','bytes','elementary_bytes','metadata_bytes','packet_sha256','decoded_sha256')}|dict(arm=arm,action=row['action'],packet_path=path,bpp=row['bytes']*8/(16*960*960)))
                    rates[(c,q,arm)].append(row['bytes']*8/(16*960*960))
        if len(window_rows)!=168:raise ValueError('complete all21×8 window bank required')
        (bank/f'window-{offset+1:06d}.json.gz').write_bytes(gzip.compress(json.dumps(serial(window_rows),allow_nan=False).encode(),mtime=0))
        print('A8_OT_WINDOW',record['name'],offset+1,bank_rows,round(time.perf_counter()-started,1),flush=True)
    ids=[record['name']+'/'+str(t) for t in range(1,257,16)];audit_selected(selected_rows,ids)
    if bank_rows!=2688 or len(repeats)!=2 or any(len(v)!=256 for v in frames.values()):raise ValueError('complete256-frame bank/trackers/repeats required')
    for (c,q,arm),values in frames.items():write_mot(out/'trackers'/f'{c}_{q}_{arm}'/'data'/f"{record['name']}.txt",values,geometry,native['width'],native['height'])
    decision=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),scope='all source-only256-frame choices and selected trackers before GT staging',choices=choices,selected_rows=selected_rows,window_sources=window_hashes)
    raw=(json.dumps(serial(decision),allow_nan=False)+'\n').encode();(out/'source-only-selections.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    # The GT accessor is reached only after every selected stream/track is sealed.
    restore_ot(record,cfg,source,expected=frozen,gt=True);gt=stage_ground_truth([plan],out/'ground_truth');save(out/'ground-truth-hashes.json',gt)
    metrics=official(package,[plan],out,[f'{c}_{q}_{arm}' for c in CODECS for q in QPS for arm in ('anchor','adaptive')])
    curves={c:{arm:dict(rate=[float(np.mean(rates[(c,q,arm)])) for q in QPS],quality=[metrics[f'{c}_{q}_{arm}']['combined']['HOTA'] for q in QPS]) for arm in ('anchor','adaptive')} for c in CODECS}
    report=dict(complete=True,sequence=record['name'],dataset=record['dataset'],native=native,curves=curves,metrics=metrics,
        comparisons={c:assess_curve(curves[c]['anchor'],curves[c]['adaptive'],qps=QPS) for c in CODECS},source_only_decisions_sha256=hashlib.sha256(raw).hexdigest(),
        source_frames=256,candidate_rows=bank_rows,source_detector_frames=256,actual_encodes=encodes,actual_decodes=decodes,decoded_detector_frames=decoded_model_frames,
        independent_anchor_repeats=repeats,model_sha256=MODEL_SHA,gt_scored=True,retuned=False,wall_seconds=time.perf_counter()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        h264_counterfactual_conditions_per_window_even_single_qp=84,h265_candidates_per_requested_qp=21)
    save(out/'evaluation-report.json',report);print('A8_OT_VIDEO_COMPLETE',record['name'],flush=True);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--worker',type=int,choices=(0,1),required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing') or not torch.cuda.is_available():raise ValueError('Kaggle GPU only')
    if a.out.exists():raise ValueError('fresh OT worker output required')
    a.out.mkdir();torch.set_num_threads(2);torch.manual_seed(303401);torch.backends.cudnn.benchmark=False
    import cv2
    cv2.setNumThreads(1)
    cfg=json.loads((ROOT/'configs/v34r1-a8-expansion.json').read_bytes());baseline=json.loads((ROOT/'configs/v34r1-kaggle.json').read_bytes());manifest=json.loads(a.manifest.read_bytes())
    records=shard(shard(cfg['ot']['records'],a.shard,4),a.worker,2);frozen={r['name']:r for r in manifest['ot']}
    detector=DetectionAnalyzer('resnet50','cuda');detector.model.transform.min_size=(960,);detector.model.transform.max_size=960;detector.model.roi_heads.detections_per_img=300
    if state_hash(detector.model.state_dict())!=MODEL_SHA:raise ValueError('frozen OT evaluator state drift')
    package=load_trackeval(ROOT/'data/TrackEval');reports=[]
    for record in records:reports.append(one_video(record,frozen[record['name']],cfg,baseline,a.out/record['name'],package,detector))
    if state_hash(detector.model.state_dict())!=MODEL_SHA:raise ValueError('frozen OT model mutated')
    save(a.out/'complete.json',dict(complete=True,physical_videos=[r['sequence'] for r in reports],frames=sum(r['source_frames'] for r in reports),gpu=torch.cuda.get_device_name(0),model_sha256=MODEL_SHA))


if __name__=='__main__':main()

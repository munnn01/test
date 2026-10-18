"""Four-QP real-codec evaluation with frozen controllers and label-free sender selection."""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import copy
import hashlib
import json
import os
import platform
import importlib.metadata
from pathlib import Path
import time
import numpy as np
import torch
from torchvision.models.video import Swin3D_T_Weights

from ..data import canon,read_image
from ..v33ap.sources import SourceRecord,read_source,serial
from ..v34.protocol import codec_factory,digest
from ..v34.preprocessing import region_maps
from ..v34.runner import input_pair,condition,infer,write_json
from ..v34.checkpoint import load_checkpoint
from ..v34.learning import domain_calibration
from ..v34.evaluation import choose_packet,coco_predictions
from ..v34.tracking import ByteTracker,track_frames
from ..v32.tracking import plan_sequences,read_sequence,write_mot,stage_ground_truth,load_trackeval,evaluate_mot
from ..coco_metrics import coco_map
from . import QPS
from .preprocessing import PROFILES,filter_clip
from .policy import shortlist,select_measured
from .tasks import Heads
from .tracking import CausalTrackingGuard
from .metrics import assess_curve,paired_ar_ci
from .audit import audit_grid,verify_source

ROOT=Path(__file__).resolve().parents[2]
ARMS=('anchor','baseline','guard_legacy','adaptive')


def load_config(path):
    cfg=json.loads(Path(path).read_text(encoding='utf-8'))
    if (cfg.get('schema')!='adaptive-vcm-v34r1-1' or tuple(cfg.get('qps',()))!=QPS
        or cfg.get('codecs')!=['h264','h265'] or cfg.get('target_bd_rate_pct')!=-15.):
        raise ValueError('prospective V34r1 four-QP/six-endpoint protocol drift')
    all_extras=cfg['extra_profiles']+[i for v in cfg.get('extra_profiles_by_task',{}).values() for i in v]
    if any(type(i) is not int or not 13<=i<len(PROFILES) for i in all_extras):
        raise ValueError('unknown unlearned filter profile')
    if (cfg.get('guard_tasks','all') not in ('target','all') or cfg.get('arms',list(ARMS)) not in
        [list(ARMS),['anchor','adaptive']]): raise ValueError('unknown guard or arm protocol')
    return cfg


def records(manifest,task,split,cfg):
    items=[row for row in manifest[task] if row['split']==split]
    if not items: raise ValueError(f'no declared {task}/{split} records')
    if task=='ar':
        mapping={canon(name):i for i,name in enumerate(Swin3D_T_Weights.KINETICS400_V1.meta['categories'])}
        for item in items:
            verify_source(item,required=split=='test')
            if item.get('kind')=='frames':
                if item.get('provider_selected_paths'):
                    from .frames import read_selected
                    rgb,recipe=read_selected(item,frames=cfg['frames'],size=cfg['ar_size'])
                else:
                    from ..v34.holdout import read_frames
                    rgb,recipe=read_frames(Path(item['path']),frames=cfg['frames'],stride=cfg['temporal_stride'],size=cfg['ar_size'])
                geometry=None; label=int(item['label'])
            else:
                rgb,geometry,recipe=read_source('ar',item,cfg)
                label=mapping[canon(item['category'])]
                if recipe['repeated_frames']: raise ValueError('short source clip; TEST padding/replacement prohibited')
            clean_item={k:v for k,v in item.items() if k not in ('label','targets','annotations')}
            record=SourceRecord('ar/'+str(item['id']),'ar',clean_item,rgb,geometry,recipe)
            record.gt_label=label
            yield record
    elif task=='od':
        for item in items:
            verify_source(item,required=split=='test')
            clean_item={**item,'id':str(item['id']),'image_id':int(item['id'])}
            rgb,geometry=read_image(clean_item,cfg['od_size'])
            yield SourceRecord('od/coco2017/'+str(item['id']),'od',clean_item,rgb,geometry,dict(kind='single-image',repeated_frames=0))
    else:
        for item in items:
            verify_source(item,required=split=='test')
            plans=plan_sequences(Path(item['path']).parent,[item['sequence']],item['frames'])
            plan=plans[0]; rgb,geometry=read_sequence(plan,cfg['ot_size'])
            for offset in range(0,len(rgb),cfg['frames']):
                clip=rgb[offset:offset+cfg['frames']]
                if len(clip)!=cfg['frames']: raise ValueError('incomplete tracking window')
                source=dict(sequence=item['sequence'],start=offset+1,width=plan['width'],height=plan['height'])
                record=SourceRecord(f"ot/{item['sequence']}/{offset+1:06d}",'ot',source,clip.copy(),geometry,
                                    dict(kind='contiguous-MOT20',repeated_frames=0,frame_indices=list(range(offset+1,offset+1+len(clip)))))
                record.tracking_plan=plan
                yield record


def source_cost(heads,record,decoded,tracking_state,cache,guard_mask=None):
    sha=hashlib.sha256(decoded.tobytes()).hexdigest()
    if sha not in cache:
        if guard_mask is not None:
            record=copy(record); record.mask=list(guard_mask)
        if tracking_state is None:
            value=heads.costs(decoded,record,source_only=True); trial=None; detections=None
        else:
            other=copy(record); other.mask=list(record.mask); other.mask[2]=False
            value=heads.costs(decoded,other,source_only=True)
            detections=heads.detections(decoded)
            trial=tracking_state.trial(detections)
            value[2]=trial.cost
        cache[sha]=(value,trial,detections)
    return cache[sha]


def ar_receiver_result(probabilities,record,*,defer=False):
    label=int(np.argmax(probabilities))
    return dict(predicted_label=label) if defer else dict(correct=int(label==record.gt_label))


def evaluate_condition(record,model,heads,maps,pair,codec_name,qp,cfg,state=None,filtered_cache=None,legacy_state=None,*,defer_ar_labels=False,stage_callback=None,packet_sink=None):
    condition_start=time.perf_counter()
    codec=codec_factory(record.task,codec_name,qp,cfg)
    executable=Path(cfg['ffmpeg_executable'])
    if not executable.is_absolute(): executable=ROOT/executable
    if not executable.is_file(): raise FileNotFoundError('pinned FFmpeg executable missing: '+str(executable))
    codec.standard.ffmpeg=str(executable)
    packets={}; timings={}; costs={}
    if filtered_cache is None: filtered_cache={}
    def encode(action):
        if action not in packets:
            start=time.perf_counter()
            reused=action in filtered_cache
            if not reused: filtered_cache[action]=filter_clip(record.rgb,maps,PROFILES[action])
            filtered=filtered_cache[action]
            filter_seconds=time.perf_counter()-start
            packets[action]=codec.roundtrip(filtered,source_shape=record.rgb.shape,task=record.task)
            timings[action]=dict(filter_seconds=filter_seconds,filter_reused=reused,**codec.last_times)
        return packets[action]
    anchor=encode(0)
    local_cfg={**cfg,'domain_condition':model.conditions==9}
    if stage_callback:stage_callback('controller')
    prediction=infer(model,pair,condition(codec_name,qp,record.mask,local_cfg,domain=record.task))
    forced=int(np.argmax(prediction['logits']))
    arms_requested=cfg.get('arms',list(ARMS))
    if 'baseline' not in arms_requested:
        baseline_action=0; baseline=anchor; baseline_details={}
    elif record.task=='od':
        baseline_action=forced; baseline=encode(forced); baseline_details={'tried_actions':[0,forced], 'policy':'frozen primary forced OD controller'}
    else:
        _,baseline_action,baseline,baseline_details=choose_packet(model,pair,codec_name,qp,record.mask,local_cfg,anchor,encode,
            domain=record.task,source_score=lambda rgb:heads.costs(rgb,record,source_only=True))
    if state is not None: state.begin(record.item['sequence'],record.source_predictions['od'])
    if legacy_state is not None: legacy_state.begin(record.item['sequence'],record.source_predictions['od'])
    guard_mask=record.mask if cfg.get('guard_tasks','all')=='all' else [record.task==t for t in ('ar','od','ot')]
    extras=cfg.get('extra_profiles_by_task',{}).get(record.task,cfg['extra_profiles'])
    candidates=shortlist(prediction,record.mask,cfg['damage_tolerance'],
        calibration=domain_calibration(model.metadata['calibration'],record.task),
        max_learned=cfg['source_shortlist'],extras=extras)
    candidate_packets=[encode(a) for a in candidates]
    rate=[p.coded_bytes/anchor.coded_bytes for p in candidate_packets]
    values=[]
    if stage_callback:stage_callback('candidate_scoring')
    for packet,ratio in zip(candidate_packets,rate):
        values.append(source_cost(heads,record,packet.decoded,state,costs,guard_mask)[0] if ratio<=cfg['measured_budget_ratio'] else [None]*3)
    selected,guard=select_measured(rate,values,guard_mask,cfg['source_guard_limits'],
                                 budget=cfg['measured_budget_ratio'],quality_weight=cfg['quality_weight'])
    legacy_indices=[i for i,a in enumerate(candidates) if a<13]
    legacy_values=[values[i] for i in legacy_indices]
    legacy_trials={}
    if legacy_state is not None:
        legacy_values=[]
        for i in legacy_indices:
            value=list(values[i])
            if rate[i]<=cfg['measured_budget_ratio']:
                sha=hashlib.sha256(candidate_packets[i].decoded.tobytes()).hexdigest()
                _,_,detections=costs[sha]
                trial=legacy_state.trial(detections)
                value[2]=trial.cost; legacy_trials[i]=trial
            legacy_values.append(value)
    legacy_choice,legacy_guard=select_measured([rate[i] for i in legacy_indices],legacy_values,
        guard_mask,cfg['source_guard_limits'],budget=cfg['measured_budget_ratio'],quality_weight=cfg['quality_weight'])
    legacy_selected=legacy_indices[legacy_choice]
    if state is not None:
        _,trial,_=source_cost(heads,record,candidate_packets[selected].decoded,state,costs,guard_mask)
        state.commit(trial)
    if legacy_state is not None: legacy_state.commit(legacy_trials[legacy_selected])
    details=dict(tried_actions=candidates,source_guard=guard,unlearned_extra_profiles=extras,
                 selection_policy='feasible learned proposals + fixed multiscale trials + source-only measured guard',
                 filter_profile=PROFILES[candidates[selected]].name)
    sender_seconds=time.perf_counter()-condition_start
    arms={'anchor':(0,anchor,{}),'baseline':(baseline_action,baseline,baseline_details),
          'guard_legacy':(candidates[legacy_selected],candidate_packets[legacy_selected],{'source_guard':legacy_guard}),
          'adaptive':(candidates[selected],candidate_packets[selected],details)}
    predictions={}; rows=[]
    if stage_callback:stage_callback('receiver')
    for arm,(action,packet,extra) in arms.items():
        if arm not in arms_requested: continue
        sha=packet.packet_sha256
        if sha not in predictions:
            if record.task=='ar': predictions[sha]=heads.action.probabilities(packet.decoded)
            elif record.task=='od': predictions[sha]=heads.detector.predict(packet.decoded.copy())
            else:
                pixel_sha=hashlib.sha256(packet.decoded.tobytes()).hexdigest()
                cached=costs.get(pixel_sha)
                predictions[sha]=cached[2] if cached and cached[2] is not None else heads.detections(packet.decoded)
        pred=predictions[sha]
        packet_path=packet_sink(packet) if packet_sink else None
        row=dict(id=record.id,task=record.task,codec=codec_name,qp=qp,arm=arm,action=action,
                 bytes=packet.coded_bytes,elementary_bytes=packet.elementary_bytes,metadata_bytes=packet.metadata_bytes,
                 bpp=packet.coded_bytes*8/np.prod(record.rgb.shape[:3]),packet_sha256=sha,
                 decoded_sha256=hashlib.sha256(packet.decoded.tobytes()).hexdigest(),**extra)
        if record.task=='ar': row.update(ar_receiver_result(pred,record,defer=defer_ar_labels))
        elif record.task=='od': row.update(image_id=record.item['image_id'],coco=coco_predictions(pred,record.item,record.geometry))
        else: row['_detections']=pred
        if packet_path is not None:row['packet_path']=packet_path
        if arm=='adaptive':
            row.update(candidate_encode_count=len(packets),candidate_search_encode_count=len(candidates),
                       candidate_work=[dict(action=a,**timings[a]) for a in packets],
                       sender_selection_seconds=sender_seconds)
        rows.append(row)
    for row in rows:
        if row['arm']=='adaptive':
            row.update(condition_total_seconds=time.perf_counter()-condition_start,
                       receiver_all_arms_seconds=time.perf_counter()-condition_start-sender_seconds)
    return rows


def summarize(rows,task,manifest,cfg,out,plans,tracks,trackeval,expected_ids=None):
    requested_arms=cfg.get('arms',list(ARMS))
    audit=audit_grid(rows,source_ids=expected_ids or sorted({r['id'] for r in rows}),arms=requested_arms,codecs=cfg['codecs'])
    curves={codec:{arm:{'rate':[],'quality':[]} for arm in requested_arms} for codec in cfg['codecs']}
    if task=='od':
        meta=json.loads((Path(manifest['_root'])/'coco/annotations/instances_val2017.json').read_text(encoding='utf-8'))
        gt=defaultdict(list)
        for annotation in meta['annotations']: gt[annotation['image_id']].append(annotation)
    official={}
    if task=='ot':
        base=out/'tracking'; base.mkdir()
        for key,frames in tracks.items():
            codec,qp,arm,sequence=key
            record=plans[sequence]
            write_mot(base/'trackers'/f'{codec}_{qp}_{arm}'/'data'/f'{sequence}.txt',frames,record['geometry'],record['width'],record['height'])
        # DEV --limit may stop within a sequence. Score exactly the observed
        # prefix, never the full declared sequence against a truncated tracker.
        observed={sequence:len(frames) for (_,_,_,sequence),frames in tracks.items()}
        plan=[dict(r['plan'],frames=observed[name]) for name,r in plans.items()]
        stage_ground_truth(plan,base/'ground_truth')
        names=sorted({f'{c}_{q}_{a}' for c,q,a,_ in tracks})
        official=evaluate_mot(load_trackeval(trackeval),plan,base,names)
    for codec in cfg['codecs']:
        for arm in requested_arms:
            for qp in cfg['qps']:
                subset=[r for r in rows if r['codec']==codec and r['qp']==qp and r['arm']==arm]
                if not subset: raise ValueError('incomplete paired evaluation grid')
                if task=='ar': quality=float(np.mean([r['correct'] for r in subset]))
                elif task=='od':
                    ids=[r['image_id'] for r in subset]
                    quality=coco_map([p for r in subset for p in r['coco']],gt,ids,meta)[0]
                else: quality=official[f'{codec}_{qp}_{arm}']['combined']['HOTA']/100
                curve=curves[codec][arm]
                curve['rate'].append(float(np.mean([r['bpp'] for r in subset])))
                curve['quality'].append(quality)
    comparisons={codec:{arm:assess_curve(curves[codec]['anchor'],curves[codec][arm],qps=cfg['qps']) for arm in requested_arms if arm!='anchor'} for codec in cfg['codecs']}
    if task=='ar' and cfg.get('ar_bootstrap_draws',0):
        for codec in cfg['codecs']:
            ci=paired_ar_ci(rows,codec,draws=cfg['ar_bootstrap_draws'],seed=cfg['seed'])
            comparisons[codec]['adaptive']=assess_curve(curves[codec]['anchor'],curves[codec]['adaptive'],qps=cfg['qps'],ci=ci)
    return dict(task=task,grid_audit=audit,curves=curves,comparisons=comparisons,official_tracking_metrics=official,
                scope='measured finite-sample task metrics; no confidence claim without independent-unit bootstrap')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/v34r1.json')
    p.add_argument('--task',choices=['ar','od','ot'],required=True)
    p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int,help='DEV smoke sources only; prohibited on TEST')
    p.add_argument('--trackeval',type=Path,default=ROOT/'data/TrackEval')
    args=p.parse_args()
    if args.limit is not None and (args.split=='test' or args.limit<1): raise ValueError('TEST truncation prohibited')
    if args.out.exists() and any(args.out.iterdir()): raise ValueError('immutable empty output directory required')
    cfg=load_config(args.config)
    manifest=json.loads(args.manifest.read_text(encoding='utf-8')); manifest['_root']=str(args.manifest.resolve().parent)
    if manifest.get('qps')!=list(QPS): raise ValueError('manifest four-QP grid drift')
    if args.task=='od' and args.split=='test':
        verify_source(dict(path=str(args.manifest.parent/'coco/annotations/instances_val2017.json'),
                           file_sha256=manifest.get('coco_annotations_sha256')))
    declared=[r for r in manifest[args.task] if r['split']==args.split]
    expected_ids=[]
    for item in declared:
        if args.task=='ot':
            if item['frames']%cfg['frames']: raise ValueError('incomplete declared OT window')
            expected_ids.extend(f"ot/{item['sequence']}/{i+1:06d}" for i in range(0,item['frames'],cfg['frames']))
        else: expected_ids.append(('ar/' if args.task=='ar' else 'od/coco2017/')+str(item['id']))
    if args.limit is not None: expected_ids=expected_ids[:args.limit]
    os.environ.setdefault('TORCH_HOME',str(ROOT/'data/torch'))
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint=ROOT/('checkpoints/v34_od_coco' if args.task=='od' else 'checkpoints/v34_r1')/f"controller_{cfg['seed']}.pth"
    model=load_checkpoint(checkpoint,device)
    if args.split=='test' and model.checkpoint_sha256!=cfg.get('controller_sha256_by_task',{}).get(args.task):
        raise ValueError('TEST requires a frozen task-specific controller hash')
    heads=Heads(device,'ar_od' if args.task=='od' else 'ar_ot',cfg)
    runtime_hashes={str(path.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted((ROOT/'adaptive_vcm').rglob('*.py'))}
    write_json(args.out/'protocol.json',dict(config=cfg,config_sha256=digest(cfg),checkpoint_sha256=model.checkpoint_sha256,
        task_models=heads.hashes(),manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        ffmpeg_sha256=hashlib.sha256((ROOT/cfg['ffmpeg_executable']).read_bytes()).hexdigest(),
        runtime_sha256=runtime_hashes,expected_source_ids=expected_ids,
        environment=dict(python=platform.python_version(),platform=platform.system(),
            packages={name:importlib.metadata.version(name) for name in ('torch','torchvision','numpy','scipy','opencv-python-headless','supervision','pycocotools')},
            gpu=torch.cuda.get_device_name() if device=='cuda' else None),
        split=args.split,task=args.task,limit=args.limit,device=device,profile_family=[serial(p.__dict__) for p in PROFILES]))
    rows=[]; states={}; trackers={}; tracks=defaultdict(list); plans={}
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    for index,record in enumerate(records(manifest,args.task,args.split,cfg)):
        if args.limit is not None and index>=args.limit: break
        source_start=time.perf_counter(); heads.source(record)
        maps=region_maps(record.rgb,record.source_predictions['od']); pair=input_pair(record.rgb,maps)
        filtered_cache={}
        source_seconds=time.perf_counter()-source_start
        for codec in cfg['codecs']:
            for qp in cfg['qps']:
                state=legacy_state=None
                if args.task=='ot':
                    state=states.setdefault((codec,qp,'adaptive'),CausalTrackingGuard())
                    if 'guard_legacy' in cfg.get('arms',list(ARMS)):
                        legacy_state=states.setdefault((codec,qp,'guard_legacy'),CausalTrackingGuard())
                condition_rows=evaluate_condition(record,model,heads,maps,pair,codec,qp,cfg,state,filtered_cache,legacy_state)
                for row in condition_rows:
                    if row['arm']=='adaptive': row['source_preparation_seconds_shared_across_8_conditions']=source_seconds
                if args.task=='ot':
                    sequence=record.item['sequence']
                    plans[sequence]=dict(plan=record.tracking_plan,geometry=record.geometry,width=record.item['width'],height=record.item['height'])
                    for row in condition_rows:
                        key=(codec,qp,row['arm'],sequence)
                        if key not in trackers: trackers[key]=ByteTracker()
                        tracks[key].extend(track_frames(row.pop('_detections'),trackers[key]))
                rows.extend(condition_rows)
        write_json(args.out/'rows.partial.json',rows)
        print(f'{args.task.upper()} {args.split}: {index+1} sources/windows, {len(rows)} paired rows, {time.perf_counter()-start:.1f}s',flush=True)
    summary=summarize(rows,args.task,manifest,cfg,args.out,plans,tracks,args.trackeval,expected_ids)
    summary['runtime']=dict(wall_seconds=time.perf_counter()-start,
        peak_torch_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
        timing_scope='includes sender source preparation, proposal/guard, all actual codec trials and receiver task scoring; model/data load excluded')
    write_json(args.out/'rows.json',rows); write_json(args.out/'summary.json',summary)
    print(json.dumps(serial(summary['comparisons']),indent=2),flush=True)


if __name__=='__main__': main()

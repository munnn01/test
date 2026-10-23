"""TRAIN codec labels, balanced replay optimization and paired identity/warm-start pilot."""
import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .quality import (capture_state, restore_state, guarded_step, identity_start,
                      quality_feasible, real_measure, protected_objective, semantic_weights)
from .screen import regularizers, task_terms, tensor_digest
from .engine import save_checkpoint, load_checkpoint, source_hashes
from .models import real_value_proxy_grad


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def log_row(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')


def case_seed(case, cfg):
    index, codec, qp = case
    return cfg['seed'] + index*8 + cfg['codecs'].index(codec)*4 + cfg['qps'].index(qp)


def balanced_schedule(records, codecs, qps, seed):
    """One epoch visits every source/point exactly once in balanced distinct-source batches."""
    families = [[i for i,r in enumerate(records) if r['task']==task] for task in ('ar','od')]
    if len(families[0]) != len(families[1]) or not families[0] or len(families[0])%2:
        raise ValueError('equal positive even source counts for AR and OD required')
    rng = random.Random(seed)
    grid = list(itertools.product(codecs,qps))
    points = {}
    for i in sum(families,[]):
        points[i] = rng.sample(grid,len(grid))
    batches = []
    for epoch in range(len(grid)):
        left,right = [rng.sample(f,len(f)) for f in families]
        for offset in range(0,len(left),2):
            indices = left[offset:offset+2]+right[offset:offset+2]
            batches.append([(i,*points[i][epoch]) for i in indices])
    return batches


@torch.no_grad()
def motion_reference(x):
    """Past-only backward optical flow, with photometric confidence for disocclusion."""
    import cv2
    result = x.clone()
    b,t,_,h,w = x.shape
    yy,xx = torch.meshgrid(torch.arange(h,device=x.device),torch.arange(w,device=x.device),indexing='ij')
    for batch in range(b):
        for frame in range(1,t):
            pair = (x[batch,frame-1:frame+1].permute(0,2,3,1).cpu().numpy()*255).round().astype(np.uint8)
            previous,current = [cv2.cvtColor(p,cv2.COLOR_RGB2GRAY) for p in pair]
            flow = cv2.calcOpticalFlowFarneback(current,previous,None,.5,2,15,2,5,1.1,0)
            flow = torch.from_numpy(flow).to(x)
            gx,gy = xx+flow[...,0], yy+flow[...,1]
            valid = ((gx>=0)&(gx<=w-1)&(gy>=0)&(gy<=h-1)).to(x)
            grid = torch.stack((2*gx/max(w-1,1)-1,2*gy/max(h-1,1)-1),-1).unsqueeze(0)
            warped = F.grid_sample(x[batch:batch+1,frame-1],grid,mode='bilinear',padding_mode='border',align_corners=True)[0]
            confidence = torch.exp(-(warped-x[batch,frame]).abs().mean(0,keepdim=True)/.05)*valid
            result[batch,frame] = x[batch,frame]+confidence*(warped-x[batch,frame])
    return result


@torch.no_grad()
def candidate_bank(x, protection, max_rms):
    """Seven TRAIN operators, causal conditional on fixed privileged offline protection maps."""
    if not 0 < max_rms <= 8/255 or protection.shape != x[:,:,:1].shape:
        raise ValueError('invalid candidate RMS or protection geometry')
    b,t,c,h,w = x.shape
    flat = x.reshape(b*t,c,h,w)
    low = F.avg_pool2d(F.pad(flat,(2,2,2,2),mode='reflect'),5,stride=1).reshape_as(x)
    important = protection.clamp(0,1)
    background = 1-important
    high = x-low
    changes = {'identity':torch.zeros_like(x),
               'background_weak':-.05*background*high,
               'background_medium':-.1*background*high,
               'background_strong':-.2*background*high,
               'dual_region':-(.2*background+.02*important)*high,
               'temporal':.1*background*(motion_reference(x)-x),
               'detail':.03*important*high-.1*background*high}
    result = {}
    for name,delta in changes.items():
        rms = delta.square().mean((2,3,4),keepdim=True).sqrt()
        delta = delta*(max_rms/rms.clamp_min(1e-12)).clamp(max=1)
        result[name] = (x+delta).clamp(0,1)
    return result


def select_candidate(rows, baseline, allowance, max_rms):
    valid = [r for r in rows if math.isfinite(r['bpp']) and r['bpp']>0
             and math.isfinite(r['total_rms']) and r['total_rms'] <= max_rms+1e-8
             and quality_feasible(r,baseline,allowance)]
    if not any(r['candidate']=='identity' for r in valid):
        raise ValueError('feasible identity candidate required')
    # Identity wins ties: never learn a pointless alteration of the input.
    return min(valid,key=lambda r:(r['bpp'],r['candidate']!='identity'))


class ReplayQueue:
    def __init__(self, capacity):
        if capacity < 1: raise ValueError('positive replay capacity required')
        self.capacity,self.cases,self.cursor = capacity,[],0

    def add(self, case):
        case = tuple(case)
        if case in self.cases: self.cases.remove(case)
        self.cases.append(case)
        self.cases = self.cases[-self.capacity:]

    def take(self, count, exclude=()):
        eligible = [c for c in self.cases if c not in exclude]
        if not eligible: return []
        selected = [eligible[(self.cursor+j)%len(eligible)] for j in range(min(count,len(eligible)))]
        self.cursor += len(selected)
        return selected


class ShortTransaction:
    """Only a full-panel pass commits a block; rollback restores Adam as well."""
    def __init__(self, model, optimizer, interval=16, minimum=4):
        if not 1 <= minimum <= interval: raise ValueError('invalid transaction interval')
        self.model,self.optimizer = model,optimizer
        self.interval,self.minimum = interval,minimum
        self.state = capture_state(model,optimizer)
        self.pending,self.attempts = 0,0
        self.counts = dict(proposals=0,accepted_proposals=0,rejected_proposals=0,
                           committed_proposals=0,rolled_back_proposals=0,rolled_back_blocks=0)

    def note(self, accepted):
        self.counts['proposals'] += 1
        self.counts['accepted_proposals' if accepted else 'rejected_proposals'] += 1
        self.pending += int(accepted)
        self.attempts += 1

    def finish(self, passed):
        if passed:
            self.counts['committed_proposals'] += self.pending
        else:
            restore_state(self.model,self.optimizer,self.state)
            self.counts['rolled_back_proposals'] += self.pending
            self.counts['rolled_back_blocks'] += 1
            self.interval = max(self.minimum,self.interval//2)
        self.pending,self.attempts = 0,0
        self.state = capture_state(self.model,self.optimizer)
        return dict(self.counts)


class RecoveryContext:
    """Only TRAIN sources are admitted; DEV is owned by the final evaluator."""
    def __init__(self, samples, refs, masks, tasks, codec, cfg):
        if not samples or len(samples)!=len(refs) or len(samples)!=len(masks):
            raise ValueError('aligned nonempty TRAIN samples/references/masks required')
        if len({r['id'] for r,_,_ in samples}) != len(samples):
            raise ValueError('duplicate TRAIN sources')
        self.samples,self.refs,self.masks = samples,refs,masks
        self.tasks,self.codec,self.cfg = tasks,codec,cfg
        self.device = torch.device(getattr(tasks,'device',samples[0][1].device))
        self.baselines = {}

    def x(self, case):
        return self.samples[case[0]][1].to(self.device)

    def protection(self, index):
        return ((self.masks[index].to(self.device)-1)/4).clamp(0,1)

    def seed(self, case):
        return case_seed(case,self.cfg)

    def all_cases(self):
        return [(i,c,q) for i in range(len(self.samples))
                for c,q in itertools.product(self.cfg['codecs'],self.cfg['qps'])]

    def anchors(self):
        chosen = []
        for task in ('ar','od'):
            chosen.extend([i for i,s in enumerate(self.samples) if s[0]['task']==task][:self.cfg['anchor_per_task']])
        return [(i,c,q) for i in chosen for c,q in itertools.product(self.cfg['codecs'],self.cfg['qps'])]

    @torch.no_grad()
    def measure(self, y, case):
        index,codec,qp = case
        row,packet = real_measure(self.tasks,self.codec,y,self.samples[index][0],self.refs[index],
                                  codec,qp,self.seed(case))
        row.update(bytes=len(packet.streams[0]),
                   stream_sha256=hashlib.sha256(packet.streams[0]).hexdigest(),
                   codec_seconds=packet.seconds)
        return row

    @torch.no_grad()
    def check(self, npp, cases, previous=None, full=False):
        rows,violations = [],[]
        for case in cases:
            index,codec,qp = case
            x = self.x(case)
            y = npp(x,codec,qp)
            rms = float((y-x).square().mean().sqrt())
            step_rms = float((y-previous[case]).square().mean().sqrt()) if previous is not None else 0.
            bounds = (math.isfinite(rms) and math.isfinite(step_rms) and rms<=self.cfg['max_total_rms']
                      and (previous is None or step_rms<=self.cfg['max_npp_update_rms']))
            row = {'case':case,'source':self.samples[index][0]['id'],'total_rms':rms,
                   'step_rms':step_rms,'bounds_passed':bounds}
            if bounds:
                measurement = self.measure(y,case)
                row.update(measurement=measurement,
                           rate_ratio=measurement['bpp']/self.baselines[case]['bpp'],
                           changed_rgb8=bool(((y*255).round()!=(x*255).round()).any()))
                row['quality_passed'] = quality_feasible(measurement,self.baselines[case],self.cfg['quality_allowance'])
            else:
                row['quality_passed'] = False
            rows.append(row)
            if not bounds or not row['quality_passed']:
                violations.append(row)
                if not full: break
        return {'passed':not violations,'violations':violations,'cases':rows,
                'mean_rate_ratio':float(np.mean([r['rate_ratio'] for r in rows]))
                                  if all('rate_ratio' in r for r in rows) else None,
                'changed_rgb8_cases':sum(r.get('changed_rgb8',False) for r in rows)}


@torch.no_grad()
def collect_candidates(context, out):
    """E0: measure each candidate with the same real-codec and task-sampling seed."""
    out = Path(out); out.mkdir(parents=True,exist_ok=False)
    cfg,labels = context.cfg,{}
    chosen_ratios,nonidentity = [],0
    for index,(record,cpu,_) in enumerate(context.samples):
        x = cpu.to(context.device)
        bank = candidate_bank(x,context.protection(index),cfg['max_total_rms'])
        for codec,qp in itertools.product(cfg['codecs'],cfg['qps']):
            case = (index,codec,qp)
            rows = []
            for candidate,y in bank.items():
                measurement = context.measure(y,case)
                row = {'split':'TRAIN','source':record['id'],'source_index':index,'task':record['task'],
                       'codec':codec,'qp':qp,'candidate':candidate,'seed':context.seed(case),
                       'total_rms':float((y-x).square().mean().sqrt()),**measurement}
                if candidate=='identity': context.baselines[case] = measurement
                row['feasible'] = quality_feasible(row,context.baselines[case],cfg['quality_allowance'])
                log_row(out/'candidates.jsonl',row); rows.append(row)
            chosen = select_candidate(rows,context.baselines[case],cfg['quality_allowance'],cfg['max_total_rms'])
            label = {'case':case,'source':record['id'],'candidate':chosen['candidate'],
                     'measurement':chosen,'rate_ratio':chosen['bpp']/context.baselines[case]['bpp']}
            labels[case] = label
            log_row(out/'labels.jsonl',label)
            chosen_ratios.append(label['rate_ratio']); nonidentity += chosen['candidate']!='identity'
        print(json.dumps({'stage':'recovery_oracle','sources':index+1,'total':len(context.samples),
                          'nonidentity_labels':nonidentity}),flush=True)
    summary = {'split':'TRAIN','source_count':len(context.samples),'cases':len(labels),
               'candidate_count':7,'real_roundtrips':len(labels)*7,'nonidentity_labels':nonidentity,
               'mean_selected_rate_ratio':float(np.mean(chosen_ratios)),
               'scope':'TRAIN candidate-bank oracle; not a deployed NPP or DEV result',
               'teacher_inputs':'source saliency/boxes; offline only'}
    write_json(out/'summary.json',summary)
    write_json(out/'identity_targets.json',[{'case':case,**row} for case,row in context.baselines.items()])
    return labels


def stored_config(cfg):
    return dict(cfg,npp_kind=cfg.get('npp_kind','regional'))


def fit_warmstart(npp, initial, proxy, context, labels, out):
    """E2 initialization: distill TRAIN labels, then backtrack to a verified TRAIN panel."""
    out = Path(out); out.mkdir(parents=True,exist_ok=False)
    cfg = context.cfg
    schedule = balanced_schedule([s[0] for s in context.samples],cfg['codecs'],cfg['qps'],cfg['seed'])
    opt = torch.optim.Adam(npp.parameters(),lr=cfg['warmstart_learning_rate'])
    for step in range(cfg['warmstart_steps']):
        batch = schedule[step%len(schedule)]
        opt.zero_grad(set_to_none=True)
        loss_value = 0.
        for case in batch:
            index,codec,qp = case
            x = context.x(case)
            with torch.no_grad():
                target = candidate_bank(x,context.protection(index),cfg['max_total_rms'])[labels[case]['candidate']]
            y = npp(x,codec,qp)
            loss = ((y-target).square()*(1+context.protection(index))).mean()/cfg['max_total_rms']**2
            (loss/len(batch)).backward(); loss_value += float(loss.detach())/len(batch)
        torch.nn.utils.clip_grad_norm_(npp.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        log_row(out/'distillation.jsonl',{'step':step+1,'loss':loss_value,'batch':batch})
        if (step+1)%32==0:
            print(json.dumps({'stage':'recovery_warmstart','step':step+1,'loss':loss_value}),flush=True)
    original = copy.deepcopy(initial.state_dict())
    proposed = copy.deepcopy(npp.state_dict())
    attempts = []
    selected_scale = None
    for scale in cfg['warmstart_scales']:
        with torch.no_grad():
            for key,value in npp.state_dict().items():
                value.copy_(original[key]+scale*(proposed[key]-original[key]))
        panel = context.check(npp,context.anchors(),full=True)
        # Warm-start must produce measurable panel savings, otherwise keep identity.
        useful = (panel['passed'] and panel['changed_rgb8_cases']>0
                  and panel['mean_rate_ratio'] < 1-cfg['minimum_rate_improvement'])
        attempts.append({'scale':scale,'panel':panel,'useful':useful})
        if useful or scale==0 and panel['passed']:
            selected_scale = scale
            break
    if selected_scale is None: raise RuntimeError('identity warm-start fallback failed')
    save_checkpoint(out/'start.pt',npp,proxy,stored_config(cfg),'recovery_warmstart',0)
    report = {'steps':cfg['warmstart_steps'],'selected_scale':selected_scale,
              'nonidentity':tensor_digest(npp)!=tensor_digest(initial),
              'attempts':attempts,'final_panel':panel,'scope':'TRAIN initialization; fixed-panel guarantee only'}
    write_json(out/'summary.json',report)
    return report


def fit_recovery(npp, proxy, context, out):
    """E1/E2 optimization with balanced gradients, replay and short atomic commits."""
    out = Path(out); out.mkdir(parents=True,exist_ok=False)
    cfg = context.cfg
    schedule = balanced_schedule([s[0] for s in context.samples],cfg['codecs'],cfg['qps'],cfg['seed'])
    optimizer = torch.optim.Adam([
        {'params':[p for name,p in npp.named_parameters() if name!='filter_strength']},
        {'params':[npp.filter_strength],'lr':cfg['filter_strength_learning_rate']}],lr=cfg['learning_rate'])
    transaction = ShortTransaction(npp,optimizer,cfg['anchor_every'],cfg['minimum_block'])
    replay = ReplayQueue(cfg['replay_capacity'])
    frozen = tensor_digest(proxy)
    initial_digest = tensor_digest(npp)
    panel = context.check(npp,context.anchors(),full=True)
    if not panel['passed']: raise ValueError('recovery start is not TRAIN-panel feasible')
    best_ratio = panel['mean_rate_ratio']
    progress_ratio = best_ratio
    selected,selected_step = copy.deepcopy(npp.state_dict()),0
    last_useful,last_commit,stop_reason = 0,0,'budget_exhausted'
    save_checkpoint(out/'start.pt',npp,proxy,stored_config(cfg),'recovery_start',0)
    for step in range(cfg['npp_steps']):
        batch = schedule[step%len(schedule)]
        hard = replay.take(cfg['replay_per_step'],exclude=set(batch))
        cases = batch+hard
        previous,loss_value,measurements = {},0.,[]
        optimizer.zero_grad(set_to_none=True)
        for case in cases:
            index,codec,qp = case
            x = context.x(case)
            torch.manual_seed(context.seed(case))
            y = npp(x,codec,qp)
            previous[case] = y.detach()
            actual = context.codec.roundtrip(y,codec,qp)
            prediction = proxy(y,codec,qp)
            decoded = real_value_proxy_grad(prediction.reconstruction,actual.reconstruction)
            rate = real_value_proxy_grad(prediction.bpp,actual.bpp)
            # The real codec and proxy must not advance detector ROI sampling RNG.
            torch.manual_seed(context.seed(case))
            terms = task_terms(context.tasks.losses(decoded,context.samples[index][0],context.refs[index]),
                               context.samples[index][0])
            baseline = context.baselines[case]
            weight = min(1.,.05+.95*step/cfg['rate_warmup'])
            objective,_ = protected_objective(rate,baseline['bpp'],terms,baseline['terms'],cfg['loss_weights'],
                cfg['quality_allowance'],penalty=cfg['quality_penalty'],rate_weight=weight)
            preservation,temporal = regularizers(x,y,context.masks[index].to(context.device))
            loss = objective+cfg['loss_weights']['pre']*preservation+cfg['loss_weights']['temp']*temporal
            if not torch.isfinite(loss): raise RuntimeError('nonfinite recovery objective')
            (loss/len(cases)).backward(); loss_value += float(loss.detach())/len(cases)
            measurements.append({'case':case,'bpp':float(actual.bpp.mean()),
                                 'terms':{k:float(v.detach()) for k,v in terms.items()}})
        grad = torch.nn.utils.clip_grad_norm_(npp.parameters(),1.,error_if_nonfinite=True)
        def check():
            report = context.check(npp,cases,previous=previous)
            # Retain hard cases from all line-search attempts, even if a smaller step passes.
            for failure in report['violations']: replay.add(failure['case'])
            return report['passed'],report
        accepted,scale,guard = guarded_step(npp,optimizer,check,attempts=cfg['guard_attempts'])
        transaction.note(accepted)
        row = {'step':step+1,'batch':batch,'replay':hard,'gradient_sources':len(cases),'loss':loss_value,
               'gradient_norm':float(grad),'accepted':accepted,'step_scale':scale,
               'measurements':measurements,'guard':guard}
        log_row(out/'training.jsonl',row)
        if (transaction.attempts>=transaction.interval or step+1==cfg['npp_steps']
                or step+1-last_useful>=cfg['stall_steps']):
            panel = context.check(npp,context.anchors(),full=True)
            for failure in panel['violations']: replay.add(failure['case'])
            counts = transaction.finish(panel['passed'])
            if panel['passed']:
                last_commit = step+1
                if panel['changed_rgb8_cases'] and panel['mean_rate_ratio'] < best_ratio:
                    best_ratio = panel['mean_rate_ratio']
                    selected,selected_step = copy.deepcopy(npp.state_dict()),step+1
                if best_ratio < progress_ratio-cfg['minimum_rate_improvement']:
                    progress_ratio = best_ratio
                    last_useful = step+1
            log_row(out/'guards.jsonl',{'step':step+1,'panel':panel,'counts':counts,
                                     'next_interval':transaction.interval,'best_panel_rate_ratio':best_ratio,
                                     'selected_step':selected_step,'last_useful_progress_step':last_useful})
            save_checkpoint(out/'last.pt',npp,proxy,stored_config(cfg),'recovery_last_committed',step+1)
            torch.save({'optimizer':optimizer.state_dict(),'counts':counts,'step':step+1},out/'optimizer.pt')
            print(json.dumps({'stage':'recovery_guard','step':step+1,'passed':panel['passed'],
                              'interval':transaction.interval,'counts':counts,
                              'best_panel_rate_ratio':best_ratio}),flush=True)
        if step+1-last_useful>=cfg['stall_steps']:
            stop_reason = 'no_useful_panel_rate_progress'; break
    npp.load_state_dict(selected,strict=True)
    panel = context.check(npp,context.anchors(),full=True)
    if not panel['passed']: raise RuntimeError('selected checkpoint failed TRAIN panel recheck')
    if tensor_digest(proxy)!=frozen: raise RuntimeError('frozen proxy changed')
    save_checkpoint(out/'selected.pt',npp,proxy,stored_config(cfg),'recovery_selected',selected_step)
    report = {'counts':dict(transaction.counts),'stop_reason':stop_reason,'selected_step':selected_step,
              'last_feasible_check_step':last_commit,'best_panel_rate_ratio':best_ratio,'final_panel':panel,
              'selected_differs_from_arm_start':tensor_digest(npp)!=initial_digest,
              'selection':'minimum mean relative rate on fixed TRAIN panel among feasible checkpoints',
              'committed_counts_scope':'transactionally accepted updates; final selected checkpoint may precede last',
              'fixed_proxy_verified':True,'proxy_tensor_sha256':frozen,
              'scope':'TRAIN guards only; no guarantee on all TRAIN/DEV sources'}
    write_json(out/'training_summary.json',report)
    return report


def validate_protocol(cfg):
    if (cfg['qps']!=[30,35,40,45] or cfg['codecs']!=['h264','h265']
            or cfg['arms']!=['replay_identity','replay_warmstart']):
        raise ValueError('paired recovery requires the fixed eight operating points and two arms')
    positive = ('npp_steps','train_count','dev_count','warmstart_steps','anchor_every','minimum_block',
                'anchor_per_task','stall_steps','replay_capacity','replay_per_step','rate_warmup','guard_attempts')
    if any(type(cfg[k]) is not int or cfg[k]<1 for k in positive):
        raise ValueError('positive integer training/check budgets required')
    if (cfg['train_count']%2 or not cfg['minimum_block']<=cfg['anchor_every']<=16
            or cfg['stall_steps']<cfg['anchor_every'] or cfg['anchor_per_task']>cfg['train_count']
            or not 0<cfg['max_npp_update_rms']<=1/255 or not 0<cfg['max_total_rms']<=8/255
            or not 0<cfg['filter_limit']<=.2 or cfg['quality_allowance']!=.02):
        raise ValueError('invalid recovery guard, source or block protocol')
    scales = cfg['warmstart_scales']
    if (not scales or scales[0]!=1 or scales[-1]!=0 or any(not math.isfinite(v) for v in scales)
            or any(a<=b for a,b in zip(scales,scales[1:]))):
        raise ValueError('warm-start scales must strictly descend from one to identity zero')
    if not 0<cfg['minimum_rate_improvement']<1:
        raise ValueError('positive finite useful-rate threshold required')


def validate_manifest(manifest, prior):
    train,dev = [{r['id'] for r in manifest[s]} for s in ('train','dev')]
    old_train,old_dev = [{r['id'] for r in prior[s]} for s in ('train','dev')]
    if (len(train)!=len(manifest['train']) or len(dev)!=len(manifest['dev']) or train&dev
            or not old_train<=train or train&old_dev or dev&old_train):
        raise ValueError('source overlap or missing calibration TRAIN source')


def main():
    from .bdrate import comparisons
    from .codec import RealCodec
    from .evaluate import evaluate
    from .run import configuration, manifest_for, samples_for
    from .tasks import TaskHeads

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ar-root','coco-root','annotations','checkpoint','config','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    cfg = configuration(args.config)
    validate_protocol(cfg)
    if not torch.cuda.is_available(): raise RuntimeError('real-data recovery requires CUDA GPU')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=False)
    receipt = {'complete':False,'started':time.time(),'config':cfg,'source_hashes':source_hashes(),
               'source_commit':os.environ.get('AROD_SOURCE_COMMIT'),'gpu':torch.cuda.get_device_name(),
               'completed_arms':[],'scope':'E0/E1/E2 TRAIN fitting and final DEV; TEST untouched; no joint video OD GT'}
    status = args.out/'run.json'
    write_json(status,receipt)
    try:
        print(json.dumps({'stage':'recovery_init','phase':'checkpoint'}),flush=True)
        base,proxy,saved = load_checkpoint(args.checkpoint,device='cuda')
        if saved['format']!=2 or saved['stage']!='proxy_calibration':
            raise ValueError('calibrated format-2 checkpoint required')
        for key in ('width','frames','size','qps','codecs','preset','fps','od_frames'):
            if saved['config'][key]!=cfg[key]: raise ValueError('checkpoint protocol differs: '+key)
        proxy.requires_grad_(False).eval()
        receipt['checkpoint_sha256'] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        manifest = manifest_for(args,cfg)
        prior = json.loads((args.checkpoint.parent/'manifest.json').read_text())
        validate_manifest(manifest,prior)
        write_json(args.out/'manifest.json',manifest)
        train = samples_for(manifest['train'],cfg)
        tasks = TaskHeads(device='cuda',size=cfg['size'],od_frames=cfg['od_frames'])
        frozen = {k:tensor_digest(getattr(tasks,k)) for k in ('ar','od')}
        refs,masks = [],[]
        for index,(record,cpu,gt) in enumerate(train):
            x = cpu.cuda()
            reference = tasks.references(x,record,gt)
            refs.append(reference); masks.append(semantic_weights(tasks,x,record,reference).cpu())
            if (index+1)%8==0:
                print(json.dumps({'stage':'recovery_references','sources':index+1,'total':len(train)}),flush=True)
        codec = RealCodec(preset=cfg['preset'],gop=cfg['frames'],fps=cfg['fps'])
        context = RecoveryContext(train,refs,masks,tasks,codec,cfg)
        labels = collect_candidates(context,args.out/'oracle_train')
        initial = identity_start(base,cfg['filter_limit'])
        receipt['identity_npp_tensor_sha256'] = tensor_digest(initial)
        results,training = {},{}
        # Finish all TRAIN operations before reading DEV pixels or computing DEV metrics.
        for arm in cfg['arms']:
            torch.manual_seed(cfg['seed'])
            npp = copy.deepcopy(initial)
            if arm=='replay_warmstart':
                receipt['warmstart'] = fit_warmstart(npp,initial,proxy,context,labels,args.out/'warmstart')
                # Full warm-start detail is already persisted; keep worker receipt small.
                receipt['warmstart'] = {k:v for k,v in receipt['warmstart'].items() if k not in ('attempts','final_panel')}
            print(json.dumps({'stage':'recovery_arm_start','arm':arm}),flush=True)
            training[arm] = fit_recovery(npp,proxy,context,args.out/arm)
            if any(tensor_digest(getattr(tasks,k))!=v for k,v in frozen.items()):
                raise RuntimeError('frozen task state changed')
            receipt['completed_arms'].append(arm)
            write_json(status,receipt)
        dev = samples_for(manifest['dev'],cfg)
        for arm in cfg['arms']:
            npp,_,_ = load_checkpoint(args.out/arm/'selected.pt',device='cuda')
            destination = args.out/arm/'evaluation'; destination.mkdir()
            print(json.dumps({'stage':'recovery_dev','arm':arm}),flush=True)
            results[arm] = evaluate(npp,proxy,codec,tasks,dev,cfg,args.annotations,destination)
        report = {'scope':receipt['scope'],'summaries':results,'training':training,
                  'bd_rates':comparisons(results),'confidence_intervals':None,'test_used':False,
                  'oracle_scope':'TRAIN-only feasibility and distillation; no DEV candidate selection'}
        write_json(args.out/'recovery_summary.json',report)
        receipt.update(complete=True,task_frozen_verified=True,proxy_frozen_verified=True)
    except BaseException as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        receipt['wall_seconds'] = time.time()-receipt['started']
        write_json(status,receipt)
        print(json.dumps({'stage':'recovery_finished','complete':receipt['complete'],
                          'wall_seconds':receipt['wall_seconds']}),flush=True)


if __name__=='__main__':
    main()

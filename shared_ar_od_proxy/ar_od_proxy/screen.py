"""Controlled TRAIN-only optimization and final DEV three-arm codec experiment."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .bdrate import comparisons
from .codec import RealCodec
from .engine import (bounded_optimizer_step, emit, load_checkpoint, save_checkpoint,
                     source_hashes, training_case)
from .evaluate import evaluate
from .models import RegionalVideoNPP, real_value_proxy_grad
from .run import configuration, manifest_for, samples_for
from .tasks import TaskHeads


def constrained_objective(rate, identity_rate, terms, identity_terms, duals, allowance):
    """Real-valued relative rate plus hinge constraints on decoded task losses."""
    if not math.isfinite(float(identity_rate)) or float(identity_rate) <= 0:
        raise ValueError('identity rate must be finite and positive')
    loss = rate.mean() / float(identity_rate)
    excess = {}
    for key, value in terms.items():
        baseline = float(identity_terms[key])
        if not math.isfinite(baseline) or baseline < 0:
            raise ValueError('identity task loss must be finite and nonnegative')
        difference = (value-baseline) / max(baseline, .1) - allowance
        loss = loss + duals[key] * difference.relu()
        excess[key] = float(difference.detach())
    return loss, excess


def update_duals(duals, excess, learning_rate, maximum):
    for key, value in excess.items():
        duals[key] = min(maximum, max(.05, duals[key]+learning_rate*value))


def case_seed(step, count, config):
    return config['seed'] + step % (count*len(config['codecs'])*len(config['qps']))


def task_terms(raw, record):
    terms = {'od': raw['od_gt']+.5*raw['od_source']}
    if record['task'] == 'ar':
        terms['ar'] = raw['ar_gt']+.2*raw['ar_source']
    return terms


def foreground_weights(x, refs):
    """Training-only preservation weights. No teacher or boxes at inference."""
    mask = x.new_zeros(x.shape[0], x.shape[1], 1, *x.shape[-2:])
    h, w = x.shape[-2:]
    for reference in refs['od']:
        for box in reference['boxes'].detach().cpu().tolist():
            x1,y1,x2,y2 = box
            x1,y1 = max(0,math.floor(x1)),max(0,math.floor(y1))
            x2,y2 = min(w,math.ceil(x2)),min(h,math.ceil(y2))
            if x2 > x1 and y2 > y1:
                # A conservative union across the short GOP; source motion adds protection below.
                mask[..., y1:y2, x1:x2] = 1
    if x.shape[1] > 1:
        motion = (x[:,1:]-x[:,:-1]).abs().mean(2,keepdim=True)
        mask[:,1:] = torch.maximum(mask[:,1:], (motion/.05).clamp(0,1))
    return 1+4*mask


def regularizers(x, y, mask=None):
    delta = y-x
    preservation = (delta.square()*(1 if mask is None else mask)).mean()
    temporal = x.sum()*0
    if x.shape[1] > 1:
        motion = (x[:,1:]-x[:,:-1]).abs().mean(2,keepdim=True).detach()
        temporal = ((delta[:,1:]-delta[:,:-1]).square()*(-10*motion).exp()).mean()
    return preservation, temporal


def tensor_digest(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fit_arm(npp, proxy, codec, tasks, samples, references, baselines, cfg, arm, out):
    out.mkdir()
    arm_config = dict(cfg, screen_arm=arm, npp_kind='regional' if arm=='regional' else 'video')
    save_checkpoint(out/'start.pt', npp, proxy, arm_config, 'screen_start', 0)
    groups = npp.parameters()
    if arm=='regional':
        # The new scalar controls a whole image; give it an explicit, recorded scale.
        groups = [{'params':[p for key,p in npp.named_parameters() if key!='filter_strength']},
                  {'params':[npp.filter_strength],
                   'lr':cfg.get('filter_strength_learning_rate',cfg['learning_rate'])}]
    optimizer = torch.optim.Adam(groups, lr=cfg['learning_rate'])
    device = next(npp.parameters()).device
    duals = {(c,q,t): {'ar':1.,'od':1.} for c in cfg['codecs'] for q in cfg['qps'] for t in ('ar','od')}
    # Masks, references and identity targets are shared constants across arms.
    masks = [foreground_weights(x.to(device), ref) for (_,x,_),ref in zip(samples,references)]
    if any(p.requires_grad for m in (proxy,tasks.ar,tasks.od) for p in m.parameters()):
        raise ValueError('all proxy/task parameters must be frozen')
    for step in range(cfg['npp_steps']):
        torch.manual_seed(case_seed(step,len(samples),cfg))  # paired with identity, every pass
        index,name,qp = training_case(step, len(samples), cfg)
        record,cpu,_ = samples[index]
        x = cpu.to(device)
        baseline = baselines[(index,name,qp)]
        npp.train()
        start = time.perf_counter()
        y = npp(x,name,qp)
        actual = codec.roundtrip(y,name,qp)
        predicted = proxy(y,name,qp)
        decoded = real_value_proxy_grad(predicted.reconstruction, actual.reconstruction)
        rate = real_value_proxy_grad(predicted.bpp, actual.bpp)
        terms = task_terms(tasks.losses(decoded,record,references[index]),record)
        preservation,temporal = regularizers(x,y,None if arm=='existing' else masks[index])
        w = cfg['loss_weights']
        active_duals = duals[(name,qp,record['task'])]
        if arm=='existing':
            objective = w['rate']*rate.mean()+sum(w[k]*v for k,v in terms.items())
            excess = {}
        else:
            objective,excess = constrained_objective(rate,baseline['bpp'],terms,baseline['terms'],
                                                     active_duals,cfg['quality_allowance'])
        loss = objective+w['pre']*preservation+w['temp']*temporal
        diagnostic = step % cfg['diagnostic_every'] == 0 or step+1==cfg['npp_steps']
        norms = {}
        if diagnostic:
            gradient_terms = {'rate': rate.mean()/(1 if arm=='existing' else baseline['bpp']), **terms}
            for key,value in gradient_terms.items():
                gradient, = torch.autograd.grad(value,y,retain_graph=True)
                norms[key] = float(gradient.norm())
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite screen loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(npp.parameters(),1.,error_if_nonfinite=True)
        rms,scale = bounded_optimizer_step(npp,optimizer,x,y.detach(),name,qp,cfg['max_npp_update_rms'])
        row = {'stage':'screen_train','arm':arm,'step':step+1,'source':record['id'],
               'task':record['task'],'codec':name,'qp':qp,'loss':float(loss.detach()),
               'actual_bpp':float(actual.bpp.mean()),'proxy_bpp':float(predicted.bpp.detach().mean()),
               'identity_bpp':baseline['bpp'],'relative_rate':float(actual.bpp.mean())/baseline['bpp'],
               'task_losses':{k:float(v.detach()) for k,v in terms.items()},'identity_losses':baseline['terms'],
               'excess':excess,'duals':dict(active_duals),'gradient_norms':norms,
               'preservation':float(preservation.detach()),'temporal':float(temporal.detach()),
               'npp_grad_norm':float(norm),'output_update_rms':rms,'step_scale':scale}
        if diagnostic:
            with torch.no_grad():
                after = npp(x,name,qp)
                post = codec.roundtrip(after,name,qp)
                post_proxy = proxy(after,name,qp)
                torch.manual_seed(case_seed(step,len(samples),cfg))
                post_terms = task_terms(tasks.losses(post.reconstruction,record,references[index]),record)
            row['actual_step'] = {'real_rate_delta':float((post.bpp-actual.bpp).mean()),
                                  'proxy_rate_delta':float((post_proxy.bpp-predicted.bpp.detach()).mean()),
                                  'real_task_deltas':{k:float(v-terms[k].detach()) for k,v in post_terms.items()}}
        update_duals(active_duals,excess,cfg['dual_learning_rate'],cfg['dual_maximum'])
        row['seconds'] = time.perf_counter()-start
        # Persist every update, but keep notebook console compact.
        if step % 32 == 0 or diagnostic or step+1==cfg['npp_steps']:
            emit(out/'training.jsonl',row)
        else:
            with (out/'training.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
        if (step+1)%256==0 or step+1==cfg['npp_steps']:
            save_checkpoint(out/'last.pt',npp,proxy,arm_config,'screen',step+1)
            torch.save({'optimizer':optimizer.state_dict(),'duals':{str(k):v for k,v in duals.items()},
                        'step':step+1,'config':arm_config},out/'optimizer.pt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('ar-root','coco-root','annotations','checkpoint','config','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('screen requires a real CUDA GPU')
    cfg = configuration(args.config)
    if cfg['arms'] != ['existing','normalized','regional'] or cfg['proxy_refresh_every'] != 0:
        raise ValueError('three fixed arms and frozen proxy required')
    if cfg['npp_steps'] % (2*cfg['train_count']*8):
        raise ValueError('budget must cover equal complete source/QP grids')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=False)
    receipt = {'complete':False,'config':cfg,'source_hashes':source_hashes(),
               'source_commit':os.environ.get('AROD_SOURCE_COMMIT'),'device':torch.cuda.get_device_name(),
               'started':time.time(),'completed_arms':[],'scope':'DEV screening; TEST untouched'}
    (args.out/'run.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    try:
        base,proxy,saved = load_checkpoint(args.checkpoint,device='cuda')
        if saved['format']!=2 or saved['stage']!='proxy_calibration':
            raise ValueError('screen requires calibrated format-2 checkpoint')
        for key in ('width','frames','size','qps','codecs','preset','fps','od_frames','train_count','loss_weights'):
            if cfg[key] != saved['config'][key]:
                raise ValueError('input checkpoint protocol differs: '+key)
        proxy.requires_grad_(False).eval()
        proxy_hash = tensor_digest(proxy)
        receipt['warm_start'] = {'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                                 'proxy_tensor_sha256':proxy_hash,'npp_tensor_sha256':tensor_digest(base)}
        manifest = manifest_for(args,cfg)
        previous = json.loads((args.checkpoint.parent/'manifest.json').read_text())
        if [r['id'] for r in manifest['train']] != [r['id'] for r in previous['train']]:
            raise ValueError('TRAIN must match calibration exactly')
        if not {r['id'] for r in previous['dev']} <= {r['id'] for r in manifest['dev']}:
            raise ValueError('expanded DEV must include previous DEV sources')
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        tasks = TaskHeads(device='cuda',size=cfg['size'],od_frames=cfg['od_frames'])
        task_hashes = {k:tensor_digest(getattr(tasks,k)) for k in ('ar','od')}
        codec = RealCodec(preset=cfg['preset'],gop=cfg['frames'],fps=cfg['fps'])
        train = samples_for(manifest['train'],cfg)
        dev = samples_for(manifest['dev'],cfg)  # validate every selected source before training
        references = [tasks.references(x.cuda(),r,gt) for r,x,gt in train]
        baselines,rows = {},[]
        with torch.no_grad():
            for step in range(len(train)*8):
                index,name,qp = training_case(step,len(train),cfg)
                torch.manual_seed(case_seed(step,len(train),cfg))
                record,x,_ = train[index]
                actual = codec.roundtrip(x.cuda(),name,qp)
                values = task_terms(tasks.losses(actual.reconstruction,record,references[index]),record)
                item = {'bpp':float(actual.bpp.mean()),'terms':{k:float(v) for k,v in values.items()}}
                baselines[(index,name,qp)] = item
                rows.append({'source':record['id'],'codec':name,'qp':qp,**item})
                if (step+1)%32==0:
                    emit(None,{'stage':'identity_targets','completed':step+1,'total':len(train)*8})
        (args.out/'identity_targets.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        receipt['task_tensor_sha256'] = task_hashes
        (args.out/'run.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        summaries = {}
        for arm in cfg['arms']:
            torch.manual_seed(cfg['seed'])
            npp = RegionalVideoNPP.from_base(base) if arm=='regional' else copy.deepcopy(base)
            receipt.setdefault('parameters',{})[arm] = sum(p.numel() for p in npp.parameters())
            fit_arm(npp,proxy,codec,tasks,train,references,baselines,cfg,arm,args.out/arm)
            if tensor_digest(proxy)!=proxy_hash or any(tensor_digest(getattr(tasks,k))!=v for k,v in task_hashes.items()):
                raise RuntimeError('frozen proxy or task state changed')
            # No intermediate DEV selection; all arms finish the predeclared budget.
            evaluation = args.out/arm/'evaluation'; evaluation.mkdir()
            summaries[arm] = evaluate(npp,proxy,codec,tasks,dev,cfg,args.annotations,evaluation)
            del npp
            receipt['completed_arms'].append(arm)
            (args.out/'run.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        report = {'scope':'DEV only; no joint video GT; no TEST claim',
                  'bd_rates':comparisons(summaries), 'summaries':summaries,
                  'confidence_intervals':None,'uncertainty':'32 sources per task; no confidence intervals estimated'}
        (args.out/'screen_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        receipt.update(complete=True,proxy_frozen_verified=True,task_frozen_verified=True)
    except BaseException as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        receipt['wall_seconds'] = time.time()-receipt['started']
        (args.out/'run.json').write_text(json.dumps(receipt,indent=2,allow_nan=False),encoding='utf-8')
        emit(None,{'complete':receipt['complete'],'wall_seconds':receipt['wall_seconds']})


if __name__=='__main__':
    main()

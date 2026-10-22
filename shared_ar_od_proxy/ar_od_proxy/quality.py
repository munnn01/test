"""TRAIN-only quality guards and on-policy proxy calibration; final DEV evaluation."""
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

from .bdrate import comparisons
from .codec import RealCodec
from .engine import emit, load_checkpoint, save_checkpoint, source_hashes, training_case
from .evaluate import evaluate
from .models import RegionalVideoNPP, real_value_proxy_grad
from .run import configuration, manifest_for, samples_for
from .screen import case_seed, foreground_weights, regularizers, task_terms, tensor_digest
from .tasks import TaskHeads


def identity_start(base, filter_limit):
    """Retain learned trunk, reset both RGB output branches for exact identity."""
    model = RegionalVideoNPP.from_base(base)
    model.filter_limit = float(filter_limit)
    if not 0 < model.filter_limit <= .9:
        raise ValueError('invalid filter limit')
    with torch.no_grad():
        for layer in (model.rgb, model.pixel[-1]):
            layer.weight.zero_(); layer.bias.zero_()
    return model


def protected_objective(rate, identity_rate, terms, baselines, weights, allowance,
                        task_weight=1., penalty=20., rate_weight=1.):
    if not math.isfinite(identity_rate) or identity_rate <= 0:
        raise ValueError('positive finite identity rate required')
    loss = rate_weight * rate.mean() / identity_rate
    excess = {}
    for key, value in terms.items():
        baseline = float(baselines[key])
        if not math.isfinite(baseline) or baseline < 0:
            raise ValueError('invalid identity task loss')
        relative = (value - baseline) / max(baseline, .1)
        violation = (relative - allowance).relu()
        # The linear task term remains active inside the allowed region.
        loss = loss + weights[key] * (task_weight * relative + penalty * violation.square())
        excess[key] = float((relative - allowance).detach())
    return loss, excess


def quality_feasible(candidate, baseline, allowance):
    if set(candidate['terms']) != set(baseline['terms']):
        return False
    for key, value in candidate['terms'].items():
        reference = float(baseline['terms'][key])
        if not math.isfinite(value) or not math.isfinite(reference) or reference < 0:
            return False
        if value > reference + allowance * max(reference, .1) + 1e-6:
            return False
    return not baseline.get('ar_correct', False) or candidate.get('ar_correct') is True


def capture_state(model, optimizer):
    return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())


def restore_state(model, optimizer, state):
    model.load_state_dict(state[0], strict=True)
    optimizer.load_state_dict(state[1])


def guarded_step(model, optimizer, check, attempts=3):
    """Check the real candidate; failed/exceptional proposals restore Adam too."""
    before = capture_state(model, optimizer)
    accepted = False
    details = {}
    try:
        optimizer.step()
        proposed = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            for attempt in range(attempts):
                scale = .5 ** attempt
                if attempt:
                    for key, value in model.state_dict().items():
                        value.copy_(before[0][key] + scale * (proposed[key] - before[0][key]))
                accepted, details = check()
                if accepted:
                    return True, scale, details
        return False, 0., details
    finally:
        if not accepted:
            restore_state(model, optimizer, before)


def refresh_indices(records, epoch, fit_per_task=4, audit_per_task=2):
    """Globally disjoint proxy-fit/audit source pools; all originate in TRAIN."""
    fit, audit = [], []
    for family in ('ar', 'od'):
        indices = [i for i, r in enumerate(records) if r['task'] == family]
        held = indices[::4]
        learn = [i for i in indices if i not in set(held)]
        if len(learn) < fit_per_task or len(held) < audit_per_task:
            raise ValueError('insufficient independent TRAIN sources for proxy refresh')
        fit.extend(learn[(epoch*fit_per_task+j) % len(learn)] for j in range(fit_per_task))
        audit.extend(held[(epoch*audit_per_task+j) % len(held)] for j in range(audit_per_task))
    return fit, audit


def refresh_acceptable(before, after):
    keys = ('rate_mape', 'sign_accuracy', 'reconstruction_l1', 'delta_l1')
    if any(after.get(k) is None or before.get(k) is None
           or not math.isfinite(after[k]) or not math.isfinite(before[k]) for k in keys):
        return False
    # Never accept directional regression in exchange for a better scalar fit.
    safe = (after['rate_mape'] <= before['rate_mape'] + .01
            and after['sign_accuracy'] >= before['sign_accuracy']
            and after['reconstruction_l1'] <= before['reconstruction_l1'] * 1.05 + 1e-5
            and after['delta_l1'] <= before['delta_l1'] * 1.05 + 1e-5)
    improved = (after['sign_accuracy'] >= before['sign_accuracy'] + .05
                or after['rate_mape'] <= before['rate_mape'] - .01
                or after['delta_l1'] < before['delta_l1'] * .9)
    for point, old in before.get('points', {}).items():
        new = after.get('points', {}).get(point)
        if new is None or new['rate_mape'] > old['rate_mape']+.02:
            safe = False
        elif old['sign_accuracy'] is not None and (new['sign_accuracy'] is None
                or new['sign_accuracy'] < old['sign_accuracy']-.25):
            safe = False
    return safe and improved


def semantic_weights(tasks, x, record, refs):
    """Offline training mask: AR input sensitivity plus OD boxes and motion."""
    weights = foreground_weights(x, refs)
    if record['task'] == 'ar':
        source = x.detach().requires_grad_()
        logits = tasks.ar_logits(source)
        target = refs['ar_logits'].argmax(-1)
        sensitivity, = torch.autograd.grad(logits.gather(1, target[:, None]).sum(), source)
        saliency = sensitivity.detach().abs().mean(2, keepdim=True)
        scale = torch.quantile(saliency.flatten(), .95).clamp_min(1e-8)
        weights = torch.maximum(weights, 1 + 8 * (saliency/scale).clamp(0, 1))
    return weights.detach()


@torch.no_grad()
def real_measure(tasks, codec, y, record, refs, name, qp, seed):
    packet = codec.roundtrip(y, name, qp)
    torch.manual_seed(seed)  # Pair TorchVision RPN/ROI sampling with identity.
    terms = task_terms(tasks.losses(packet.reconstruction, record, refs), record)
    result = {'terms': {k: float(v) for k, v in terms.items()}, 'bpp': float(packet.bpp.mean())}
    if record['task'] == 'ar':
        result['ar_correct'] = tasks.ar_logits(packet.reconstruction).argmax(-1).item() == record['ar_label']
    return result, packet


def reconstruction_jvp(proxy, pair, device, create_graph=False):
    inputs = pair['inputs'].to(device).float()
    mid, direction = (inputs[0:1] + inputs[1:2])/2, (inputs[1:2] - inputs[0:1])/2
    _, tangent = torch.autograd.functional.jvp(
        lambda z: proxy(z, pair['codec'], pair['qp']).reconstruction,
        mid, direction, create_graph=create_graph)
    decoded = pair['decoded'].to(device).float()/255
    target = (decoded[1:2] - decoded[0:1])/2
    return tangent, target


def collect_refresh(npp, proxy, codec, samples, indices, cfg, epoch, keep_recon):
    """Symmetric finite probes of current NPP outputs, never DEV or TEST."""
    device = next(npp.parameters()).device
    rows, recon = [], []
    grid = list(itertools.product(cfg['codecs'], cfg['qps']))
    with torch.no_grad():
        for position, index in enumerate(indices):
            record, cpu, _ = samples[index]
            x = cpu.to(device)
            for point, (name, qp) in enumerate(grid):
                y = npp(x, name, qp)
                # Alternate the actual cumulative NPP direction and smooth random directions.
                direction = y-x
                kind = 'npp_residual'
                if (point+epoch) % 2 or float(direction.square().mean()) < 1e-12:
                    generator = torch.Generator(device=device).manual_seed(cfg['seed']+index*97+point+epoch*997)
                    noise = torch.randn(x.shape[0]*x.shape[1],3,8,8,device=device,generator=generator)
                    direction = F.interpolate(noise,x.shape[-2:],mode='bilinear',align_corners=False).reshape_as(x)
                    kind = 'smooth_random'
                direction = direction / direction.square().mean().sqrt().clamp_min(1e-8)
                eps = cfg['probe_rms']
                inputs = torch.cat(((y-eps*direction).clamp(0,1), (y+eps*direction).clamp(0,1)),0)
                packets = [codec.roundtrip(z[None],name,qp) for z in inputs]
                target = torch.cat([p.bpp for p in packets]).cpu()
                rows.append({'features':proxy.rate_head.features(inputs).cpu(),
                             'pixels':x.shape[1]*x.shape[3]*x.shape[4],
                             'point':proxy.rate_head.point(name,qp), 'target':target,
                             'source':record['id'],'codec':name,'qp':qp,'direction':kind,
                             'stream_sha256':[hashlib.sha256(p.streams[0]).hexdigest() for p in packets]})
                # Two points per source, rotated so both codecs and all QPs occur across sources.
                if keep_recon and point in (position % 4, 4+position % 4):
                    recon.append({'inputs':inputs.cpu(), 'decoded':torch.cat([
                        (p.reconstruction.cpu()*255).round().byte() for p in packets]),
                        'codec':name,'qp':qp,'source':record['id']})
            emit(None,{'stage':'refresh_collect','epoch':epoch,'source':record['id'],'pairs':len(rows)})
    return rows, recon


def rate_tensors(rows, device):
    return (torch.cat([r['features'] for r in rows]).to(device),
            torch.tensor([r['pixels'] for r in rows for _ in range(2)],device=device),
            torch.tensor([r['point'] for r in rows for _ in range(2)],device=device),
            torch.cat([r['target'] for r in rows]).to(device))


def proxy_metrics(proxy, rows, recon):
    device = next(proxy.parameters()).device
    f, p, q, target = rate_tensors(rows, device)
    with torch.no_grad():
        pred = proxy.rate_head.from_features(f,p,q)
        real_delta = target.reshape(-1,2).diff(dim=1).flatten()
        pred_delta = pred.reshape(-1,2).diff(dim=1).flatten()
        usable = real_delta.abs() > (16/p[::2])  # More than two stream bytes.
        sign = float((pred_delta[usable].sign()==real_delta[usable].sign()).float().mean()) if usable.any() else None
        points = {}
        for point in sorted(set(r['point'] for r in rows)):
            selected = q==point
            pair_selected = q[::2]==point
            active = usable & pair_selected
            points[str(point)] = {'rate_mape':float(((pred[selected]-target[selected]).abs()/target[selected]).mean()),
                'sign_accuracy':float((pred_delta[active].sign()==real_delta[active].sign()).float().mean()) if active.any() else None,
                'sign_count':int(active.sum())}
        l1 = []
        for pair in recon:
            prediction = proxy(pair['inputs'].to(device),pair['codec'],pair['qp']).reconstruction
            l1.append(float(F.l1_loss(prediction,pair['decoded'].to(device).float()/255)))
    tangents = []
    for pair in recon:
        tangent, actual = reconstruction_jvp(proxy,pair,device)
        tangents.append(float(F.l1_loss(tangent,actual)))
    return {'rate_mape':float(((pred-target).abs()/target).mean()), 'sign_accuracy':sign,
            'sign_count':int(usable.sum()), 'reconstruction_l1':float(np.mean(l1)),
            'delta_l1':float(np.mean(tangents)), 'points':points}


def fit_reconstruction(proxy, pairs, cfg):
    """Clip only reconstruction/JVP parameters; rate-era gradients cannot enter."""
    device = next(proxy.parameters()).device
    proxy.requires_grad_(True)
    proxy.rate_head.requires_grad_(False)
    proxy.entropy_scale.requires_grad_(False); proxy.entropy_mean.requires_grad_(False)
    for p in proxy.parameters(): p.grad = None
    parameters = [p for p in proxy.parameters() if p.requires_grad]
    opt = torch.optim.Adam(parameters,lr=1e-4)
    try:
        for step in range(cfg['refresh_reconstruction_steps']):
            pair = pairs[step % len(pairs)]
            predicted = proxy(pair['inputs'].to(device),pair['codec'],pair['qp']).reconstruction
            target_rec = pair['decoded'].to(device).float()/255
            tangent, actual = reconstruction_jvp(proxy,pair,device,create_graph=True)
            loss = F.l1_loss(predicted,target_rec) + 5*F.mse_loss(predicted,target_rec)
            loss = loss + .1*F.l1_loss(tangent/cfg['probe_rms'],actual/cfg['probe_rms'])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True); opt.step()
    finally:
        proxy.requires_grad_(False).eval()
        for p in proxy.parameters(): p.grad = None


def refresh_proxy(npp, proxy, codec, samples, cfg, epoch, out):
    fit_ids, audit_ids = refresh_indices([s[0] for s in samples],epoch)
    fit, fit_rec = collect_refresh(npp,proxy,codec,samples,fit_ids,cfg,epoch,True)
    audit, audit_rec = collect_refresh(npp,proxy,codec,samples,audit_ids,cfg,epoch,True)
    before_state = copy.deepcopy(proxy.state_dict())
    before = proxy_metrics(proxy,audit,audit_rec)
    device = next(proxy.parameters()).device
    accepted = False
    try:
        proxy.rate_head.requires_grad_(True)
        opt = torch.optim.Adam(proxy.rate_head.parameters(),lr=.005)
        f,p,q,target = rate_tensors(fit,device)
        for _ in range(cfg['refresh_rate_steps']):
            predicted = proxy.rate_head.from_features(f,p,q).clamp_min(1e-9).log()
            truth = target.log()
            delta_pred = predicted.reshape(-1,2).diff(dim=1)
            delta_true = truth.reshape(-1,2).diff(dim=1)
            loss = F.smooth_l1_loss(predicted,truth) + .1*F.smooth_l1_loss(delta_pred/.02,delta_true/.02)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(proxy.rate_head.parameters(),1.,error_if_nonfinite=True); opt.step()
        fit_reconstruction(proxy,fit_rec,cfg)
        after = proxy_metrics(proxy,audit,audit_rec)
        accepted = refresh_acceptable(before,after)
        row = {'stage':'proxy_refresh','epoch':epoch,'accepted':accepted,'before':before,'after':after,
               'fit_sources':[samples[i][0]['id'] for i in fit_ids],
               'audit_sources':[samples[i][0]['id'] for i in audit_ids]}
        torch.save({'fit':fit,'audit':audit},out/f'refresh_{epoch}_rate_pairs.pt')
        emit(out/'refresh.jsonl',row)
        return row
    finally:
        if not accepted: proxy.load_state_dict(before_state,strict=True)
        proxy.requires_grad_(False).eval()
        for p in proxy.parameters(): p.grad = None


def candidate_check(npp, tasks, codec, x, previous, record, refs, name, qp, seed, baseline, cfg):
    """The loss/correctness and both displacement bounds are measured, not inferred."""
    with torch.no_grad():
        y = npp(x,name,qp)
        step_rms = float((y-previous).square().mean().sqrt())
        total_rms = float((y-x).square().mean().sqrt())
        bounds = (math.isfinite(step_rms) and math.isfinite(total_rms)
                  and step_rms <= cfg['max_npp_update_rms'] and total_rms <= cfg['max_total_rms'])
        if not bounds:
            return False, {'bounds_passed':False,'step_rms':step_rms,'total_rms':total_rms}
        result, _ = real_measure(tasks,codec,y,record,refs,name,qp,seed)
        passed = quality_feasible(result,baseline,cfg['quality_allowance'])
        return passed, {'bounds_passed':True,'quality_passed':passed,
                        'step_rms':step_rms,'total_rms':total_rms,'measurement':result}


def direction_diagnostics(y, prediction, packet, terms, tasks, codec, record, refs, name, qp, seed, baseline, cfg):
    """Separate rate and task descent directions, evaluated at real codec outputs."""
    task_value = sum(cfg['loss_weights'][k]*v/max(baseline['terms'][k],.1) for k,v in terms.items())
    gr, = torch.autograd.grad(prediction.bpp.mean()/baseline['bpp'],y,retain_graph=True)
    gt, = torch.autograd.grad(task_value,y,retain_graph=True)
    rows = []
    for label,gradient in (('rate',gr),('task',gt)):
        direction = -gradient.detach()/gradient.detach().square().mean().sqrt().clamp_min(1e-12)
        for eps in (.5/255,2/255):
            candidate = (y.detach()+eps*direction).clamp(0,1)
            result,_ = real_measure(tasks,codec,candidate,record,refs,name,qp,seed)
            displacement = candidate-y.detach()
            rows.append({'direction':label,'requested_rms':eps,
                         'real_rate_delta':result['bpp']-float(packet.bpp.mean()),
                         'linear_proxy_relative_rate_delta':float((gr.detach()*displacement).sum()),
                         'linear_proxy_task_delta':float((gt.detach()*displacement).sum()),
                         'real_task_deltas':{k:result['terms'][k]-float(v.detach()) for k,v in terms.items()}})
    return rows


def anchor_cases(samples, cfg):
    indices = []
    for family in ('ar','od'):
        indices.extend([i for i,s in enumerate(samples) if s[0]['task']==family][:cfg['anchor_per_task']])
    return [(i,c,q) for i in indices for c,q in itertools.product(cfg['codecs'],cfg['qps'])]


@torch.no_grad()
def check_anchors(npp, proxy, codec, tasks, samples, refs, baselines, cfg):
    device = next(npp.parameters()).device
    violations = []
    for index,name,qp in anchor_cases(samples,cfg):
        record,cpu,_ = samples[index]
        x = cpu.to(device)
        y = npp(x,name,qp)
        total = float((y-x).square().mean().sqrt())
        step = index*8+proxy.rate_head.point(name,qp)
        result,_ = real_measure(tasks,codec,y,record,refs[index],name,qp,case_seed(step,len(samples),cfg))
        if (not math.isfinite(total) or total>cfg['max_total_rms']
                or not quality_feasible(result,baselines[(index,name,qp)],cfg['quality_allowance'])):
            violations.append({'source':record['id'],'codec':name,'qp':qp,'total_rms':total,
                               'measurement':result})
    return violations


def fit_quality(npp, proxy, codec, tasks, samples, refs, masks, baselines, cfg, arm, out):
    out.mkdir()
    device = next(npp.parameters()).device
    stored = dict(cfg,npp_kind='regional',quality_arm=arm)
    save_checkpoint(out/'start.pt',npp,proxy,stored,'quality_start',0)
    groups = [{'params':[p for name,p in npp.named_parameters() if name!='filter_strength']},
              {'params':[npp.filter_strength], 'lr':cfg['filter_strength_learning_rate']}]
    optimizer = torch.optim.Adam(groups,lr=cfg['learning_rate'])
    block = capture_state(npp,optimizer)
    block_accepts = 0
    counts = {'proposals':0,'accepted_proposals':0,'rejected_proposals':0,
              'rolled_back_proposals':0,'rolled_back_blocks':0,'committed_proposals':0}
    refreshes = []
    frozen = tensor_digest(proxy)
    for step in range(cfg['npp_steps']):
        if step and arm=='quality_refresh' and step % cfg['proxy_refresh_every']==0:
            refreshes.append(refresh_proxy(npp,proxy,codec,samples,cfg,step//cfg['proxy_refresh_every']-1,out))
            save_checkpoint(out/f'proxy_version_{step}.pt',npp,proxy,stored,'quality_proxy_refresh',step)
        npp.train(); proxy.eval()
        index,name,qp = training_case(step,len(samples),cfg)
        record,cpu,_ = samples[index]
        x = cpu.to(device)
        baseline = baselines[(index,name,qp)]
        seed = case_seed(step,len(samples),cfg)
        start = time.perf_counter()
        torch.manual_seed(seed)
        y = npp(x,name,qp)
        actual = codec.roundtrip(y,name,qp)
        prediction = proxy(y,name,qp)
        decoded = real_value_proxy_grad(prediction.reconstruction,actual.reconstruction)
        rate = real_value_proxy_grad(prediction.bpp,actual.bpp)
        terms = task_terms(tasks.losses(decoded,record,refs[index]),record)
        rate_weight = min(1., .05+.95*step/cfg['rate_warmup'])
        objective,excess = protected_objective(rate,baseline['bpp'],terms,baseline['terms'],
            cfg['loss_weights'],cfg['quality_allowance'],penalty=cfg['quality_penalty'],rate_weight=rate_weight)
        preservation,temporal = regularizers(x,y,masks[index])
        loss = objective + cfg['loss_weights']['pre']*preservation + cfg['loss_weights']['temp']*temporal
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite quality loss')
        diagnostic = None
        if step%cfg.get('diagnostic_every',127)==0:
            diagnostic = direction_diagnostics(y,prediction,actual,terms,tasks,codec,record,refs[index],
                                                name,qp,seed,baseline,cfg)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(npp.parameters(),1.,error_if_nonfinite=True)
        accepted,scale,details = guarded_step(npp,optimizer,
            lambda: candidate_check(npp,tasks,codec,x,y.detach(),record,refs[index],name,qp,seed,baseline,cfg),
            attempts=cfg['guard_attempts'])
        counts['proposals'] += 1
        counts['accepted_proposals' if accepted else 'rejected_proposals'] += 1
        block_accepts += int(accepted)
        row = {'stage':'quality_train','arm':arm,'step':step+1,'source':record['id'],'task':record['task'],
               'codec':name,'qp':qp,'loss':float(loss.detach()),'rate_weight':rate_weight,
               'actual_bpp':float(actual.bpp.mean()),'proxy_bpp':float(prediction.bpp.detach().mean()),
               'identity_bpp':baseline['bpp'],'task_losses':{k:float(v.detach()) for k,v in terms.items()},
               'identity_losses':baseline['terms'],'excess':excess,'npp_grad_norm':float(grad),
               'accepted':accepted,'step_scale':scale,'guard':details,'seconds':time.perf_counter()-start}
        if diagnostic is not None: row['direction_probes'] = diagnostic
        # Logging every proposal is essential when rejection rates differ between arms.
        with (out/'training.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        if step%32==0 or step+1==cfg['npp_steps']:
            emit(None,row)
        if (step+1)%cfg['anchor_every']==0 or step+1==cfg['npp_steps']:
            violations = check_anchors(npp,proxy,codec,tasks,samples,refs,baselines,cfg)
            if violations:
                restore_state(npp,optimizer,block)
                counts['rolled_back_proposals'] += block_accepts
                counts['rolled_back_blocks'] += 1
            else:
                counts['committed_proposals'] += block_accepts
            block = capture_state(npp,optimizer)
            block_accepts = 0
            emit(out/'guards.jsonl',{'stage':'quality_anchor','arm':arm,'step':step+1,
                                     'passed':not violations,'violations':violations,'counts':dict(counts)})
            save_checkpoint(out/'last.pt',npp,proxy,stored,'quality',step+1)
            torch.save({'optimizer':optimizer.state_dict(),'counts':counts,'step':step+1},out/'optimizer.pt')
    if arm=='quality_fixed' and tensor_digest(proxy)!=frozen:
        raise RuntimeError('fixed proxy changed')
    report = {'counts':counts,'refreshes':refreshes,'proxy_tensor_sha256':tensor_digest(proxy),
              'fixed_proxy_verified':arm!='quality_fixed' or tensor_digest(proxy)==frozen,
              'guard_scope':'current proposal and fixed TRAIN anchors; no DEV quality guarantee',
              'final_anchor_verified':not check_anchors(npp,proxy,codec,tasks,samples,refs,baselines,cfg)}
    if not report['final_anchor_verified']:
        raise RuntimeError('final TRAIN anchor guard failed after rollback')
    (out/'training_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ar-root','coco-root','annotations','checkpoint','config','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('quality recovery requires real CUDA GPU')
    cfg = configuration(args.config)
    if (cfg['arms']!=['quality_fixed','quality_refresh'] or not 0<cfg['filter_limit']<=.2
            or not 0<cfg['max_total_rms']<=8/255
            or cfg['npp_steps']%(2*cfg['train_count']*8)
            or cfg['proxy_refresh_every']%cfg['anchor_every']
            or cfg['npp_steps']%cfg['anchor_every']):
        raise ValueError('invalid paired quality experiment protocol')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=False)
    receipt = {'complete':False,'started':time.time(),'config':cfg,'source_hashes':source_hashes(),
               'source_commit':os.environ.get('AROD_SOURCE_COMMIT'),'device':torch.cuda.get_device_name(),
               'scope':'DEV recovery experiment; TEST untouched; no joint video OD GT','completed_arms':[]}
    status = args.out/'run.json'
    status.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    try:
        emit(None,{'stage':'quality_init','phase':'checkpoint'})
        base,calibrated,saved = load_checkpoint(args.checkpoint,device='cuda')
        if saved['format']!=2 or saved['stage']!='proxy_calibration':
            raise ValueError('calibrated format-2 checkpoint required')
        for key in ('width','frames','size','qps','codecs','preset','fps','od_frames'):
            if saved['config'][key]!=cfg[key]: raise ValueError('checkpoint codec protocol differs: '+key)
        calibrated.requires_grad_(False).eval()
        receipt['checkpoint_sha256'] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        manifest = manifest_for(args,cfg)
        prior = json.loads((args.checkpoint.parent/'manifest.json').read_text())
        train_ids,dev_ids = ({r['id'] for r in manifest[s]} for s in ('train','dev'))
        if not {r['id'] for r in prior['train']}<=train_ids:
            raise ValueError('expanded TRAIN must retain calibration TRAIN sources')
        if train_ids & {r['id'] for r in prior['dev']} or dev_ids & {r['id'] for r in prior['train']}:
            raise ValueError('calibration TRAIN/DEV leakage')
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        emit(None,{'stage':'quality_init','phase':'load_samples','train':len(manifest['train']),'dev':len(manifest['dev'])})
        train = samples_for(manifest['train'],cfg)
        dev = samples_for(manifest['dev'],cfg)
        tasks = TaskHeads(device='cuda',size=cfg['size'],od_frames=cfg['od_frames'])
        hashes = {k:tensor_digest(getattr(tasks,k)) for k in ('ar','od')}
        codec = RealCodec(preset=cfg['preset'],gop=cfg['frames'],fps=cfg['fps'])
        refs,masks = [],[]
        for index,(record,cpu,gt) in enumerate(train):
            x = cpu.cuda()
            reference = tasks.references(x,record,gt)
            refs.append(reference); masks.append(semantic_weights(tasks,x,record,reference))
            if index%16==0: emit(None,{'stage':'quality_references','completed':index+1,'total':len(train)})
        baselines,rows = {},[]
        for step in range(len(train)*8):
            index,name,qp = training_case(step,len(train),cfg)
            record,cpu,_ = train[index]
            result,_ = real_measure(tasks,codec,cpu.cuda(),record,refs[index],name,qp,case_seed(step,len(train),cfg))
            baselines[(index,name,qp)] = result
            rows.append({'source':record['id'],'codec':name,'qp':qp,**result})
            if (step+1)%32==0: emit(None,{'stage':'quality_identity','completed':step+1,'total':len(train)*8})
        (args.out/'identity_targets.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        # Each arm has identical reset trunk/output weights, masks, source schedule and proposal budget.
        summaries,training = {},{}
        initial = identity_start(base,cfg['filter_limit'])
        receipt['initial_npp_tensor_sha256'] = tensor_digest(initial)
        for arm in cfg['arms']:
            torch.manual_seed(cfg['seed'])
            npp,proxy = copy.deepcopy(initial),copy.deepcopy(calibrated)
            training[arm] = fit_quality(npp,proxy,codec,tasks,train,refs,masks,baselines,cfg,arm,args.out/arm)
            if any(tensor_digest(getattr(tasks,k))!=v for k,v in hashes.items()):
                raise RuntimeError('frozen task state changed')
            evaluation = args.out/arm/'evaluation'; evaluation.mkdir()
            summaries[arm] = evaluate(npp,proxy,codec,tasks,dev,cfg,args.annotations,evaluation)
            receipt['completed_arms'].append(arm)
            status.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
            del npp,proxy
        report = {'scope':receipt['scope'],'bd_rates':comparisons(summaries),'summaries':summaries,
                  'training':training,'confidence_intervals':None,'test_used':False}
        (args.out/'quality_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        receipt.update(complete=True,task_frozen_verified=True)
    except BaseException as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        receipt['wall_seconds'] = time.time()-receipt['started']
        status.write_text(json.dumps(receipt,indent=2,allow_nan=False),encoding='utf-8')
        emit(None,{'complete':receipt['complete'],'wall_seconds':receipt['wall_seconds']})


if __name__=='__main__':
    main()

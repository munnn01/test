"""Continuous real-forward/proxy-backward training and fixed-to-adaptive QP ablation."""
import copy
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from .engine import save_checkpoint
from .models import real_value_proxy_grad
from .quality import protected_objective
from .recovery import balanced_schedule,log_row,write_json,stored_config
from .screen import regularizers,task_terms,tensor_digest


def split_train(records,fit_count,selection_count,calibration_records):
    """Use deterministic prefix for fitting; subsequent TRAIN sources only for selection."""
    if len({r['id'] for r in records})!=len(records):raise ValueError('duplicate sources')
    fit_ids,held_ids=set(),set()
    for task in ('ar','od'):
        family=[r for r in records if r['task']==task]
        if len(family)!=fit_count+selection_count:raise ValueError('incorrect TRAIN family size')
        fit_ids.update(r['id'] for r in family[:fit_count])
        held_ids.update(r['id'] for r in family[fit_count:])
    if fit_ids&held_ids or not {r['id'] for r in calibration_records}<=fit_ids:
        raise ValueError('selection overlaps calibration fitting or calibration sources missing')
    return [r for r in records if r['id'] in fit_ids],[r for r in records if r['id'] in held_ids]


def phase_qps(arm,step,cfg):
    if arm not in ('continuous','curriculum'):raise ValueError('unknown paper arm')
    return [cfg['fixed_qp']] if arm=='curriculum' and step<cfg['fixed_steps'] else cfg['qps']


def training_schedule(records,arm,step,cfg):
    # Generate one shared source/codec ordering. Sampling a smaller QP grid would
    # consume RNG differently and confound the QP ablation with different batches.
    schedule=balanced_schedule(records,cfg['codecs'],cfg['qps'],cfg['seed'])
    if phase_qps(arm,step,cfg)==[cfg['fixed_qp']]:
        schedule=[[(i,c,cfg['fixed_qp']) for i,c,_ in batch] for batch in schedule]
    return schedule


def continuous_step(model,optimizer):
    """Quality constraints affect checkpoint selection, never the optimizer trajectory."""
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError('nonfinite NPP parameters after Adam')
    return float(norm)


def selection_report(rows,baselines,allowance,max_rms):
    cases=[tuple(r['case']) for r in rows]
    if len(set(cases))!=len(cases) or set(cases)!=set(baselines):
        raise ValueError('selection must cover every held-out codec/QP case exactly once')
    groups=defaultdict(list);rates=[];bounds=True
    for r in rows:
        case=tuple(r['case']);base=baselines[case];actual=r['measurement']
        rms=r['total_rms'];bounds &= math.isfinite(rms) and rms<=max_rms
        ratio=actual['bpp']/base['bpp'];rates.append(ratio)
        bounds &= math.isfinite(ratio) and ratio>0
        if set(actual['terms'])!=set(base['terms']):raise ValueError('task terms differ')
        for term,value in actual['terms'].items():
            relative=(value-base['terms'][term])/max(base['terms'][term],.1)
            groups[(case[1],case[2],r['task'],term)].append(relative)
    metrics=[]
    for key,values in sorted(groups.items()):
        mean=sum(values)/len(values)
        metrics.append({'codec':key[0],'qp':key[1],'source_task':key[2],'term':key[3],
                        'mean_relative_loss_delta':mean,'passed':math.isfinite(mean) and mean<=allowance+1e-6})
    correct=[]
    for codec,qp in sorted({tuple(r['case'][1:]) for r in rows if r['task']=='ar'}):
        group=[r for r in rows if r['task']=='ar' and tuple(r['case'][1:])==(codec,qp)]
        before=sum(bool(baselines[tuple(r['case'])]['ar_correct']) for r in group)
        after=sum(bool(r['measurement']['ar_correct']) for r in group)
        correct.append({'codec':codec,'qp':qp,'baseline_correct':before,'correct':after,'passed':after>=before})
    return {'passed':bool(bounds and all(m['passed'] for m in metrics+correct)),
            'bounds_passed':bool(bounds),'groups':metrics,'ar_correct_counts':correct,
            'mean_rate_ratio':sum(rates)/len(rates),'changed_rgb8_cases':sum(r['changed_rgb8'] for r in rows)}


@torch.no_grad()
def measure_selection(npp,context):
    rows=[]
    for case in context.all_cases():
        x=context.x(case);y=npp(x,case[1],case[2])
        rows.append({'case':case,'task':context.samples[case[0]][0]['task'],
                     'total_rms':float((y-x).square().mean().sqrt()),
                     'changed_rgb8':bool(((y*255).round()!=(x*255).round()).any()),
                     'measurement':context.measure(y,case)})
    report=selection_report(rows,context.baselines,context.cfg['quality_allowance'],context.cfg['max_total_rms'])
    report['cases']=rows
    return report


def sample_loss(npp,proxy,context,case,step):
    index,codec,qp=case;x=context.x(case)
    y=npp(x,codec,qp)
    actual=context.codec.roundtrip(y,codec,qp)
    prediction=proxy(y,codec,qp)
    decoded=real_value_proxy_grad(prediction.reconstruction,actual.reconstruction)
    rate=real_value_proxy_grad(prediction.bpp,actual.bpp)
    torch.manual_seed(context.seed(case))
    terms=task_terms(context.tasks.losses(decoded,context.samples[index][0],context.refs[index]),context.samples[index][0])
    baseline=context.baselines[case];cfg=context.cfg
    objective,_=protected_objective(rate,baseline['bpp'],terms,baseline['terms'],cfg['loss_weights'],
        cfg['quality_allowance'],penalty=cfg['quality_penalty'],rate_weight=min(1.,.05+.95*step/cfg['rate_warmup']))
    preservation,temporal=regularizers(x,y,context.masks[index].to(context.device))
    loss=objective+cfg['loss_weights']['pre']*preservation+cfg['loss_weights']['temp']*temporal
    if not torch.isfinite(loss):raise RuntimeError('nonfinite continuous objective')
    return loss,{'case':case,'bpp':float(actual.bpp.mean()),'output_rms':float((y-x).detach().square().mean().sqrt()),
                 'terms':{k:float(v.detach()) for k,v in terms.items()}}


def directional_audit(npp,proxy,context,path):
    """Probe local proxy directions against real codec; never optimize or select on this audit."""
    original=tensor_digest(npp);rows=[]
    indices=[next(i for i,s in enumerate(context.samples) if s[0]['task']==task) for task in ('ar','od')]
    for case in [c for c in context.all_cases() if c[0] in indices]:
        index,codec,qp=case;record=context.samples[index][0]
        with torch.no_grad():y=npp(context.x(case),codec,qp)
        real=context.measure(y,case)
        actual=context.codec.roundtrip(y,codec,qp)
        y=y.detach().requires_grad_();prediction=proxy(y,codec,qp)
        decoded=real_value_proxy_grad(prediction.reconstruction,actual.reconstruction)
        torch.manual_seed(context.seed(case))
        terms=task_terms(context.tasks.losses(decoded,record,context.refs[index]),record)
        objective=sum(context.cfg['loss_weights'][k]*v/max(real['terms'][k],.1) for k,v in terms.items())
        grads={'task':torch.autograd.grad(objective,y,retain_graph=True)[0].detach(),
               'rate':torch.autograd.grad(prediction.bpp.mean(),y)[0].detach()}
        mse=float((prediction.reconstruction.detach()-actual.reconstruction).square().mean())
        row={'case':case,'source':record['id'],'task':record['task'],'real':real,
             'proxy_reconstruction_l1':float((prediction.reconstruction.detach()-actual.reconstruction).abs().mean()),
             'proxy_reconstruction_psnr_db':-10*math.log10(max(mse,1e-12)),
             'proxy_rate_relative_error':float(prediction.bpp.detach().mean())/real['bpp']-1,
             'real_decoded_source_psnr_db':-10*math.log10(max(float((actual.reconstruction-context.x(case)).square().mean()),1e-12)),
             'directions':[]}
        for name,gradient in grads.items():
            rms=float(gradient.square().mean().sqrt())
            if not math.isfinite(rms):raise RuntimeError('nonfinite directional gradient')
            direction=-gradient/max(rms,1e-12)
            for epsilon in (.5/255,2/255):
                moved=(y.detach()+epsilon*direction).clamp(0,1)
                result=context.measure(moved,case)
                row['directions'].append({'objective':name,'epsilon_rms':epsilon,'gradient_rms':rms,
                    'proxy_linear_objective_delta':float((gradient*(moved-y.detach())).sum()),
                    'actual_step_rms':float((moved-y.detach()).square().mean().sqrt()),
                    'real_rate_relative_delta':result['bpp']/real['bpp']-1,
                    'real_task_relative_deltas':{k:(v-real['terms'][k])/max(real['terms'][k],.1) for k,v in result['terms'].items()},
                    'measurement':result})
        rows.append(row);log_row(path,row)
        print(json.dumps({'stage':'paper_direction_audit','case':case,'rows':len(rows)}),flush=True)
        del y,prediction,decoded,objective,terms,grads
    if tensor_digest(npp)!=original:raise RuntimeError('audit changed NPP')
    return {'cases':len(rows),'path':Path(path).name,'scope':'two TRAIN-selection sources; local finite-step diagnostic, not DEV'}


def fit_continuous(npp,proxy,context,selection,out,arm):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    cfg=context.cfg;initial=tensor_digest(npp);proxy_hash=tensor_digest(proxy)
    optimizer=torch.optim.Adam([
        {'params':[p for name,p in npp.named_parameters() if name!='filter_strength'],'lr':cfg['learning_rate']},
        {'params':[npp.filter_strength],'lr':cfg['filter_strength_learning_rate']}])
    initial_lrs=[g['lr'] for g in optimizer.param_groups]
    selected=copy.deepcopy(npp.state_dict());selected_step=0;best_ratio=1.
    save_checkpoint(out/'start.pt',npp,proxy,stored_config(cfg),'paper_start',0)
    before_audit=directional_audit(npp,proxy,selection,out/'directions_start.jsonl')
    records=[s[0] for s in context.samples];active_qps=None
    for step in range(cfg['npp_steps']):
        qps=phase_qps(arm,step,cfg)
        if qps!=active_qps:
            schedule=training_schedule(records,arm,step,cfg);active_qps=qps
        for group,lr in zip(optimizer.param_groups,initial_lrs):
            group['lr']=lr*(.1 if step>=cfg['lr_decay_step'] else 1.)
        batch=schedule[step%len(schedule)]
        optimizer.zero_grad(set_to_none=True);loss_value=0.;measurements=[]
        for case in batch:
            loss,measurement=sample_loss(npp,proxy,context,case,step)
            (loss/len(batch)).backward();loss_value+=float(loss.detach())/len(batch)
            measurements.append(measurement)
        grad=continuous_step(npp,optimizer)
        log_row(out/'training.jsonl',{'step':step+1,'qps':qps,'batch':batch,'loss':loss_value,
            'gradient_norm':grad,'learning_rates':[g['lr'] for g in optimizer.param_groups],'measurements':measurements})
        if (step+1)%16==0:
            print(json.dumps({'stage':'paper_training','arm':arm,'step':step+1,'loss':loss_value,'qps':qps}),flush=True)
        if (step+1)%cfg['selection_every']==0 or step+1==cfg['npp_steps']:
            panel=measure_selection(npp,selection)
            if panel['passed'] and panel['changed_rgb8_cases'] and panel['mean_rate_ratio']<best_ratio:
                selected=copy.deepcopy(npp.state_dict());selected_step=step+1;best_ratio=panel['mean_rate_ratio']
            log_row(out/'selection.jsonl',{'step':step+1,'selected_step':selected_step,'panel':panel})
            save_checkpoint(out/'last.pt',npp,proxy,stored_config(cfg),'paper_last',step+1)
            torch.save({'optimizer':optimizer.state_dict(),'step':step+1},out/'optimizer.pt')
            print(json.dumps({'stage':'paper_selection','arm':arm,'step':step+1,
                'passed':panel['passed'],'mean_rate_ratio':panel['mean_rate_ratio'],'selected_step':selected_step}),flush=True)
    # Audit last trajectory even if selection falls back to identity; preserve evidence of failure.
    after_audit=directional_audit(npp,proxy,selection,out/'directions_last.jsonl')
    last_digest=tensor_digest(npp)
    npp.load_state_dict(selected,strict=True)
    final=measure_selection(npp,selection)
    if not final['passed']:raise RuntimeError('selected checkpoint failed held-out recheck')
    if tensor_digest(proxy)!=proxy_hash:raise RuntimeError('proxy changed')
    save_checkpoint(out/'selected.pt',npp,proxy,stored_config(cfg),'paper_selected',selected_step)
    report={'steps':cfg['npp_steps'],'rollback_steps':0,'selected_step':selected_step,
        'best_selection_rate_ratio':best_ratio,'final_selection':final,'fixed_proxy_verified':True,
        'selected_differs_from_start':tensor_digest(npp)!=initial,'last_differs_from_start':last_digest!=initial,
        'start_audit':before_audit,'last_audit':after_audit,'selection':'minimum feasible held-out TRAIN mean rate ratio; identity fallback',
        'scope':'continuous surrogate training; aggregate held-out TRAIN selection is not a DEV guarantee'}
    write_json(out/'training_summary.json',report)
    return report


def validate_config(cfg):
    if (cfg['arms']!=['continuous','curriculum'] or cfg['npp_steps']!=512 or cfg['fixed_steps']!=128
            or cfg['fixed_qp']!=35 or cfg['fit_count']!=32 or cfg['selection_count']!=8
            or cfg['train_count']!=40 or cfg['selection_every']!=128 or cfg['lr_decay_step']!=384
            or cfg['large_base_channels']!=29 or cfg['checkpoint_frames'] is not True
            or cfg['quality_allowance']!=.02 or cfg['max_total_rms']!=6/255):
        raise ValueError('paper pilot requires fixed matched capacity, split and equal training budgets')


def main():
    import argparse,hashlib,json,os,random,time
    import numpy as np
    from .engine import load_checkpoint,source_hashes
    from .capacity import model_profile
    from .large_npp import LargeRegionalVideoNPP
    from .quality import identity_start,semantic_weights
    from .recovery import RecoveryContext,validate_manifest
    from .bdrate import comparisons
    from .codec import RealCodec
    from .evaluate import evaluate
    from .run import configuration,manifest_for,samples_for
    from .tasks import TaskHeads

    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('ar-root','coco-root','annotations','checkpoint','config','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',choices=('continuous','curriculum'),required=True)
    args=parser.parse_args();cfg=configuration(args.config);validate_config(cfg)
    if not torch.cuda.is_available():raise RuntimeError('paper training experiment requires real CUDA GPU')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    args.out.mkdir(parents=True,exist_ok=False)
    receipt={'complete':False,'started':time.time(),'config':cfg,'paper_arm':args.arm,
             'source_commit':os.environ.get('AROD_SOURCE_COMMIT'),'source_hashes':source_hashes(),
             'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),
             'precision':'FP32; TF32 disabled; cuDNN deterministic; full deterministic algorithms not enforced',
             'scope':'TRAIN continuous/QP curriculum pilot, final DEV; single seed; no joint video OD GT; TEST untouched'}
    status=args.out/'run.json';write_json(status,receipt)
    try:
        base,proxy,saved=load_checkpoint(args.checkpoint,device='cuda')
        if saved['format']!=2 or saved['stage']!='proxy_calibration':
            raise ValueError('calibrated format2 checkpoint required')
        for key in ('width','frames','size','qps','codecs','preset','fps','od_frames'):
            if cfg[key]!=saved['config'][key]:raise ValueError('checkpoint protocol differs: '+key)
        proxy.requires_grad_(False).eval()
        proxy_hash=tensor_digest(proxy)
        receipt.update(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                       proxy_tensor_sha256=proxy_hash)
        manifest=manifest_for(args,cfg)
        validate_manifest(manifest,json.loads((args.checkpoint.parent/'manifest.json').read_text()))
        fit_records,selection_records=split_train(manifest['train'],cfg['fit_count'],cfg['selection_count'],
            json.loads((args.checkpoint.parent/'manifest.json').read_text())['train'])
        manifest.update(fit=fit_records,selection=selection_records)
        write_json(args.out/'manifest.json',manifest)
        # Common parent initialization is explicitly seeded independent of dataset/ROI RNG.
        torch.manual_seed(cfg['seed'])
        parent=identity_start(base,cfg['filter_limit'])
        receipt['common_parent_tensor_sha256']=tensor_digest(parent)
        torch.manual_seed(cfg['seed']+1)
        npp=LargeRegionalVideoNPP.from_regional(parent,cfg['large_base_channels'],cfg['checkpoint_frames'])
        arm_cfg=dict(cfg,npp_kind='large_regional',paper_arm=args.arm)
        count=sum(p.numel() for p in npp.parameters())
        if count!=9_895_548:raise ValueError('matched capacity differs')
        receipt.update(npp_config=arm_cfg,npp_initial_tensor_sha256=tensor_digest(npp),parameters=count)
        write_json(status,receipt)
        print(json.dumps({'stage':'paper_init','arm':args.arm,'parameters':count,
                          'proxy_width':cfg['width'],'common_parent_sha256':receipt['common_parent_tensor_sha256']}),flush=True)
        train=samples_for(manifest['fit'],cfg)
        held=samples_for(manifest['selection'],cfg)
        video=next(x for record,x,_ in train if record['task']=='ar').cuda()
        receipt['model_profile']=model_profile(npp,video,'h264',35)
        write_json(args.out/'model_profile.json',receipt['model_profile'])
        print(json.dumps({'stage':'paper_profile','arm':args.arm,**receipt['model_profile']}),flush=True)
        del video,parent,base
        tasks=TaskHeads(device='cuda',size=cfg['size'],od_frames=cfg['od_frames'])
        task_hashes={name:tensor_digest(getattr(tasks,name)) for name in ('ar','od')}
        codec=RealCodec(preset=cfg['preset'],gop=cfg['frames'],fps=cfg['fps'])
        contexts={}
        for label,samples in (('fit',train),('selection',held)):
            refs,masks=[],[]
            for index,(record,cpu,gt) in enumerate(samples):
                x=cpu.cuda();reference=tasks.references(x,record,gt)
                refs.append(reference)
                masks.append(semantic_weights(tasks,x,record,reference).cpu() if label=='fit' else torch.ones_like(cpu[:,:,:1]))
                if (index+1)%8==0:print(json.dumps({'stage':'paper_references','split':label,'sources':index+1,'total':len(samples)}),flush=True)
            context=RecoveryContext(samples,refs,masks,tasks,codec,arm_cfg)
            baseline_rows=[]
            for case in context.all_cases():
                value=context.measure(context.x(case),case)
                context.baselines[case]=value
                baseline_rows.append({'case':case,'source':samples[case[0]][0]['id'],**value})
                if len(baseline_rows)%32==0:
                    print(json.dumps({'stage':'paper_identity','split':label,'cases':len(baseline_rows),'total':len(samples)*8}),flush=True)
            write_json(args.out/(label+'_identity_targets.json'),baseline_rows)
            contexts[label]=context
        torch.cuda.reset_peak_memory_stats()
        print(json.dumps({'stage':'paper_train_start','arm':args.arm,'budget':cfg['npp_steps']}),flush=True)
        training=fit_continuous(npp,proxy,contexts['fit'],contexts['selection'],args.out/args.arm,args.arm)
        receipt['training_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        receipt['training_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        if tensor_digest(proxy)!=proxy_hash or any(tensor_digest(getattr(tasks,k))!=v for k,v in task_hashes.items()):
            raise RuntimeError('fixed proxy/task state changed')
        # Only the selected TRAIN checkpoint is evaluated; no DEV-driven model selection.
        dev=samples_for(manifest['dev'],cfg)
        destination=args.out/args.arm/'evaluation';destination.mkdir()
        summary=evaluate(npp,proxy,codec,tasks,dev,cfg,args.annotations,destination)
        report={'paper_arm':args.arm,'parameters':count,'model_profile':receipt['model_profile'],
                'training':training,'summary':summary,'bd_rates':comparisons({args.arm:summary}),
                'test_used':False,'confidence_intervals':None,'scope':receipt['scope']}
        write_json(args.out/'paper_summary.json',report)
        receipt.update(complete=True,proxy_frozen_verified=True,task_frozen_verified=True)
    except BaseException as error:
        receipt['error']=f'{type(error).__name__}: {error}';raise
    finally:
        receipt['wall_seconds']=time.time()-receipt['started'];write_json(status,receipt)
        print(json.dumps({'stage':'paper_finished','arm':args.arm,'complete':receipt['complete'],
                          'wall_seconds':receipt['wall_seconds']}),flush=True)


if __name__=='__main__':main()

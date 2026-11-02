"""Original-NPP teacher warm-up and matched TRAIN-calibrated task-gain experiments."""
import argparse,copy,gc,hashlib,json,math,os,random,shutil,time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from . import tri_task_study as study
from . import sam_teacher_recipe as recipe
from .assets import digest,records_for_task,dataset_mount
from .training import rng_state,restore_rng,config_digest,records_digest,log_event
from .proxy_recovery import Budget,BudgetExpired,model_digest
from .sam_power_recover import state_digest,verify_parent,validate_sources,PARENT_SOURCE
from .sam_power_study import config as old_config
from .sam_structured import SAMGaussianNPP,SAMTask,StructuredMaskCache,prepare_structured_masks
from .tri_task_models import PaperNPP
from .tri_task_data import load,enrich_masks,PUBLIC_PROXY
from .codec_content_rate import NALCodec,ContentProxy,gradient_balance
from .coco_content_rate_study import calibrate_content,audit_content,_relaxed
from .protocol import QPS,real_forward,proxy_lambda
from .tasks import freeze

def rng_digest():
    state=rng_state();h=hashlib.sha256()
    h.update(repr(state['python']).encode());n=state['numpy']
    h.update(repr((n[0],*n[2:])).encode());h.update(n[1].tobytes());h.update(state['torch'].cpu().numpy().tobytes())
    for v in state['cuda']:h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()

@contextmanager
def preserve_rng():
    saved=rng_state()
    try:yield
    finally:restore_rng(saved)

def seeded_task(task,x,target,seed,receipt=False):
    with preserve_rng():
        random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed)
        if x.is_cuda:torch.cuda.manual_seed_all(seed)
        before=rng_digest();loss=task.loss(x,target)
    return (loss,before) if receipt else loss

def npp_train(npp):
    npp.train();study.freeze_npp_bn(npp)

def validate_checkpoint(saved,cfg,runtime,source,shared):
    if (saved.get('schema')!=1 or saved.get('source_sha256')!=source or saved.get('config')!=cfg
        or saved.get('config_sha256')!=config_digest(cfg) or saved.get('runtime')!=runtime
        or saved.get('runtime_sha256')!=config_digest(runtime) or saved.get('shared_sha256')!=shared):
        raise ValueError('checkpoint source/config/cohort/shared state differs')
    if saved.get('phase')!='task' or saved.get('arm') not in recipe.ARMS or not saved.get('adaptive'):
        raise ValueError('adaptive task-phase checkpoint required')
    if not 0<=saved.get('task_updates',-1)<=cfg.get('task_steps',1536):raise ValueError('task counter differs')
    if not saved.get('optimizer') or not saved.get('rng') or not saved.get('gains'):raise ValueError('optimizer/RNG/gains missing')
    if saved.get('gains_sha256')!=config_digest(saved['gains']):raise ValueError('checkpoint task gains differ')
    if state_digest(saved['npp'])!=saved.get('npp_sha256') or state_digest(saved['proxy'])!=saved.get('proxy_sha256'):
        raise ValueError('checkpoint model/proxy content differs')


def prepare_maps(rows,cfg,out,budget):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    return prepare_structured_masks(rows,cfg,out,budget)

def validate_parent_cohort(runtime,cohort):
    identity=cohort['identity']
    if (runtime['train']!=identity['train'] or runtime['proxy_dev']!=identity['dev']
        or runtime['datasets']!=identity['datasets']):
        raise ValueError('parent training annotations or dataset versions differ')

def copy_codec_cache(anchor,dest,cfg,rows,task,annotation):
    anchor=Path(anchor);dest=Path(dest);identity=json.loads((anchor/'evaluation_identity.json').read_text())
    identity.pop('npp_sha256')
    expected=dict(config_sha256=config_digest(cfg),sources=[r['id'] for r in rows],records_sha256=records_digest(rows),
        task_sha256=task.weight_digest,annotation_sha256=digest(annotation) if annotation else None)
    if identity!=expected:raise ValueError('codec-only cache inference/cohort/task/annotation differs')
    target=dest/'cache/codec_only'
    if not target.exists():shutil.copytree(anchor/'cache/codec_only',target)

def evaluate(npp,cfg,rows,task,codec,out,annotation,budget,anchor=None):
    with preserve_rng():
        if anchor is not None:copy_codec_cache(anchor,out,cfg,rows,task,annotation)
        result=study.evaluate(cfg,rows,npp,task,codec,Path(out),annotation,budget)
        if not result.get('complete'):raise BudgetExpired('evaluation incomplete')
    return result

def student_fidelity(npp,samples,cfg,budget):
    device=next(npp.parameters()).device;old=npp.training;npp.eval();teacher=[];source=[]
    with torch.no_grad():
        for x,_,_ in samples:
            budget.require('student_fidelity');x=x.to(device);y=npp(x,35);t=recipe.teacher_target(x,cfg['teacher_sigma'])
            teacher.append(float((y-t).square().mean()));source.append(float((y-x).square().mean()))
    npp.train(old)
    if old:study.freeze_npp_bn(npp)
    psnr=lambda v:-10*math.log10(max(stat_mean(v),1e-12))
    return dict(teacher_psnr_db=psnr(teacher),source_psnr_db=psnr(source),sources=len(samples))
def stat_mean(values):return sum(values)/len(values)

def warm_student(npp,samples,dev,cfg,budget,out,runtime=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);device=next(npp.parameters()).device
    optimizer=torch.optim.Adam(npp.parameters(),lr=cfg['warmup_lr']);npp_train(npp);passed=False;probe={};updates=0
    for step in range(cfg['warmup_steps']):
        if budget.remaining()<cfg['evaluation_reserve']+1800:break
        x=samples[random.randrange(len(samples))][0].to(device);target=recipe.teacher_target(x,cfg['teacher_sigma'])
        optimizer.zero_grad(set_to_none=True);loss=(npp(x,35)-target).square().mean();study.finite_step(loss,npp,optimizer);updates+=1
        if updates==1 or updates%127==0:log_event(out,'teacher_warm',step=updates,mse=float(loss.detach()),lr=cfg['warmup_lr'])
        if updates%cfg['warmup_probe_every']==0 or updates==cfg['warmup_steps']:
            probe=student_fidelity(npp,dev,cfg,budget)
            passed=updates>=cfg['warmup_min_steps'] and probe['teacher_psnr_db']>=cfg['teacher_psnr_gate'] and probe['source_psnr_db']>=cfg['collapse_psnr']
            torch.save(dict(phase='teacher',config=cfg,config_sha256=config_digest(cfg),runtime=runtime,
                source_sha256=os.environ.get('PAPER_SOURCE_SHA256'),npp=npp.state_dict(),optimizer=optimizer.state_dict(),
                rng=rng_state(),teacher_updates=updates,adaptive=npp.adaptive),'%s/teacher_last.pt'%out)
            log_event(out,'teacher_fidelity',updates=updates,passed=passed,**probe)
            if passed:break
    return dict(passed=passed,updates=updates,optimizer_lr=cfg['warmup_lr'],**probe)

def refit_mixture(proxy,npp,samples,cfg,codec,budget,out):
    proxy.requires_grad_(False);proxy.bank.train();device=next(proxy.bank.parameters()).device
    before={k:v.detach().cpu().clone() for k,v in proxy.bank.state_dict().items() if '.g_s.' not in k}
    optimizers={}
    for q in QPS:
        decoder=proxy.bank.networks[str(q)].g_s;decoder.requires_grad_(True)
        optimizers[q]=torch.optim.Adam(decoder.parameters(),lr=1e-4)
    counts=dict(x=0,npp=0,gaussian=0);updates=0;old=npp.training;npp.eval()
    for i in range(cfg['proxy_steps']):
        if budget.remaining()<cfg['evaluation_reserve']+1200:break
        q=QPS[i%4];variant=('x','npp','gaussian')[(i//4)%3]
        x=samples[(i//12)%len(samples)][0].to(device)
        with torch.no_grad():
            z=x if variant=='x' else npp(x,q) if variant=='npp' else recipe.teacher_target(x,cfg['teacher_sigma'])
            actual=codec(z,cfg['codecs'][0],q)
        optimizers[q].zero_grad(set_to_none=True);prediction=proxy(z,q)
        loss=proxy_lambda(q)*(prediction.reconstruction-actual.reconstruction).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite mixed reconstruction loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(proxy.bank.networks[str(q)].g_s.parameters(),10.,error_if_nonfinite=True)
        optimizers[q].step();counts[variant]+=1;updates+=1
        if i<4 or i%51==0:log_event(out,'teacher_proxy_fit',step=updates,qp=q,variant=variant,loss=float(loss.detach()))
    for k,v in before.items():
        if not torch.equal(v,proxy.bank.state_dict()[k].detach().cpu()):raise RuntimeError('frozen analysis/entropy changed')
    freeze(proxy);npp.train(old)
    if old:study.freeze_npp_bn(npp)
    return dict(updates=updates,target=cfg['proxy_steps'],variant_counts=counts,analysis_entropy_unchanged=True)

def parameter_norm(loss,npp):
    params=[p for p in npp.parameters() if p.requires_grad]
    grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    selected=[g for g in grads if g is not None]
    if not selected or any(not torch.isfinite(v).all() for v in selected):raise RuntimeError('missing/nonfinite calibration gradients')
    value=float(torch.sqrt(sum(v.detach().float().square().sum() for v in selected)))
    if value<=0:raise RuntimeError('zero calibration gradient')
    return value

def calibrate_gains(npp,proxy,task,samples,cfg,codec,budget,out):
    rows=[];device=next(npp.parameters()).device;npp_train(npp)
    with preserve_rng():
        for i,(x,target,meta) in enumerate(samples[:cfg['gain_sources']]):
            x=x.to(device)
            for q in QPS:
                budget.require('TRAIN_task_gain_calibration');seed=cfg['seed']+i*101+q
                y=npp(x,q);p=_relaxed(proxy,y,q,seed);actual=codec(y,cfg['codecs'][0],q)
                t=seeded_task(task,real_forward(p.reconstruction,actual.reconstruction),target,seed)
                original=cfg['lambda_scale']*study.task_lambda(q)*t
                rows.append(dict(split='train',source=meta['id'],qp=q,rate_norm=parameter_norm(p.bpp,npp),task_norm=parameter_norm(original,npp)))
    gains=recipe.fit_task_gains(rows,cfg['gain_target_ratio'],cfg['gain_min'],cfg['gain_max'])
    (Path(out)/'task_gain_calibration.json').write_text(json.dumps(dict(rows=rows,gains=gains),indent=2))
    return gains

def finite_probe(npp,proxy,task,samples,maps,cfg,arm,gains,budget,codec=None):
    codec=codec or getattr(proxy,'codec',None)
    if codec is None:raise ValueError('explicit real codec required for finite probes')
    rows=[];device=next(proxy.bank.parameters()).device;old=npp.training;npp.eval()
    with preserve_rng():
        for i,(sample,target,meta) in enumerate(samples[:2]):
            x=sample.to(device);mask=maps[i]
            for q in QPS:
                budget.require('finite_task_objective_probe');seed=cfg['seed']+9000+i*101+q
                with torch.no_grad():z=npp(x,q)
                v=z.detach().requires_grad_(True);p=_relaxed(proxy,v,q,seed);actual=codec(v,cfg['codecs'][0],q)
                t=seeded_task(task,real_forward(p.reconstruction,actual.reconstruction),target,seed)
                total,_=recipe.objective(cfg,arm,real_forward(p.bpp,p.bpp.new_tensor(actual.bpp)),t,x,v,q,mask,gains)
                gradients=dict(task=torch.autograd.grad(t,v,retain_graph=True)[0],full=torch.autograd.grad(total,v)[0])
                for component,gradient in gradients.items():
                    if not torch.isfinite(gradient).all() or gradient.abs().sum()==0:raise RuntimeError('invalid '+component+' pixel gradient')
                    for eps in cfg['probe_epsilons']:
                        values=[]
                        for sign in (-1,1):
                            perturbed=(z+sign*eps*gradient.sign()).clamp(0,1)
                            with torch.no_grad():
                                real=codec(perturbed,cfg['codecs'][0],q)
                                native,rng_hash=seeded_task(task,real.reconstruction,target,seed,True)
                                paid=native.new_tensor(real.bpp)
                                full,_=recipe.objective(cfg,arm,paid,native,x,perturbed,q,mask,gains)
                            values.append(dict(task=float(native),full=float(full),bpp=real.bpp,rng=rng_hash))
                        minus,plus=values;delta=plus[component]-minus[component]
                        tolerance=max(1e-8,1e-5*max(abs(minus[component]),abs(plus[component])))
                        rows.append(dict(source=meta['id'],qp=q,epsilon=eps,component=component,minus=minus,plus=plus,
                            aligned=delta>tolerance,tied=abs(delta)<=tolerance,rng_minus=minus['rng'],rng_plus=plus['rng']))
    npp.train(old)
    if old:study.freeze_npp_bn(npp)
    eligible=[r for r in rows if r['component']=='full' and not r['tied']]
    task_rows=[r for r in rows if r['component']=='task' and not r['tied']]
    alignment=sum(r['aligned'] for r in eligible)/max(1,len(eligible))
    gate=dict(passed=len(eligible)>=cfg['full_direction_eligible'] and alignment>=cfg['full_direction_min'],
        untied=len(eligible),alignment=alignment,total=sum(r['component']=='full' for r in rows))
    return dict(rows=rows,gate=gate,task_diagnostic=dict(untied=len(task_rows),alignment=sum(r['aligned'] for r in task_rows)/max(1,len(task_rows))),
        scope='finite clamped pixel probes; not a global derivative proof')

def save_task(path,npp,proxy,optimizer,cfg,runtime,arm,gains,shared,step,candidates):
    data=dict(schema=1,config=cfg,config_sha256=config_digest(cfg),runtime=runtime,runtime_sha256=config_digest(runtime),
        source_sha256=os.environ.get('PAPER_SOURCE_SHA256'),phase='task',arm=arm,adaptive=npp.adaptive,task_updates=step,
        npp=npp.state_dict(),npp_sha256=model_digest(npp),proxy=proxy.state_dict(),proxy_sha256=model_digest(proxy),
        optimizer=optimizer.state_dict(),rng=rng_state(),gains=gains,gains_sha256=config_digest(gains),shared_sha256=shared,candidates=candidates)
    temp=Path(path).with_suffix('.writing');torch.save(data,temp);os.replace(temp,path)

def train_arm(npp,proxy,task,codec,samples,maps,cfg,arm,gains,budget,out,runtime,shared,rank_callback,dev=None,resume=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);device=next(npp.parameters()).device
    optimizer=torch.optim.Adam(npp.parameters(),lr=cfg['lr']);step=0;candidates={};reason=None
    if resume is not None:
        validate_checkpoint(resume,cfg,runtime,os.environ.get('PAPER_SOURCE_SHA256'),shared)
        if resume['arm']!=arm or resume['gains']!=gains:raise ValueError('resume arm or frozen gains differ')
        npp.load_state_dict(resume['npp']);proxy.load_state_dict(resume['proxy']);optimizer.load_state_dict(resume['optimizer'])
        restore_rng(resume['rng']);step=resume['task_updates'];candidates=dict(resume.get('candidates',{}))
    initial={k:v.detach().cpu().clone() for k,v in npp.named_parameters()}
    proxy_sha=model_digest(proxy);task_sha=task.weight_digest;npp_train(npp)
    # last.pt precedes potentially interrupted ranking. Retry that boundary
    # before another task update, including a resume at the final task step.
    if step and step%cfg['checkpoint_every']==0 and step not in candidates:
        with preserve_rng():candidates[step]=rank_callback(npp,step)
        npp_train(npp)
        save_task(out/'last.pt',npp,proxy,optimizer,cfg,runtime,arm,gains,shared,step,candidates)
    fidelity={}
    while step<cfg['task_steps']:
        if budget.remaining()<cfg['evaluation_reserve']:reason='task_budget';break
        i=random.randrange(len(samples));x,target,meta=samples[i];x=x.to(device);q=QPS[step%4]
        optimizer.zero_grad(set_to_none=True);y=npp(x,q);p=_relaxed(proxy,y,q,cfg['seed']+200000+step)
        actual=codec(y,cfg['codecs'][0],q);t=seeded_task(task,real_forward(p.reconstruction,actual.reconstruction),target,cfg['seed']+100000+step)
        rate=real_forward(p.bpp,p.bpp.new_tensor(actual.bpp));loss,parts=recipe.objective(cfg,arm,rate,t,x,y,q,maps[i],gains)
        if step%128<4:
            coeff=recipe.task_coefficient(cfg,arm,q,gains);balance=gradient_balance(rate,coeff*t,loss-rate-coeff*t,npp.parameters())
            log_event(out,'teacher_component_gradients',arm=arm,step=step+1,qp=q,**balance)
        norm=study.finite_step(loss,npp,optimizer);step+=1
        if step==1 or step%cfg['log_every']==0:
            log_event(out,'teacher_task_train',arm=arm,step=step,qp=q,source=meta['id'],loss=float(loss.detach()),
                paid_bpp=actual.bpp,task_loss=float(t.detach()),gradient_norm=norm,lr=optimizer.param_groups[0]['lr'],
                task_gain=float(parts['task_gain']))
        if step%128==0:
            fidelity=student_fidelity(npp,dev or samples[:2],cfg,budget)
            if fidelity['source_psnr_db']<cfg['collapse_psnr']:reason='source_fidelity_collapse';break
        if step%cfg['checkpoint_every']==0:
            snap=out/('npp_'+str(step)+'.pt');torch.save(dict(npp=npp.state_dict(),npp_sha256=model_digest(npp),task_updates=step,
                config_sha256=config_digest(cfg),runtime_sha256=config_digest(runtime),source_sha256=os.environ.get('PAPER_SOURCE_SHA256'),shared_sha256=shared),snap)
            save_task(out/'last.pt',npp,proxy,optimizer,cfg,runtime,arm,gains,shared,step,candidates)
            with preserve_rng():candidates[step]=rank_callback(npp,step)
            npp_train(npp)
            save_task(out/'last.pt',npp,proxy,optimizer,cfg,runtime,arm,gains,shared,step,candidates)
    if reason is None:reason='task_schedule_finished'
    save_task(out/'last.pt',npp,proxy,optimizer,cfg,runtime,arm,gains,shared,step,candidates)
    if model_digest(proxy)!=proxy_sha or model_digest(task.model)!=task_sha:raise RuntimeError('frozen proxy/task changed')
    delta=float(torch.sqrt(sum((v.detach().cpu()-initial[k]).square().sum() for k,v in npp.named_parameters())))
    result=dict(training_updates=step,stop_reason=reason,candidates=candidates,theta_delta_norm=delta,
        proxy_sha256=proxy_sha,task_sha256=task_sha,last_fidelity=fidelity,optimizer_lr=optimizer.param_groups[0]['lr'])
    (out/'training.json').write_text(json.dumps(result,indent=2));return result

def read_json(path):return json.loads(Path(path).read_text())
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--proxy',type=Path,required=True);parser.add_argument('--proxy-sha256',required=True)
    parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--mask-parent',type=Path,required=True)
    parser.add_argument('--input-spec',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();cfg=read_json(args.config);spec=read_json(args.input_spec);out=args.out;out.mkdir(parents=True,exist_ok=True)
    if cfg!=recipe.config(cfg['task'],cfg['codecs'][0]):raise ValueError('teacher recipe differs')
    source=os.environ.get('PAPER_SOURCE_SHA256')
    if not source:raise ValueError('source manifest required')
    budget=Budget(cfg['max_session_seconds']);started=time.time()
    report=dict(source_sha256=source,config=cfg,experiment_complete=False,phase='setup',arms={},tracking='DEFERRED')
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        if args.proxy_sha256!=PUBLIC_PROXY['sha256'] or digest(args.proxy)!=args.proxy_sha256:raise ValueError('public proxy input differs')
        validate_sources(Path(__file__).resolve().parents[1],spec['source_hashes'],PARENT_SOURCE)
        for name,sha in spec['checkpoint_files'].items():
            if digest(args.parent/name)!=sha:raise ValueError('parent checkpoint blob differs: '+name)
        prior=verify_parent(args.parent,spec,old_config(cfg['task'],'control',cfg['codecs'][0]))
        for name,sha in spec['mask_files'].items():
            if digest(args.mask_parent/name)!=sha:raise ValueError('SAM parent report differs')
        mask_cloud=read_json(args.mask_parent/'cloud_status.json');mask_report=read_json(args.mask_parent/'foreground/study_run.json')
        if (not mask_cloud['bootstrap_complete'] or mask_cloud['worker_exit_code']!=0
            or mask_cloud['source_sha256']!=spec['mask_source_sha256'] or mask_report['state']['task_updates']!=768
            or mask_report['warmup_npp_sha256']!=prior['control/study_run.json']['warmup_npp_sha256']
            or mask_report['frozen_proxy_sha256']!=prior['control/study_run.json']['frozen_proxy_sha256']):
            raise ValueError('SAM cache parent lineage differs')
        report['parent']=dict(ref=spec['ref'],mask_ref=spec['mask_ref'],checkpoint_files=spec['checkpoint_files'])
        original_train,original_val,inventory=records_for_task('od',Path('/kaggle/input'))
        old_cohort=prior['control/cohort.json']
        train,dev,rank,final=recipe.cohorts(original_train,original_val,old_cohort,cfg)
        annotation_root=dataset_mount(Path('/kaggle/input'),'awsaf49/coco-2017-dataset')
        valpaths=list(annotation_root.rglob('instances_val2017.json'))
        if len(valpaths)!=1:raise ValueError('one official COCO VAL annotation required')
        val_annotation=valpaths[0];train_annotation=val_annotation.with_name('instances_train2017.json')
        if cfg['task']=='is':
            train=enrich_masks(train,train_annotation);dev=enrich_masks(dev,train_annotation)
            rank=enrich_masks(rank,train_annotation);final=enrich_masks(final,val_annotation)
        partitions=dict(train=train,proxy_dev=dev,ranking=rank,final_validation=final)
        runtime={name:records_digest(rows) for name,rows in partitions.items()};runtime['datasets']=inventory['dataset_versions']
        validate_parent_cohort(runtime,old_cohort)
        report['runtime']=runtime
        write_json(out/'cohorts.json',dict(identity=runtime,previous_val_excluded=old_cohort['validation'],
            **{name:[r['id'] for r in rows] for name,rows in partitions.items()}))
        target_cache=out/'shared_sam_maps'
        if target_cache.exists():raise ValueError('fresh output required; do not silently mix prior caches')
        shutil.copytree(args.mask_parent/'shared_sam_maps',target_cache)
        verifier=StructuredMaskCache(target_cache,cfg['sam2'])
        for row in train:
            x,_,_=load(row,cfg);verifier.load(row,x)
        cache,cache_info=prepare_maps(train+dev[:2],cfg,out/'mask_preparation',budget)
        report['mask_sources']=len(cache_info['sources']);report['mask_rendered_sha256']=cache_info['rendered_cohort_sha256']
        samples=[load(row,cfg) for row in train];dev_samples=[load(row,cfg) for row in dev]
        maps=[cache.load(row,sample[0]) for row,sample in zip(train,samples)]
        dev_maps=[cache.load(row,sample[0]) for row,sample in zip(dev[:2],dev_samples[:2])]
        codec=NALCodec(cfg['codecs'][0],preset=cfg['preset'],gop=cfg['gop'],fps=cfg['fps'])
        parent=torch.load(args.parent/'shared_warmup.pt',map_location='cpu',weights_only=False)
        from .sam_power_recover import migrate_warm
        migrate_warm(parent,prior['control/study_run.json'],prior['control/config.json'],PARENT_SOURCE,PARENT_SOURCE)
        npp=PaperNPP().to('cuda');npp.load_state_dict(parent['npp'])
        from .image_proxy import ImageProxyBank
        from .image_proxy_audit import restore_proxy_state
        bank=ImageProxyBank(cfg['proxy_quality'],pretrained=False)
        restore_proxy_state(bank,{k[len('bank.'):]:v for k,v in parent['proxy'].items() if k.startswith('bank.')})
        proxy=ContentProxy(bank,codec).to('cuda');proxy.load_state_dict(parent['proxy']);freeze(proxy)
        if model_digest(proxy)!=prior['control/study_run.json']['frozen_proxy_sha256']:raise ValueError('restored proxy differs')
        del parent;gc.collect()
        report['phase']='teacher_warmup';log_event(out,'teacher_ready',task=cfg['task'],codec=cfg['codecs'][0],runtime=runtime)
        warm=warm_student(npp,samples,dev_samples,cfg,budget,out/'warmup',runtime);report['teacher_warmup']=warm
        write_json(out/'teacher_warmup.json',warm)
        if not warm['passed']:raise RuntimeError('teacher fidelity gate failed; no task optimization')
        npp=PaperNPP.from_fixed(npp);npp_train(npp)
        report['phase']='proxy_refit';proxy_out=out/'proxy';proxy_out.mkdir(exist_ok=True)
        fit=refit_mixture(proxy,npp,samples,cfg,codec,budget,proxy_out);report['proxy_fit']=fit
        if fit['updates']!=cfg['proxy_steps']:raise BudgetExpired('mixed proxy fit incomplete')
        report['content_calibration']=calibrate_content(proxy,npp,samples,cfg,codec,proxy_out,budget)
        initial=audit_content(proxy,npp,dev_samples,cfg,codec,proxy_out,budget,'initial');report['initial_proxy_gate']=initial['gate']
        if not initial['gate']['passed']:raise RuntimeError('held-out content proxy gate failed')
        freeze(proxy);proxy_sha=model_digest(proxy);warm_sha=model_digest(npp)
        task=study.initialize_task(lambda:SAMTask(cfg['task'],device='cuda'));write_json(out/'analyzer.json',task.receipt())
        gains=calibrate_gains(npp,proxy,task,samples,cfg,codec,budget,out)
        if model_digest(proxy)!=proxy_sha or model_digest(npp)!=warm_sha:raise RuntimeError('gain calibration changed frozen/warm state')
        report['task_gains']=gains;report['phase']='native_gradient_probes'
        probes={}
        for arm in recipe.ARMS:
            probes[arm]=finite_probe(npp,proxy,task,dev_samples[:2],dev_maps,cfg,arm,gains,budget,codec=codec)
            write_json(out/(arm+'_finite_probe.json'),probes[arm])
        report['finite_probe_gates']={arm:v['gate'] for arm,v in probes.items()}
        if not all(v['gate']['passed'] for v in probes.values()):raise RuntimeError('native full-objective gate failed before paired task training')
        shared_state={k:v.detach().cpu().clone() for k,v in npp.state_dict().items()};shared_rng=rng_state()
        shared_file=out/'shared_warmup.pt'
        torch.save(dict(config=cfg,runtime=runtime,source_sha256=source,npp=shared_state,adaptive=True,proxy=proxy.state_dict(),
            rng=shared_rng,gains=gains,teacher_warmup=warm,npp_sha256=warm_sha,proxy_sha256=proxy_sha),shared_file)
        shared_sha=digest(shared_file);report['shared_warmup_sha256']=shared_sha
        report['warm_npp_sha256']=warm_sha;report['frozen_proxy_sha256']=proxy_sha
        report['phase']='TRAIN_ranking_baselines'
        gaussian=SAMGaussianNPP(cfg['teacher_sigma']).to('cuda')
        rank_anchor=out/'ranking/gaussian'
        rank_gaussian=evaluate(gaussian,cfg,rank,task,codec,rank_anchor,train_annotation,budget)
        rank_warm=evaluate(npp,cfg,rank,task,codec,out/'ranking/warm',train_annotation,budget,rank_anchor)
        candidates={}
        for arm in recipe.ARMS:
            report['phase']='task_'+arm
            student=PaperNPP(adaptive=True).to('cuda');student.load_state_dict(shared_state);restore_rng(shared_rng)
            callback=lambda model,step,arm=arm:evaluate(model,cfg,rank,task,codec,out/'ranking'/arm/str(step),train_annotation,budget,rank_anchor)
            result=train_arm(student,proxy,task,codec,samples,maps,cfg,arm,gains,budget,out/arm,runtime,shared_sha,callback,dev_samples)
            result['gate_passed']=probes[arm]['gate']['passed'];report['arms'][arm]=result;candidates[arm]=result['candidates']
            del student;gc.collect();torch.cuda.empty_cache()
            if result['training_updates']!=cfg['task_steps'] or result['stop_reason']!='task_schedule_finished':raise BudgetExpired('paired task schedule incomplete')
        selection=recipe.rank_candidates(candidates,rank_warm,rank_gaussian);report['selection']=selection
        write_json(out/'selection.json',selection);report['phase']='new_final_VAL'
        final_anchor=out/'final_validation/gaussian'
        final_gaussian=evaluate(gaussian,cfg,final,task,codec,final_anchor,val_annotation,budget)
        report['gaussian_final_metrics']=final_gaussian
        for arm in recipe.ARMS:
            selected=selection['selected_steps'][arm];snapshot=torch.load(out/arm/('npp_'+str(selected)+'.pt'),map_location='cpu',weights_only=False)
            if (snapshot['source_sha256']!=source or snapshot['config_sha256']!=config_digest(cfg)
                or snapshot['runtime_sha256']!=config_digest(runtime) or snapshot['shared_sha256']!=shared_sha
                or snapshot['task_updates']!=selected or state_digest(snapshot['npp'])!=snapshot['npp_sha256']):
                raise ValueError('selected snapshot identity differs')
            student=PaperNPP(adaptive=True).to('cuda');student.load_state_dict(snapshot['npp']);del snapshot
            audit=audit_content(proxy,student,dev_samples,cfg,codec,out/arm,budget,'selected')
            r=report['arms'][arm];r['selected_step']=selected;r['selected_proxy_gate']=audit['gate']
            r['gate_passed']=r['gate_passed'] and audit['gate']['passed']
            r['final_metrics']=evaluate(student,cfg,final,task,codec,out/'final_validation'/arm,val_annotation,budget,final_anchor)
            torch.save(dict(config=cfg,runtime=runtime,source_sha256=source,npp=student.state_dict(),adaptive=True,
                selected_task_updates=selected,total_training_updates=cfg['task_steps'],npp_sha256=model_digest(student),
                shared_sha256=shared_sha,proxy_sha256=proxy_sha,gains=gains),out/arm/'selected.pt')
            del student;gc.collect();torch.cuda.empty_cache()
        if model_digest(proxy)!=proxy_sha or model_digest(task.model)!=task.weight_digest:raise RuntimeError('frozen proxy/task drifted')
        report['final_common_bd']=recipe.final_common_bd(dict(gaussian=final_gaussian,**{arm:report['arms'][arm]['final_metrics'] for arm in recipe.ARMS}))
        report['experiment_complete']=recipe.experiment_complete(report['arms']);report['phase']='complete' if report['experiment_complete'] else 'final_gate_failed'
        if not report['experiment_complete']:raise RuntimeError('trial incomplete or selected proxy gate failed')
    except BaseException as error:
        report['error']=type(error).__name__+': '+str(error)
        raise
    finally:
        report['wall_seconds']=time.time()-started;report['budget_remaining']=budget.remaining()
        write_json(out/'study_results.json',report)
        log_event(out,'teacher_end',complete=report['experiment_complete'],phase=report['phase'],error=report.get('error'))

if __name__=='__main__':main()

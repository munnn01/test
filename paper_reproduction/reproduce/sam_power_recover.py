"""Recover completed controls and run the two pending SAM arms from verified warm-up."""
import argparse,gc,hashlib,json,os,shutil,sys
from pathlib import Path
import torch
from . import sam_power_study as power
from . import tri_task_study as study
from .sam_structured import SAMGaussianNPP,SAMTask,structured_objective,prepare_structured_masks
from .codec_content_rate import NALCodec,ContentProxy
from .coco_content_rate_study import reconstruction_fit
from .proxy_recovery import Budget
from .tensor_bytes import raw_tensor_bytes
from .training import config_digest,records_digest
from .assets import digest,dataset_mount,records_for_task
from .tri_task_data import select_cohorts,enrich_masks
from .kaggle import source_files

PARENT_SOURCE='4fa546b69bc80dc1498f8490d92a639c53b1f907c6856fd883f8bb6803df9ffc'
ANCHOR="        self.anchor=nn.Parameter(torch.zeros(1),requires_grad=False)\n"

class FlatParent:
    def __init__(self,root):self.root=Path(root)
    def __truediv__(self,name):
        allowed={'cloud_status.json','control/study_run.json','control/config.json','control/cohort.json','control/analyzer.json','control/validation/metrics.json','shared_warmup.pt','control/last.pt'}
        if name not in allowed:raise ValueError('undeclared parent member')
        return self.root/name.replace('/','__')

def state_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(str(tuple(value.shape)).encode());h.update(raw_tensor_bytes(value))
    return h.hexdigest()

def validate_sources(root,manifest,old_source):
    if old_source!=PARENT_SOURCE or config_digest(manifest)!=old_source:raise ValueError('parent source manifest differs')
    current=source_files(root)
    for name,sha in manifest.items():
        if not name.startswith('reproduce/') or not name.endswith('.py'):continue
        data=(Path(root)/name).read_bytes()
        if name=='reproduce/sam_structured.py':
            # Exact one-line device compatibility fix is the only old training-source change.
            anchor=ANCHOR.replace('\n','\r\n').encode() if b'\r\n' in data else ANCHOR.encode()
            if data.count(anchor)!=1:raise ValueError('Gaussian anchor migration differs')
            data=data.replace(anchor,b'',1)
        if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('training source changed: '+name)
    return current

def migrate_warm(parent,report,cfg,old_source,new_source):
    if parent.get('source_sha256')!=old_source or parent.get('config')!=cfg:raise ValueError('warm source/config differs')
    if (parent.get('optimizer_phase')!='warmup' or parent.get('npp_adaptive') is not False
        or parent['state'].get('task_updates')!=0 or not parent['state'].get('warmup_converged')):
        raise ValueError('task-free converged fixed warm-up required')
    if not all(parent.get(k) for k in ('npp','proxy','optimizer','rng')):raise ValueError('warm snapshot incomplete')
    if state_digest(parent['npp'])!=parent.get('npp_sha256') or state_digest(parent['npp'])!=report['warmup_npp_sha256']:
        raise ValueError('warm NPP digest differs')
    if state_digest(parent['proxy'])!=report['frozen_proxy_sha256']:raise ValueError('warm proxy digest differs')
    result=dict(parent);result['source_sha256']=new_source
    return result

def verify_parent(parent,spec,cfg):
    reports={}
    for name,sha in spec['files'].items():
        path=parent/name
        if digest(path)!=sha:raise ValueError('parent report differs: '+name)
        reports[name]=json.loads(path.read_text())
    report=reports['control/study_run.json'];saved_cfg=reports['control/config.json']
    plain=dict(saved_cfg);plain.pop('runtime_identity')
    cloud=reports['cloud_status.json'];metrics=reports['control/validation/metrics.json']
    if plain!=cfg or cloud['source_sha256']!=PARENT_SOURCE or cloud.get('worker_exit_code')!=1:raise ValueError('parent protocol/status differs')
    if (report.get('error')!='StopIteration: ' or report.get('execution_complete') is not False
        or report['state']['task_updates']!=cfg['task_steps'] or report['state']['stop_reason']!='task_schedule_finished'
        or not report.get('task_trained') or not metrics.get('complete') or len(metrics['curves'])!=8
        or report['metrics']!=metrics or not report['content_screen']['passed'] or not report['final_content_screen']['passed']):
        raise ValueError('completed screened control required')
    checkpoint=torch.load(parent/'control/last.pt',map_location='cpu',weights_only=False)
    if (checkpoint.get('source_sha256')!=PARENT_SOURCE or checkpoint.get('config')!=saved_cfg
        or checkpoint.get('config_sha256')!=config_digest(saved_cfg)
        or checkpoint.get('optimizer_phase')!='task' or checkpoint.get('npp_adaptive') is not True
        or checkpoint.get('state')!=report['state']
        or state_digest(checkpoint['npp'])!=report['npp_sha256'] or checkpoint.get('npp_sha256')!=report['npp_sha256']
        or state_digest(checkpoint['proxy'])!=report['frozen_proxy_sha256']):
        raise ValueError('completed control checkpoint differs')
    del checkpoint
    return reports

def same_codec_baseline(a,b):
    def anchors(value):
        return sorted([r for r in value['curves'] if r['arm']=='codec_only'],key=lambda r:r['qp'])
    if anchors(a)!=anchors(b):raise ValueError('recovery codec/task baseline changed')

def finish_classical(cfg,parent,out,report,analyzer):
    budget=Budget(1800)
    train0,val0,inventory=records_for_task('od',Path('/kaggle/input'))
    train,dev,validation,_=select_cohorts(train0,val0,cfg)
    root=dataset_mount(Path('/kaggle/input'),'awsaf49/coco-2017-dataset')
    annotations=list(root.rglob('instances_val2017.json'))
    if len(annotations)!=1:raise ValueError('one official VAL annotation required')
    annotation=annotations[0]
    if cfg['task']=='is':
        train=enrich_masks(train,annotation.with_name('instances_train2017.json'))
        dev=enrich_masks(dev,annotation.with_name('instances_train2017.json'));validation=enrich_masks(validation,annotation)
    identity=dict(train=records_digest(train),dev=records_digest(dev),validation=records_digest(validation),datasets=inventory['dataset_versions'])
    if identity!=cfg['runtime_identity']:raise ValueError('recovery cohorts differ')
    task=study.initialize_task(lambda:SAMTask(cfg['task'],device='cuda'))
    if task.receipt()!=analyzer:raise ValueError('frozen task differs')
    codec=NALCodec(cfg['codecs'][0],preset=cfg['preset'],gop=cfg['gop'],fps=cfg['fps'])
    report['classical_baselines']={}
    for sigma in cfg['classical_sigmas']:
        gaussian=SAMGaussianNPP(sigma).to('cuda')
        result=study.evaluate(cfg,validation,gaussian,task,codec,out/'classical'/str(sigma),annotation,budget)
        if not result.get('complete'):raise RuntimeError('Gaussian evaluation budget exceeded')
        same_codec_baseline(report['metrics'],result)
        report['classical_baselines'][str(sigma)]=result
    del task;gc.collect();torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--proxy',required=True);parser.add_argument('--proxy-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--recovery-spec',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text());spec=json.loads(args.recovery_spec.read_text())
    if cfg!=power.config(cfg['task'],'control',cfg['codecs'][0]):raise ValueError('recovery recipe differs')
    validate_sources(Path(__file__).resolve().parents[1],spec['source_hashes'],PARENT_SOURCE)
    if spec.get('dataset_ref'):args.parent=FlatParent(args.parent)
    old=verify_parent(args.parent,spec,cfg);saved_cfg=old['control/config.json'];report=old['control/study_run.json']
    warm=torch.load(args.parent/'shared_warmup.pt',map_location='cpu',weights_only=False)
    migrated=migrate_warm(warm,report,saved_cfg,PARENT_SOURCE,os.environ['PAPER_SOURCE_SHA256'])
    args.out.mkdir(parents=True,exist_ok=True);warm_path=args.out/'shared_warmup.pt';torch.save(migrated,warm_path)
    receipt=dict(parent_ref=spec['ref'],parent_source=PARENT_SOURCE,source=os.environ['PAPER_SOURCE_SHA256'],
        parent_warm_file_sha256=digest(args.parent/'shared_warmup.pt'),warm_file_sha256=digest(warm_path),
        npp_sha256=state_digest(warm['npp']),proxy_sha256=state_digest(warm['proxy']),
        control_task_updates_rerun=0,optimizer_rng_preserved=True,training_source_compatibility='only frozen Gaussian device anchor')
    (args.out/'recovery_receipt.json').write_text(json.dumps(receipt,indent=2))
    del warm,migrated
    control=args.out/'control';control.mkdir(exist_ok=True)
    for name in spec['files']:
        if name.startswith('control/'):
            dest=args.out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(args.parent/name,dest)
    # Never copy the failed Gaussian cache: its model identity has changed.
    finish_classical(saved_cfg,args.parent,control,report,old['control/analyzer.json'])
    report['previous_error']=report.pop('error');report['execution_complete']=True;report['recovery']=receipt
    (control/'study_run.json').write_text(json.dumps(report,indent=2))
    study.config=lambda task,arm:power.config(task,arm,cfg['codecs'][0])
    study.CocoTask=SAMTask;study.mask_objective=structured_objective;study.prepare_masks=prepare_structured_masks
    codecs=[]
    def factory(**kwargs):
        codec=NALCodec(cfg['codecs'][0],**kwargs);codecs.append(codec);return codec
    study.RealCodec=factory;study.FrameProxy=lambda bank:ContentProxy(bank,codecs[-1]);study.fit_video_proxy=reconstruction_fit
    reports={'control':report}
    for arm in power.ARMS[1:]:
        configuration=args.out/(arm+'_config.json');configuration.write_text(json.dumps(power.config(cfg['task'],arm,cfg['codecs'][0])))
        sys.argv=['tri_task_study','--config',str(configuration),'--proxy',args.proxy,'--proxy-sha256',args.proxy_sha256,
            '--out',str(args.out/arm),'--warm-start',str(warm_path)]
        study.main();reports[arm]=json.loads((args.out/arm/'study_run.json').read_text())
        if (reports[arm]['warmup_npp_sha256']!=report['warmup_npp_sha256']
            or reports[arm]['frozen_proxy_sha256']!=report['frozen_proxy_sha256']):raise ValueError('recovered arm warm/proxy differs')
        gc.collect();torch.cuda.empty_cache()
    status=power.study_completion(reports,cfg['task_steps'],cfg['classical_sigmas'])
    result=dict(task=cfg['task'],codec=cfg['codecs'][0],**status,recovery=receipt,
        arms={arm:dict(state=r['state'],metrics=r.get('metrics'),content_screen=r.get('content_screen')) for arm,r in reports.items()},
        classical_baselines=report['classical_baselines'],scope=cfg['scope'])
    (args.out/'three_arm_results.json').write_text(json.dumps(result,indent=2))
    if not status['complete']:raise RuntimeError('recovered study incomplete')

if __name__=='__main__':main()

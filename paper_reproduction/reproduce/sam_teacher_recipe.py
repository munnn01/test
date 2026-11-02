"""Training-only teacher and task-gain controls; deployed original NPP is unchanged."""
import hashlib,math,statistics
import torch
from .sam_power_study import config as previous_config
from .sam_structured import SAMGaussianNPP,structured_objective
from .protocol import QPS,task_lambda

ARMS=('reference','balanced')
STEPS=(512,1024,1536)

def config(task,codec='h264'):
    cfg=previous_config(task,'foreground',codec)
    cfg.update(study_profile='sam_teacher_balance_v1',experiment_arms=list(ARMS),teacher_sigma=.8,
        warmup_lr=1e-4,warmup_steps=3000,warmup_min_steps=1000,warmup_probe_every=500,teacher_psnr_gate=35.,
        task_steps=1536,proxy_steps=512,ranking_sources=32,validation_sources=256,
        max_session_seconds=10200,evaluation_reserve=2700,checkpoint_every=512,log_every=127,
        gain_sources=8,gain_target_ratio=.5,gain_min=.05,gain_max=4.,
        full_direction_min=.625,full_direction_eligible=8,probe_epsilons=[1/255,2/255],boundary_weight=0.,
        checkpoint_selection='joint_common_AP_TRAIN_ranking_32_task_snapshots',
        scope='new256COCO_VAL_excludes_previous64_independent_intra_images')
    cfg['sam2'].update(foreground_extra=1.,gamma=0.)
    cfg['classical_sigmas']=[.8]
    cfg['fidelity_notes']+=['Training-only Gaussian0.8 warm target, no Gaussian module at deployment.',
        'Two task-gain arms share exact warm student/proxy/RNG; original lambda/8 input conditioning retained.',
        '32extra TRAIN rank sources and256new VAL sources exclude prior fitting/dev/VAL sources.']
    return cfg

def teacher_target(x,sigma):
    return SAMGaussianNPP(sigma).to(device=x.device,dtype=x.dtype)(x.detach(),35).detach()

def fit_task_gains(rows,target=.5,low=.05,high=4.):
    if not 0<target or not 0<low<=high:raise ValueError('positive gain bounds/target required')
    if any(r.get('split')!='train' for r in rows):raise ValueError('task gains may fit TRAIN only')
    gains={}
    for q in QPS:
        values=[]
        for row in rows:
            if row['qp']!=q:continue
            a,b=float(row['rate_norm']),float(row['task_norm'])
            if not (math.isfinite(a) and math.isfinite(b) and a>0 and b>0):raise ValueError('finite nonzero gradient norms required')
            values.append(a/b/target)
        if len(values)<2:raise ValueError('all4QP need repeated TRAIN probes')
        gains[str(q)]=min(high,max(low,statistics.median(values)))
    return gains

def task_coefficient(cfg,arm,qp,gains):
    if arm not in ARMS or qp not in QPS:raise ValueError('declared arm/QP required')
    gain=1. if arm=='reference' else float(gains[str(qp)])
    if not math.isfinite(gain) or not cfg['gain_min']<=gain<=cfg['gain_max']:raise ValueError('invalid frozen task gain')
    return cfg['lambda_scale']*task_lambda(qp)*gain

def objective(cfg,arm,rate,task,x,y,qp,maps,gains):
    base,parts=structured_objective(cfg,rate,task,x,y,qp,maps)
    coefficient=task_coefficient(cfg,arm,qp,gains)
    original=cfg['lambda_scale']*task_lambda(qp)
    parts.update(task_gain=rate.new_tensor(coefficient/original),task_coefficient=rate.new_tensor(coefficient))
    return base+(coefficient-original)*task,parts

def cohorts(train,validation,previous,cfg):
    def lookup(rows):
        values={r['id']:r for r in rows}
        if len(values)!=len(rows):raise ValueError('duplicate source IDs')
        return values
    index=lookup(train);valindex=lookup(validation)
    try:a=[index[s] for s in previous['train']];b=[index[s] for s in previous['dev']]
    except KeyError as e:raise ValueError('pinned old TRAIN cohort missing') from e
    excluded={r['id'] for r in a+b}
    if len(a)!=64 or len(b)!=8 or len(excluded)!=72:raise ValueError('exact disjoint64TRAIN/8dev required')
    key=lambda r:hashlib.sha256((str(cfg['seed'])+'/'+r['id']).encode()).hexdigest()
    rank=sorted((r for r in train if r['id'] not in excluded),key=key)[:cfg['ranking_sources']]
    final=sorted((r for r in validation if r['id'] not in set(previous['validation'])),key=key)[:cfg['validation_sources']]
    if len(rank)!=cfg['ranking_sources'] or len(final)!=cfg['validation_sources']:raise ValueError('enough new disjoint ranking/VAL sources required')
    if not set(previous['validation'])<=set(valindex):raise ValueError('old VAL sources missing')
    return a,b,rank,final

def _curve(metrics):
    if not metrics.get('complete') or len(metrics.get('curves',[]))!=8:raise ValueError('complete4QP paired curve required')
    rows=metrics['curves']
    if {(r['arm'],r['qp']) for r in rows}!={(arm,q) for arm in ('npp','codec_only') for q in QPS}:raise ValueError('complete unique QP points required')
    if any(not math.isfinite(r['bpp']) or r['bpp']<=0 or not math.isfinite(r['quality']) for r in rows):raise ValueError('finite RD curve required')
    return [r for r in rows if r['arm']=='npp'],[r for r in rows if r['arm']=='codec_only']

def rank_candidates(candidates,warm,gaussian):
    from scipy.interpolate import PchipInterpolator
    if set(candidates)!=set(ARMS) or any(set(v)!=set(STEPS) for v in candidates.values()):raise ValueError('all6task snapshots required')
    all_curves={('warm',0):warm,('gaussian',0):gaussian}
    all_curves.update({(arm,step):m for arm,values in candidates.items() for step,m in values.items()})
    functions={};bounds=[];baseline=None
    for key,metrics in all_curves.items():
        rows,anchors=_curve(metrics)
        if baseline is None:baseline=anchors
        if anchors!=baseline:raise ValueError('ranking codec-only anchors differ')
        for current in (rows,anchors):
            current=sorted(current,key=lambda r:r['quality'])
            if any(b['quality']<=a['quality'] for a,b in zip(current,current[1:])):raise ValueError('distinct quality points required')
            bounds.append((current[0]['quality'],current[-1]['quality']))
        current=sorted(rows,key=lambda r:r['quality'])
        functions[key]=PchipInterpolator([r['quality'] for r in current],[math.log(r['bpp']) for r in current])
    lo=max(v[0] for v in bounds);hi=min(v[1] for v in bounds)
    if lo>=hi:raise ValueError('no joint AP interval; no candidate may be silently excluded')
    scores={k:float(f.integrate(lo,hi)/(hi-lo)) for k,f in functions.items()}
    selected={arm:min(STEPS,key=lambda s:scores[(arm,s)]) for arm in ARMS}
    return dict(selected_steps=selected,common_quality_interval=[lo,hi],
        scores={arm:{str(s):scores[(arm,s)] for s in STEPS} for arm in ARMS},
        warm_score=scores[('warm',0)],gaussian_score=scores[('gaussian',0)],
        improved_over_warm={arm:scores[(arm,selected[arm])]<scores[('warm',0)]-1e-9 for arm in ARMS})

def experiment_complete(rows):
    if set(rows)!=set(ARMS):return False
    for row in rows.values():
        if row.get('training_updates')!=1536 or row.get('selected_step') not in STEPS or not row.get('gate_passed'):return False
        try:_curve(row.get('final_metrics',{}))
        except ValueError:return False
        if any(r.get('sources')!=256 for r in row['final_metrics']['curves']):return False
    return True

def final_common_bd(metrics):
    """All final methods use identical native codec anchors and one AP domain."""
    from scipy.interpolate import PchipInterpolator
    functions={};bounds=[];baseline=None
    for name,value in metrics.items():
        rows,anchors=_curve(value)
        if baseline is None:baseline=anchors
        if anchors!=baseline:raise ValueError('final codec-only anchors differ')
        for key,current in ((name,rows),('codec_only',anchors)):
            current=sorted(current,key=lambda r:r['quality'])
            if any(b['quality']<=a['quality'] for a,b in zip(current,current[1:])):raise ValueError('distinct final quality points required')
            bounds.append((current[0]['quality'],current[-1]['quality']))
            functions[key]=PchipInterpolator([r['quality'] for r in current],[math.log(r['bpp']) for r in current])
    if not metrics:raise ValueError('final methods required')
    lo=max(v[0] for v in bounds);hi=min(v[1] for v in bounds)
    if lo>=hi:raise ValueError('no final common AP interval')
    anchor=functions['codec_only'].integrate(lo,hi)
    return dict(quality_interval=[lo,hi],percent={name:100*math.expm1(float((functions[name].integrate(lo,hi)-anchor)/(hi-lo))) for name in metrics})

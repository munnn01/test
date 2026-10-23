"""One-arm real-codec capacity experiment with common initialization and controls."""
import argparse,copy,hashlib,itertools,json,os,random,time
from pathlib import Path

import numpy as np
import torch

from .engine import load_checkpoint,source_hashes
from .large_npp import LargeRegionalVideoNPP
from .quality import identity_start,semantic_weights
from .recovery import (RecoveryContext,fit_recovery,validate_manifest,validate_protocol,
                       write_json,log_row)
from .screen import tensor_digest


def validate_config(cfg):
    if (cfg['arms']!=['small','matched','large']
            or cfg['capacity_bases']!={'small':None,'matched':29,'large':32}
            or cfg['stall_steps']<cfg['npp_steps'] or cfg['npp_steps']!=512
            or cfg.get('checkpoint_frames') is not True):
        raise ValueError('capacity comparison requires three specified equal-budget arms and checkpointing')
    validate_protocol(dict(cfg,arms=['replay_identity','replay_warmstart']))


def build_model(parent,cfg,arm):
    if arm not in cfg['capacity_bases']:raise ValueError('unknown capacity arm')
    base=cfg['capacity_bases'][arm]
    stored=dict(cfg,capacity_arm=arm,npp_kind='regional' if base is None else 'large_regional')
    if base is None:
        return copy.deepcopy(parent),stored
    stored.update(large_base_channels=base)
    return LargeRegionalVideoNPP.from_regional(parent,base,cfg['checkpoint_frames']),stored


def model_profile(npp,video,codec,qp):
    """Functional full-clip forward/backward without an optimizer step or teacher updates."""
    device=next(npp.parameters()).device
    before=tensor_digest(npp)
    npp.train();npp.zero_grad(set_to_none=True)
    if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    y=npp(video,codec,qp)
    assert torch.equal(y,video),'capacity model must start at exact identity'
    y.square().mean().backward()
    gradients=[p.grad for p in npp.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError('invalid capacity-model gradient')
    if device.type=='cuda':torch.cuda.synchronize()
    seconds=time.perf_counter()-started
    result={'parameters':sum(p.numel() for p in npp.parameters()),
            'trainable_parameters':sum(p.numel() for p in npp.parameters() if p.requires_grad),
            'identity_exact':True,'gradient_finite':True,'profile_geometry':list(video.shape),
            'npp_forward_backward_seconds':seconds,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved() if device.type=='cuda' else None}
    npp.zero_grad(set_to_none=True)
    if tensor_digest(npp)!=before:raise RuntimeError('model profile changed parameters/buffers')
    return result


def main():
    from .bdrate import comparisons
    from .codec import RealCodec
    from .evaluate import evaluate
    from .run import configuration,manifest_for,samples_for
    from .tasks import TaskHeads

    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('ar-root','coco-root','annotations','checkpoint','config','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',choices=('small','matched','large'),required=True)
    args=parser.parse_args();cfg=configuration(args.config);validate_config(cfg)
    if not torch.cuda.is_available():raise RuntimeError('capacity experiment requires real CUDA GPU')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    args.out.mkdir(parents=True,exist_ok=False)
    receipt={'complete':False,'started':time.time(),'config':cfg,'capacity_arm':args.arm,
             'source_commit':os.environ.get('AROD_SOURCE_COMMIT'),'source_hashes':source_hashes(),
             'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),
             'precision':'FP32; TF32 disabled; cuDNN deterministic; full deterministic algorithms not enforced',
             'scope':'TRAIN capacity pilot, final DEV; single seed; no joint video OD GT; TEST untouched'}
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
        write_json(args.out/'manifest.json',manifest)
        # Common parent initialization is explicitly seeded independent of dataset/ROI RNG.
        torch.manual_seed(cfg['seed'])
        parent=identity_start(base,cfg['filter_limit'])
        receipt['common_parent_tensor_sha256']=tensor_digest(parent)
        torch.manual_seed(cfg['seed']+1)
        npp,arm_cfg=build_model(parent,cfg,args.arm)
        count=sum(p.numel() for p in npp.parameters())
        if args.arm=='matched' and not 9_420_000<=count<=10_500_000:raise ValueError('matched capacity below target')
        if args.arm=='large' and count<11_500_000:raise ValueError('large capacity below target')
        receipt.update(npp_config=arm_cfg,npp_initial_tensor_sha256=tensor_digest(npp),parameters=count)
        write_json(status,receipt)
        print(json.dumps({'stage':'capacity_init','arm':args.arm,'parameters':count,
                          'proxy_width':cfg['width'],'common_parent_sha256':receipt['common_parent_tensor_sha256']}),flush=True)
        train=samples_for(manifest['train'],cfg)
        video=next(x for record,x,_ in train if record['task']=='ar').cuda()
        receipt['model_profile']=model_profile(npp,video,'h264',35)
        write_json(args.out/'model_profile.json',receipt['model_profile'])
        print(json.dumps({'stage':'capacity_profile','arm':args.arm,**receipt['model_profile']}),flush=True)
        del video,parent,base
        tasks=TaskHeads(device='cuda',size=cfg['size'],od_frames=cfg['od_frames'])
        task_hashes={name:tensor_digest(getattr(tasks,name)) for name in ('ar','od')}
        refs,masks=[],[]
        for index,(record,cpu,gt) in enumerate(train):
            x=cpu.cuda();reference=tasks.references(x,record,gt)
            refs.append(reference);masks.append(semantic_weights(tasks,x,record,reference).cpu())
            if (index+1)%8==0:print(json.dumps({'stage':'capacity_references','sources':index+1,'total':len(train)}),flush=True)
        codec=RealCodec(preset=cfg['preset'],gop=cfg['frames'],fps=cfg['fps'])
        context=RecoveryContext(train,refs,masks,tasks,codec,arm_cfg)
        baseline_rows=[]
        for case in context.all_cases():
            value=context.measure(context.x(case),case)
            context.baselines[case]=value
            baseline_rows.append({'case':case,'source':train[case[0]][0]['id'],**value})
            if len(baseline_rows)%32==0:
                print(json.dumps({'stage':'capacity_identity','cases':len(baseline_rows),'total':len(train)*8}),flush=True)
        write_json(args.out/'identity_targets.json',baseline_rows)
        torch.cuda.reset_peak_memory_stats()
        print(json.dumps({'stage':'capacity_train_start','arm':args.arm,'budget':cfg['npp_steps']}),flush=True)
        training=fit_recovery(npp,proxy,context,args.out/args.arm)
        receipt['training_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        receipt['training_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        if tensor_digest(proxy)!=proxy_hash or any(tensor_digest(getattr(tasks,k))!=v for k,v in task_hashes.items()):
            raise RuntimeError('fixed proxy/task state changed')
        # Only the selected TRAIN checkpoint is evaluated; no DEV-driven model selection.
        dev=samples_for(manifest['dev'],cfg)
        destination=args.out/args.arm/'evaluation';destination.mkdir()
        summary=evaluate(npp,proxy,codec,tasks,dev,cfg,args.annotations,destination)
        report={'capacity_arm':args.arm,'parameters':count,'model_profile':receipt['model_profile'],
                'training':training,'summary':summary,'bd_rates':comparisons({args.arm:summary}),
                'test_used':False,'confidence_intervals':None,'scope':receipt['scope']}
        write_json(args.out/'capacity_summary.json',report)
        receipt.update(complete=True,proxy_frozen_verified=True,task_frozen_verified=True)
    except BaseException as error:
        receipt['error']=f'{type(error).__name__}: {error}';raise
    finally:
        receipt['wall_seconds']=time.time()-receipt['started'];write_json(status,receipt)
        print(json.dumps({'stage':'capacity_finished','arm':args.arm,'complete':receipt['complete'],
                          'wall_seconds':receipt['wall_seconds']}),flush=True)


if __name__=='__main__':main()

"""Offline SAM instance interior/edge maps and frozen native task loss."""
import gc,json,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .sam2_masks import MaskCache,tensor_digest,build_teacher,mask_objective
from .training import config_digest,rng_state,restore_rng,log_event
from .tri_task_data import load
from .tri_task_models import CocoTask
from .coco_codec_study import weighted_task_loss
from . import tri_task_study as study

def structured_sam_maps(rows,size,settings):
    h,w=size;accepted=[]
    for row in rows:
        m=torch.as_tensor(row['segmentation'],dtype=torch.float32)
        if m.shape!=(h,w) or not torch.isfinite(m).all() or m.min()<0 or m.max()>1:raise ValueError('invalid SAM mask')
        iou=float(row['predicted_iou']);stability=float(row['stability_score'])
        if not (math.isfinite(iou) and math.isfinite(stability) and 0<=iou<=1 and 0<=stability<=1):
            raise ValueError('invalid SAM confidence')
        area=float(m.mean())
        if settings['min_area_fraction']<=area<=settings['max_area_fraction'] and iou>=settings['pred_iou_thresh'] and stability>=settings['stability_score_thresh']:
            accepted.append((iou*stability,m))
    accepted=sorted(accepted,key=lambda v:v[0],reverse=True)[:settings['max_masks']]
    fg=torch.zeros(h,w);edge=torch.zeros(h,w)
    for confidence,m in accepted:
        fg=torch.maximum(fg,confidence*m);v=m[None,None]
        e=(F.max_pool2d(v,3,1,1)+F.max_pool2d(-v,3,1,1))[0,0].clamp(0,1)
        edge=torch.maximum(edge,confidence*e)
    k=settings['smoothing'];fg=F.avg_pool2d(fg[None,None],k,1,k//2)[None].clamp(0,1)
    edge=edge[None,None,None].clamp(0,1)
    maps=torch.cat((fg,edge),dim=2)
    return maps,dict(accepted=len(accepted),fg_coverage=float(fg.mean()),edge_coverage=float(edge.mean()),empty=not accepted)

class StructuredMaskCache(MaskCache):
    def __init__(self,root,settings):
        super().__init__(root,dict(settings,structured_maps_schema='instance_core_edge_v1'))
    def store(self,row,x,mask):
        if mask.shape!=(*x.shape[:2],2,*x.shape[-2:]) or not torch.isfinite(mask).all() or mask.min()<0 or mask.max()>1:
            raise ValueError('aligned two-channel SAM map required')
        mask=mask.detach().cpu().to(torch.float16).contiguous();identity=self.identity(row,x)
        torch.save(dict(identity=identity,identity_sha256=config_digest(identity),mask=mask,mask_sha256=tensor_digest(mask)),self.path(row))
    def load(self,row,x):
        value=torch.load(self.path(row),map_location='cpu',weights_only=False);m=value['mask'];identity=self.identity(row,x)
        if value['identity']!=identity or value['identity_sha256']!=config_digest(identity) or value['mask_sha256']!=tensor_digest(m):
            raise ValueError('SAM map cache identity/content differs')
        if m.shape!=(*x.shape[:2],2,*x.shape[-2:]) or not torch.isfinite(m).all() or m.min()<0 or m.max()>1:
            raise ValueError('SAM map shape/range differs')
        return m.float()

def prepare_structured_masks(rows,cfg,out,budget):
    control=cfg['sam_arm']=='control'
    root=out/'sam_maps' if control else out.parent/'shared_sam_maps'
    cache=StructuredMaskCache(root,cfg['sam2']);saved=rng_state();teacher=None;records=[]
    try:
        for i,row in enumerate(rows):
            budget.require('structured_sam_cache');x,_,_=load(row,cfg)
            if cache.path(row).is_file():
                maps=cache.load(row,x)
                part=dict(cached=True,fg_coverage=float(maps[:,:,0].mean()),edge_coverage=float(maps[:,:,1].mean()))
            elif control:
                maps=torch.zeros(*x.shape[:2],2,*x.shape[-2:]);part=dict(control=True,fg_coverage=0.,edge_coverage=0.)
                cache.store(row,x,maps)
            else:
                if teacher is None:teacher=build_teacher(cfg['sam2'],out)
                rgb=(x[0,0].permute(1,2,0)*255).round().byte().numpy()
                with torch.inference_mode():generated=teacher.generate(rgb)
                maps,part=structured_sam_maps(generated,x.shape[-2:],cfg['sam2']);cache.store(row,x,maps)
            records.append(dict(source=row['id'],pixels_sha256=tensor_digest(x),**part))
            if i==0 or (i+1)%8==0:log_event(out,'sam_structured_cache',sources=i+1,total=len(rows),arm=cfg['sam_arm'])
    finally:
        del teacher;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        restore_rng(saved)
    info=dict(sources=records,root=str(root),schema='instance_core_edge_v1',
        rendered_cohort_sha256=config_digest([(r['source'],r['pixels_sha256']) for r in records]))
    (out/'sam_cache.json').write_text(json.dumps(info,indent=2));return cache,info

class SAMTask(CocoTask):
    def loss(self,x,target):
        self.model.train()
        for m in self.model.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
        try:
            parts=self.model([x[0,0]],[{k:v.to(x.device) for k,v in target.items()}])
            study._active_task_components={k:float(v.detach()) for k,v in parts.items()}
            return weighted_task_loss(parts,3. if self.task=='is' else 1.)
        finally:self.model.eval()

def structured_objective(cfg,rate,task,x,y,q,maps):
    if maps.shape!=(*y.shape[:2],2,*y.shape[-2:]) or not torch.isfinite(maps).all() or maps.min()<0 or maps.max()>1:
        raise ValueError('aligned SAM interior/edge maps required')
    maps=maps.detach().to(y.device);fg=maps[:,:,:1];edge=maps[:,:,1:2];interior=fg*(1-edge)
    loss,parts=mask_objective(cfg,rate,task,x,y,q,interior)
    boundary=((y-x).square()*edge).sum()/(edge.sum()*3).clamp_min(1)
    parts.update(boundary_distortion=boundary,sam_foreground_coverage=fg.mean(),sam_edge_coverage=edge.mean())
    for k,v in getattr(study,'_active_task_components',{}).items():parts['native_'+k]=rate.new_tensor(v)
    return loss+cfg['boundary_weight']*boundary,parts

class SAMGaussianNPP(nn.Module):
    def __init__(self,sigma):
        super().__init__()
        if not math.isfinite(sigma) or sigma<=0:raise ValueError('positive sigma required')
        r=math.ceil(3*sigma);v=torch.arange(-r,r+1,dtype=torch.float32)
        v=torch.exp(-v.square()/(2*sigma*sigma));v=v/v.sum()
        self.register_buffer('kernel',(v[:,None]*v[None,:])[None,None].repeat(3,1,1,1))
        self.anchor=nn.Parameter(torch.zeros(1),requires_grad=False)
    def forward(self,x,qp):
        b,t,c,h,w=x.shape;r=self.kernel.shape[-1]//2
        return F.conv2d(F.pad(x.reshape(b*t,c,h,w),(r,r,r,r),mode='replicate'),self.kernel,groups=3).reshape_as(x).clamp(0,1)

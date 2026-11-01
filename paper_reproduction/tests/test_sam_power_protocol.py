import json
from pathlib import Path
import torch
import pytest
from reproduce.sam_power_study import config,ARMS
from reproduce.coco_codec_study import validate_warm_start
from reproduce.sam_structured import structured_objective
from reproduce.sam_power_cloud import prepare,verify

def test_three_arms_share_protocol_and_differ_by_guidance_only():
    saved_cfg=config('is','control','h264');saved_cfg['runtime_identity']={'train':'same'}
    saved=dict(config=saved_cfg,source_sha256='s',optimizer_phase='warmup',state=dict(task_updates=0,warmup_converged=True),
        npp={'a':1},proxy={'a':1},optimizer={'a':1},rng={'a':1})
    for arm in ARMS:
        target=config('is',arm,'h264');target['runtime_identity']={'train':'same'}
        validate_warm_start(saved,target,'s')
        assert target['rate_profile']=='nal_content_v2_power' and not target['identity_fallback']
        assert target['classical_sigmas']==[.5,.8] and target['sam2']['gamma']==0
    assert config('is','foreground')['boundary_weight']==0 and config('is','sam_boundary')['boundary_weight']==2

def test_boundary_loss_uses_cached_sam_channel_and_has_gradient():
    from reproduce import tri_task_study as study
    study._codec_edge=torch.ones(1,1,1,8,8)
    x=torch.zeros(1,1,3,8,8);y=torch.full_like(x,.1,requires_grad=True)
    maps=torch.zeros(1,1,2,8,8);maps[:,:,1,:,3:5]=1
    cfg=config('is','sam_boundary')
    loss,parts=structured_objective(cfg,y.new_tensor(.1),y.mean(),x,y,35,maps)
    assert parts['boundary_distortion'].item()==pytest.approx(.01)
    zero,_=structured_objective(cfg,y.new_tensor(.1),y.mean(),x,y,35,torch.zeros_like(maps))
    assert loss>zero
    loss.backward();assert y.grad.abs().sum()>0

def test_three_arm_cloud_config_and_gate_tamper(tmp_path):
    root=Path(__file__).resolve().parents[1]
    for task,codec in (('is','h264'),('od','h265')):
        p=tmp_path/(task+codec);prepare(root,p,'testowner','sam-power',config(task,'control',codec))
        _,job=verify(p,root)
        code=''.join(json.loads((p/'notebook.ipynb').read_text())['cells'][0]['source'])
        assert "'reproduce.sam_power_study'" in code and job['config']['rate_profile']=='nal_content_v2_power'
        job['config']['rate_median_error_max']=10
        (p/'job.json').write_text(json.dumps(job))
        with pytest.raises(ValueError):verify(p)

def test_overall_completion_requires_all_declared_gaussian_baselines():
    from reproduce.sam_power_study import study_completion
    full=dict(task_trained=True,metrics={'complete':True},state=dict(task_updates=768,stop_reason='task_schedule_finished'))
    reports={arm:dict(full) for arm in ARMS}
    reports['control']['classical_baselines']={'0.5':{'complete':True},'0.8':{'complete':False}}
    status=study_completion(reports,768,[.5,.8])
    assert status['trained_arms_complete'] and not status['classical_complete'] and not status['complete']
    reports['control']['classical_baselines']['0.8']['complete']=True
    assert study_completion(reports,768,[.5,.8])['complete']

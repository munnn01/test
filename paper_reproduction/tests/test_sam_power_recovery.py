import copy
import pytest
import torch
from reproduce.sam_structured import SAMGaussianNPP
from reproduce.evaluation import processed_video

def test_gaussian_runs_through_actual_processed_video_path():
    model=SAMGaussianNPP(.5)
    x=torch.rand(1,3,3,12,14)
    y=processed_video(model,x,35,1)
    assert y.shape==x.shape and torch.isfinite(y).all()
    assert not any(p.requires_grad for p in model.parameters())

def test_warm_checkpoint_migration_preserves_weights_and_rejects_mismatch():
    from reproduce.sam_power_recover import migrate_warm,state_digest
    cfg={'task':'od','sam_arm':'control','runtime_identity':{'train':'same'}}
    state={'network.weight':torch.tensor([1.,2.])};proxy={'content':torch.tensor([3.])}
    parent=dict(source_sha256='old',config=cfg,optimizer_phase='warmup',npp_adaptive=False,
        state=dict(task_updates=0,warmup_converged=True),npp=state,npp_sha256=state_digest(state),
        proxy=proxy,optimizer={'state':1},rng={'python':1})
    report=dict(warmup_npp_sha256=state_digest(state),frozen_proxy_sha256=state_digest(proxy))
    migrated=migrate_warm(parent,report,cfg,'old','new')
    assert migrated['source_sha256']=='new' and parent['source_sha256']=='old'
    assert state_digest(migrated['npp'])==state_digest(state)
    wrong=copy.deepcopy(parent);wrong['state']['task_updates']=1
    with pytest.raises(ValueError):migrate_warm(wrong,report,cfg,'old','new')
    with pytest.raises(ValueError):migrate_warm(parent,report,cfg,'other','new')
    wrong=copy.deepcopy(report);wrong['frozen_proxy_sha256']='bad'
    with pytest.raises(ValueError):migrate_warm(parent,wrong,cfg,'old','new')

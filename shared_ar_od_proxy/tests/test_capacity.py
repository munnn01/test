import copy,json
from pathlib import Path

import pytest
import torch


def large_class():
    from ar_od_proxy.large_npp import LargeRegionalVideoNPP
    return LargeRegionalVideoNPP


def test_large_parameter_counts_reach_and_exceed_paper_without_padding():
    cls=large_class()
    counts=[]
    for base in (29,32):
        model=cls(width=12,base_channels=base)
        count=sum(p.numel() for p in model.parameters())
        assert all(p.requires_grad for p in model.parameters())
        counts.append(count)
        del model
    assert 9_420_000<=counts[0]<=10_500_000
    assert counts[1]>=11_500_000 and counts[1]>counts[0]


def test_large_identity_shapes_and_causality_with_active_unet_output():
    cls=large_class();torch.set_num_threads(2);torch.manual_seed(11)
    model=cls(width=4,base_channels=4).eval()
    x=torch.rand(1,3,3,48,64)
    for c in ('h264','h265'):
        for q in (30,35,40,45):
            torch.testing.assert_close(model(x,c,q),x,rtol=0,atol=0)
    with torch.no_grad():model.unet_rgb.weight.normal_(std=.002)
    first=model(x,'h264',35)
    changed=x.clone();changed[:,2]=1-changed[:,2]
    second=model(changed,'h264',35)
    assert first.shape==x.shape and first.min()>=0 and first.max()<=1
    torch.testing.assert_close(first[:,:2],second[:,:2],rtol=0,atol=0)
    assert (first-x).abs().max()>0


def test_frame_checkpoint_matches_outputs_and_gradients():
    cls=large_class();torch.set_num_threads(2);torch.manual_seed(12)
    direct=cls(width=4,base_channels=4,checkpoint_frames=False).train()
    with torch.no_grad():direct.unet_rgb.weight.normal_(std=.01)
    checkpointed=copy.deepcopy(direct);checkpointed.checkpoint_frames=True
    x=torch.rand(1,3,3,32,32)
    expected=direct(x,'h265',40);actual=checkpointed(x,'h265',40)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    expected.square().mean().backward();actual.square().mean().backward()
    nonzero=[]
    for (na,a),(nb,b) in zip(direct.named_parameters(),checkpointed.named_parameters()):
        assert na==nb and (a.grad is None)==(b.grad is None)
        if a.grad is not None:
            assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-8)
            if a.grad.abs().sum()>0:nonzero.append(na)
    assert any(n.startswith('unet_down') for n in nonzero)
    assert any(n.startswith('unet_temporal') for n in nonzero)


def test_large_initial_parent_and_checkpoint_keep_proxy_width_independent(tmp_path):
    cls=large_class();torch.set_num_threads(2)
    from ar_od_proxy.models import VideoNPP,VideoProxy
    from ar_od_proxy.quality import identity_start
    from ar_od_proxy.engine import save_checkpoint,load_checkpoint
    from ar_od_proxy.recovery import stored_config
    parent=identity_start(VideoNPP(4),.2)
    model=cls.from_regional(parent,base_channels=4)
    for name,value in parent.state_dict().items():
        torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
    cfg={'width':4,'npp_kind':'large_regional','large_base_channels':4,'filter_limit':.2,'checkpoint_frames':True}
    assert stored_config(cfg)['npp_kind']=='large_regional'
    with torch.no_grad():model.unet_rgb.bias.fill_(.01)
    path=tmp_path/'large.pt'
    save_checkpoint(path,model,VideoProxy(4),stored_config(cfg),'capacity',0)
    restored,proxy,saved=load_checkpoint(path)
    assert isinstance(restored,cls) and proxy.encoder[0].out_channels==4
    assert restored.base_channels==4 and restored.checkpoint_frames
    x=torch.rand(1,2,3,32,32)
    torch.testing.assert_close(model(x,'h264',45),restored(x,'h264',45),rtol=0,atol=0)


def test_capacity_protocol_requires_three_fresh_equal_budget_arms():
    from ar_od_proxy import capacity
    from ar_od_proxy.run import configuration
    root=Path(__file__).resolve().parents[1]
    cfg=configuration(root/'ar_od_proxy/configs/capacity_pilot.json')
    capacity.validate_config(cfg)
    assert cfg['arms']==['small','matched','large'] and cfg['width']==12
    assert cfg['npp_steps']==512 and cfg['stall_steps']>=cfg['npp_steps']
    assert cfg['capacity_bases']=={'small':None,'matched':29,'large':32}
    for override in ({'capacity_bases':{'small':None,'matched':4,'large':8}},
                     {'stall_steps':128},{'arms':['large']}):
        with pytest.raises(ValueError):capacity.validate_config(dict(cfg,**override))


def test_capacity_models_share_exact_parent_and_proxy_config():
    from ar_od_proxy import capacity
    from ar_od_proxy.models import VideoNPP
    from ar_od_proxy.quality import identity_start
    from ar_od_proxy.screen import tensor_digest
    parent=identity_start(VideoNPP(4),.2)
    cfg={'width':4,'filter_limit':.2,'capacity_bases':{'small':None,'matched':4,'large':5},'checkpoint_frames':True}
    digest=tensor_digest(parent)
    for arm in ('small','matched','large'):
        model,stored=capacity.build_model(parent,cfg,arm)
        assert stored['width']==4 and stored['capacity_arm']==arm
        assert tensor_digest(parent)==digest
        for name,value in parent.state_dict().items():
            torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
        x=torch.rand(1,2,3,32,32)
        torch.testing.assert_close(model(x,'h265',35),x,rtol=0,atol=0)


def test_capacity_payload_pins_arm_and_requires_explicit_valid_variant(tmp_path):
    from ar_od_proxy.kaggle import prepare,verify_payload
    root=Path(__file__).resolve().parents[1]
    kwargs={'action':'capacity','baseline_sha256':'b'*64,'checkpoint_dataset':'example/private-checkpoint'}
    prepare(root,tmp_path/'payload','a'*40,'example','capacity-matched',capacity_arm='matched',**kwargs)
    _,job=verify_payload(tmp_path/'payload','example/capacity-matched')
    code=''.join(json.loads((tmp_path/'payload/notebook.ipynb').read_text())['cells'][0]['source'])
    assert job['capacity_arm']=='matched' and 'ar_od_proxy.capacity' in code
    assert "CAPACITY_ARM = 'matched'" in code and 'capacity_pilot.json' in code
    with pytest.raises(ValueError):
        prepare(root,tmp_path/'bad','a'*40,'example','capacity-bad',capacity_arm='unknown',**kwargs)
    with pytest.raises(ValueError):
        prepare(root,tmp_path/'missing','a'*40,'example','capacity-missing',**kwargs)

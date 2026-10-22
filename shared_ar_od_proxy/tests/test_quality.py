import copy
import importlib
import json
from pathlib import Path

import pytest
import torch


def quality_module():
    from ar_od_proxy import quality
    return quality


def test_quality_objective_protects_task_even_inside_allowance():
    q = quality_module()
    rate = torch.tensor(2., requires_grad=True)
    task = torch.tensor(.9, requires_grad=True)
    loss, excess = q.protected_objective(rate, 2., {'ar': task}, {'ar': 1.},
                                         {'ar': 2.}, .02, task_weight=1., penalty=10.)
    loss.backward()
    assert rate.grad > 0 and task.grad > 0 and excess['ar'] < 0
    above = torch.tensor(1.2, requires_grad=True)
    loss, _ = q.protected_objective(rate, 2., {'ar': above}, {'ar': 1.},
                                    {'ar': 2.}, .02, task_weight=1., penalty=10.)
    loss.backward()
    assert above.grad > task.grad


def test_real_guard_refuses_task_violation_flip_and_cumulative_drift():
    q = quality_module()
    baseline = {'terms': {'ar': 1., 'od': 2.}, 'ar_correct': True}
    good = {'terms': {'ar': 1.01, 'od': 2.02}, 'ar_correct': True}
    assert q.quality_feasible(good, baseline, .02)
    for bad in [dict(good, terms={'ar': 1.1, 'od': 2.}),
                dict(good, ar_correct=False), dict(good, terms={'ar': float('nan'), 'od': 2.})]:
        assert not q.quality_feasible(bad, baseline, .02)
    assert not q.quality_feasible({'terms': {'ar': 1.}}, baseline, .02)


def test_identity_measurement_rejects_nan_immediately_with_source_and_point():
    q=quality_module()
    from ar_od_proxy.codec import RealCodec
    class InvalidTasks:
        def losses(self,x,record,refs):
            zero=x.sum()*0
            return {'ar_gt':zero,'ar_source':zero,'od_gt':zero+float('nan'),'od_source':zero}
    x=torch.rand(1,1,3,32,32)
    with pytest.raises(ValueError,match='fixture/bad.*h264.*30.*od'):
        q.real_measure(InvalidTasks(),RealCodec(preset='ultrafast',gop=1),x,
                       {'id':'fixture/bad','task':'od'},{},'h264',30,17)


def test_rejected_adam_displacement_restores_parameters_and_moments():
    q = quality_module()
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.1)
    model(torch.ones(1, 1)).sum().backward(); optimizer.step(); optimizer.zero_grad()
    before = q.capture_state(model, optimizer)
    model(torch.ones(1, 1)).sum().backward()
    accepted, scale, details = q.guarded_step(model, optimizer, lambda: (False, {'bad': True}), attempts=3)
    assert not accepted and scale == 0 and details['bad']
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[0][key], rtol=0, atol=0)
    for key, state in optimizer.state_dict()['state'].items():
        for field, value in state.items():
            torch.testing.assert_close(value, before[1]['state'][key][field], rtol=0, atol=0)


def test_backtracking_accepts_smaller_displacement():
    q = quality_module()
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad(): model.weight.zero_()
    opt = torch.optim.Adam(model.parameters(), lr=1.)
    (-model(torch.ones(1, 1)).sum()).backward()
    accepted, scale, _ = q.guarded_step(model, opt, lambda: (model.weight.item() <= .3, {}), attempts=4)
    assert accepted and 0 < scale < 1 and 0 < model.weight.item() <= .3


def test_identity_reset_filter_cap_and_checkpoint_roundtrip(tmp_path):
    q = quality_module()
    from ar_od_proxy.models import VideoNPP, VideoProxy
    from ar_od_proxy.engine import save_checkpoint, load_checkpoint
    torch.set_num_threads(2)
    base = VideoNPP(4)
    with torch.no_grad(): base.rgb.bias.fill_(1.)
    npp = q.identity_start(base, .2)
    x = torch.rand(1, 2, 3, 32, 32)
    for c in ('h264', 'h265'):
        for qp in (30, 35, 40, 45):
            torch.testing.assert_close(npp(x, c, qp), x, rtol=0, atol=0)
    with torch.no_grad(): npp.filter_strength.fill_(10.)
    cfg = {'width': 4, 'npp_kind': 'regional', 'filter_limit': .2}
    path = tmp_path/'quality.pt'
    save_checkpoint(path, npp, VideoProxy(4), cfg, 'quality', 1)
    restored, _, _ = load_checkpoint(path)
    assert restored.filter_limit == .2
    torch.testing.assert_close(restored(x, 'h265', 40), npp(x, 'h265', 40), rtol=0, atol=0)


def test_refresh_fit_and_audit_sources_never_overlap_and_are_train_only():
    q = quality_module()
    records = [{'id': f'{family}/{i}', 'task': family} for i in range(16) for family in ('ar', 'od')]
    fit_all, audit_all = set(), set()
    for epoch in range(8):
        fit, audit = q.refresh_indices(records, epoch, fit_per_task=3, audit_per_task=2)
        assert len(fit) == 6 and len(audit) == 4 and not set(fit) & set(audit)
        fit_all.update(fit); audit_all.update(audit)
    assert not fit_all & audit_all


def test_refresh_gate_rejects_absolute_or_directional_regression():
    q = quality_module()
    before = {'rate_mape': .1, 'sign_accuracy': .6, 'reconstruction_l1': .02, 'delta_l1': .005}
    assert q.refresh_acceptable(before, dict(before, sign_accuracy=.8))
    assert not q.refresh_acceptable(before, dict(before, rate_mape=.2, sign_accuracy=.8))
    assert not q.refresh_acceptable(before, dict(before, rate_mape=.05, sign_accuracy=.4))
    assert not q.refresh_acceptable(before, dict(before, sign_accuracy=None))
    assert not q.refresh_acceptable(before, dict(before, sign_accuracy=.8, delta_l1=.02))


def test_quality_cloud_config_keeps_four_qps_and_expands_sources(tmp_path):
    from ar_od_proxy.kaggle import prepare, verify_payload
    from ar_od_proxy.run import configuration
    root = Path(__file__).resolve().parents[1]
    cfg = configuration(root/'ar_od_proxy/configs/quality_recovery.json')
    assert cfg['train_count'] >= 64 and cfg['dev_count'] >= 128
    assert cfg['arms'] == ['quality_fixed', 'quality_refresh']
    assert cfg['filter_limit'] <= .2 and cfg['max_total_rms'] <= 8/255
    prepare(root, tmp_path/'payload', 'a'*40, 'example', 'quality-recovery', action='quality',
            baseline_sha256='b'*64, checkpoint_dataset='example/private-checkpoint')
    _, job = verify_payload(tmp_path/'payload', 'example/quality-recovery')
    code = ''.join(json.loads((tmp_path/'payload/notebook.ipynb').read_text())['cells'][0]['source'])
    assert job['action'] == 'quality' and 'ar_od_proxy.quality' in code and 'quality_recovery.json' in code


def test_reconstruction_jvp_fits_the_input_jacobian_with_parameter_gradients():
    q = quality_module()
    from ar_od_proxy.models import ProxyOutput, VideoProxy
    class LinearProxy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.scale = torch.nn.Parameter(torch.tensor(.3))
        def forward(self, x, codec, qp):
            return ProxyOutput(self.scale*x, x.mean((1,2,3,4)))
    proxy = LinearProxy()
    inputs = torch.stack([torch.full((1,3,8,8),.2), torch.full((1,3,8,8),.8)])
    pair = {'inputs':inputs,'decoded':(inputs*.8*255).round().byte(),'codec':'h264','qp':35}
    optimizer = torch.optim.SGD(proxy.parameters(),lr=1.)
    before = None
    for _ in range(10):
        tangent, target = q.reconstruction_jvp(proxy,pair,'cpu',create_graph=True)
        loss = (tangent-target).square().mean()
        if before is None: before = loss.item()
        optimizer.zero_grad(); loss.backward()
        assert proxy.scale.grad is not None and proxy.scale.grad.abs() > 0
        optimizer.step()
    assert loss.item() < before*.1
    # Exercise the actual causal STE proxy too, including higher-order gradients.
    real_proxy = VideoProxy(4)
    tangent, target = q.reconstruction_jvp(real_proxy,pair,'cpu',create_graph=True)
    (tangent-target).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in real_proxy.decoder.parameters())


def test_guard_restores_state_when_measurement_raises():
    q = quality_module()
    model = torch.nn.Linear(1,1)
    optimizer = torch.optim.Adam(model.parameters())
    original = q.capture_state(model,optimizer)
    model(torch.ones(1,1)).sum().backward()
    def fail(): raise RuntimeError('codec failure')
    with pytest.raises(RuntimeError,match='codec failure'):
        q.guarded_step(model,optimizer,fail)
    for k,v in model.state_dict().items(): torch.testing.assert_close(v,original[0][k],rtol=0,atol=0)
    assert optimizer.state_dict()['state'] == original[1]['state']


def test_reconstruction_refresh_ignores_stale_frozen_rate_gradients():
    q = quality_module()
    assert hasattr(q,'fit_reconstruction'), 'separate reconstruction clipping from frozen rate head'
    from ar_od_proxy.models import VideoProxy
    torch.set_num_threads(2); torch.manual_seed(42)
    a=VideoProxy(4); b=copy.deepcopy(a)
    for p in b.rate_head.parameters(): p.grad=torch.full_like(p,1e6)
    inputs=torch.rand(2,1,3,16,16)
    pair={'inputs':inputs,'decoded':(inputs*255).round().byte(),'codec':'h264','qp':35}
    cfg={'refresh_reconstruction_steps':2,'probe_rms':2/255}
    q.fit_reconstruction(a,[pair],cfg)
    q.fit_reconstruction(b,[pair],cfg)
    for key,value in a.state_dict().items(): torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)
    assert all(p.grad is None and not p.requires_grad for p in b.parameters())


def test_quality_training_uses_real_codecs_and_saves_only_guarded_states(tmp_path):
    q = quality_module()
    from ar_od_proxy.models import VideoNPP, VideoProxy
    from ar_od_proxy.codec import RealCodec
    from ar_od_proxy.engine import load_checkpoint
    torch.set_num_threads(2); torch.manual_seed(92)
    class Tasks:
        def __init__(self):
            self.ar = torch.nn.Conv2d(3,2,1).requires_grad_(False)
            self.od = torch.nn.Conv2d(3,2,1).requires_grad_(False)
        def losses(self,x,record,refs):
            assert isinstance(refs,dict) and 'indices' in refs
            # A random multiplier detects unpaired ROI/RPN sampling seeds.
            value = self.od(x[:,0]).square().mean()*(1+torch.rand(()))
            zero = x.sum()*0
            return {'ar_gt':zero,'ar_source':zero,'od_gt':value,'od_source':zero}
    x = torch.rand(1,1,3,32,32)
    record = {'id':'fixture/image','task':'od'}
    samples = [(record,x,None)]
    refs = [{'od':[{'boxes':torch.empty(0,4),'labels':torch.empty(0,dtype=torch.long)}],'indices':[0]}]
    tasks = Tasks(); codec = RealCodec(preset='ultrafast',gop=1)
    cfg = {'width':4,'seed':21,'qps':[30,35,40,45],'codecs':['h264','h265'],
           'npp_steps':8,'anchor_every':8,'anchor_per_task':1,'proxy_refresh_every':16,
           'filter_limit':.2,'learning_rate':2e-5,'filter_strength_learning_rate':2e-4,
           'max_total_rms':6/255,'max_npp_update_rms':1/255,'rate_warmup':4,
           'quality_allowance':.02,'quality_penalty':20.,'guard_attempts':3,
           'loss_weights':{'ar':4.,'od':2.,'pre':50.,'temp':1.}}
    baselines = {}
    for point,(c,qp) in enumerate(__import__('itertools').product(cfg['codecs'],cfg['qps'])):
        baselines[(0,c,qp)] = q.real_measure(tasks,codec,x,record,refs[0],c,qp,q.case_seed(point,1,cfg))[0]
    for arm in ('quality_fixed','quality_refresh'):
        model = q.identity_start(VideoNPP(4),.2)
        proxy = VideoProxy(4).requires_grad_(False).eval()
        digest = q.tensor_digest(proxy)
        report = q.fit_quality(model,proxy,codec,tasks,samples,refs,[torch.ones_like(x[:,:,:1])],baselines,cfg,arm,tmp_path/arm)
        counts = report['counts']
        assert counts['proposals']==8
        assert counts['accepted_proposals']+counts['rejected_proposals']==8
        assert counts['committed_proposals']+counts['rolled_back_proposals']==counts['accepted_proposals']
        assert report['final_anchor_verified'] and q.tensor_digest(proxy)==digest
        restored,_,metadata = load_checkpoint(tmp_path/arm/'last.pt')
        assert metadata['step']==8 and restored.filter_limit==.2
        torch.testing.assert_close(model(x,'h264',30),restored(x,'h264',30),rtol=0,atol=0)
        rows = [json.loads(s) for s in (tmp_path/arm/'training.jsonl').read_text().splitlines()]
        assert all(not r['accepted'] or r['guard']['quality_passed'] for r in rows)

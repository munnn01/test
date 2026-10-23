import copy,json
from pathlib import Path

import pytest
import torch


def test_split_train_preserves_fit_and_excludes_calibration_from_selection():
    from ar_od_proxy.paper_training import split_train
    records=[{'id':f'{t}{i}','task':t} for i in range(40) for t in ('ar','od')]
    fit,held=split_train(records,32,8,records[:16])
    assert fit==records[:64] and held==records[64:]
    with pytest.raises(ValueError):split_train(records,32,8,[records[-1]])
    with pytest.raises(ValueError):split_train(records[:-1],32,8,[])


def test_curriculum_changes_only_qp_schedule_and_covers_full_grid():
    from ar_od_proxy.paper_training import phase_qps,training_schedule
    cfg={'qps':[30,35,40,45],'codecs':['h264','h265'],'fixed_qp':35,'fixed_steps':128,'seed':7}
    assert phase_qps('continuous',0,cfg)==cfg['qps']
    assert phase_qps('curriculum',127,cfg)==[35]
    assert phase_qps('curriculum',128,cfg)==cfg['qps']
    records=[{'id':f'{t}{i}','task':t} for i in range(4) for t in ('ar','od')]
    schedule=training_schedule(records,'curriculum',128,cfg)
    assert {tuple(c[1:]) for b in schedule for c in b}=={(c,q) for c in cfg['codecs'] for q in cfg['qps']}
    assert all(sum(records[c[0]]['task']=='ar' for c in b)==2 for b in schedule)
    common=training_schedule(records,'continuous',0,cfg)
    fixed=training_schedule(records,'curriculum',0,cfg)
    assert [[c[:2] for c in b] for b in common]==[[c[:2] for c in b] for b in fixed]
    assert {c[2] for b in fixed for c in b}=={35}
    assert training_schedule(records,'curriculum',128,cfg)==common


def row(i,term,rate=1.,correct=True,task='ar',qp=35,codec='h264'):
    return {'case':[i,codec,qp],'task':task,'total_rms':.001,'changed_rgb8':True,
            'measurement':{'terms':{'ar':term} if task=='ar' else {'od':term},'ar_correct':correct if task=='ar' else None,'bpp':rate}}


def test_selection_aggregates_sources_but_never_hides_bad_qp_or_ar_flip():
    from ar_od_proxy.paper_training import selection_report
    baseline={(i,'h264',35):{'terms':{'ar':1.},'ar_correct':True,'bpp':1.} for i in range(2)}
    rows=[row(0,1.1,.9),row(1,.9,.9)]
    result=selection_report(rows,baseline,.02,6/255)
    assert result['passed'] and result['mean_rate_ratio']==pytest.approx(.9)
    bad=copy.deepcopy(rows);bad[0]['measurement']['ar_correct']=False
    assert not selection_report(bad,baseline,.02,6/255)['passed']
    bad=copy.deepcopy(rows);bad[1]['measurement']['terms']['ar']=1.
    assert not selection_report(bad,baseline,.02,6/255)['passed']
    bad=copy.deepcopy(rows);bad[0]['total_rms']=float('nan')
    assert not selection_report(bad,baseline,.02,6/255)['passed']
    with pytest.raises(ValueError):selection_report(rows[:1],baseline,.02,6/255)


def test_optimizer_does_not_rollback_quality_and_rejects_nonfinite_gradients():
    from ar_od_proxy.paper_training import continuous_step
    model=torch.nn.Linear(1,1,bias=False);model.weight.data.fill_(1)
    opt=torch.optim.Adam(model.parameters(),lr=.1)
    for _ in range(2):
        opt.zero_grad();model(torch.ones(1,1)).sum().backward();continuous_step(model,opt)
    assert model.weight.item()<.85 and opt.state[model.weight]['step']==2
    before=model.weight.detach().clone();model.weight.grad.fill_(float('nan'))
    with pytest.raises(RuntimeError):continuous_step(model,opt)
    assert torch.equal(before,model.weight)


def test_paper_payload_pins_valid_arm_and_configuration(tmp_path):
    from ar_od_proxy.kaggle import prepare,verify_payload
    root=Path(__file__).resolve().parents[1]
    kwargs=dict(action='paper-training',baseline_sha256='b'*64,checkpoint_dataset='example/private-checkpoint')
    prepare(root,tmp_path/'ok','a'*40,'example','paper-curriculum',paper_arm='curriculum',**kwargs)
    _,job=verify_payload(tmp_path/'ok','example/paper-curriculum')
    code=''.join(json.loads((tmp_path/'ok/notebook.ipynb').read_text())['cells'][0]['source'])
    assert job['paper_arm']=='curriculum' and 'ar_od_proxy.paper_training' in code
    assert "PAPER_ARM = 'curriculum'" in code
    with pytest.raises(ValueError):prepare(root,tmp_path/'bad','a'*40,'example','paper-invalid',paper_arm='unknown',**kwargs)
    with pytest.raises(ValueError):prepare(root,tmp_path/'missing','a'*40,'example','paper-missing',**kwargs)


def test_failed_selection_preserves_trajectory_and_keeps_identity_fallback(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from ar_od_proxy import paper_training as p
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.w=torch.nn.Parameter(torch.ones(1));self.filter_strength=torch.nn.Parameter(torch.zeros(1))
    model=Tiny();proxy=torch.nn.Linear(1,1).requires_grad_(False)
    cfg={'qps':[30,35,40,45],'codecs':['h264','h265'],'seed':1,'fixed_qp':35,'fixed_steps':2,
         'npp_steps':4,'learning_rate':.1,'filter_strength_learning_rate':.1,'lr_decay_step':3,'selection_every':2}
    samples=[({'id':f'{t}{i}','task':t},None,None) for i in range(2) for t in ('ar','od')]
    context=SimpleNamespace(cfg=cfg,samples=samples)
    checkpoints={};seen=[]
    monkeypatch.setattr(p,'save_checkpoint',lambda path,npp,*args:checkpoints.__setitem__(path.name,npp.w.item()))
    monkeypatch.setattr(p,'directional_audit',lambda *args:{'mock_audit':True})
    def measure(npp,selection):
        seen.append(npp.w.item())
        return {'passed':npp.w.item()>.99,'changed_rgb8_cases':1,'mean_rate_ratio':.9}
    monkeypatch.setattr(p,'measure_selection',measure)
    monkeypatch.setattr(p,'sample_loss',lambda npp,*args:(npp.w.square().sum(),{}))
    result=p.fit_continuous(model,proxy,context,context,tmp_path/'fit','continuous')
    assert seen[1]<seen[0]<1. and seen[-1]==1.
    assert checkpoints['last.pt']<.8 and checkpoints['selected.pt']==1.
    assert result['rollback_steps']==0 and result['last_differs_from_start'] and result['selected_step']==0


def test_packaged_paper_config_and_cross_qp_constraint():
    from ar_od_proxy.paper_training import validate_config,selection_report
    from ar_od_proxy.run import configuration
    cfg=configuration(Path(__file__).resolve().parents[1]/'ar_od_proxy/configs/paper_training.json')
    validate_config(cfg)
    baseline={(0,'h264',q):{'terms':{'ar':1.},'ar_correct':True,'bpp':1.} for q in (30,45)}
    rows=[row(0,.5,.9,qp=30),row(0,1.1,.9,qp=45)]
    assert not selection_report(rows,baseline,.02,6/255)['passed']


def test_direction_audit_matches_known_real_descent_without_changing_model(tmp_path):
    from types import SimpleNamespace
    from ar_od_proxy.paper_training import directional_audit
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.ones(1))
        def forward(self,x,*args):return self.scale*x
    class Proxy(torch.nn.Module):
        def forward(self,x,*args):return SimpleNamespace(reconstruction=.5*x,bpp=x.mean().reshape(1))
    class Tasks:
        def losses(self,x,record,refs):
            value=x.square().mean();zero=value*0
            return {'ar_gt':value if record['task']=='ar' else zero,'ar_source':zero,
                    'od_gt':value if record['task']=='od' else zero,
                    'od_source':value if record['task']=='ar' else zero}
    class Context:
        samples=[({'id':t,'task':t},None,None) for t in ('ar','od')]
        cfg={'loss_weights':{'ar':1.,'od':1.}};refs=[None,None];tasks=Tasks()
        codec=SimpleNamespace(roundtrip=lambda x,*args:Proxy()(x.detach()))
        def all_cases(self):return [(i,'h264',35) for i in range(2)]
        def x(self,case):return torch.full((1,1,3,4,4),.5)
        def seed(self,case):return 1
        def measure(self,y,case):
            value=float((.5*y).square().mean())
            terms={'ar':value,'od':.5*value} if case[0]==0 else {'od':value}
            return {'bpp':float(y.mean()),'terms':terms,'ar_correct':True}
    model=Model();path=tmp_path/'audit.jsonl'
    result=directional_audit(model,Proxy(),Context(),path)
    assert result['cases']==2 and model.scale.item()==1. and model.scale.grad is None
    for line in path.read_text().splitlines():
        row_data=json.loads(line)
        assert row_data['proxy_reconstruction_l1']==0.
        for probe in row_data['directions']:
            assert probe['proxy_linear_objective_delta']<0 and probe['real_rate_relative_delta']<0
            assert all(v<0 for v in probe['real_task_relative_deltas'].values())

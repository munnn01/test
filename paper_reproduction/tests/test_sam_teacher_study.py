import copy,importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import pytest,torch
from torch import nn
from reproduce.training import config_digest,records_digest,rng_state

def api():
    assert importlib.util.find_spec('reproduce.sam_teacher_study') is not None,'teacher study is not implemented'
    return __import__('reproduce.sam_teacher_study',fromlist=['*'])

class Proxy(nn.Module):
    def __init__(self,wrong=False):
        super().__init__();self.bank=nn.Linear(1,1);self.wrong=wrong
    def forward(self,x,q):
        y=-x if self.wrong else x
        return SimpleNamespace(reconstruction=y,bpp=.1+.01*x.mean(),raw_bpp=.1+.01*x.mean())

class Codec:
    def __call__(self,x,codec,q,directory=None):
        return SimpleNamespace(reconstruction=x.detach(),bpp=float(.1+.01*x.detach().mean()))

class Task:
    def loss(self,x,target):return x.square().mean()+torch.rand((),device=x.device)
class Identity(nn.Module):
    def forward(self,x,q):return x

def test_seeded_task_preserves_outer_rng_and_pairs_stochastic_task_calls():
    a=api();x=torch.ones(1,1,3,4,5,requires_grad=True);before=a.rng_digest()
    v,one=a.seeded_task(Task(),x,{},101,True);w,two=a.seeded_task(Task(),x,{},101,True)
    assert a.rng_digest()==before and one==two and torch.equal(v,w)
    v.backward();assert x.grad.abs().sum()>0

@pytest.mark.parametrize('wrong',[False,True])
def test_finite_gate_detects_wrong_task_proxy_jacobian(wrong):
    a=api();from reproduce.sam_teacher_recipe import config
    cfg=config('od','h264');x=torch.full((1,1,3,6,7),.5)
    samples=[(x,{},dict(id='d'+str(i))) for i in range(2)]
    maps=[torch.ones(1,1,2,6,7)*.2 for _ in samples];gains={str(q):1. for q in cfg['qps']}
    proxy=Proxy(wrong);before=a.rng_digest()
    report=a.finite_probe(Identity(),proxy,Task(),samples,maps,cfg,'reference',gains,a.Budget(100),codec=Codec())
    assert report['gate']['passed'] is (not wrong)
    assert len(report['rows'])==32 and report['gate']['untied']>=8
    assert all(r['rng_minus']==r['rng_plus'] for r in report['rows'])
    assert a.rng_digest()==before

def test_checkpoint_rejects_wrong_source_cohort_model_or_proxy():
    a=api();cfg={'recipe':'r'};runtime={'cohort':'same'};n={'p':torch.tensor([2.])};proxy={'b':torch.tensor([3.])}
    saved=dict(schema=1,config=cfg,config_sha256=config_digest(cfg),runtime=runtime,runtime_sha256=config_digest(runtime),
        source_sha256='s',npp=n,npp_sha256=a.state_digest(n),proxy=proxy,proxy_sha256=a.state_digest(proxy),
        optimizer={'state':1},rng=rng_state(),phase='task',arm='reference',task_updates=512,adaptive=True,
        gains={'30':1.,'35':1.,'40':1.,'45':1.},shared_sha256='shared')
    saved['gains_sha256']=config_digest(saved['gains'])
    a.validate_checkpoint(saved,cfg,runtime,'s','shared')
    with pytest.raises(ValueError):a.validate_checkpoint(saved,cfg,runtime,'wrong','shared')
    with pytest.raises(ValueError):a.validate_checkpoint(saved,cfg,{'cohort':'other'},'s','shared')
    wrong=copy.deepcopy(saved);wrong['npp']['p'][0]=9
    with pytest.raises(ValueError):a.validate_checkpoint(wrong,cfg,runtime,'s','shared')
    wrong=copy.deepcopy(saved);wrong['gains']['30']=2.
    with pytest.raises(ValueError):a.validate_checkpoint(wrong,cfg,runtime,'s','shared')
    wrong=copy.deepcopy(saved);wrong['proxy']['b'][0]=9
    with pytest.raises(ValueError):a.validate_checkpoint(wrong,cfg,runtime,'s','shared')

def test_codec_cache_reuse_checks_every_inference_identity(tmp_path):
    a=api();from reproduce.training import config_digest,records_digest
    cfg={'gop':16};rows=[dict(id='x')];task=SimpleNamespace(weight_digest='task')
    anchor=tmp_path/'anchor';(anchor/'cache/codec_only/30').mkdir(parents=True)
    (anchor/'cache/codec_only/30/point.json').write_text('{"source":"x"}')
    ident=dict(config_sha256=config_digest(cfg),sources=['x'],records_sha256=records_digest(rows),npp_sha256='gauss',
        task_sha256='task',annotation_sha256=None)
    (anchor/'evaluation_identity.json').write_text(json.dumps(ident))
    dest=tmp_path/'candidate';a.copy_codec_cache(anchor,dest,cfg,rows,task,None)
    assert (dest/'cache/codec_only/30/point.json').read_bytes()==(anchor/'cache/codec_only/30/point.json').read_bytes()
    with pytest.raises(ValueError):a.copy_codec_cache(anchor,tmp_path/'bad',{'gop':8},rows,task,None)

class TinyNPP(nn.Module):
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.tensor(.9));self.adaptive=True
    def forward(self,x,q):return (x*self.scale).clamp(0,1)

def test_teacher_warm_resets_lr_and_checks_both_fidelities(tmp_path):
    a=api();from reproduce.sam_teacher_recipe import config
    cfg=config('od');cfg.update(warmup_steps=6,warmup_min_steps=2,warmup_probe_every=2,warmup_lr=.1,teacher_psnr_gate=20.)
    npp=TinyNPP();x=torch.full((1,1,3,6,7),.5);samples=[(x,{},dict(id='x'))]
    r=a.warm_student(npp,samples,samples,cfg,a.Budget(10000),tmp_path)
    assert r['passed'] and r['updates']>=2 and r['optimizer_lr']==.1 and r['teacher_psnr_db']>=20.

def test_decoder_mix_changes_only_synthesis_and_includes_all_three_variants(tmp_path):
    a=api();from reproduce.sam_teacher_recipe import config
    class Net(nn.Module):
        def __init__(self):super().__init__();self.g_s=nn.Conv2d(3,3,1);self.analysis=nn.Parameter(torch.ones(1))
    class MixProxy(nn.Module):
        def __init__(self):
            super().__init__();self.bank=nn.Module();self.bank.networks=nn.ModuleDict({str(q):Net() for q in (30,35,40,45)})
        def forward(self,x,q):return SimpleNamespace(reconstruction=self.bank.networks[str(q)].g_s(x[0])[None])
    cfg=config('od');cfg.update(proxy_steps=12);proxy=MixProxy();before={k:v.clone() for k,v in proxy.bank.state_dict().items()}
    x=torch.rand(1,1,3,6,7);r=a.refit_mixture(proxy,TinyNPP(),[(x,{},dict(id='x'))],cfg,Codec(),a.Budget(10000),tmp_path)
    assert r['updates']==12 and set(r['variant_counts'])=={'x','npp','gaussian'} and all(v==4 for v in r['variant_counts'].values())
    assert all(torch.equal(v,proxy.bank.state_dict()[k]) for k,v in before.items() if '.g_s.' not in k)
    assert any(not torch.equal(v,proxy.bank.state_dict()[k]) for k,v in before.items() if '.g_s.' in k)
    assert not any(p.requires_grad for p in proxy.parameters())

def test_task_loop_updates_actual_npp_and_exports_resumable_checkpoint(tmp_path,monkeypatch):
    a=api();from reproduce.sam_teacher_recipe import config
    monkeypatch.setenv('PAPER_SOURCE_SHA256','s')
    cfg=config('od');cfg.update(task_steps=4,checkpoint_every=2,evaluation_reserve=2,log_every=1)
    npp=TinyNPP();proxy=Proxy();proxy.requires_grad_(False);task=Task();task.model=nn.Identity()
    task.weight_digest=a.model_digest(task.model)
    x=torch.full((1,1,3,6,7),.5);samples=[(x,{},dict(id='x'))];maps=[torch.zeros(1,1,2,6,7)]
    gains={str(q):1. for q in cfg['qps']};before=npp.scale.detach().clone()
    r=a.train_arm(npp,proxy,task,Codec(),samples,maps,cfg,'reference',gains,a.Budget(10000),tmp_path,{'c':'r'},'shared',lambda model,step:dict(complete=True,step=step))
    assert r['training_updates']==4 and set(r['candidates'])=={2,4} and not torch.equal(npp.scale,before)
    saved=torch.load(tmp_path/'last.pt',map_location='cpu',weights_only=False)
    a.validate_checkpoint(saved,cfg,{'c':'r'},'s','shared')
    assert saved['task_updates']==4 and saved['optimizer'] and saved['rng']

def test_probe_requires_explicit_codec_instead_of_synthetic_fallback():
    a=api();from reproduce.sam_teacher_recipe import config
    cfg=config('od');x=torch.full((1,1,3,6,7),.5);samples=[(x,{},dict(id='d'+str(i))) for i in range(2)]
    with pytest.raises(ValueError,match='codec'):
        a.finite_probe(Identity(),Proxy(),Task(),samples,[torch.zeros(1,1,2,6,7)]*2,cfg,'reference',{str(q):1. for q in cfg['qps']},a.Budget(100))

def test_parent_training_annotations_and_dataset_version_must_match():
    a=api();old={'identity':dict(train='train',dev='dev',datasets={'coco':2})}
    runtime=dict(train='train',proxy_dev='dev',datasets={'coco':2})
    a.validate_parent_cohort(runtime,old)
    with pytest.raises(ValueError):a.validate_parent_cohort(dict(runtime,train='changed annotations'),old)
    with pytest.raises(ValueError):a.validate_parent_cohort(dict(runtime,datasets={'coco':3}),old)

def test_offline_cached_masks_can_prepare_into_a_fresh_output_directory(tmp_path,monkeypatch):
    a=api()
    from reproduce.sam_teacher_recipe import config
    from reproduce import sam_structured
    cfg=config('od');x=torch.rand(1,1,3,6,7);image=tmp_path/'image.bin';image.write_bytes(b'pixels')
    row=dict(id='one',path=str(image));root=tmp_path/'shared_sam_maps'
    cache=sam_structured.StructuredMaskCache(root,cfg['sam2']);cache.store(row,x,torch.zeros(1,1,2,6,7))
    monkeypatch.setattr(sam_structured,'load',lambda row,cfg:(x,{},row))
    out=tmp_path/'fresh_preparation'
    result,info=a.prepare_maps([row],cfg,out,a.Budget(1000))
    assert out.joinpath('sam_cache.json').exists() and info['sources'][0]['cached']
    assert torch.equal(result.load(row,x),torch.zeros(1,1,2,6,7))

@pytest.mark.parametrize('interrupted_step',[2,4])
def test_resume_retries_pending_boundary_ranking_before_more_task_updates(tmp_path,monkeypatch,interrupted_step):
    a=api();from reproduce.sam_teacher_recipe import config
    monkeypatch.setenv('PAPER_SOURCE_SHA256','s')
    cfg=config('od');cfg.update(task_steps=4,checkpoint_every=2,evaluation_reserve=2,log_every=1)
    npp=TinyNPP();proxy=Proxy();proxy.requires_grad_(False);task=Task();task.model=nn.Identity();task.weight_digest=a.model_digest(task.model)
    x=torch.full((1,1,3,6,7),.5);samples=[(x,{},dict(id='x'))];maps=[torch.zeros(1,1,2,6,7)]
    gains={str(q):1. for q in cfg['qps']}
    def interrupted(model,step):
        if step==interrupted_step:raise a.BudgetExpired('ranking interrupted')
        return dict(complete=True,step=step)
    with pytest.raises(a.BudgetExpired):
        a.train_arm(npp,proxy,task,Codec(),samples,maps,cfg,'reference',gains,a.Budget(10000),tmp_path,{'c':'r'},'shared',interrupted)
    saved=torch.load(tmp_path/'last.pt',map_location='cpu',weights_only=False);assert saved['task_updates']==interrupted_step
    called=[]
    def ranked(model,step):called.append(step);return dict(complete=True,step=step)
    result=a.train_arm(npp,proxy,task,Codec(),samples,maps,cfg,'reference',gains,a.Budget(10000),tmp_path,{'c':'r'},'shared',ranked,resume=saved)
    assert called[0]==interrupted_step and set(result['candidates'])=={2,4} and result['training_updates']==4

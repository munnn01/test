import math
import pytest
import torch
import importlib.util
def api():
    assert importlib.util.find_spec("reproduce.sam_teacher_recipe") is not None,"teacher recipe is not implemented"
    return __import__("reproduce.sam_teacher_recipe",fromlist=["*"])

def test_recipe_keeps_original_pipeline_and_has_bounded_two_arm_study():
    recipe=api()
    for task in ('is','od'):
        for codec in ('h264','h265'):
            cfg=recipe.config(task,codec)
            assert cfg['qps']==[30,35,40,45] and not cfg['identity_fallback']
            assert cfg['teacher_sigma']==.8 and cfg['boundary_weight']==0 and cfg['sam2']['foreground_extra']==1
            assert cfg['task_steps']==1536 and cfg['ranking_sources']==32 and cfg['validation_sources']==256
            assert cfg['experiment_arms']==['reference','balanced'] and cfg['input_condition']=='released_lambda_over_8'
            assert cfg['warmup_lr']==1e-4 and cfg['lr']==3e-5 and cfg['max_session_seconds']==10200

def test_gaussian_target_is_detached_and_preserves_video_geometry():
    recipe=api()
    x=torch.rand(1,1,3,13,17,requires_grad=True)
    y=recipe.teacher_target(x,.8)
    assert y.shape==x.shape and not y.requires_grad and y.min()>=0 and y.max()<=1
    assert not torch.equal(x,y)

def test_gains_fit_train_only_and_balance_weighted_gradient_ratio():
    recipe=api()
    rows=[dict(split='train',qp=q,rate_norm=r,task_norm=1.) for q,r in zip((30,35,40,45),(2.,.5,.1,.01)) for _ in range(3)]
    gains=recipe.fit_task_gains(rows,.5)
    assert gains=={'30':4.,'35':1.,'40':.2,'45':.05}
    bad=[dict(rows[0],split='val')]+rows[1:]
    with pytest.raises(ValueError):recipe.fit_task_gains(bad,.5)
    with pytest.raises(ValueError):recipe.fit_task_gains([dict(v,rate_norm=0.) for v in rows],.5)

def test_balanced_objective_changes_only_task_term_and_has_gradient():
    recipe=api()
    cfg=recipe.config('od','h264');x=torch.zeros(1,1,3,6,7);y=torch.full_like(x,.1,requires_grad=True)
    maps=torch.zeros(1,1,2,6,7);maps[:,:,:1]=.5
    rate=y.square().mean();task=2*y.mean();gains={str(q):.25 for q in cfg['qps']}
    a,_=recipe.objective(cfg,'reference',rate,task,x,y,45,maps,gains)
    b,parts=recipe.objective(cfg,'balanced',rate,task,x,y,45,maps,gains)
    from reproduce.protocol import task_lambda
    assert float(a-b)==pytest.approx(float(.75*cfg['lambda_scale']*task_lambda(45)*task))
    assert parts['task_gain'].item()==.25
    b.backward();assert torch.isfinite(y.grad).all() and y.grad.abs().sum()>0

def test_cohorts_exclude_training_ranking_and_all_previous_val_sources():
    recipe=api()
    train=[dict(id='t'+str(i)) for i in range(120)];val=[dict(id='v'+str(i)) for i in range(400)]
    previous=dict(train=[r['id'] for r in train[:64]],dev=[r['id'] for r in train[64:72]],validation=[r['id'] for r in val[:64]])
    parts=recipe.cohorts(train,val,previous,recipe.config('od','h264'))
    assert [len(v) for v in parts]==[64,8,32,256]
    ids=[{r['id'] for r in rows} for rows in parts]
    assert not ids[0]&ids[1] and not ids[0]&ids[2] and not ids[1]&ids[2]
    assert not ids[3]&set(previous['validation'])
    assert recipe.cohorts(train,val,previous,recipe.config('od','h264'))==parts

def curves(scale,offset=0.,sources=32):
    rows=[]
    for q,rate,quality in zip((30,35,40,45),(1.,.6,.3,.1),(.8,.65,.5,.3)):
        rows+=[dict(arm='npp',qp=q,bpp=rate*scale,quality=quality+offset,sources=sources),
               dict(arm='codec_only',qp=q,bpp=rate,quality=quality,sources=sources)]
    return dict(complete=True,curves=rows)

def test_ranking_has_one_shared_interval_across_both_arms_and_cannot_choose_zero():
    recipe=api()
    metrics={'reference':{512:curves(.9),1024:curves(.95),1536:curves(1.)},
             'balanced':{512:curves(.85,.01),1024:curves(.8,.02),1536:curves(.88,-.01)}}
    ranked=recipe.rank_candidates(metrics,curves(.7),curves(.75))
    assert ranked['selected_steps']=={'reference':512,'balanced':1024}
    assert ranked['common_quality_interval']==pytest.approx([.32,.79])
    assert not ranked['improved_over_warm']['balanced']
    bad={'reference':dict(metrics['reference']),'balanced':dict(metrics['balanced'])};bad['balanced'][1536]=curves(.9,2.)
    with pytest.raises(ValueError):recipe.rank_candidates(bad,curves(.7),curves(.75))

def test_budget_or_incomplete_candidates_cannot_complete():
    recipe=api()
    rows={arm:dict(training_updates=1536,selected_step=512,final_metrics=curves(.9,sources=256),gate_passed=True) for arm in ('reference','balanced')}
    assert recipe.experiment_complete(rows)
    rows['balanced']['training_updates']=1024;assert not recipe.experiment_complete(rows)
    rows['balanced']['training_updates']=1536;rows['balanced']['selected_step']=0;assert not recipe.experiment_complete(rows)

def test_final_comparison_has_one_shared_AP_domain_and_native_byte_accounting():
    recipe=api()
    r=recipe.final_common_bd({'reference':curves(.9,sources=256),'balanced':curves(.8,sources=256),'gaussian':curves(.75,sources=256)})
    assert r['quality_interval']==pytest.approx([.3,.8])
    assert r['percent']==pytest.approx({'reference':-10.,'balanced':-20.,'gaussian':-25.})

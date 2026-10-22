import copy
import json
from pathlib import Path

import pytest
import torch


def recovery():
    from ar_od_proxy import recovery
    return recovery


def test_candidate_bank_is_bounded_identity_and_causal_given_fixed_teacher_maps():
    r = recovery()
    torch.set_num_threads(2)
    x = torch.rand(1, 3, 3, 32, 32)
    protection = torch.zeros_like(x[:, :, :1])
    bank = r.candidate_bank(x, protection, max_rms=6/255)
    assert len(bank) == 7 and next(iter(bank)) == 'identity'
    torch.testing.assert_close(bank['identity'], x, rtol=0, atol=0)
    for name, y in bank.items():
        assert y.shape == x.shape and 0 <= y.min() <= y.max() <= 1
        assert (y-x).square().mean().sqrt() <= 6/255+1e-7
        changed = x.clone(); changed[:, 2] = 1-x[:, 2]
        torch.testing.assert_close(r.candidate_bank(changed, protection, 6/255)[name][:, :2],
                                   y[:, :2], rtol=0, atol=0)
    protected = r.candidate_bank(x, torch.ones_like(protection), 6/255)
    torch.testing.assert_close(protected['background_weak'], x, rtol=0, atol=0)


def test_balanced_schedule_covers_every_source_point_and_changes_by_seed():
    r = recovery()
    records = [{'id':f'{t}/{i}', 'task':t} for i in range(4) for t in ('ar', 'od')]
    a = r.balanced_schedule(records, ['h264','h265'], [30,35,40,45], seed=9)
    b = r.balanced_schedule(records, ['h264','h265'], [30,35,40,45], seed=10)
    assert a == r.balanced_schedule(records, ['h264','h265'], [30,35,40,45], seed=9)
    assert a != b and len(a) == 16
    flat = [case for batch in a for case in batch]
    assert len(set(flat)) == 64
    for batch in a:
        assert len(batch) == 4 and len({i for i,_,_ in batch}) == 4
        assert sorted(records[i]['task'] for i,_,_ in batch) == ['ar','ar','od','od']


def test_selection_never_uses_infeasible_candidate_for_rate_saving():
    r = recovery()
    baseline = {'terms':{'ar':1.,'od':1.}, 'ar_correct':True, 'bpp':1.}
    rows = [{'candidate':'identity', **baseline, 'total_rms':0.},
            {'candidate':'unsafe', **dict(baseline, bpp=.5, terms={'ar':1.2,'od':1.}), 'total_rms':.001},
            {'candidate':'good', **dict(baseline, bpp=.9), 'total_rms':.001}]
    assert r.select_candidate(rows, baseline, .02, 6/255)['candidate'] == 'good'
    rows[-1]['ar_correct'] = False
    assert r.select_candidate(rows, baseline, .02, 6/255)['candidate'] == 'identity'
    rows[-1]['ar_correct'] = True; rows[-1]['total_rms'] = 1.
    assert r.select_candidate(rows, baseline, .02, 6/255)['candidate'] == 'identity'


def test_replay_is_unique_bounded_and_returns_valid_hard_cases():
    r = recovery()
    replay = r.ReplayQueue(3)
    for i in [0,1,2,1,3]: replay.add((i,'h264',35))
    assert len(replay.cases) == 3 and len(set(replay.cases)) == 3
    chosen = replay.take(2, exclude={(3,'h264',35)})
    assert len(chosen) == 2 and all(c in replay.cases and c[0] != 3 for c in chosen)


def test_short_transaction_restores_adam_and_only_counts_kept_updates():
    r = recovery()
    model = torch.nn.Linear(1,1,bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=.1)
    transaction = r.ShortTransaction(model,optimizer, interval=16, minimum=4)
    original = copy.deepcopy(model.state_dict())
    for _ in range(2):
        optimizer.zero_grad(); model(torch.ones(1,1)).sum().backward(); optimizer.step()
        transaction.note(True)
    assert transaction.finish(False)['rolled_back_proposals'] == 2
    torch.testing.assert_close(model.weight, original['weight'], rtol=0, atol=0)
    assert optimizer.state_dict()['state'] == {} and transaction.interval == 8
    optimizer.zero_grad(); model(torch.ones(1,1)).sum().backward(); optimizer.step()
    transaction.note(True)
    counts = transaction.finish(True)
    assert counts['committed_proposals'] == 1 and counts['accepted_proposals'] == 3
    assert counts['committed_proposals']+counts['rolled_back_proposals'] == counts['accepted_proposals']


def test_recovery_payload_has_new_runner_config_and_private_checkpoint(tmp_path):
    from ar_od_proxy.kaggle import prepare,verify_payload
    from ar_od_proxy.run import configuration
    root = Path(__file__).resolve().parents[1]
    cfg = configuration(root/'ar_od_proxy/configs/recovery_pilot.json')
    assert cfg['train_count'] == 32 and cfg['npp_steps'] == 512
    assert cfg['anchor_every'] == 16 and cfg['stall_steps'] == 128
    assert cfg['arms'] == ['replay_identity','replay_warmstart']
    prepare(root,tmp_path/'payload','a'*40,'example','recovery-pilot', action='recovery',
            baseline_sha256='b'*64,checkpoint_dataset='example/private-checkpoint')
    _,job = verify_payload(tmp_path/'payload','example/recovery-pilot')
    code = ''.join(json.loads((tmp_path/'payload/notebook.ipynb').read_text())['cells'][0]['source'])
    assert job['action'] == 'recovery' and 'ar_od_proxy.recovery' in code
    assert 'recovery_pilot.json' in code


class ToyTasks:
    def losses(self,x,record,refs):
        zero = x.sum()*0
        od = .1+x.square().mean()*.001*(1+torch.rand((),device=x.device))
        return {'ar_gt':zero+.1 if record['task']=='ar' else zero,
                'ar_source':zero,'od_gt':od if record['task']=='od' else zero,
                'od_source':od if record['task']=='ar' else zero}

    def ar_logits(self,x):
        return torch.stack((x.mean().reshape(1)+1,x.mean().reshape(1)*0),1)


def small_config():
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'ar_od_proxy/configs/recovery_pilot.json').read_text())
    return dict(cfg,width=4,frames=16,size=32,train_count=2,npp_steps=4,
                anchor_every=2,minimum_block=1,anchor_per_task=1,stall_steps=4,
                warmstart_steps=2)


def test_oracle_labels_are_real_codec_train_only_and_identity_paired(tmp_path):
    r=recovery()
    from ar_od_proxy.codec import RealCodec
    torch.manual_seed(4); torch.set_num_threads(2)
    x=torch.rand(1,1,3,32,32)
    samples=[({'id':'image/a','task':'od'},x,None)]
    refs=[{}]; masks=[torch.ones_like(x[:,:,:1])]
    cfg=small_config()
    context=r.RecoveryContext(samples,refs,masks,ToyTasks(),RealCodec(preset='ultrafast',gop=16),cfg)
    labels=r.collect_candidates(context,tmp_path/'oracle')
    assert len(labels)==8 and len(context.baselines)==8
    rows=[json.loads(line) for line in (tmp_path/'oracle/candidates.jsonl').read_text().splitlines()]
    assert len(rows)==56 and all(row['source']=='image/a' and row['split']=='TRAIN' for row in rows)
    for row in rows:
        assert row['bytes'] > 0 and len(row['stream_sha256'])==64
    for case,label in labels.items():
        assert r.quality_feasible(label['measurement'],context.baselines[case],.02)
    assert (tmp_path/'oracle/summary.json').is_file()


def test_balanced_recovery_and_warmstart_keep_proxy_frozen_and_checkpoint_guarded(tmp_path):
    r=recovery()
    from ar_od_proxy.models import VideoNPP,VideoProxy
    from ar_od_proxy.codec import CodecOutput
    from ar_od_proxy.engine import load_checkpoint
    torch.manual_seed(8); torch.set_num_threads(2)
    class Codec:
        @torch.no_grad()
        def roundtrip(self,x,c,q):
            decoded=(x*255).round()/255
            return CodecOutput(decoded, x.new_tensor([1.+float(x.square().mean())]), [b'fixture'],0.)
    samples=[({'id':f'{t}/{i}','task':t,'ar_label':0},torch.rand(1,2 if t=='ar' else 1,3,32,32),None)
             for i in range(2) for t in ('ar','od')]
    cfg=dict(small_config(),npp_steps=12)
    refs=[{} for _ in samples]; masks=[torch.ones_like(x[:,:,:1]) for _,x,_ in samples]
    context=r.RecoveryContext(samples,refs,masks,ToyTasks(),Codec(),cfg)
    labels=r.collect_candidates(context,tmp_path/'oracle')
    initial=r.identity_start(VideoNPP(4),.2)
    proxy=VideoProxy(4).requires_grad_(False).eval()
    digest=r.tensor_digest(proxy)
    warm=copy.deepcopy(initial)
    result=r.fit_warmstart(warm,initial,proxy,context,labels,tmp_path/'warm')
    assert result['final_panel']['passed']
    summary=r.fit_recovery(warm,proxy,context,tmp_path/'arm')
    assert summary['counts']['proposals'] == 4
    assert summary['fixed_proxy_verified'] and r.tensor_digest(proxy)==digest
    assert summary['final_panel']['passed']
    counts=summary['counts']
    assert counts['accepted_proposals']==counts['committed_proposals']+counts['rolled_back_proposals']
    restored,_,saved=load_checkpoint(tmp_path/'arm/selected.pt')
    assert saved['stage']=='recovery_selected'
    torch.testing.assert_close(restored(samples[0][1],'h264',30),warm(samples[0][1],'h264',30),rtol=0,atol=0)
    rows=[json.loads(line) for line in (tmp_path/'arm/training.jsonl').read_text().splitlines()]
    assert all(len(row['batch'])==4 and row['gradient_sources']>=4 for row in rows)
    assert (tmp_path/'arm/optimizer.pt').is_file()


def test_runner_refuses_leakage_and_invalid_budget_before_training():
    r=recovery()
    cfg=small_config()
    r.validate_protocol(cfg)
    for override in ({'anchor_every':256},{'arms':['anything']},
                     {'warmstart_scales':[1.,.5]},{'stall_steps':0},{'replay_per_step':0}):
        with pytest.raises(ValueError): r.validate_protocol(dict(cfg,**override))
    prior={'train':[{'id':'ar/old'}],'dev':[{'id':'ar/dev'}]}
    valid={'train':[{'id':'ar/old'},{'id':'od/train'}], 'dev':[{'id':'ar/dev'}]}
    r.validate_manifest(valid,prior)
    with pytest.raises(ValueError):
        r.validate_manifest({'train':valid['train']+[{'id':'ar/dev'}],'dev':[]},prior)
    with pytest.raises(ValueError):
        r.validate_manifest({'train':[{'id':'od/train'}],'dev':valid['dev']},prior)


def test_selection_keeps_smaller_gain_than_stall_threshold(tmp_path,monkeypatch):
    r=recovery()
    from ar_od_proxy.models import VideoNPP,VideoProxy
    from ar_od_proxy.codec import CodecOutput
    torch.manual_seed(34); torch.set_num_threads(2)
    class Codec:
        @torch.no_grad()
        def roundtrip(self,x,c,q):
            return CodecOutput((x*255).round()/255,x.new_tensor([1.]),[b'fixture'],0.)
    samples=[({'id':f'{t}/{i}','task':t,'ar_label':0},torch.rand(1,1,3,32,32),None)
             for i in range(2) for t in ('ar','od')]
    cfg=dict(small_config(),npp_steps=1,anchor_every=1,minimum_block=1,stall_steps=1)
    masks=[torch.ones_like(x[:,:,:1]) for _,x,_ in samples]
    context=r.RecoveryContext(samples,[{} for _ in samples],masks,ToyTasks(),Codec(),cfg)
    context.baselines={case:context.measure(context.x(case),case) for case in context.all_cases()}
    npp=r.identity_start(VideoNPP(4),.2)
    initial=r.tensor_digest(npp)
    proxy=VideoProxy(4).requires_grad_(False).eval()
    def check(model,cases,previous=None,full=False):
        changed=r.tensor_digest(model)!=initial
        return {'passed':True,'violations':[],'cases':[],
                'mean_rate_ratio':.99995 if changed else 1.,'changed_rgb8_cases':int(changed)}
    monkeypatch.setattr(context,'check',check)
    report=r.fit_recovery(npp,proxy,context,tmp_path/'arm')
    assert report['counts']['committed_proposals']==1
    assert report['selected_step']==1 and report['best_panel_rate_ratio']==.99995
    assert r.tensor_digest(npp)!=initial

import json
from pathlib import Path

import numpy as np
import pytest
import torch


def test_regional_starts_exactly_at_base_and_can_learn_filter(tmp_path):
    from ar_od_proxy.models import VideoNPP, VideoProxy, RegionalVideoNPP
    from ar_od_proxy.engine import save_checkpoint, load_checkpoint
    torch.set_num_threads(2)
    base = VideoNPP(4)
    model = RegionalVideoNPP.from_base(base)
    x = torch.rand(1, 2, 3, 32, 32)
    for codec in ('h264', 'h265'):
        for qp in (30, 35, 40, 45):
            torch.testing.assert_close(model(x, codec, qp), base(x, codec, qp), rtol=0, atol=0)
    model(x, 'h264', 35).square().mean().backward()
    assert model.filter_strength.grad.abs().sum() > 0
    with torch.no_grad():
        model.filter_strength.fill_(.2)
    expected = model(x, 'h265', 40)
    path = tmp_path/'regional.pt'
    cfg = {'width':4, 'npp_kind':'regional'}
    save_checkpoint(path, model, VideoProxy(4), cfg, 'screen', 1)
    restored, _, _ = load_checkpoint(path)
    assert isinstance(restored, RegionalVideoNPP)
    torch.testing.assert_close(restored(x, 'h265', 40), expected, rtol=0, atol=0)
    # Causal operation: a future frame cannot change the first output.
    altered = x.clone(); altered[:, 1] = 0
    torch.testing.assert_close(model(x, 'h265', 40)[:, 0], model(altered, 'h265', 40)[:, 0])


def test_normalized_objective_is_rate_scale_invariant_and_penalizes_quality_excess():
    from ar_od_proxy.screen import constrained_objective
    rate = torch.tensor(2., requires_grad=True)
    ar = torch.tensor(.9, requires_grad=True)
    od = torch.tensor(1.2, requires_grad=True)
    loss, excess = constrained_objective(rate, 2., {'ar':ar, 'od':od},
                                        {'ar':1., 'od':1.}, {'ar':2., 'od':3.}, .02)
    scaled, _ = constrained_objective(rate*10, 20., {'ar':ar, 'od':od},
                                      {'ar':1., 'od':1.}, {'ar':2., 'od':3.}, .02)
    torch.testing.assert_close(loss, scaled)
    assert loss.item() == pytest.approx(1+3*.18)
    loss.backward()
    assert rate.grad > 0 and ar.grad == 0 and od.grad > 0
    assert excess['ar'] < 0 and excess['od'] > 0
    with pytest.raises(ValueError, match='identity'):
        constrained_objective(rate, 0., {'ar':ar}, {'ar':1.}, {'ar':1.}, .02)


def test_duals_are_separate_bounded_and_relax_after_recovery():
    from ar_od_proxy.screen import update_duals
    a, b = {'ar':1., 'od':1.}, {'ar':1., 'od':1.}
    update_duals(a, {'ar':1., 'od':-.5}, .2, 5.)
    assert b == {'ar':1., 'od':1.}
    assert a['ar'] > 1 and 0 < a['od'] < 1
    for _ in range(100):
        update_duals(a, {'ar':10., 'od':-10.}, .2, 5.)
    assert a == {'ar':5., 'od':.05}


def test_identity_and_candidate_sampling_seed_matches_on_every_training_pass():
    from ar_od_proxy.screen import case_seed
    cfg = {'seed':27,'codecs':['h264','h265'],'qps':[30,35,40,45]}
    for case in range(32*8):
        for epoch in (0,1,11):
            assert case_seed(case+epoch*256,32,cfg) == 27+case


def test_bdrate_exact_proportional_saving_and_refuses_bad_curves():
    from ar_od_proxy.bdrate import bd_rate
    rates = np.array([4., 3., 2., 1.])
    quality = [.9, .8, .6, .4]
    result = bd_rate(rates, quality, rates*.8, quality)
    assert result['percent'] == pytest.approx(-20., abs=1e-8)
    assert result['quality_overlap'] == [.4, .9]
    for bad in ([.9,.8,.8,.4], [.9,.6,.8,.4], [.4,.6,.8,.9]):
        invalid = bd_rate(rates, bad, rates*.8, quality)
        assert invalid['percent'] is None and invalid['reason']
    assert bd_rate(rates, [.3,.2,.15,.1], rates*.8, quality)['percent'] is None
    assert bd_rate([4,3,2], [.9,.8,.6], rates*.8, quality)['percent'] is None


def test_screen_cloud_action_uses_owned_private_checkpoint_and_pinned_config(tmp_path):
    from ar_od_proxy.kaggle import prepare, verify_payload
    root = Path(__file__).resolve().parents[1]
    payload = tmp_path/'payload'
    prepare(root, payload, 'a'*40, 'example', 'bdrate-screen', action='screen',
            baseline_sha256='b'*64, checkpoint_dataset='example/calibrated-private-input')
    metadata, job = verify_payload(payload, 'example/bdrate-screen')
    code = ''.join(json.loads((payload/'notebook.ipynb').read_text())['cells'][0]['source'])
    assert metadata['is_private'] and not metadata['kernel_sources']
    assert 'ar_od_proxy.screen' in code and 'bdrate_screen.json' in code
    cfg = json.loads((root/'ar_od_proxy/configs/bdrate_screen.json').read_text())
    assert cfg['npp_steps'] == 3072 and cfg['dev_count'] == 32
    assert cfg['arms'] == ['existing', 'normalized', 'regional']
    assert cfg['qps'] == [30,35,40,45] and cfg['codecs'] == ['h264','h265']


def test_notebook_embeds_exact_sources_and_logs_each_startup_phase(tmp_path):
    import ast
    import base64
    import hashlib
    import zlib
    from ar_od_proxy.kaggle import prepare
    root = Path(__file__).resolve().parents[1]
    folder = tmp_path/'payload'
    job = prepare(root,folder,'a'*40,'example','bdrate-screen',action='screen',
                  baseline_sha256='b'*64,checkpoint_dataset='example/private-checkpoint')
    code = ''.join(json.loads((folder/'notebook.ipynb').read_text())['cells'][0]['source'])
    constants = {n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(code).body
                 if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant)}
    files = json.loads(zlib.decompress(base64.b64decode(constants['SOURCE_BUNDLE'])))
    assert set(files)==set(job['source_hashes'])
    for name, content in files.items():
        assert hashlib.sha256(content.encode('utf-8')).hexdigest()==job['source_hashes'][name]
    assert constants['CHECKPOINT_DATASET']=='example/private-checkpoint'
    for phase in ('sources_verified','dependencies_ready','inputs_ready','runner_start'):
        assert phase in code
    assert 'timeout=180' in code


def test_each_arm_runs_same_real_codec_grid_and_saves_frozen_proxy(tmp_path):
    from torch import nn
    from ar_od_proxy.screen import fit_arm
    from ar_od_proxy.models import VideoNPP, VideoProxy, RegionalVideoNPP
    from ar_od_proxy.codec import RealCodec
    from ar_od_proxy.engine import load_checkpoint
    torch.set_num_threads(2)

    class Tasks:
        def __init__(self):
            self.ar = nn.Conv2d(3,2,1).requires_grad_(False)
            self.od = nn.Conv2d(3,2,1).requires_grad_(False)
        def losses(self,x,record,refs):
            return {'ar_gt':x.sum()*0, 'ar_source':x.sum()*0,
                    'od_gt':self.od(x[:,0]).square().mean(),'od_source':x.sum()*0}

    x = torch.rand(1,1,3,32,32)
    samples = [({'id':'fixture/image','task':'od'},x,None)]
    references = [{'od':[{'boxes':torch.tensor([[4.,4.,24.,24.]]), 'labels':torch.tensor([1])}], 'indices':[0]}]
    proxy = VideoProxy(4).requires_grad_(False).eval()
    original_proxy = {k:v.clone() for k,v in proxy.state_dict().items()}
    base = VideoNPP(4)
    tasks = Tasks()
    codec = RealCodec(preset='ultrafast',gop=1)
    baselines = {}
    for name in ('h264','h265'):
        for qp in (30,35,40,45):
            packet = codec.roundtrip(x,name,qp)
            baselines[(0,name,qp)] = {'bpp':float(packet.bpp.mean()),
                                     'terms':{'od':float(tasks.losses(packet.reconstruction,{}, {})['od_gt'])}}
    cfg = {'seed':9,'width':4,'npp_steps':8,'learning_rate':2e-5,'diagnostic_every':1,
           'codecs':['h264','h265'],'qps':[30,35,40,45],'max_npp_update_rms':1/255,
           'loss_weights':{'rate':1.,'ar':1.,'od':1.,'pre':.5,'temp':.1},
           'quality_allowance':.02,'dual_learning_rate':.05,'dual_maximum':20.}
    schedules = []
    for arm in ('existing','normalized','regional'):
        model = RegionalVideoNPP.from_base(base) if arm=='regional' else __import__('copy').deepcopy(base)
        folder = tmp_path/arm
        fit_arm(model,proxy,codec,tasks,samples,references,baselines,cfg,arm,folder)
        rows = [json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
        schedules.append([(r['source'],r['codec'],r['qp']) for r in rows])
        assert len(rows)==8 and all(0 < r['output_update_rms'] <= 1/255 for r in rows)
        assert all('actual_step' in r and r['gradient_norms']['rate'] > 0 for r in rows)
        restored, saved_proxy, metadata = load_checkpoint(folder/'last.pt')
        assert metadata['step']==8 and (folder/'optimizer.pt').is_file()
        assert all(torch.equal(v,saved_proxy.state_dict()[k]) for k,v in original_proxy.items())
        torch.testing.assert_close(restored(x,'h265',45),model(x,'h265',45),rtol=0,atol=0)
    assert schedules[0]==schedules[1]==schedules[2]
    assert all(torch.equal(v,proxy.state_dict()[k]) for k,v in original_proxy.items())

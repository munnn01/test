import importlib
import json
from pathlib import Path

import pytest
import torch

from ar_od_proxy.models import VideoNPP, VideoProxy


def test_calibrated_warm_start_preserves_weights_and_freezes_proxy(tmp_path):
    engine = importlib.import_module('ar_od_proxy.engine')
    assert hasattr(engine, 'load_calibrated_start'), 'missing calibrated training entry point'
    cfg = {'width':4, 'learning_rate':1e-4, 'proxy_refresh_every':128,
           'frames':16, 'qps':[30,35,40,45], 'codecs':['h264','h265']}
    npp, proxy = VideoNPP(4), VideoProxy(4)
    with torch.no_grad():
        npp.rgb.bias.fill_(.1)
        proxy.rate_head.log_header.fill_(4.)
    path=tmp_path/'calibrated.pt'
    engine.save_checkpoint(path,npp,proxy,cfg,'proxy_calibration',1024)
    target=dict(cfg,learning_rate=2e-5,proxy_refresh_every=0,max_npp_update_rms=1/255)
    restored, frozen, meta=engine.load_calibrated_start(path,target)
    for k,v in npp.state_dict().items():
        torch.testing.assert_close(v,restored.state_dict()[k])
    for k,v in proxy.state_dict().items():
        torch.testing.assert_close(v,frozen.state_dict()[k])
    assert all(not p.requires_grad for p in frozen.parameters())
    assert meta['stage']=='proxy_calibration'
    with pytest.raises(ValueError,match='config'):
        engine.load_calibrated_start(path,dict(target,frames=32))
    with pytest.raises(ValueError,match='refresh'):
        engine.load_calibrated_start(path,dict(target,proxy_refresh_every=128))


def test_uncalibrated_checkpoint_cannot_warm_start_calibrated_training(tmp_path):
    engine=importlib.import_module('ar_od_proxy.engine')
    assert hasattr(engine,'load_calibrated_start'), 'missing calibrated-stage validation'
    cfg={'width':4,'proxy_refresh_every':0}
    path=tmp_path/'wrong.pt'
    engine.save_checkpoint(path,VideoNPP(4),VideoProxy(4),cfg,'npp',1)
    with pytest.raises(ValueError,match='calibrat'):
        engine.load_calibrated_start(path,cfg)


def test_proposed_parameter_update_is_backtracked_to_output_rms_bound():
    from ar_od_proxy import engine
    assert hasattr(engine,'bounded_optimizer_step'), 'missing output-step guard'
    model=VideoNPP(4)
    x=torch.rand(1,2,3,32,32)
    previous=model(x,'h264',35).detach()
    optimizer=torch.optim.Adam(model.parameters(),lr=1.)
    loss=model(x,'h264',35).mean()
    loss.backward()
    rms,scale=engine.bounded_optimizer_step(model,optimizer,x,previous,'h264',35,1/255)
    with torch.no_grad():
        actual=(model(x,'h264',35)-previous).square().mean().sqrt().item()
    assert 0 < actual <= 1/255
    assert actual==pytest.approx(rms)
    assert 0 < scale < 1


def test_deploy_notebook_mounts_calibration_and_selects_warm_start_command(tmp_path):
    from ar_od_proxy import kaggle
    directory=tmp_path/'payload'
    kaggle.prepare(Path(__file__).resolve().parents[1],directory,'a'*40,'example','calibrated-deploy',
                   action='train-calibrated',baseline_sha256='b'*64)
    metadata,job=kaggle.verify_payload(directory,'example/calibrated-deploy')
    code=''.join(json.loads((directory/'notebook.ipynb').read_text())['cells'][0]['source'])
    assert metadata['kernel_sources']==['qktttttttttt/ar-od-proxy-calibration-qp4-20261022']
    assert 'calibrated.pt' in code and 'calibrated_train.json' in code
    assert "'--checkpoint'" in code and "'train-calibrated'" in code
    assert job['action']=='train-calibrated' and 'b'*64 in code


def test_deploy_config_keeps_grid_and_uses_small_bounded_updates():
    from ar_od_proxy.run import configuration
    path=Path(__file__).resolve().parents[1]/'ar_od_proxy/configs/calibrated_train.json'
    assert path.is_file(), 'missing calibrated deployment config'
    cfg=configuration(path)
    assert cfg['qps']==[30,35,40,45] and cfg['codecs']==['h264','h265']
    assert cfg['learning_rate']==2e-5 and cfg['max_npp_update_rms']==1/255
    assert cfg['npp_steps']==256 and cfg['proxy_refresh_every']==0


@pytest.mark.parametrize('bound',[None,0.,-1.,float('nan'),float('inf'),4/255])
def test_calibrated_training_cannot_silently_omit_or_disable_update_bound(tmp_path,bound):
    from ar_od_proxy import engine
    cfg={'width':4,'proxy_refresh_every':0,'learning_rate':2e-5}
    path=tmp_path/'calibrated.pt'
    engine.save_checkpoint(path,VideoNPP(4),VideoProxy(4),cfg,'proxy_calibration',1024)
    requested=dict(cfg)
    if bound is not None:
        requested['max_npp_update_rms']=bound
    with pytest.raises(ValueError,match='RMS'):
        engine.load_calibrated_start(path,requested)


def test_deploy_can_use_an_owned_private_checkpoint_dataset_on_another_account(tmp_path):
    from ar_od_proxy import kaggle
    directory=tmp_path/'payload'
    dataset='example/calibrated-private-input'
    kaggle.prepare(Path(__file__).resolve().parents[1],directory,'a'*40,'example','calibrated-deploy',
                   action='train-calibrated',baseline_sha256='b'*64,checkpoint_dataset=dataset)
    metadata,job=kaggle.verify_payload(directory,'example/calibrated-deploy')
    assert metadata['kernel_sources']==[]
    assert metadata['dataset_sources']==kaggle.DATASETS+[dataset]
    assert job['checkpoint_dataset']==dataset
    with pytest.raises(ValueError,match='dataset'):
        kaggle.prepare(Path(__file__).resolve().parents[1],tmp_path/'bad','a'*40,'example','calibrated-deploy',
                       action='train-calibrated',baseline_sha256='b'*64,checkpoint_dataset='someone/other-input')

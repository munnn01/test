import hashlib,importlib.util,json,shutil
from pathlib import Path
import pytest
from reproduce.training import config_digest
from reproduce.sam_teacher_recipe import config

def api():
    assert importlib.util.find_spec('reproduce.sam_teacher_cloud') is not None,'teacher cloud not implemented'
    return __import__('reproduce.sam_teacher_cloud',fromlist=['*'])

def setup_payload(tmp_path,monkeypatch):
    cloud=api();import reproduce.sam_power_recover as recovery
    root=tmp_path/'source';(root/'reproduce').mkdir(parents=True);(root/'reproduce/run.py').write_text('pass\n')
    manifest={'reproduce/run.py':hashlib.sha256((root/'reproduce/run.py').read_bytes()).hexdigest()}
    source=config_digest(manifest);monkeypatch.setattr(cloud,'PARENT_SOURCE',source);monkeypatch.setattr(recovery,'PARENT_SOURCE',source)
    oldnames=['cloud_status.json','control/study_run.json','control/config.json','control/cohort.json','control/analyzer.json','control/validation/metrics.json']
    spec=dict(ref='owner/sam-power-is-h264-20261101/1',parent_transport_version=2,source_sha256=source,source_hashes=manifest,
        files={n:'a'*64 for n in oldnames},checkpoint_files={n:'b'*64 for n in ('shared_warmup.pt','control/last.pt')},
        mask_ref='owner/sam-recover-is-h264-20261101/3',mask_source_sha256='9fa98ecc542dc0bc193c38cfa9dcb0b47f4da5cd44d24916815f3d6d964fa69c',
        mask_files={n:'c'*64 for n in ('cloud_status.json','foreground/study_run.json','foreground/sam_cache.json')})
    folder=tmp_path/'payload';cloud.prepare(root,folder,'owner','sam-teacher-is-h264',config('is'),spec)
    return cloud,root,folder,spec

def test_teacher_cloud_binds_both_private_owned_artifacts_and_exact_recipe(tmp_path,monkeypatch):
    cloud,root,p,spec=setup_payload(tmp_path,monkeypatch)
    meta,job=cloud.verify(p,root)
    assert meta['kernel_sources']==['owner/sam-power-is-h264-20261101','owner/sam-recover-is-h264-20261101']
    assert job['config']['task_steps']==1536 and job['config']['experiment_arms']==['reference','balanced']
    code=''.join(json.loads((p/'notebook.ipynb').read_text())['cells'][0]['source'])
    assert 'reproduce.sam_teacher_study' in code and '--mask-parent' in code
    with pytest.raises(ValueError):cloud.validate_input(dict(spec,mask_ref='different/sam-recover-is-h264-20261101/3'),'owner',config('is'))

@pytest.mark.parametrize('mutation',['extra_cell','code_file','model_sources','private','config','checkpoint_hash'])
def test_teacher_cloud_rejects_payload_mutations_even_with_refreshed_checksums(tmp_path,monkeypatch,mutation):
    cloud,root,p,spec=setup_payload(tmp_path,monkeypatch)
    job=json.loads((p/'job.json').read_text());meta=json.loads((p/'kernel-metadata.json').read_text())
    nb=json.loads((p/'notebook.ipynb').read_text())
    if mutation=='extra_cell':nb['cells'].append(dict(cell_type='code',source=['print(1)'],metadata={},outputs=[],execution_count=None))
    elif mutation=='code_file':meta['code_file']='other.py'
    elif mutation=='model_sources':meta['model_sources']=['external/model/framework/variation/1']
    elif mutation=='private':meta['is_private']=False
    elif mutation=='config':job['config']['teacher_sigma']=.5
    else:job['training_inputs']['checkpoint_files']['shared_warmup.pt']='invalid'
    for name,value in (('notebook.ipynb',nb),('kernel-metadata.json',meta)):(p/name).write_text(json.dumps(value))
    for name in job['payload_hashes']:job['payload_hashes'][name]=hashlib.sha256((p/name).read_bytes()).hexdigest()
    (p/'job.json').write_text(json.dumps(job))
    with pytest.raises(ValueError):cloud.verify(p)

@pytest.mark.parametrize('mutation',['extra_cell','execution_metadata','outputs','model_sources','competition_sources','enable_gpu'])
def test_remote_verifier_rejects_undeclared_execution_and_inputs(tmp_path,monkeypatch,mutation):
    cloud,root,p,spec=setup_payload(tmp_path,monkeypatch)
    meta=json.loads((p/'kernel-metadata.json').read_text());nb=json.loads((p/'notebook.ipynb').read_text())
    remote_meta=json.loads(json.dumps(meta));remote_nb=json.loads(json.dumps(nb))
    remote_meta['dataset_sources']=[r.rsplit('/',1)[0] for r in meta['dataset_sources']]
    remote_meta['code_file']='sam-teacher-is-h264.ipynb'
    remote_nb['cells'][0]['source']=''.join(remote_nb['cells'][0]['source'])
    cloud.verify_remote(meta,nb,remote_meta,remote_nb)
    if mutation=='extra_cell':remote_nb['cells'].append(dict(cell_type='code',source='print(1)',metadata={},execution_count=None,outputs=[]))
    elif mutation=='execution_metadata':remote_nb['cells'][0]['execution_count']=1
    elif mutation=='outputs':remote_nb['cells'][0]['outputs']=[dict(output_type='stream',name='stdout',text='modified')]
    elif mutation=='enable_gpu':remote_meta['enable_gpu']=False
    else:remote_meta[mutation]=['undeclared']
    with pytest.raises(ValueError):cloud.verify_remote(meta,nb,remote_meta,remote_nb)

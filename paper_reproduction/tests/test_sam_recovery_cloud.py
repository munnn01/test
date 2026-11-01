import hashlib,json
from pathlib import Path
import pytest
from reproduce.sam_power_study import config
from reproduce import sam_recovery_cloud as cloud
from reproduce import sam_power_recover as recovery
from reproduce.training import config_digest

@pytest.mark.parametrize('mutation',['code','extra_cell','code_file','language','kernel_type','model_sources','competition_sources'])
def test_recovery_notebook_binds_same_owner_parent_and_rejects_code_tamper(tmp_path,monkeypatch,mutation):
    root=Path(__file__).resolve().parents[1]
    manifest={'reproduce/run.py':hashlib.sha256((root/'reproduce/run.py').read_bytes()).hexdigest()}
    sha=config_digest(manifest);monkeypatch.setattr(cloud,'PARENT_SOURCE',sha);monkeypatch.setattr(recovery,'PARENT_SOURCE',sha)
    names=['cloud_status.json','control/study_run.json','control/config.json','control/cohort.json','control/analyzer.json','control/validation/metrics.json']
    spec=dict(ref='owner/sam-power-is-h264-20261101/1',source_sha256=sha,source_hashes=manifest,files={n:'a'*64 for n in names})
    p=tmp_path/'payload';cloud.prepare(root,p,'owner','sam-recover-is-h264',config('is','control'),spec)
    meta,job=cloud.verify(p,root);assert meta['kernel_sources']==[spec['ref']]
    wrong=dict(spec,ref=spec['ref'].replace('owner/','different/'))
    with pytest.raises(ValueError):cloud.validate_spec(wrong,'owner')
    nb=json.loads((p/'notebook.ipynb').read_text())
    if mutation=='code':nb['cells'][0]['source'].append("print('unbound mutation')\n")
    elif mutation=='extra_cell':nb['cells'].append(dict(cell_type='code',source=["print('extra execution')"],metadata={},execution_count=None,outputs=[]))
    else:
        meta[mutation]=['undeclared/source/1'] if mutation.endswith('_sources') else 'other'
        (p/'kernel-metadata.json').write_text(json.dumps(meta))
    (p/'notebook.ipynb').write_text(json.dumps(nb))
    for name in job['payload_hashes']:job['payload_hashes'][name]=hashlib.sha256((p/name).read_bytes()).hexdigest()
    (p/'job.json').write_text(json.dumps(job))
    with pytest.raises(ValueError):cloud.verify(p)

def test_exact_baseline_match_required():
    from reproduce.sam_power_recover import same_codec_baseline
    a={'curves':[dict(arm='codec_only',qp=30,bytes=100,quality=.4)]}
    same_codec_baseline(a,a)
    with pytest.raises(ValueError):same_codec_baseline(a,{'curves':[dict(arm='codec_only',qp=30,bytes=101,quality=.4)]})

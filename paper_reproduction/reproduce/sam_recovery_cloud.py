"""Canonical recovery notebooks with pinned same-owner predecessor outputs."""
import ast,hashlib,json,re
from pathlib import Path
from .kaggle import prepare as native_prepare,source_files,BOOTSTRAP
from .assets import pinned_datasets
from .proxy_cloud import OLD_COMMAND
from .image_study_cloud import STUDY_COMMAND
from .image_proxy_cloud import validate_legacy
from .sam_power_study import config
from .sam_power_recover import PARENT_SOURCE,validate_sources
from .tri_task_data import PUBLIC_PROXY

COMMAND=STUDY_COMMAND.replace('reproduce.image_study','reproduce.sam_power_recover')+"""
    recovery_file=OUT/'recovery_spec.json';recovery_file.write_text(json.dumps(RECOVERY_SPEC))
    owner,slug,version=RECOVERY_SPEC['ref'].split('/')
    roots=[pathlib.Path('/kaggle/input')/slug,pathlib.Path('/kaggle/input/notebooks')/owner/slug,pathlib.Path('/kaggle/input/kernels')/owner/slug]
    parents=set()
    for root in roots:
        for base in (root,root/version):
            parent=base/'paper_results'
            report=parent/'control/study_run.json'
            if report.is_file() and stream_sha(report)==RECOVERY_SPEC['files']['control/study_run.json']:parents.add(parent.resolve())
    if len(parents)!=1:raise ValueError('pinned completed-control parent missing or ambiguous')
    args+=['--parent',str(parents.pop()),'--recovery-spec',str(recovery_file)]
"""

def validate_spec(spec,account):
    if not re.fullmatch(re.escape(account)+r'/sam-power-(?:is|od)-h26[45]-20261101/1',spec.get('ref','')):raise ValueError('same-owner pinned predecessor required')
    if spec.get('source_sha256')!=PARENT_SOURCE:raise ValueError('parent source differs')
    if hashlib.sha256(json.dumps(spec['source_hashes'],sort_keys=True,separators=(',',':')).encode()).hexdigest()!=PARENT_SOURCE:raise ValueError('parent manifest differs')
    expected={'cloud_status.json','control/study_run.json','control/config.json','control/cohort.json','control/analyzer.json','control/validation/metrics.json'}
    if set(spec['files'])!=expected or any(not re.fullmatch('[0-9a-f]{64}',v) for v in spec['files'].values()):raise ValueError('exact parent report hashes required')

def code_for(c):
    prefix=''.join(k+'='+repr(c[k])+'\n' for k in ('TASK','MODE','MANIFEST','SOURCE_SHA','RESUME_DATASET','RESUME_SHA','BUNDLE','STUDY','INPUT_SPEC','RECOVERY_SPEC'))
    return prefix+BOOTSTRAP.replace("if TASK in ('od','ic'):","if TASK in ('od','ic','is'):").replace(OLD_COMMAND,COMMAND)

def prepare(root,directory,account,slug,cfg,spec):
    if cfg!=config(cfg['task'],'control',cfg['codecs'][0]):raise ValueError('recovery recipe differs')
    validate_spec(spec,account);validate_sources(root,spec['source_hashes'],PARENT_SOURCE)
    directory=Path(directory);job=native_prepare(root,directory,account,slug,'od')
    nb=json.loads((directory/'notebook.ipynb').read_text());c={}
    for node in ast.parse(''.join(nb['cells'][0]['source'])).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('TASK','MODE','MANIFEST','SOURCE_SHA','RESUME_DATASET','RESUME_SHA','BUNDLE'):
            c[node.targets[0].id]=ast.literal_eval(node.value)
    c.update(TASK=cfg['task'],STUDY=cfg,INPUT_SPEC=PUBLIC_PROXY,RECOVERY_SPEC=spec)
    nb['cells'][0]['source']=code_for(c).splitlines(True)
    meta=json.loads((directory/'kernel-metadata.json').read_text());meta['kernel_sources']=[spec['ref']]
    meta['dataset_sources']=pinned_datasets('od')+[PUBLIC_PROXY['ref']]
    job.update(task=cfg['task'],config=cfg,input_spec=PUBLIC_PROXY,recovery_spec=spec)
    for name,value in (('notebook.ipynb',nb),('kernel-metadata.json',meta)):
        (directory/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    job['payload_hashes']={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ('notebook.ipynb','kernel-metadata.json')}
    (directory/'job.json').write_text(json.dumps(job,indent=2));verify(directory,root);return job

def verify(directory,root=None):
    directory=Path(directory);job=json.loads((directory/'job.json').read_text());meta=json.loads((directory/'kernel-metadata.json').read_text())
    cfg=job['config'];spec=job['recovery_spec'];validate_spec(spec,job['handle'].split('/')[0])
    if cfg!=config(cfg['task'],'control',cfg['codecs'][0]) or job['input_spec']!=PUBLIC_PROXY:raise ValueError('recipe/input differs')
    if spec['ref'].split('/')[1]!='sam-power-'+cfg['task']+'-'+cfg['codecs'][0]+'-20261101':raise ValueError('parent task/codec differs')
    if set(job['payload_hashes'])!={'notebook.ipynb','kernel-metadata.json'}:raise ValueError('both payload hashes required')
    for name,sha in job['payload_hashes'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=sha:raise ValueError('payload checksum differs')
    notebook=json.loads((directory/'notebook.ipynb').read_text())
    if (notebook.get('nbformat')!=4 or notebook.get('nbformat_minor')!=5 or len(notebook.get('cells',[]))!=1
        or notebook['cells'][0].get('cell_type')!='code'):raise ValueError('exactly one canonical code cell required')
    expected_meta=dict(id=job['handle'],title=job['handle'].split('/')[1],code_file='notebook.ipynb',
        language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,
        dataset_sources=pinned_datasets('od')+[PUBLIC_PROXY['ref']],kernel_sources=[spec['ref']],competition_sources=[],model_sources=[])
    if meta!=expected_meta:raise ValueError('execution metadata or undeclared inputs differ')
    code=''.join(notebook['cells'][0]['source']);c={}
    names={'TASK','MODE','MANIFEST','SOURCE_SHA','RESUME_DATASET','RESUME_SHA','BUNDLE','STUDY','INPUT_SPEC','RECOVERY_SPEC'}
    for node in ast.parse(code).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in names:c[node.targets[0].id]=ast.literal_eval(node.value)
    if (set(c)!=names or c['TASK']!=cfg['task'] or c['MODE']!='full' or c['STUDY']!=cfg or c['INPUT_SPEC']!=PUBLIC_PROXY or c['RECOVERY_SPEC']!=spec
        or c['RESUME_DATASET'] is not None or c['RESUME_SHA'] is not None or c['SOURCE_SHA']!=job['source_sha256'] or c['MANIFEST']!=job['source_hashes']
        or meta['id']!=job['handle'] or meta['is_private'] is not True or meta['enable_gpu'] is not True or meta['enable_internet'] is not True
        or meta['dataset_sources']!=pinned_datasets('od')+[PUBLIC_PROXY['ref']] or meta['kernel_sources']!=[spec['ref']]
        or code!=code_for(c)):raise ValueError('recovery cloud binding differs')
    validate_legacy(c['BUNDLE'],c['MANIFEST'],c['SOURCE_SHA'])
    if root is not None:
        if source_files(root)!=job['source_hashes']:raise ValueError('source changed since packaging')
        validate_sources(root,spec['source_hashes'],PARENT_SOURCE)
    compile(code,'recovery-bootstrap','exec');return meta,job

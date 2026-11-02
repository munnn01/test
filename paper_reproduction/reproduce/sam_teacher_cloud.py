"""Immutable private notebooks for the unchanged-pipeline teacher/gain trial."""
import ast,hashlib,json,re
from pathlib import Path
from .kaggle import prepare as native_prepare,BOOTSTRAP,source_files
from .proxy_cloud import OLD_COMMAND
from .image_study_cloud import STUDY_COMMAND
from .image_proxy_cloud import validate_legacy
from .sam_teacher_recipe import config
from .sam_power_recover import PARENT_SOURCE,validate_sources
from .tri_task_data import PUBLIC_PROXY
from .assets import pinned_datasets

MASK_SOURCE='9fa98ecc542dc0bc193c38cfa9dcb0b47f4da5cd44d24916815f3d6d964fa69c'
CONSTANTS=('TASK','MODE','MANIFEST','SOURCE_SHA','RESUME_DATASET','RESUME_SHA','BUNDLE','STUDY','INPUT_SPEC','TRAIN_INPUT')
COMMAND=STUDY_COMMAND.replace('reproduce.image_study','reproduce.sam_teacher_study')+"""
    training_file=OUT/'training_inputs.json';training_file.write_text(json.dumps(TRAIN_INPUT))
    def owned_input(ref,member,sha):
        owner,slug,version=ref.split('/')
        roots=[pathlib.Path('/kaggle/input')/slug,pathlib.Path('/kaggle/input/notebooks')/owner/slug,pathlib.Path('/kaggle/input/kernels')/owner/slug]
        matches=set()
        for root in roots:
            for base in (root,root/version):
                parent=base/'paper_results';path=parent/member
                if path.is_file() and stream_sha(path)==sha:matches.add(parent.resolve())
        if len(matches)!=1:raise ValueError('exact same-owner parent input missing or ambiguous')
        return str(matches.pop())
    parent=owned_input(TRAIN_INPUT['ref'],'control/study_run.json',TRAIN_INPUT['files']['control/study_run.json'])
    masks=owned_input(TRAIN_INPUT['mask_ref'],'foreground/sam_cache.json',TRAIN_INPUT['mask_files']['foreground/sam_cache.json'])
    args+=['--parent',parent,'--mask-parent',masks,'--input-spec',str(training_file)]
"""

def validate_input(spec,account,cfg):
    key=cfg['task']+'-'+cfg['codecs'][0]
    if spec.get('ref')!=account+'/sam-power-'+key+'-20261101/1' or spec.get('parent_transport_version')!=2:raise ValueError('exact same-owner parent artifact required')
    if not re.fullmatch(re.escape(account+'/sam-recover-'+key+'-20261101/')+r'[1-9][0-9]*',spec.get('mask_ref','')):raise ValueError('same-owner pinned SAM source required')
    if spec.get('source_sha256')!=PARENT_SOURCE or spec.get('mask_source_sha256')!=MASK_SOURCE:raise ValueError('known parent source required')
    manifest=spec.get('source_hashes',{})
    if hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=PARENT_SOURCE:raise ValueError('parent source manifest differs')
    groups={
        'files':{'cloud_status.json','control/study_run.json','control/config.json','control/cohort.json','control/analyzer.json','control/validation/metrics.json'},
        'checkpoint_files':{'shared_warmup.pt','control/last.pt'},
        'mask_files':{'cloud_status.json','foreground/study_run.json','foreground/sam_cache.json'}}
    for name,keys in groups.items():
        values=spec.get(name,{})
        if set(values)!=keys or any(not re.fullmatch('[0-9a-f]{64}',v) for v in values.values()):raise ValueError('exact parent member checksums required')

def notebook(code):
    return dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},
        cells=[dict(cell_type='code',id='teacher-balanced-study',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))])

def code_for(c):
    return ''.join(k+'='+repr(c[k])+'\n' for k in CONSTANTS)+BOOTSTRAP.replace("if TASK in ('od','ic'):","if TASK in ('od','ic','is'):").replace(OLD_COMMAND,COMMAND)

def metadata(handle,spec):
    return dict(id=handle,title=handle.split('/')[1],code_file='notebook.ipynb',language='python',kernel_type='notebook',
        is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=pinned_datasets('od')+[PUBLIC_PROXY['ref']],
        kernel_sources=[spec['ref'].rsplit('/',1)[0],spec['mask_ref'].rsplit('/',1)[0]],competition_sources=[],model_sources=[])

def prepare(root,directory,account,slug,cfg,spec):
    if cfg!=config(cfg['task'],cfg['codecs'][0]):raise ValueError('teacher recipe differs')
    validate_input(spec,account,cfg);validate_sources(root,spec['source_hashes'],PARENT_SOURCE)
    directory=Path(directory);job=native_prepare(root,directory,account,slug,'od')
    native=''.join(json.loads((directory/'notebook.ipynb').read_text())['cells'][0]['source']);c={}
    for node in ast.parse(native).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in CONSTANTS:
            c[node.targets[0].id]=ast.literal_eval(node.value)
    c.update(TASK=cfg['task'],STUDY=cfg,INPUT_SPEC=PUBLIC_PROXY,TRAIN_INPUT=spec)
    code=code_for(c);nb=notebook(code);meta=metadata(job['handle'],spec)
    job.update(task=cfg['task'],config=cfg,input_spec=PUBLIC_PROXY,training_inputs=spec)
    for name,value in (('notebook.ipynb',nb),('kernel-metadata.json',meta)):(directory/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    job['payload_hashes']={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ('notebook.ipynb','kernel-metadata.json')}
    (directory/'job.json').write_text(json.dumps(job,indent=2));verify(directory,root);return job

def verify(directory,root=None):
    directory=Path(directory);job=json.loads((directory/'job.json').read_text());meta=json.loads((directory/'kernel-metadata.json').read_text())
    cfg=job['config'];spec=job['training_inputs'];validate_input(spec,job['handle'].split('/')[0],cfg)
    if cfg!=config(cfg['task'],cfg['codecs'][0]) or job['input_spec']!=PUBLIC_PROXY:raise ValueError('teacher recipe/public input differs')
    if set(job['payload_hashes'])!={'notebook.ipynb','kernel-metadata.json'}:raise ValueError('both payload checksums required')
    for name,sha in job['payload_hashes'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=sha:raise ValueError('payload checksum differs')
    nb=json.loads((directory/'notebook.ipynb').read_text())
    if len(nb.get('cells',[]))!=1 or nb['cells'][0].get('cell_type')!='code':raise ValueError('single code cell required')
    code=''.join(nb['cells'][0]['source']);c={}
    for node in ast.parse(code).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in CONSTANTS:
            if node.targets[0].id in c:raise ValueError('duplicate source-binding constant')
            c[node.targets[0].id]=ast.literal_eval(node.value)
    if (set(c)!=set(CONSTANTS) or c['TASK']!=cfg['task'] or c['MODE']!='full' or c['STUDY']!=cfg
        or c['INPUT_SPEC']!=PUBLIC_PROXY or c['TRAIN_INPUT']!=spec or c['RESUME_DATASET'] is not None or c['RESUME_SHA'] is not None
        or c['SOURCE_SHA']!=job['source_sha256'] or c['MANIFEST']!=job['source_hashes'] or code!=code_for(c)
        or nb!=notebook(code) or meta!=metadata(job['handle'],spec)):
        raise ValueError('canonical teacher cloud binding differs')
    validate_legacy(c['BUNDLE'],c['MANIFEST'],c['SOURCE_SHA'])
    if root is not None:
        if source_files(root)!=job['source_hashes']:raise ValueError('source changed since packaging')
        validate_sources(root,spec['source_hashes'],PARENT_SOURCE)
    compile(code,'verified-teacher-bootstrap','exec');return meta,job

def verify_remote(expected_meta,expected_nb,actual_meta,actual_nb):
    """Allow Kaggle's filename/source serialization, reject executable changes."""
    for key in ('id','language','kernel_type','is_private','enable_gpu','enable_internet'):
        if actual_meta.get(key)!=expected_meta[key]:raise ValueError('remote execution metadata differs: '+key)
    if actual_meta.get('enable_tpu',False):raise ValueError('undeclared TPU')
    if actual_meta.get('code_file') not in ('notebook.ipynb',expected_meta['id'].split('/')[1]+'.ipynb'):
        raise ValueError('remote notebook filename differs')
    for key in ('dataset_sources','kernel_sources','competition_sources','model_sources'):
        wanted=expected_meta[key]
        if key=='dataset_sources':wanted=[r.rsplit('/',1)[0] for r in wanted]
        got=actual_meta.get(key,[])
        if not isinstance(got,list) or len(got)!=len(wanted) or set(got)!=set(wanted):raise ValueError('remote attachments differ: '+key)
    def normalize(value):
        value=json.loads(json.dumps(value))
        for cell in value.get('cells',[]):
            content=cell.get('source')
            if isinstance(content,list) and all(isinstance(v,str) for v in content):cell['source']=''.join(content)
            elif not isinstance(content,str):raise ValueError('invalid remote cell source')
        return value
    if normalize(actual_nb)!=normalize(expected_nb):raise ValueError('remote whole notebook differs')
    return True


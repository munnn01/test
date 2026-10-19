import hashlib,json,os,pathlib,shutil,subprocess,sys
PIN='db6d8b5b162d6e0303fc9a94422c32648d34c98d'
ANCHORS={'merge_od.py': 'eac5a9f68ee0b0a70399a9ac13aa7bb72ef284be575bebacdfaf291795b5b0ce', 'od-inputs750.tgz': 'fa772dc6092f6474b7c6b65398b8cfcced764bcd5bb07dc0961f7ece3daf9652', 'test_a8_od_merge.py': '7eb366b0405dd7d8097b89df8cbe93c257ab502aed26c21e0fd2588027430879'}
ROOT=pathlib.Path('/kaggle/temp/adaptive_preprocessing');ROOT.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['git','clone','-q','https://github.com/munnn01/test.git',str(ROOT)],check=True)
subprocess.run(['git','-C',str(ROOT),'checkout','-q',PIN],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps','numpy==2.1.3','scipy==1.16.3','pycocotools==2.0.11'],check=True)
os.environ.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
matches=list(pathlib.Path('/kaggle/input').rglob('od-inputs750.tgz'))
if len(matches)!=1:raise ValueError('one declared OD input snapshot required')
data=matches[0].parent
for name,digest in ANCHORS.items():
    h=hashlib.sha256()
    with (data/name).open('rb') as stream:
        while block:=stream.read(4*1024*1024):h.update(block)
    if h.hexdigest()!=digest:raise ValueError('immutable input/script hash drift: '+name)
execution=pathlib.Path('/kaggle/temp/od-merge-execution750');execution.mkdir()
for name in ('merge_od.py','test_a8_od_merge.py'):shutil.copyfile(data/name,execution/name)
testlog=pathlib.Path('/kaggle/working/od-merge-tests.log')
with testlog.open('w') as log:
    subprocess.run([sys.executable,str(execution/'test_a8_od_merge.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
print(testlog.read_text(),flush=True)
annotations=list(pathlib.Path('/kaggle/input').rglob('instances_val2017.json'))
if len(annotations)!=1:raise ValueError('one original COCO validation annotation required')
subprocess.run([sys.executable,'-u',str(execution/'merge_od.py'),'--archive',str(data/'od-inputs750.tgz'),'--archive-sha256',ANCHORS['od-inputs750.tgz'],'--annotation',str(annotations[0]),'--out','/kaggle/working/od-merged750'],cwd=ROOT,check=True)

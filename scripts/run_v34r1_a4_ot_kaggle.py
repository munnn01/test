"""Kaggle CPU: authenticated A3 prediction reuse, frozen A4 replay, no new inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts')); sys.path.insert(0,str(ROOT))
from run_v34r1_kaggle import run
from adaptive_vcm.v34r1_a4.cloud_record import record_cloud_run


def main():
    p=argparse.ArgumentParser(); p.add_argument('--commit',required=True); args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'): raise ValueError('Kaggle only')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=args.commit: raise ValueError('immutable commit required')
    work=Path('/kaggle/working'); base=work/'a4-ot-calibration'
    status=dict(complete=False,commit=args.commit,test_scored=False,new_model_inferences=0,new_encodes=0)
    with record_cloud_run(base,work/'v34r1-a4-ot-calibration.tgz',status,logs=[work/'a4-block-tests.log',work/'a4-ot-analysis.log']):
        run([sys.executable,'-m','pip','install','-q','--no-deps','numpy==2.1.3','scipy==1.16.3',
            'opencv-python==4.12.0.88','opencv-python-headless==4.12.0.88','supervision==0.27.0',
            'pycocotools==2.0.11','tabulate==0.9.0'])
        os.environ.update(PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg')
        trackeval=ROOT/'data/TrackEval'; run(['git','clone','-q','https://github.com/JonathonLuiten/TrackEval.git',trackeval])
        run(['git','-C',trackeval,'checkout','-q','12c8791b303e0a0b50f753af204249e622d0281a']); os.environ['V34_TRACKEVAL']=str(trackeval)
        testlog=work/'a4-block-tests.log'; run([sys.executable,'-m','unittest','tests.test_v34r1_a3_hota','tests.test_v34r1_a3_tracking',
            'tests.test_v34r1_a3_core','tests.test_v34r1_a4_rd_fit','tests.test_v34r1_a4_marginal_ap',
            'tests.test_v34r1_a4_cloud_record','-v'],log=testlog)
        cfg=json.loads((ROOT/'configs/v34r1-a4-ot-rd-fit.json').read_text()); old=json.loads((ROOT/'configs/v34r1-a3-ot-probe.json').read_text())
        archives=list(Path('/kaggle/input').rglob('v34r1-a3-ot-dev.tgz'))
        if len(archives)!=1 or hashlib.sha256(archives[0].read_bytes()).hexdigest()!=cfg['parent_archive_sha256']: raise ValueError('exact complete parent archive required')
        cache=ROOT/'data/a4-parent'; cache.mkdir()
        with tarfile.open(archives[0]) as tar:
            for member in tar:
                name=PurePosixPath(member.name); target=(cache/member.name).resolve()
                if name.is_absolute() or '..' in name.parts or '\\' in member.name or ':' in member.name or not target.is_relative_to(cache): raise ValueError('unsafe cache member')
                if member.isdir(): target.mkdir(parents=True,exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with tar.extractfile(member) as src,target.open('wb') as dst: shutil.copyfileobj(src,dst)
                else: raise ValueError('special cache member')
        zips=list(Path('/kaggle/input').rglob('mot17-source-cache.zip'))
        if len(zips)!=1 or hashlib.sha256(zips[0].read_bytes()).hexdigest()!=old['source_cache_sha256']: raise ValueError('exact frozen source metadata cache required')
        sources=ROOT/'data/a4-mot17-metadata'; sources.mkdir()
        with zipfile.ZipFile(zips[0]) as zipped:
            raw=zipped.read('source-manifest.json')
            if hashlib.sha256(raw).hexdigest()!=old['source_manifest_sha256']: raise ValueError('source manifest drift')
            for seq in json.loads(raw)['sequences']:
                if seq['split']=='reserved_test': continue
                for member in ('seqinfo.ini','gt/gt.txt'):
                    relative=seq['name']+'/'+member; data=zipped.read(relative)
                    if hashlib.sha256(data).hexdigest()!=seq['files'][member]: raise ValueError('original sequence metadata drift')
                    target=(sources/relative).resolve()
                    if not target.is_relative_to(sources): raise ValueError('source path escape')
                    target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
        run([sys.executable,'-u','-m','adaptive_vcm.v34r1_a4.ot_calibration','--cache',str(cache/'a3-ot-dev'),'--sources',str(sources),'--out',str(base)],log=work/'a4-ot-analysis.log')


if __name__=='__main__': main()

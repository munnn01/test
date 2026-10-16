"""Private pinned Kaggle CPU replay, zero new real codec/detector work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from run_v34r1_kaggle import run
from adaptive_vcm.v34r1_a4.cloud_record import record_cloud_run,require_sha256,kernel_artifact,unpack_archive


def main():
    p=argparse.ArgumentParser(); p.add_argument('--commit',required=True); args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'): raise ValueError('Kaggle only')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=args.commit: raise ValueError('immutable revision required')
    work=Path('/kaggle/working'); base=work/'a5-ot-soft'; blocklog=work/'a5-ot-block-tests.log'; analysis=work/'a5-ot-analysis.log'
    with record_cloud_run(base,work/'v34r1-a5-ot-soft.tgz',dict(commit=args.commit,test_scored=False,new_model_inferences=0,new_encodes=0),logs=[blocklog,analysis]):
        run([sys.executable,'-m','pip','install','-q','--no-deps','numpy==2.1.3','scipy==1.16.3',
            'opencv-python==4.12.0.88','opencv-python-headless==4.12.0.88','supervision==0.27.0','pycocotools==2.0.11','tabulate==0.9.0'])
        os.environ.update(PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg')
        trackeval=ROOT/'data/TrackEval'; trackeval.parent.mkdir(parents=True,exist_ok=True)
        run(['git','clone','-q','https://github.com/JonathonLuiten/TrackEval.git',trackeval])
        run(['git','-C',trackeval,'checkout','-q','12c8791b303e0a0b50f753af204249e622d0281a']); os.environ['V34_TRACKEVAL']=str(trackeval)
        run([sys.executable,'-m','unittest','tests.test_v34r1_a5_ot_choice','tests.test_v34r1_a3_hota',
            'tests.test_v34r1_a3_tracking','tests.test_v34r1_a3_core','tests.test_v34r1_a4_rd_fit','tests.test_v34r1_a4_cloud_record','-v'],log=blocklog)
        cfg=json.loads((ROOT/'configs/v34r1-a5-ot-soft.json').read_text()); old=json.loads((ROOT/'configs/v34r1-a3-ot-probe.json').read_text())
        archive=kernel_artifact('/kaggle/input','v34r1-a3-ot-dev.tgz'); require_sha256(archive.read_bytes(),cfg['parent_archive_sha256'])
        cache=ROOT/'data/a5-parent'; unpack_archive(archive,cache)
        zipped_path=kernel_artifact('/kaggle/input','mot17-source-cache.zip'); require_sha256(zipped_path.read_bytes(),old['source_cache_sha256'])
        sources=ROOT/'data/a5-mot17-metadata'; sources.mkdir()
        with zipfile.ZipFile(zipped_path) as zipped:
            raw=zipped.read('source-manifest.json'); require_sha256(raw,old['source_manifest_sha256'])
            for seq in json.loads(raw)['sequences']:
                if seq['split']=='reserved_test': continue
                for member in ('seqinfo.ini','gt/gt.txt'):
                    relative=seq['name']+'/'+member; data=zipped.read(relative); require_sha256(data,seq['files'][member])
                    target=(sources/relative).resolve()
                    if not target.is_relative_to(sources): raise ValueError('original metadata path escape')
                    target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
        run([sys.executable,'-u','-m','adaptive_vcm.v34r1_a5.ot_replay','--cache',str(cache/'a3-ot-dev'),
            '--sources',str(sources),'--out',str(base)],log=analysis)


if __name__=='__main__': main()

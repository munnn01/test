"""Pinned CPU-only risk-critic training from an exact completed cloud label archive."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from adaptive_vcm.v34r1_a4.cloud_record import record_cloud_run,kernel_artifact,unpack_archive
from run_v34r1_kaggle import run
from prepare_v34r1_cloud_data import mounted


def main():
    p=argparse.ArgumentParser(); p.add_argument('--commit',required=True); p.add_argument('--labels-sha256',required=True); args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'): raise ValueError('Kaggle-only real training')
    if not re.fullmatch('[a-f0-9]{64}',args.labels_sha256): raise ValueError('exact parent archive SHA required')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=args.commit: raise ValueError('immutable code revision required')
    work=Path('/kaggle/working'); base=work/'a4-od-training'; log=work/'a4-training-block-tests.log'; analysis=work/'a4-training-analysis.log'
    with record_cloud_run(base,work/'v34r1-a4-od-training.tgz',dict(commit=args.commit,label_archive_sha256=args.labels_sha256,
        test_scored=False,fresh_dev64_scored=False,new_encodes=0,new_task_model_inferences=0,mode='critic_training'),logs=[log,analysis]):
        run([sys.executable,'-m','pip','install','-q','--no-deps','numpy==2.1.3','scipy==1.16.3','pycocotools==2.0.11',
             'scikit-learn==1.7.2','joblib==1.5.2','threadpoolctl==3.6.0'])
        os.environ.update(PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        run([sys.executable,'-m','unittest','tests.test_v34r1_a4_risk','tests.test_v34r1_a4_marginal_ap',
            'tests.test_v34r1_a4_cloud_record','tests.test_v34r1_a4_rd_fit','tests.test_v34r1_a3_core','-v'],log=log)
        source=ROOT/'results/v34r1_a3/od-dev'; run([sys.executable,str(ROOT/'scripts/verify_v34r1_a3.py'),'--root',str(source)])
        print('A4_LOCATE_NOTEBOOK_INPUT',flush=True)
        archive=kernel_artifact('/kaggle/input','v34r1-a4-od-labels.tgz')
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=args.labels_sha256: raise ValueError('exact completed label archive required')
        cache=ROOT/'data/a4-labels'; unpack_archive(archive,cache)
        annotation=mounted('annotations/instances_val2017.json')
        if annotation is None: raise FileNotFoundError('official frozen COCO annotations missing')
        run([sys.executable,'-u','-m','adaptive_vcm.v34r1_a4.od_train','--labels',str(cache/'a4-od-labels'),
             '--annotation',str(annotation),'--out',str(base)],log=analysis)


if __name__=='__main__': main()

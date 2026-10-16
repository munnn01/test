"""Cloud CPU label-generation job; real GT scoring never runs on the laptop."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from adaptive_vcm.v34r1_a4.cloud_record import record_cloud_run
from run_v34r1_kaggle import run
from prepare_v34r1_cloud_data import mounted


def main():
    p=argparse.ArgumentParser(); p.add_argument('--commit',required=True); args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'): raise ValueError('Kaggle cloud only')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=args.commit: raise ValueError('immutable source revision required')
    work=Path('/kaggle/working'); base=work/'a4-od-labels'; log=work/'a4-label-block-tests.log'; analysis=work/'a4-label-analysis.log'
    with record_cloud_run(base,work/'v34r1-a4-od-labels.tgz',dict(commit=args.commit,test_scored=False,new_encodes=0,new_model_inferences=0),logs=[log,analysis]):
        run([sys.executable,'-m','pip','install','-q','--no-deps','numpy==2.1.3','scipy==1.16.3','pycocotools==2.0.11'])
        os.environ.update(PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        run([sys.executable,'-m','unittest','tests.test_v34r1_a4_marginal_ap','tests.test_v34r1_a4_cloud_record','tests.test_v34r1_a3_core','-v'],log=log)
        source=ROOT/'results/v34r1_a3/od-dev'
        run([sys.executable,str(ROOT/'scripts/verify_v34r1_a3.py'),'--root',str(source)])
        annotation=mounted('annotations/instances_val2017.json')
        if annotation is None: raise FileNotFoundError('official frozen COCO annotations missing')
        run([sys.executable,'-u','-m','adaptive_vcm.v34r1_a4.od_labels','--source',str(source),'--annotation',str(annotation),'--out',str(base)],log=analysis)


if __name__=='__main__': main()

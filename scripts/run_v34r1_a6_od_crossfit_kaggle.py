"""Pinned private Kaggle CPU: cohort-local labels, OOF selection, final lock."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from adaptive_vcm.v34r1_a4.cloud_record import record_cloud_run
from run_v34r1_kaggle import run
from prepare_v34r1_cloud_data import mounted


def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'):raise ValueError('Kaggle only')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=args.commit:raise ValueError('immutable revision required')
    work=Path('/kaggle/working');base=work/'a6-od-crossfit';tests=work/'a6-od-tests.log';analysis=work/'a6-od-analysis.log'
    with record_cloud_run(base,work/'v34r1-a6-od-crossfit.tgz',dict(commit=args.commit,pilot_scored=False,test_scored=False,new_task_model_inferences=0,new_encodes=0),logs=[tests,analysis]):
        base.mkdir();run([sys.executable,'-m','pip','install','-q','--no-deps','numpy==2.1.3','scipy==1.16.3','pycocotools==2.0.11',
            'scikit-learn==1.7.2','joblib==1.5.2','threadpoolctl==3.6.0'])
        os.environ.update(PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg')
        run([sys.executable,'-m','unittest','tests.test_v34r1_a6_contracts','tests.test_v34r1_a4_risk','tests.test_v34r1_a4_marginal_ap',
            'tests.test_v34r1_a4_rd_fit','tests.test_v34r1_a4_cloud_record','tests.test_v34r1_a3_core','-v'],log=tests)
        annotation=mounted('annotations/instances_val2017.json')
        if annotation is None:raise FileNotFoundError('official COCO metadata missing')
        run([sys.executable,'-u','-m','adaptive_vcm.v34r1_a6.od_crossfit','--out',str(base/'training'),'--annotation',str(annotation)],log=analysis)


if __name__=='__main__':main()

"""Source-only CPU freeze before any fixed1000/1000/10 model evaluation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from adaptive_vcm.v34r1_a4.cloud_record import record_cloud_run
from run_v34r1_kaggle import run
from prepare_v34r1_a8_sources import freeze_all


def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);a=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=a.commit:raise ValueError('pinned Kaggle checkout required')
    base=Path('/kaggle/working/a8-source-freeze');log=base.parent/'a8-source-tests.log'
    with record_cloud_run(base,base.parent/'v34r1-a8-source-freeze.tgz',dict(commit=a.commit,source_only=True,model_inferences=0,codec_trials=0,test_scored=False),logs=[log]):
        run([sys.executable,'-m','pip','install','-q','--no-deps','numpy==2.1.3','opencv-python==4.12.0.88','opencv-python-headless==4.12.0.88'])
        os.environ.update(PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        run([sys.executable,'-m','unittest','tests.test_v34r1_a8_contracts','tests.test_v34r1_a6_ar_source','tests.test_v34r1_a4_cloud_record','-v'],log=log)
        freeze_all(base)


if __name__=='__main__':main()

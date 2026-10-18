"""Private Kaggle CPU shared-QP H264 probe with exact A6 replay controls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from adaptive_vcm.v34r1_a4.cloud_record import record_cloud_run,require_sha256,unpack_archive
from run_v34r1_kaggle import run
from prepare_v34r1_a6_mot17 import restore


def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--bundle-sha256',required=True);args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'):raise ValueError('Kaggle only')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=args.commit:raise ValueError('immutable revision required')
    work=Path('/kaggle/working');base=work/'a7-ot-shared-qp';blocklog=work/'a7-ot-replay-tests.log';analysis=work/'a7-ot-replay-analysis.log'
    with record_cloud_run(base,work/'v34r1-a7-ot-shared-qp.tgz',dict(commit=args.commit,bundle_sha256=args.bundle_sha256,test_scored=False,new_model_inferences=0,new_encodes=0),logs=[blocklog,analysis]):
        run([sys.executable,'-m','pip','install','-q','--no-deps','numpy==2.1.3','scipy==1.16.3',
            'opencv-python==4.12.0.88','opencv-python-headless==4.12.0.88','supervision==0.27.0','pycocotools==2.0.11','tabulate==0.9.0'])
        os.environ.update(PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg')
        trackeval=ROOT/'data/TrackEval';trackeval.parent.mkdir(parents=True,exist_ok=True)
        run(['git','clone','-q','https://github.com/JonathonLuiten/TrackEval.git',trackeval])
        run(['git','-C',trackeval,'checkout','-q','12c8791b303e0a0b50f753af204249e622d0281a']);os.environ['V34_TRACKEVAL']=str(trackeval)
        run([sys.executable,'-m','unittest','tests.test_v34r1_a7_shared','tests.test_v34r1_a7_goal','tests.test_v34r1_a6_contracts','tests.test_v34r1_a6_ot_grid','tests.test_v34r1_a5_ot_choice',
            'tests.test_v34r1_a3_hota','tests.test_v34r1_a3_tracking','tests.test_v34r1_a3_core','tests.test_v34r1_a4_rd_fit','tests.test_v34r1_a4_cloud_record','-v'],log=blocklog)
        candidates=[p/'v34r1-a6-ot-cache-bundle.tgz' for p in (Path('/kaggle/input/v34r1-a6-ot-dev-cache'),
            Path('/kaggle/input/datasets/nguyenhoanglan1232/v34r1-a6-ot-dev-cache'))]
        found=[p for p in candidates if p.is_file()]
        if len(found)!=1:raise ValueError('one declared private measured-cache dataset required')
        require_sha256(found[0].read_bytes(),args.bundle_sha256);cache=ROOT/'data/v34r1_a7/cache-merged';unpack_archive(found[0],cache)
        cfg=json.loads((ROOT/'configs/v34r1-a6-ot21.json').read_text());sources=ROOT/'data/v34r1_a7/metadata-only'
        restore(sources,[r['name'] for r in cfg['pilot_sequences']],metadata_only=True)
        run([sys.executable,'-u','-m','adaptive_vcm.v34r1_a7.ot_replay','--cache',str(cache/'a6-ot-cache-bundle'),
            '--sources',str(sources),'--out',str(base)],log=analysis)


if __name__=='__main__':main()

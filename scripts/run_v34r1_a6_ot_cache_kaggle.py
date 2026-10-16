"""Pinned private two-T4, one-video OT full-bank source-only cache."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from adaptive_vcm.v34r1_a4.cloud_record import record_cloud_run
from adaptive_vcm.v34r1_a6.evidence import publication
from run_v34r1_kaggle import run
from run_v34r1_a3_kaggle import last_progress
from prepare_v34r1_a6_mot17 import restore


def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--sequence',required=True);args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'):raise ValueError('Kaggle only')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=args.commit:raise ValueError('immutable revision required')
    work=Path('/kaggle/working');tag=args.sequence.lower();base=work/('a6-ot-'+tag);workers=[];logs=[]
    status=dict(commit=args.commit,sequence=args.sequence,test_scored=False,gt_scored=False,gt_extracted=False)
    with record_cloud_run(base,work/('v34r1-a6-ot-'+tag+'.tgz'),status):
        base.mkdir()
        try:
            protocol=json.loads((ROOT/'configs/v34r1-a6-ot21.json').read_text())
            if args.sequence not in {r['name'] for r in protocol['pilot_sequences']}:raise ValueError('declared DEV video required')
            publication(ROOT/'results/v34r1_a3/ot-dev',protocol['parent_a3_publication_sha256'])
            run([sys.executable,'-m','pip','install','-q','--no-deps','numpy==2.1.3','scipy==1.16.3',
                'opencv-python==4.12.0.88','opencv-python-headless==4.12.0.88','supervision==0.27.0','pycocotools==2.0.11',
                'imageio-ffmpeg==0.6.0','tabulate==0.9.0'])
            import torch,torchvision,imageio_ffmpeg
            if not torch.cuda.is_available() or torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in (0,1)):raise ValueError('two T4 GPUs required')
            if str(torch.__version__)!='2.11.0+cu128' or str(torchvision.__version__)!='0.26.0+cu128':raise ValueError('pinned torch image drift')
            cfg=json.loads((ROOT/'configs/v34r1-kaggle.json').read_text());ffmpeg=Path(imageio_ffmpeg.get_ffmpeg_exe());sha=hashlib.sha256(ffmpeg.read_bytes()).hexdigest()
            if sha!=cfg['expected_ffmpeg_sha256']:raise ValueError('pinned FFmpeg drift')
            codec=ROOT/cfg['ffmpeg_executable'];codec.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ffmpeg,codec);codec.chmod(0o755)
            os.environ.update(TORCH_HOME=str(ROOT/'data/torch'),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
                MPLBACKEND='Agg',PYTHONPATH=str(ROOT),PATH=str(codec.parent)+os.pathsep+os.environ['PATH'])
            run([sys.executable,'-m','unittest','tests.test_v34r1_a6_contracts','tests.test_v34r1_a6_ot_grid','tests.test_v34r1_blocks',
                'tests.test_v34r1_a3_core','-v'],log=base/'block-tests.log')
            source=ROOT/'data/v34r1_a6'/args.sequence;manifest=restore(source,[args.sequence]);(base/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            environment=dict(python=sys.version,ffmpeg_sha256=sha,packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','scipy','supervision')})
            (base/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
            from torchvision.models.detection import fasterrcnn_resnet50_fpn,FasterRCNN_ResNet50_FPN_Weights
            downloaded=fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1);del downloaded
            for i,c in enumerate(protocol['codecs']):
                log=open(base/f'worker{i}.log','w',encoding='utf-8');logs.append(log)
                workers.append(subprocess.Popen([sys.executable,'-u','-m','adaptive_vcm.v34r1_a6.ot_cache','--root',str(source),'--sequence',args.sequence,
                    '--codec',c,'--out',str(base/f'worker{i}')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i)),stdout=log,stderr=subprocess.STDOUT))
            while any(p.poll() is None for p in workers):
                if any(p.poll() not in (None,0) for p in workers):raise RuntimeError('OT GPU worker failed')
                print('A6_OT_PROGRESS',args.sequence,json.dumps({str(i):last_progress(base/f'worker{i}.log') for i in (0,1)}),flush=True);time.sleep(30)
            if any(p.returncode for p in workers):raise RuntimeError('OT GPU worker failed')
            from adaptive_vcm.v34r1_a6.contracts import audit_ot21
            import gzip
            rows=[r for i in (0,1) for r in json.loads(gzip.decompress((base/f'worker{i}/rows.json.gz').read_bytes()))]
            split=next(r['split'] for r in protocol['pilot_sequences'] if r['name']==args.sequence)
            audit_ot21(rows,{args.sequence:split})
            (base/'cache-complete.json').write_text(json.dumps(dict(rows=len(rows),physical_videos=1,unique_source_frames=64,full21=True,
                qps=protocol['qps'],codecs=protocol['codecs'],source_task_frames=128,source_frames_duplicated_across_codec_workers=True,gt_extracted=False),indent=2)+'\n')
        finally:
            for process in workers:
                if process.poll() is None:process.terminate()
            for process in workers:
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            for log in logs:log.close()
            for i in range(len(workers)):
                path=base/f'worker{i}.log'
                if path.exists():print(path.read_text(errors='replace')[-4000:],flush=True)


if __name__=='__main__':main()

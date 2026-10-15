"""Cloud-only reproducible setup, block verification and frozen four-QP TEST."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]


def run(argv,*,log=None):
    print('RUN', ' '.join(map(str,argv)),flush=True)
    output=open(log,'w',encoding='utf-8') if log else None
    try:
        process=subprocess.Popen(list(map(str,argv)),cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='',flush=True)
            if output: output.write(line); output.flush()
        if process.wait(): raise RuntimeError('subprocess failed: '+str(argv[0:4]))
    finally:
        if output: output.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True,choices=['ar','od','ot'])
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'): raise ValueError('Kaggle cloud root required; no laptop execution')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=args.commit:
        raise ValueError('immutable checkout mismatch')
    work=Path('/kaggle/working'); out=work/(args.task+'-test'); out.mkdir(exist_ok=True)
    started=time.time(); status={'task':args.task,'complete':False,'commit':args.commit}
    try:
        run([sys.executable,'-m','pip','install','-q','--no-deps','numpy==2.1.3','scipy==1.16.3',
             'opencv-python==4.12.0.88','opencv-python-headless==4.12.0.88','supervision==0.27.0',
             'pycocotools==2.0.11','imageio-ffmpeg==0.6.0','tabulate==0.9.0','pytest==9.1.1'])
        import torch,torchvision,imageio_ffmpeg
        if not torch.cuda.is_available(): raise RuntimeError('Kaggle GPU required; CPU fallback prohibited')
        if torch.cuda.device_count()!=2 or 'T4' not in torch.cuda.get_device_name(0):
            raise ValueError('pinned Kaggle T4x2 accelerator drift')
        if str(torch.__version__)!='2.11.0+cu128' or str(torchvision.__version__)!='0.26.0+cu128':
            raise ValueError('pinned Kaggle torch/torchvision image drift')
        cfg=json.loads((ROOT/'configs/v34r1-kaggle.json').read_text())
        source=Path(imageio_ffmpeg.get_ffmpeg_exe())
        if hashlib.sha256(source.read_bytes()).hexdigest()!=cfg['expected_ffmpeg_sha256']:
            raise ValueError('pinned Linux FFmpeg binary drift')
        codec=ROOT/cfg['ffmpeg_executable']; codec.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,codec); codec.chmod(0o755)
        os.environ.update(TORCH_HOME=str(ROOT/'data/torch'),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                          MPLBACKEND='Agg',PYTHONPATH=str(ROOT),PATH=str(codec.parent)+os.pathsep+os.environ['PATH'])
        trackeval=ROOT/'data/TrackEval'
        run(['git','clone','-q','https://github.com/JonathonLuiten/TrackEval.git',trackeval])
        run(['git','-C',trackeval,'checkout','-q','12c8791b303e0a0b50f753af204249e622d0281a'])
        os.environ['V34_TRACKEVAL']=str(trackeval)
        run([sys.executable,'-m','pytest','tests','-q'],log=work/'block-tests.log')
        run([sys.executable,'scripts/reproduce_results.py','--verify-only'])
        run([sys.executable,'scripts/prepare_v34r1_cloud_data.py','--task',args.task],log=work/'source-restore.log')
        # runner requires an empty output; this setup never puts files there.
        run([sys.executable,'-u','-m','adaptive_vcm.v34r1.runner','--manifest','data/v34r1/validation_manifest.json',
             '--config','configs/v34r1-kaggle.json','--task',args.task,'--split','test','--out',out],log=work/'benchmark.log')
        status['complete']=True
    except Exception as error:
        status.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        status['wall_seconds']=time.time()-started
        (work/'cloud-status.json').write_text(json.dumps(status,indent=2))
        archive=work/f'v34r1-{args.task}-test.tgz'
        with tarfile.open(archive,'w:gz') as tar:
            if out.is_dir(): tar.add(out,arcname=args.task+'-test')
            for name in ('block-tests.log','source-restore.log','benchmark.log','cloud-status.json'):
                if (work/name).is_file(): tar.add(work/name,arcname=name)
        print('CLOUD_STATUS',json.dumps(status),flush=True)


if __name__=='__main__': main()

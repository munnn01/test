"""Pinned two-T4 OT DEV cache and independent causal policy replays, no TEST inference."""
import argparse
import gzip
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
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_v34r1_kaggle import run
from run_v34r1_a3_kaggle import last_progress


def main():
    p=argparse.ArgumentParser(); p.add_argument('--commit',required=True); args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'): raise ValueError('Kaggle cloud only')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=args.commit: raise ValueError('immutable source revision required')
    work=Path('/kaggle/working'); base=work/'a3-ot-dev'; base.mkdir()
    status=dict(complete=False,commit=args.commit,test_scored=False); started=time.time(); workers=[]; logs=[]
    try:
        run([sys.executable,'-m','pip','install','-q','--no-deps','numpy==2.1.3','scipy==1.16.3',
            'opencv-python==4.12.0.88','opencv-python-headless==4.12.0.88','supervision==0.27.0',
            'pycocotools==2.0.11','imageio-ffmpeg==0.6.0','tabulate==0.9.0','pytest==9.1.1'])
        import torch,torchvision,imageio_ffmpeg
        if not torch.cuda.is_available() or torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in (0,1)):
            raise ValueError('two Kaggle T4 GPUs required')
        if str(torch.__version__)!='2.11.0+cu128' or str(torchvision.__version__)!='0.26.0+cu128': raise ValueError('pinned torch image drift')
        cfg=json.loads((ROOT/'configs/v34r1-kaggle.json').read_text()); protocol=json.loads((ROOT/'configs/v34r1-a3-ot-probe.json').read_text())
        ffmpeg=Path(imageio_ffmpeg.get_ffmpeg_exe()); sha=hashlib.sha256(ffmpeg.read_bytes()).hexdigest()
        if sha!=cfg['expected_ffmpeg_sha256']: raise ValueError('pinned codec binary drift')
        codec=ROOT/cfg['ffmpeg_executable']; codec.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ffmpeg,codec); codec.chmod(0o755)
        os.environ.update(TORCH_HOME=str(ROOT/'data/torch'),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
            MPLBACKEND='Agg',PYTHONPATH=str(ROOT),PATH=str(codec.parent)+os.pathsep+os.environ['PATH'])
        trackeval=ROOT/'data/TrackEval'
        run(['git','clone','-q','https://github.com/JonathonLuiten/TrackEval.git',trackeval])
        run(['git','-C',trackeval,'checkout','-q','12c8791b303e0a0b50f753af204249e622d0281a'])
        os.environ['V34_TRACKEVAL']=str(trackeval)
        run([sys.executable,'-m','pytest','tests','-q'],log=base/'block-tests.log')
        run([sys.executable,'scripts/reproduce_results.py','--verify-only'])
        candidates=list(Path('/kaggle/input').rglob('mot17-source-cache.zip'))
        if len(candidates)!=1: raise ValueError('exactly one private frozen MOT17 source-cache kernel input required')
        archive=candidates[0]
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=protocol['source_cache_sha256']: raise ValueError('private source cache hash drift')
        source=ROOT/'data/v34r1_a3/mot17'; source.mkdir(parents=True)
        with zipfile.ZipFile(archive) as zipped:
            raw=zipped.read('source-manifest.json')
            if hashlib.sha256(raw).hexdigest()!=protocol['source_manifest_sha256']: raise ValueError('frozen source manifest hash drift')
            manifest=json.loads(raw); (source/'source-manifest.json').write_bytes(raw)
            # Reserved TEST is never extracted into this DEV job.
            for seq in manifest['sequences']:
                if seq['split']=='reserved_test': continue
                for member,expected in seq['files'].items():
                    relative=seq['name']+'/'+member; target=(source/relative).resolve()
                    if not target.is_relative_to(source): raise ValueError('source member escape')
                    data=zipped.read(relative)
                    if hashlib.sha256(data).hexdigest()!=expected: raise ValueError('original MOT17 member hash drift')
                    target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
            (base/'source-manifest.json').write_bytes(raw)
        status['environment']=dict(python=sys.version,ffmpeg_sha256=sha,
            packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','scipy','supervision')})
        from torchvision.models.detection import fasterrcnn_resnet50_fpn,FasterRCNN_ResNet50_FPN_Weights
        downloaded=fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1); del downloaded
        for worker in (0,1):
            log=open(base/f'worker{worker}.log','w',encoding='utf-8'); logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(worker))
            workers.append(subprocess.Popen([sys.executable,'-u','-m','adaptive_vcm.v34r1_a3.ot_cache','--root',str(source),
                '--worker',str(worker),'--out',str(base/f'worker{worker}')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(process.poll() is None for process in workers):
            failed=[i for i,p in enumerate(workers) if p.poll() not in (None,0)]
            if failed: raise RuntimeError('OT GPU worker failed: '+str(failed))
            print('OT_CACHE_PROGRESS',json.dumps({str(i):last_progress(base/f'worker{i}.log') for i in (0,1)}),flush=True)
            time.sleep(30)
        if any(p.returncode for p in workers): raise RuntimeError('OT GPU worker failed')
        run([sys.executable,'-u','-m','adaptive_vcm.v34r1_a3.ot_analyze','--root',str(base),'--sources',str(source)],log=base/'analysis.log')
        status['complete']=True
    except Exception as error:
        status.update(error_type=type(error).__name__,error=str(error))
        for process in workers:
            if process.poll() is None: process.terminate()
        for process in workers:
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        for i in range(len(workers)):
            path=base/f'worker{i}.log'
            if path.exists(): print(path.read_text(errors='replace')[-8000:],flush=True)
        raise
    finally:
        for log in logs: log.close()
        status['wall_seconds']=time.time()-started
        (base/'cloud-status.json').write_text(json.dumps(status,indent=2)+'\n')
        with tarfile.open(work/'v34r1-a3-ot-dev.tgz','w:gz') as tar:
            for path in sorted(base.rglob('*')):
                if path.is_file() and 'ground_truth' not in path.relative_to(base).parts: tar.add(path,arcname=path.relative_to(work))
        print('OT_A3_CLOUD_STATUS',json.dumps(status),flush=True)


if __name__=='__main__': main()

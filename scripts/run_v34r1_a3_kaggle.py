"""Pinned Kaggle setup: verify blocks, hash fresh sources, run exhaustive DEV on2T4s."""
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
sys.path.insert(0,str(ROOT/'scripts'))
from run_v34r1_kaggle import run


def main():
    p=argparse.ArgumentParser(); p.add_argument('--commit',required=True); args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'): raise ValueError('Kaggle cloud only')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=args.commit: raise ValueError('immutable source revision required')
    work=Path('/kaggle/working'); base=work/'a3-od-dev'; base.mkdir()
    status=dict(complete=False,commit=args.commit,test_scored=False); started=time.time(); workers=[]; logs=[]
    try:
        run([sys.executable,'-m','pip','install','-q','--no-deps','numpy==2.1.3','scipy==1.16.3',
            'opencv-python==4.12.0.88','opencv-python-headless==4.12.0.88','supervision==0.27.0',
            'pycocotools==2.0.11','imageio-ffmpeg==0.6.0','tabulate==0.9.0','pytest==9.1.1'])
        import torch,torchvision,imageio_ffmpeg
        if not torch.cuda.is_available() or torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in (0,1)):
            raise ValueError('two Kaggle T4 GPUs required')
        if str(torch.__version__)!='2.11.0+cu128' or str(torchvision.__version__)!='0.26.0+cu128': raise ValueError('pinned torch image drift')
        cfg=json.loads((ROOT/'configs/v34r1-kaggle.json').read_text())
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
        from prepare_v34r1_cloud_data import mounted
        from adaptive_vcm.data import read_image
        protocol_path=ROOT/'configs/v34r1-a3-od-probe.json'; protocol=json.loads(protocol_path.read_text())
        annotation=mounted('annotations/instances_val2017.json')
        if annotation is None or hashlib.sha256(annotation.read_bytes()).hexdigest()!=protocol['annotation_sha256']:
            raise ValueError('mounted original COCO annotations unavailable/drifted')
        shutil.copy2(annotation,base/'instances_val2017.json')
        manifest=dict(source_id_protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),dev=[],reserved_test=[])
        for split,items in protocol['cohorts'].items():
            for item in items:
                source=mounted('val2017/'+item['file_name'])
                if source is None: raise FileNotFoundError('frozen source unavailable: '+item['file_name'])
                row=dict(item,path=str(source),split=split,file_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
                pixels,_=read_image(row,320)
                row['canonical_sha256']=hashlib.sha256(pixels.tobytes()).hexdigest()
                manifest['reserved_test' if split=='test' else 'dev'].append(row)
        all_sources=manifest['dev']+manifest['reserved_test']
        for key in ('file_sha256','canonical_sha256'):
            hashes=[r[key] for r in all_sources]
            if len(hashes)!=len(set(hashes)): raise ValueError('duplicate fresh source '+key+'; frozen IDs cannot be replaced')
        manifest_file=base/'source-manifest.json'; manifest_file.write_text(json.dumps(manifest,indent=2)+'\n')
        print('A3_SOURCE_FREEZE',json.dumps(dict(dev=len(manifest['dev']),test_reserved=len(manifest['reserved_test']),
              sha256=hashlib.sha256(manifest_file.read_bytes()).hexdigest())),flush=True)
        status['environment']=dict(python=sys.version,packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','scipy','pycocotools')},ffmpeg_sha256=sha)
        # Serialize the initial weights download before two independent readers.
        from torchvision.models.detection import fasterrcnn_resnet50_fpn,FasterRCNN_ResNet50_FPN_Weights
        downloaded=fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1); del downloaded
        for worker in (0,1):
            log=open(base/f'worker{worker}.log','w',encoding='utf-8'); logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(worker))
            workers.append(subprocess.Popen([sys.executable,'-u','-m','adaptive_vcm.v34r1_a3.probe','--manifest',str(manifest_file),
                '--worker',str(worker),'--out',str(base/f'worker{worker}')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(process.poll() is None for process in workers):
            failed=[i for i,process in enumerate(workers) if process.poll() not in (None,0)]
            if failed: raise RuntimeError('A3 GPU worker failed: '+str(failed))
            print('A3_PROGRESS',json.dumps({str(i):last_progress(base/f'worker{i}.log') for i in (0,1)}),flush=True)
            time.sleep(30)
        if any(process.returncode for process in workers): raise RuntimeError('A3 worker failed; see retained worker logs')
        run([sys.executable,'-u','-m','adaptive_vcm.v34r1_a3.analyze','--root',str(base)],log=base/'analysis.log')
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
        with tarfile.open(work/'v34r1-a3-od-dev.tgz','w:gz') as tar:
            # Omit raw media and copyrighted annotation content, retain its SHA.
            for path in sorted(base.rglob('*')):
                if path.is_file() and path.name!='instances_val2017.json': tar.add(path,arcname=path.relative_to(work))
        print('A3_CLOUD_STATUS',json.dumps(status),flush=True)


def last_progress(path):
    if not path.exists(): return 'starting'
    lines=path.read_text(errors='replace').splitlines()
    return lines[-1][-400:] if lines else 'loading'


if __name__=='__main__': main()

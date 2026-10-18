"""Private T4x2 shard: fixed AR250, OD250, OT3/3/2/2 with isolated task artifacts."""
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from adaptive_vcm.v34r1_a4.cloud_record import record_cloud_run,require_sha256
from adaptive_vcm.v34r1_a6.evidence import publication
from adaptive_vcm.v34r1_a8.contracts import shard
from run_v34r1_kaggle import run
from run_v34r1_a3_kaggle import last_progress
from prepare_v34r1_a8_sources import restore_ar,restore_od,mounted,dance_zip


def workers(base,commands):
    processes=[];logs=[]
    try:
        for i,command in enumerate(commands):
            log=open(base/f'worker{i}.log','w',encoding='utf-8');logs.append(log)
            processes.append(subprocess.Popen([sys.executable,'-u',*map(str,command)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i)),stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('GPU worker failed; retain checkpoints')
            print('A8_PROGRESS',base.name,json.dumps({str(i):last_progress(base/f'worker{i}.log') for i in range(len(processes))}),flush=True);time.sleep(30)
        if any(p.returncode for p in processes):raise RuntimeError('GPU worker failed; retain checkpoints')
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()
        for i in range(len(processes)):
            path=base/f'worker{i}.log'
            if path.exists():print(path.read_text(errors='replace')[-3000:],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--shard',type=int,choices=range(4),required=True);p.add_argument('--source-manifest-sha256',required=True);a=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=a.commit:raise ValueError('immutable Kaggle source revision required')
    work=Path('/kaggle/working');base=work/('a8-half-shard'+str(a.shard));testlog=work/'a8-block-tests.log'
    states={};status=dict(commit=a.commit,shard=a.shard,source_manifest_sha256=a.source_manifest_sha256,evaluation_requested=True,tasks=states)
    with record_cloud_run(base,work/('v34r1-a8-half-shard'+str(a.shard)+'.tgz'),status,logs=[testlog]):
        base.mkdir();cfg_raw=(ROOT/'configs/v34r1-a8-expansion.json').read_bytes();cfg=json.loads(cfg_raw)
        source_raw=(ROOT/'results/v34r1_a8/source-freeze/source-manifest.json').read_bytes();require_sha256(source_raw,a.source_manifest_sha256);manifest=json.loads(source_raw)
        if manifest['selection_protocol_sha256']!=hashlib.sha256(cfg_raw).hexdigest() or manifest['physical_counts']!=dict(ar=1000,od=1000,ot=10):raise ValueError('global freeze and sample declaration drift')
        if cfg['ot']['h264']!=dict(kind='shared_qp',weight=8) or cfg['ot']['h265']!=dict(kind='independent_qp',weight=32):raise ValueError('fixed codec route required')
        publication(ROOT/'results/v34r1_a7/ot-shared-qp',cfg['ot']['parent_a7_publication_sha256'])
        publication(ROOT/'results/v34r1_a6/ot-full21',cfg['ot']['parent_a6_publication_sha256'])
        publication(ROOT/'results/v34r1_a6/od-crossfit',cfg['od']['parent_publication_sha256'])
        require_sha256((ROOT/cfg['od']['policy_path']).read_bytes(),cfg['od']['policy_sha256'])
        require_sha256((ROOT/'configs/v34r1-kaggle.json').read_bytes(),cfg['base_config_sha256'])
        run([sys.executable,'-m','pip','install','-q','--no-deps','numpy==2.1.3','scipy==1.16.3','opencv-python==4.12.0.88',
            'opencv-python-headless==4.12.0.88','supervision==0.27.0','pycocotools==2.0.11','imageio-ffmpeg==0.6.0','tabulate==0.9.0'])
        import torch,torchvision,imageio_ffmpeg
        if (not torch.cuda.is_available() or torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in (0,1))
            or str(torch.__version__)!='2.11.0+cu128' or str(torchvision.__version__)!='0.26.0+cu128'):raise ValueError('pinned Kaggle T4x2 runtime drift')
        baseline=json.loads((ROOT/'configs/v34r1-kaggle.json').read_bytes());ffmpeg=Path(imageio_ffmpeg.get_ffmpeg_exe());require_sha256(ffmpeg.read_bytes(),baseline['expected_ffmpeg_sha256'])
        codec=ROOT/baseline['ffmpeg_executable'];codec.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ffmpeg,codec);codec.chmod(0o755)
        os.environ.update(TORCH_HOME=str(ROOT/'data/torch'),PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg',PATH=str(codec.parent)+os.pathsep+os.environ['PATH'])
        trackeval=ROOT/'data/TrackEval';trackeval.parent.mkdir(parents=True,exist_ok=True)
        run(['git','clone','-q','https://github.com/JonathonLuiten/TrackEval.git',trackeval]);run(['git','-C',trackeval,'checkout','-q','12c8791b303e0a0b50f753af204249e622d0281a']);os.environ['V34_TRACKEVAL']=str(trackeval)
        run([sys.executable,'-m','unittest','tests.test_v34r1_a8_contracts','tests.test_v34r1_a8_ar_receiver','tests.test_v34r1_a8_sources','tests.test_v34r1_a8_evidence','tests.test_v34r1_a8_merge',
            'tests.test_v34r1_a7_shared','tests.test_v34r1_a7_goal','tests.test_v34r1_a6_contracts','tests.test_v34r1_a6_ot_grid',
            'tests.test_v34r1_a5_ot_choice','tests.test_v34r1_a3_hota','tests.test_v34r1_a3_tracking','tests.test_v34r1_a3_core','tests.test_v34r1_blocks','tests.test_v34r1_a4_cloud_record','-v'],log=testlog)
        from torchvision.models.video import Swin3D_T_Weights
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        for weights in (Swin3D_T_Weights.KINETICS400_V1,FasterRCNN_ResNet50_FPN_Weights.COCO_V1):
            cached=weights.get_state_dict(progress=True,check_hash=True);del cached
        (base/'environment.json').write_text(json.dumps(dict(python=sys.version,ffmpeg_sha256=baseline['expected_ffmpeg_sha256'],packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','scipy','supervision','pycocotools')},gpus=[torch.cuda.get_device_name(i) for i in (0,1)]),indent=2)+'\n')
        errors={}
        for task in ('ar','od','ot'):
            folder=base/task;folder.mkdir();started=time.perf_counter()
            try:
                if task=='ar':
                    records=[restore_ar(r) for r in shard(manifest['ar'],a.shard,4)];commands=[]
                    for i in (0,1):
                        part=dict(qps=cfg['qps'],codecs=cfg['codecs'],ar=shard(records,i,2),od=[],ot=[]);mp=folder/f'worker{i}-manifest.json';mp.write_text(json.dumps(part,indent=2)+'\n')
                        commands.append(['-m','adaptive_vcm.v34r1_a8.ar_worker','--manifest',mp,'--out',folder/f'worker{i}'])
                    workers(folder,commands)
                elif task=='od':
                    records=[restore_od(r) for r in shard(manifest['od'],a.shard,4)];mp=folder/'source-manifest.json';mp.write_text(json.dumps(dict(role='a8_shard250',images=records,
                        source_id_protocol_sha256=hashlib.sha256(cfg_raw).hexdigest(),frozen_policy_sha256=cfg['od']['policy_sha256']),indent=2)+'\n')
                    workers(folder,[['-m','adaptive_vcm.v34r1_a8.od_worker','--manifest',mp,'--worker',i,'--out',folder/f'worker{i}'] for i in (0,1)])
                    annotation=mounted('annotations/instances_val2017.json')
                    if annotation is None:raise FileNotFoundError('original COCO annotations missing')
                    run([sys.executable,'-u','-m','adaptive_vcm.v34r1_a8.od_evaluate','--root',folder,'--annotation',annotation],log=folder/'evaluation.log')
                else:
                    dance_zip(cfg) # One validated parent download before either child accesses DanceTrack.
                    mp=folder/'source-manifest.json';mp.write_bytes(source_raw)
                    workers(folder,[['-m','adaptive_vcm.v34r1_a8.ot_worker','--manifest',mp,'--shard',a.shard,'--worker',i,'--out',folder/f'worker{i}'] for i in (0,1)])
                states[task]=dict(complete=True,wall_seconds=time.perf_counter()-started)
            except Exception as error:
                errors[task]=dict(error_type=type(error).__name__,error=str(error));states[task]=dict(complete=False,**errors[task],wall_seconds=time.perf_counter()-started)
            (folder/'task-status.json').write_text(json.dumps(states[task],indent=2)+'\n')
            (base/'task-status.json').write_text(json.dumps(states,indent=2)+'\n');print('A8_TASK_STATUS',task,json.dumps(states[task]),flush=True)
        if errors:raise RuntimeError('one or more tasks failed; preserve every completed task and source checkpoint: '+json.dumps(errors))
        (base/'complete.json').write_text(json.dumps(dict(complete=True,shard=a.shard,ar=250,od=250,ot=len(shard(cfg['ot']['records'],a.shard,4)),tasks=states),indent=2)+'\n')
        print('A8_GPU_SHARD_COMPLETE',a.shard,flush=True)


if __name__=='__main__':main()

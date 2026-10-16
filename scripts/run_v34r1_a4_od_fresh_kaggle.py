"""Pinned two-T4 one-shot fresh64 confirmation of an already locked OD critic."""
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
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from adaptive_vcm.v34r1_a4.cloud_record import record_cloud_run,require_sha256,kernel_artifact,unpack_archive
from run_v34r1_kaggle import run
from run_v34r1_a3_kaggle import last_progress
from prepare_v34r1_cloud_data import mounted


def main():
    p=argparse.ArgumentParser(); p.add_argument('--commit',required=True); p.add_argument('--training-sha256',required=True); args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'): raise ValueError('Kaggle cloud only')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=args.commit: raise ValueError('immutable source revision required')
    work=Path('/kaggle/working'); base=work/'a4-od-fresh-dev'; blocklog=work/'a4-fresh-block-tests.log'; analysis=work/'a4-fresh-analysis.log'
    with record_cloud_run(base,work/'v34r1-a4-od-fresh-dev.tgz',dict(commit=args.commit,training_archive_sha256=args.training_sha256,test_scored=False),logs=[blocklog,analysis]):
        base.mkdir()
        run([sys.executable,'-m','pip','install','-q','--no-deps','numpy==2.1.3','scipy==1.16.3','pycocotools==2.0.11',
             'opencv-python==4.12.0.88','opencv-python-headless==4.12.0.88','imageio-ffmpeg==0.6.0'])
        os.environ.update(PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg',TORCH_HOME=str(ROOT/'data/torch'))
        run([sys.executable,'-m','unittest','tests.test_v34r1_a4_fresh_gate','tests.test_v34r1_a4_risk',
             'tests.test_v34r1_a4_marginal_ap','tests.test_v34r1_a4_sender_costs','tests.test_v34r1_a4_cloud_record','tests.test_v34r1_a3_core','tests.test_v34r1_blocks','-v'],log=blocklog)
        archive=kernel_artifact('/kaggle/input','v34r1-a4-od-training.tgz')
        require_sha256(archive.read_bytes(),args.training_sha256); cache=ROOT/'data/a4-training'; unpack_archive(archive,cache)
        models=cache/'a4-od-training'; status=json.loads((models/'cloud-status.json').read_text()); frozen_raw=(models/'frozen-policy.json').read_bytes(); frozen=json.loads(frozen_raw)
        if not status['complete'] or status['test_scored'] or status['fresh_dev64_scored'] or frozen['policy']['kind']!='risk_forest': raise ValueError('complete pre-fresh improvement policy required')
        protocol_raw=(ROOT/'configs/v34r1-a4-od-next-dev.json').read_bytes(); reservation=json.loads(protocol_raw)
        require_sha256(protocol_raw,frozen['fresh_dev_id_protocol_sha256'])
        require_sha256((ROOT/'configs/v34r1-a4-od-risk.json').read_bytes(),frozen['training_protocol_sha256'])
        if frozen['policy']['model'] not in ('extra_trees_depth6','extra_trees_depth10'): raise ValueError('prespecified model required')
        model_file=models/(frozen['policy']['model']+'.json'); require_sha256(model_file.read_bytes(),frozen['policy']['model_sha256'])
        locked=base/'locked-model'; locked.mkdir(); shutil.copyfile(model_file,locked/model_file.name); (locked/'frozen-policy.json').write_bytes(frozen_raw)
        cfg=json.loads((ROOT/'configs/v34r1-kaggle.json').read_text())
        if cfg['qps']!=[30,35,40,45] or cfg['codecs']!=['h264','h265']: raise ValueError('exact four-QP grid required')
        annotation=mounted('annotations/instances_val2017.json')
        if annotation is None: raise FileNotFoundError('original COCO annotations missing')
        require_sha256(annotation.read_bytes(),reservation['annotation_sha256'])
        import torch,torchvision,imageio_ffmpeg
        if not torch.cuda.is_available() or torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in (0,1)): raise ValueError('two Kaggle T4 GPUs required')
        if str(torch.__version__)!='2.11.0+cu128' or str(torchvision.__version__)!='0.26.0+cu128': raise ValueError('pinned torch runtime drift')
        ffmpeg=Path(imageio_ffmpeg.get_ffmpeg_exe()); require_sha256(ffmpeg.read_bytes(),cfg['expected_ffmpeg_sha256'])
        codec=ROOT/cfg['ffmpeg_executable']; codec.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ffmpeg,codec); codec.chmod(0o755)
        os.environ['PATH']=str(codec.parent)+os.pathsep+os.environ.get('PATH','')
        from adaptive_vcm.data import read_image
        prior_raw=(ROOT/'results/v34r1_a3/od-dev/source-manifest.json').read_bytes(); prior=json.loads(prior_raw)
        prior_sources=prior['dev']+prior['reserved_test']; old_ids={r['id'] for r in prior_sources}
        manifest=dict(role='fresh_dev64',source_id_protocol_sha256=hashlib.sha256(protocol_raw).hexdigest(),
            frozen_policy_sha256=hashlib.sha256(frozen_raw).hexdigest(),training_archive_sha256=args.training_sha256,
            prior_source_manifest_sha256=hashlib.sha256(prior_raw).hexdigest(),fresh_dev=[])
        if len(reservation['selected'])!=64 or len({r['id'] for r in reservation['selected']})!=64: raise ValueError('complete frozen64 IDs required')
        for item in reservation['selected']:
            if item['id'] in old_ids: raise ValueError('fresh source overlaps FIT/CAL/reserved TEST')
            source=mounted('val2017/'+item['file_name'])
            if source is None: raise FileNotFoundError('frozen image missing; never substitute another ID')
            row=dict(item,path=str(source),split='fresh_dev',file_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            pixels,_=read_image(row,320); row['canonical_sha256']=hashlib.sha256(pixels.tobytes()).hexdigest(); manifest['fresh_dev'].append(row)
        for key in ('file_sha256','canonical_sha256'):
            hashes=[r[key] for r in prior_sources+manifest['fresh_dev']]
            if len(hashes)!=len(set(hashes)): raise ValueError('duplicate source byte/canonical pixels')
        manifest_path=base/'source-manifest.json'; manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        (base/'environment.json').write_text(json.dumps(dict(python=sys.version,ffmpeg_sha256=cfg['expected_ffmpeg_sha256'],
            packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','scipy','pycocotools')}),indent=2)+'\n')
        print('A4_FRESH_SOURCE_FREEZE',hashlib.sha256(manifest_path.read_bytes()).hexdigest(),flush=True)
        from torchvision.models.detection import fasterrcnn_resnet50_fpn,FasterRCNN_ResNet50_FPN_Weights
        weights=fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1); del weights
        workers=[]; logs=[]
        try:
            for worker in (0,1):
                log=open(base/f'worker{worker}.log','w',encoding='utf-8'); logs.append(log)
                workers.append(subprocess.Popen([sys.executable,'-u','-m','adaptive_vcm.v34r1_a4.od_probe','--manifest',str(manifest_path),
                    '--worker',str(worker),'--out',str(base/f'worker{worker}')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(worker)),stdout=log,stderr=subprocess.STDOUT))
            while any(p.poll() is None for p in workers):
                if any(p.poll() not in (None,0) for p in workers): raise RuntimeError('fresh GPU worker failed')
                print('A4_FRESH_PROGRESS',json.dumps({str(i):last_progress(base/f'worker{i}.log') for i in (0,1)}),flush=True); time.sleep(30)
            if any(p.returncode for p in workers): raise RuntimeError('fresh GPU worker failed')
        finally:
            for process in workers:
                if process.poll() is None: process.terminate()
            for process in workers:
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            for log in logs: log.close()
        run([sys.executable,'-u','-m','adaptive_vcm.v34r1_a4.od_fresh','--root',str(base),'--models',str(locked),'--annotation',str(annotation)],log=analysis)


if __name__=='__main__': main()

"""Prepare immutable public-source notebooks; keep Kaggle credentials local."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zlib


DATASETS = ["qktttttttttt/kineticscleaned", "awsaf49/coco-2017-dataset"]


def source_files(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in ("__pycache__", ".pytest_cache", "outputs", "payloads") for part in relative.parts):
            continue
        if (relative.parts[0] in ("ar_od_proxy", "configs", "docs", "tests")
                and path.suffix in (".py", ".json", ".md")) or relative.as_posix() in ("README.md", "pyproject.toml", ".gitignore"):
            result[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


BOOTSTRAP = r'''
import base64, hashlib, json, os, pathlib, subprocess, sys, tarfile, time, traceback, zlib
ROOT = pathlib.Path('/kaggle/working/shared_ar_od_proxy')
OUT = pathlib.Path('/kaggle/working/ar_od_proxy_results')
STATUS = pathlib.Path('/kaggle/working/cloud_status.json')
receipt = {'commit': COMMIT, 'source_hashes': MANIFEST, 'complete': False, 'started': time.time()}
embedded_sources = json.loads(zlib.decompress(base64.b64decode(SOURCE_BUNDLE)))

def phase(name, **details):
    print(json.dumps({'stage':'bootstrap_phase','phase':name,
                      'seconds':time.time()-receipt['started'], **details}), flush=True)

def fetch(item):
    name, expected = item
    destination = ROOT / name
    if not destination.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('unsafe source path')
    content = embedded_sources[name].encode('utf-8')
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('published source hash mismatch: ' + name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)

try:
    import torch, torchvision
    if not torch.cuda.is_available():
        raise RuntimeError('A free GPU is required; no CPU or synthetic-data fallback')
    receipt.update(gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__), torchvision=str(torchvision.__version__))
    STATUS.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({'stage': 'bootstrap', 'gpu': receipt['gpu'], 'commit': COMMIT}), flush=True)
    ROOT.mkdir(exist_ok=False)
    if set(embedded_sources) != set(MANIFEST):
        raise ValueError('embedded source inventory mismatch')
    for item in MANIFEST.items():
        fetch(item)
    phase('sources_verified', files=len(MANIFEST))
    phase('dependencies_start')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pycocotools>=2.0.7', 'scipy>=1.10'], check=True, timeout=180)
    phase('dependencies_ready')
    import cv2
    inputs = pathlib.Path('/kaggle/input')
    kin_options = [inputs/'kineticscleaned', inputs/'datasets/qktttttttttt/kineticscleaned']
    kin = next((p for p in kin_options if p.is_dir()), None)
    if kin is None:
        raise FileNotFoundError('mounted kineticscleaned dataset not found')
    phase('input_discovery_start')
    coco_options = [inputs/'coco-2017-dataset', inputs/'datasets/awsaf49/coco-2017-dataset']
    annotations = {p.resolve():p for root in coco_options for prefix in ('coco2017','')
                   if (p := root/prefix/'annotations/instances_val2017.json').is_file()}
    if not annotations:
        phase('annotation_fallback_scan')
        annotations = {p.resolve():p for p in inputs.rglob('instances_val2017.json')}
    annotations = list(annotations.values())
    if len(annotations) != 1:
        raise ValueError('exactly one COCO validation annotation is required')
    annotation = annotations[0]
    images = annotation.parent.parent / 'val2017'
    if not images.is_dir():
        alternatives = [p for p in annotation.parent.parent.rglob('val2017') if p.is_dir()]
        if len(alternatives) != 1:
            raise FileNotFoundError('one COCO val2017 image directory required')
        images = alternatives[0]
    env = dict(os.environ, AROD_SOURCE_COMMIT=COMMIT, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    command = [sys.executable, '-u', '-m', 'ar_od_proxy.run', 'train', '--config', 'ar_od_proxy/configs/pilot.json',
               '--ar-root', str(kin), '--coco-root', str(images), '--annotations', str(annotation), '--out', str(OUT)]
    if ACTION == 'calibrate':
        candidates = list(inputs.rglob('last.pt'))
        matches = [p for p in candidates if hashlib.sha256(p.read_bytes()).hexdigest() == BASELINE_SHA256]
        if len(matches) != 1:
            raise ValueError('baseline checkpoint hash mismatch or ambiguous baseline')
        command = [sys.executable, '-u', '-m', 'ar_od_proxy.calibration',
                   '--ar-root', str(kin), '--coco-root', str(images), '--annotations', str(annotation),
                   '--checkpoint', str(matches[0]), '--out', str(OUT)]
    elif ACTION in ('train-calibrated', 'screen', 'quality'):
        if CHECKPOINT_DATASET:
            owner, slug = CHECKPOINT_DATASET.split('/')
            folders = [inputs/slug, inputs/'datasets'/owner/slug]
            candidates = list({p.resolve():p for folder in folders
                               if (p := folder/'calibrated.pt').is_file()}.values())
        else:
            phase('checkpoint_fallback_scan')
            candidates = list(inputs.rglob('calibrated.pt'))
        matches = [p for p in candidates if hashlib.sha256(p.read_bytes()).hexdigest() == BASELINE_SHA256]
        if len(matches) != 1:
            raise ValueError('calibrated checkpoint hash mismatch or ambiguous checkpoint')
        command = [sys.executable, '-u', '-m', 'ar_od_proxy.run', 'train-calibrated',
                   '--config', 'ar_od_proxy/configs/calibrated_train.json',
                   '--ar-root', str(kin), '--coco-root', str(images), '--annotations', str(annotation),
                   '--checkpoint', str(matches[0]), '--out', str(OUT)]
        if ACTION == 'screen':
            command = [sys.executable, '-u', '-m', 'ar_od_proxy.screen',
                       '--config', 'ar_od_proxy/configs/bdrate_screen.json',
                       '--ar-root', str(kin), '--coco-root', str(images), '--annotations', str(annotation),
                       '--checkpoint', str(matches[0]), '--out', str(OUT)]
        elif ACTION == 'quality':
            command = [sys.executable, '-u', '-m', 'ar_od_proxy.quality',
                       '--config', 'ar_od_proxy/configs/quality_recovery.json',
                       '--ar-root', str(kin), '--coco-root', str(images), '--annotations', str(annotation),
                       '--checkpoint', str(matches[0]), '--out', str(OUT)]
    phase('inputs_ready', kinetics=str(kin), annotations=str(annotation))
    phase('runner_start', action=ACTION)
    with pathlib.Path('/kaggle/working/training_console.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f'pipeline exited {code}; inspect training_console.log and run.json')
    report = json.loads((OUT/'run.json').read_text())
    if report.get('complete') is not True:
        raise RuntimeError('pipeline did not produce a successful completion receipt')
    receipt['complete'] = True
except BaseException as error:
    receipt['error'] = f'{type(error).__name__}: {error}'
    raise
finally:
    receipt['wall_seconds'] = time.time() - receipt['started']
    STATUS.write_text(json.dumps(receipt, indent=2))
    if OUT.is_dir():
        with tarfile.open('/kaggle/working/ar_od_proxy_results.tgz', 'w:gz') as archive:
            archive.add(OUT, arcname=OUT.name)
    print(json.dumps({'cloud_complete': receipt['complete'], 'wall_seconds': receipt['wall_seconds']}), flush=True)
'''


def prepare(root, directory, commit, account, slug, action="train", baseline_sha256=None, checkpoint_dataset=None):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("an immutable full commit SHA is required")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", account) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{5,70}", slug):
        raise ValueError("invalid Kaggle handle")
    manifest = source_files(root)
    if "ar_od_proxy/run.py" not in manifest or "ar_od_proxy/configs/pilot.json" not in manifest:
        raise ValueError("complete pipeline sources required")
    if action not in ("train", "calibrate", "train-calibrated", "screen", "quality"):
        raise ValueError("unsupported experiment action")
    if action != "train" and not re.fullmatch(r"[0-9a-f]{64}", baseline_sha256 or ""):
        raise ValueError("calibration/deployment requires the exact input checkpoint SHA256")
    kernel_sources = {"train": [], "calibrate": ["qktttttttttt/ar-od-proxy-qp4-20261022"],
                      "train-calibrated": ["qktttttttttt/ar-od-proxy-calibration-qp4-20261022"],
                      "screen": ["qktttttttttt/ar-od-proxy-calibration-qp4-20261022"],
                      "quality": ["qktttttttttt/ar-od-proxy-calibration-qp4-20261022"]}[action]
    if checkpoint_dataset is not None:
        if (action not in ("train-calibrated", "screen", "quality") or not re.fullmatch(
                re.escape(account) + r"/[a-z0-9][a-z0-9-]{5,70}", checkpoint_dataset)):
            raise ValueError("checkpoint dataset must belong to the target account for calibrated deployment")
        kernel_sources = []
    datasets = DATASETS + ([checkpoint_dataset] if checkpoint_dataset else [])
    contents = {name:(Path(root)/name).read_bytes().decode('utf-8') for name in manifest}
    for name, content in contents.items():
        if hashlib.sha256(content.encode('utf-8')).hexdigest() != manifest[name]:
            raise ValueError('source changed while preparing notebook: '+name)
    bundle = base64.b64encode(zlib.compress(json.dumps(contents).encode('utf-8'),9)).decode('ascii')
    code = (f"COMMIT = {commit!r}\nMANIFEST = {manifest!r}\nACTION = {action!r}\n"
            f"BASELINE_SHA256 = {baseline_sha256!r}\nCHECKPOINT_DATASET = {checkpoint_dataset!r}\n"
            f"SOURCE_BUNDLE = {bundle!r}\n" + BOOTSTRAP)
    compile(code, "kaggle-bootstrap", "exec")
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": [{"cell_type": "code", "id": "immutable-pilot", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}]}
    metadata = {"id": f"{account}/{slug}", "title": slug, "code_file": "notebook.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_internet": True, "dataset_sources": datasets,
                "kernel_sources": kernel_sources,
                "competition_sources": [], "model_sources": []}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name, content in (("notebook.ipynb", notebook), ("kernel-metadata.json", metadata)):
        (directory / name).write_text(json.dumps(content, indent=2), encoding="utf-8")
    job = {"commit": commit, "handle": metadata["id"], "source_hashes": manifest,
           "action": action, "baseline_sha256": baseline_sha256, "checkpoint_dataset": checkpoint_dataset,
           "payload_hashes": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                              for name in ("notebook.ipynb", "kernel-metadata.json")}}
    (directory / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    return job


def verify_payload(directory, handle):
    directory = Path(directory)
    job = json.loads((directory / "job.json").read_text(encoding="utf-8"))
    for name, expected in job["payload_hashes"].items():
        if name not in ("notebook.ipynb", "kernel-metadata.json"):
            raise ValueError("unexpected payload hash entry")
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError("payload hash mismatch")
    metadata = json.loads((directory / "kernel-metadata.json").read_text(encoding="utf-8"))
    expected_datasets = DATASETS + ([job["checkpoint_dataset"]] if job.get("checkpoint_dataset") else [])
    if (metadata["id"] != handle or job["handle"] != handle or metadata["is_private"] is not True
            or metadata["enable_gpu"] is not True or metadata["dataset_sources"] != expected_datasets):
        raise ValueError("private GPU payload/account/dataset mismatch")
    return metadata, job


def pool_environment(pool, account):
    value = json.loads(Path(pool).read_text(encoding="utf-8-sig"))[account]
    environment = dict(os.environ)
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        environment.pop(name, None)
    if isinstance(value, str):
        secret = value
        environment["KAGGLE_API_TOKEN"] = secret
    elif isinstance(value, dict) and value.get("key"):
        secret = value["key"]
        environment.update(KAGGLE_USERNAME=value.get("username", account), KAGGLE_KEY=secret)
    else:
        raise ValueError("unsupported external Kaggle credential format")
    environment.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return environment, secret


def kaggle_call(arguments, pool, account):
    environment, secret = pool_environment(pool, account)
    result = subprocess.run([sys.executable, "-m", "kaggle", *arguments], env=environment,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    output = (result.stdout + result.stderr).replace(secret, "[REDACTED]")
    print(output, flush=True)
    if result.returncode:
        raise RuntimeError(f"Kaggle CLI failed with exit {result.returncode}; no submission success inferred")
    return output


def validate_submission(output, handle):
    lowered = output.lower()
    if any(marker in lowered for marker in ("kernel push error", "not valid dataset sources", "invalid dataset source")):
        raise ValueError("Kaggle submission rejected or datasets were not attached")
    version = re.search(r"Kernel version (\d+) successfully pushed", output)
    url = re.search(r"https://(?:www\.)?kaggle\.com/code/([A-Za-z0-9_-]+/[a-z0-9-]+)", output)
    if version is None or int(version.group(1)) <= 0 or url is None or url.group(1) != handle:
        raise ValueError("Kaggle submission needs a positive version and matching notebook URL")
    return {"version": int(version.group(1)), "url": url.group(0), "acknowledged": True,
            "running_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "status", "download"))
    parser.add_argument("--account", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--commit")
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--experiment", choices=("train", "calibrate", "train-calibrated", "screen", "quality"), default="train")
    parser.add_argument("--baseline-sha256")
    parser.add_argument("--checkpoint-dataset", help="Optional owned private input dataset for another pool account")
    args = parser.parse_args()
    handle = f"{args.account}/{args.slug}"
    if args.action == "prepare":
        job = prepare(Path(__file__).parents[1], args.directory, args.commit or "", args.account, args.slug,
                      action=args.experiment, baseline_sha256=args.baseline_sha256,
                      checkpoint_dataset=args.checkpoint_dataset)
        print(json.dumps({"prepared": str(args.directory), "handle": handle, "commit": job["commit"]}))
    elif args.action == "submit":
        verify_payload(args.directory, handle)
        output = kaggle_call(["kernels", "push", "-p", str(args.directory), "--accelerator", "NvidiaTeslaT4",
                              "--timeout", "21600" if json.loads((args.directory/'job.json').read_text())['action']=='quality'
                              else "10800" if json.loads((args.directory/'job.json').read_text())['action']=='screen' else "7200"], args.pool, args.account)
        (args.directory / "submission.txt").write_text(output, encoding="utf-8")
        receipt = validate_submission(output, handle)
        (args.directory / "submission.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    elif args.action == "status":
        output = kaggle_call(["kernels", "status", handle], args.pool, args.account)
        args.directory.mkdir(parents=True, exist_ok=True)
        (args.directory / "latest-status.txt").write_text(output, encoding="utf-8")
    else:
        args.directory.mkdir(parents=True, exist_ok=True)
        kaggle_call(["kernels", "output", handle, "-p", str(args.directory)], args.pool, args.account)


if __name__ == "__main__":
    main()

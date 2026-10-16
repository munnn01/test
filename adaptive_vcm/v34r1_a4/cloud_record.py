"""Preserve setup failures and successful artifacts in immutable cloud runs."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import tarfile
import time


def require_sha256(raw,expected):
    actual=hashlib.sha256(raw).hexdigest()
    if actual!=expected: raise ValueError('frozen input SHA256 mismatch')
    return actual


def kernel_artifact(root,name):
    """Search notebook outputs, skipping unrelated modern/legacy media mounts."""
    root=Path(root)
    if Path(name).name!=name or name in ('','.','..'): raise ValueError('artifact basename required')
    skip={'datasets','coco-2017-dataset','kinetics400-rgb256-f10-val-data'}
    matches=[]
    for base in root.iterdir():
        if base.name in skip: continue
        if base.is_dir(): matches.extend(p for p in base.rglob(name) if p.is_file())
        elif base.name==name and base.is_file(): matches.append(base)
    if len(matches)!=1: raise ValueError('exactly one notebook artifact required: '+name)
    return matches[0]


def unpack_archive(archive,destination):
    destination=Path(destination).resolve()
    if destination.exists(): raise ValueError('fresh immutable extraction directory required')
    destination.mkdir(parents=True); seen=set()
    with tarfile.open(archive) as tar:
        for member in tar:
            pure=PurePosixPath(member.name); name=pure.as_posix(); target=(destination/name).resolve()
            if (pure.is_absolute() or '..' in pure.parts or '\\' in member.name or ':' in member.name
                or name in ('','.','..') or name in seen or not target.is_relative_to(destination)):
                raise ValueError('unsafe/duplicate cloud archive path')
            seen.add(name)
            if member.isdir(): target.mkdir(parents=True,exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as src,target.open('xb') as dst: shutil.copyfileobj(src,dst)
            else: raise ValueError('cloud archive links/special files forbidden')


def assert_same_files(actual,expected):
    def hashes(root):
        root=Path(root)
        return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    a,b=hashes(actual),hashes(expected)
    if not b or a!=b: raise ValueError('exact parent tracker file-set/bytes parity failed')


@contextmanager
def record_cloud_run(base,archive,status,*,logs=()):
    base,archive=Path(base),Path(archive)
    if base.exists() or archive.exists(): raise ValueError('fresh immutable cloud output required')
    started=time.time(); status=dict(status,complete=False)
    try:
        yield
        status['complete']=True
    except Exception as error:
        status.update(error_type=type(error).__name__,error=str(error)); raise
    finally:
        base.mkdir(parents=True,exist_ok=True); status['wall_seconds']=time.time()-started
        (base/'cloud-status.json').write_text(json.dumps(status,indent=2)+'\n')
        for log in logs:
            log=Path(log)
            if log.is_file(): shutil.copyfile(log,base/log.name)
        with tarfile.open(archive,'w:gz') as tar:
            for path in sorted(base.rglob('*')):
                if path.is_file() and 'ground_truth' not in path.relative_to(base).parts:
                    tar.add(path,arcname=path.relative_to(base.parent))
        print('A4_CLOUD_STATUS',json.dumps(status),flush=True)

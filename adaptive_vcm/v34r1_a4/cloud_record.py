"""Preserve setup failures and successful artifacts in immutable cloud runs."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time


def require_sha256(raw,expected):
    actual=hashlib.sha256(raw).hexdigest()
    if actual!=expected: raise ValueError('frozen input SHA256 mismatch')
    return actual


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

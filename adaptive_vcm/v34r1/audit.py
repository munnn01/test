"""Fail closed if paired source/codec/QP/arm measurements are missing or duplicated."""
from collections import Counter
from itertools import product
import hashlib
from pathlib import Path
from . import QPS


def audit_grid(rows, *, source_ids, arms, codecs=('h264','h265')):
    ids=list(source_ids)
    if not ids or len(ids)!=len(set(ids)):
        raise ValueError('source IDs must be nonempty and unique')
    expected=set(product(ids,codecs,QPS,arms))
    observed=Counter((r['id'],r['codec'],r['qp'],r['arm']) for r in rows)
    missing=expected-observed.keys()
    extra=observed.keys()-expected
    duplicates=[k for k,n in observed.items() if n!=1]
    if missing or extra or duplicates:
        raise ValueError(f'incomplete paired grid: missing={len(missing)}, extra={len(extra)}, duplicates={len(duplicates)}')
    for r in rows:
        if (r['bytes']<=0 or r['bytes']!=r['elementary_bytes']+r['metadata_bytes']
            or not r.get('packet_sha256') or not r.get('decoded_sha256')):
            raise ValueError('invalid measured packet accounting')
    return dict(source_count=len(ids),paired_rows=len(rows),qps=list(QPS),codecs=list(codecs),arms=list(arms),complete=True)


def verify_source(item, *, required=True):
    """Validate frozen byte hashes before decoding; hashes do not expose GT labels."""
    path=Path(item['path'])
    expected=item.get('file_sha256')
    if expected is None:
        if required: raise ValueError('frozen source hash inventory is required')
        return
    checks={path:expected} if isinstance(expected,str) else {}
    if isinstance(expected,dict):
        root=path.resolve()
        if item.get('kind')=='frames':
            declared=set(item.get('provider_window_files',[]))
            actual={file.name for file in root.glob('img_*.jpg')}
            if not declared or declared!=set(expected) or actual!=declared:
                raise ValueError('AR source frame inventory changed')
        if 'sequence' in item:
            used={'seqinfo.ini','gt/gt.txt',*[f'img1/{i:06d}.jpg' for i in range(1,item['frames']+1)]}
            if not used.issubset(expected): raise ValueError('OT used-frame hash inventory is incomplete')
        for name,digest in expected.items():
            target=(root/name).resolve()
            if not target.is_relative_to(root): raise ValueError('source inventory escapes its root')
            checks[target]=digest
    if not checks: raise ValueError('empty source hash inventory')
    for target,digest in checks.items():
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise ValueError('frozen source SHA-256 mismatch: '+str(target))

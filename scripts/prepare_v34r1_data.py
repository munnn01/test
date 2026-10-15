"""Download deterministic small cohorts from the user-selected K400 and official COCO/MOT sources."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import threading
from urllib.parse import quote
import zipfile
import requests

KAGGLE = 'https://www.kaggle.com/api/v1/datasets'
COCO = 'https://s3.amazonaws.com/images.cocodataset.org'
MOT = 'https://motchallenge.net/data/MOT20.zip'


def partition(identifier):
    n = int(hashlib.md5(identifier.encode()).hexdigest()[:8], 16) % 10
    return 'test' if n == 0 else 'dev' if n == 1 else 'train'


def rank(identifier):
    return hashlib.sha256(('v34r1-20261015:' + identifier).encode()).hexdigest()


def download(job):
    url, dest, expected = job
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected is None or dest.stat().st_size == expected):
        return dict(path=str(dest), bytes=dest.stat().st_size, sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    part = dest.with_name(dest.name + '.part')
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(20, 90), stream=True)
            response.raise_for_status()
            with part.open('wb') as handle:
                for chunk in response.iter_content(1024*1024):
                    handle.write(chunk)
            if expected is not None and part.stat().st_size != expected:
                raise ValueError('download size mismatch: ' + dest.name)
            if part.stat().st_size == 0:
                raise ValueError('empty download: ' + dest.name)
            part.replace(dest)
            return dict(path=str(dest), bytes=dest.stat().st_size, sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        except (requests.RequestException, OSError, ValueError):
            if attempt == 2: raise


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        head = requests.head(url, timeout=30)
        head.raise_for_status()
        self.length = int(head.headers['Content-Length'])
        self.position = 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position+offset if whence == 1 else self.length+offset
        if position < 0: raise ValueError('negative seek')
        self.position = position
        return position
    def read(self, size=-1):
        if self.position >= self.length: return b''
        end = self.length-1 if size < 0 else min(self.length-1, self.position+size-1)
        if end < self.position: return b''
        expected_range = f'bytes {self.position}-{end}/{self.length}'
        r = requests.get(self.url, headers={'Range':f'bytes={self.position}-{end}'}, timeout=90)
        r.raise_for_status()
        if r.status_code != 206 or r.headers.get('Content-Range') != expected_range:
            raise RuntimeError('server does not honor the exact requested range')
        raw = r.content
        if len(raw) != end-self.position+1: raise ValueError('truncated range')
        self.position = end+1
        return raw


def save_zip_member(archive, member, destination):
    destination = Path(destination)
    info = archive.getinfo(member)
    if destination.exists() and destination.stat().st_size == info.file_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = archive.read(member)  # zipfile verifies CRC before returning.
    destination.write_bytes(raw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/v34r1'))
    p.add_argument('--kinetics-catalog', type=Path, required=True)
    p.add_argument('--train', type=int, default=64)
    p.add_argument('--dev', type=int, default=16)
    p.add_argument('--test', type=int, default=64)
    p.add_argument('--mot-frames', type=int, default=256)
    args = p.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(args.kinetics_catalog.read_text(encoding='utf-8'))
    video_rows = []
    for row in catalog['files']:
        path = Path(row['name'])
        if path.suffix.lower() == '.mp4':
            identifier = path.parent.name+'/'+path.name
            video_rows.append({**row, 'id': identifier, 'category':path.parent.name, 'split':partition(identifier)})
    manifests = dict(schema='v34r1-data-1', qps=[30,35,40,45],
        kinetics_source=dict(dataset='qktttttttttt/kineticscleaned',version=4,official_path_split='train',
            scope='controller-held-out hash split; not independent official K400 validation'), ar=[], od=[], ot=[])
    jobs = []
    for split in ('train','dev','test'):
        chosen = sorted([r for r in video_rows if r['split']==split], key=lambda r:rank(r['id']))[:getattr(args,split)]
        if len(chosen) != getattr(args,split): raise ValueError('insufficient K400 records')
        for row in chosen:
            destination = root/'kinetics'/row['category']/Path(row['name']).name
            url = KAGGLE+'/download/qktttttttttt/kineticscleaned/'+quote(row['name'],safe='')+'?datasetVersionNumber=4'
            jobs.append((url,destination,row['bytes']))
            manifests['ar'].append({**row,'path':str(destination)})
    print(f'K400: frozen {len(jobs)} video IDs before inference',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(download,jobs),1):
            if i%16==0 or i==len(jobs): print('K400 downloaded',i,'/',len(jobs),flush=True)
    annotation = root/'coco/annotations/instances_val2017.json'
    if not annotation.exists():
        with zipfile.ZipFile(HTTPRangeFile(COCO+'/annotations/annotations_trainval2017.zip')) as archive:
            save_zip_member(archive,'annotations/instances_val2017.json',annotation)
    meta = json.loads(annotation.read_text(encoding='utf-8'))
    jobs = []
    for split in ('train','dev','test'):
        chosen = sorted([r for r in meta['images'] if partition('coco2017/'+str(r['id']))==split],key=lambda r:rank(str(r['id'])))[:getattr(args,split)]
        for row in chosen:
            destination = root/'coco/val2017'/row['file_name']
            manifests['od'].append({**row,'split':split,'path':str(destination)})
            jobs.append((COCO+'/val2017/'+row['file_name'],destination,None))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,result in enumerate(pool.map(download,jobs),1):
            if i%32==0 or i==len(jobs): print('COCO downloaded',i,'/',len(jobs),flush=True)
    print('MOT20: fetching only declared sequence frames and GT using exact byte ranges',flush=True)
    with zipfile.ZipFile(HTTPRangeFile(MOT)) as archive:
        names = archive.namelist()
        for split,sequence in [('train','MOT20-02'),('cal','MOT20-05'),('dev','MOT20-03'),('test','MOT20-01')]:
            suffix = sequence+'/seqinfo.ini'
            prefix = next(name[:-len('seqinfo.ini')] for name in names if name.endswith(suffix))
            destination = root/'mot20'/sequence
            for member in ('seqinfo.ini','gt/gt.txt'):
                save_zip_member(archive,prefix+member,destination/member)
            for frame in range(1,args.mot_frames+1):
                member = f'img1/{frame:06d}.jpg'
                save_zip_member(archive,prefix+member,destination/member)
                if frame%64==0: print(sequence,frame,'/',args.mot_frames,flush=True)
            manifests['ot'].append(dict(sequence=sequence,split=split,path=str(destination),frames=args.mot_frames))
    for task in ('ar','od'):
        for row in manifests[task]:
            row['file_sha256'] = hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()
    manifests['coco_annotations_sha256'] = hashlib.sha256(annotation.read_bytes()).hexdigest()
    manifests['catalog_sha256'] = hashlib.sha256(args.kinetics_catalog.read_bytes()).hexdigest()
    (root/'manifest.json').write_text(json.dumps(manifests,indent=2),encoding='utf-8')
    print('Data ready:',root/'manifest.json',flush=True)


if __name__=='__main__': main()

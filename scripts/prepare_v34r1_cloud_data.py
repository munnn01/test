"""Restore the frozen TEST inputs on Kaggle; verify original byte hashes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
import zipfile
import requests
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adaptive_vcm.v34r1.audit import verify_source
from prepare_v34r1_data import HTTPRangeFile,COCO,MOT
from prepare_v34r1_mot_fast import fetch_member


def checked_copy(source,destination,sha):
    raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha: raise ValueError('mounted source hash mismatch: '+str(source))
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,destination)


def mounted(relative):
    bases=[Path('/kaggle/input/kinetics400-rgb256-f10-val-data'),Path('/kaggle/input/coco-2017-dataset'),
           Path('/kaggle/input/datasets/jizeyong/kinetics400-rgb256-f10-val-data'),
           Path('/kaggle/input/datasets/awsaf49/coco-2017-dataset')]
    for base in bases:
        for prefix in ('','coco2017','coco2017/coco2017'):
            file=base/prefix/relative
            if file.is_file(): return file
    return None


def http_file(url,destination,sha):
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest()==sha: return
        raise ValueError('existing source hash mismatch')
    destination.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(8):
        try:
            response=requests.get(url,timeout=(20,90)); response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest()!=sha: raise ValueError('remote source hash mismatch')
            temporary=destination.with_name(destination.name+'.part')
            temporary.write_bytes(response.content); temporary.replace(destination); return
        except (requests.RequestException,ValueError):
            if attempt==7: raise
            time.sleep(min(30,2**attempt))


def ar_frame(job):
    yt,name,destination,sha=job
    source=mounted(yt+'/'+name)
    if source and hashlib.sha256(source.read_bytes()).hexdigest()==sha:
        checked_copy(source,destination,sha); return
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest()==sha: return
    payload=dict(ownerSlug='jizeyong',datasetSlug='kinetics400-rgb256-f10-val-data',datasetVersionNumber=1,fileName=yt+'/'+name)
    for attempt in range(10):
        response=requests.post('https://www.kaggle.com/api/v1/datasets.DatasetApiService/DownloadDataset',json=payload,timeout=(20,90))
        if response.status_code==429: time.sleep(min(60,2**attempt)); continue
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest()!=sha: raise ValueError('AR provider byte hash mismatch')
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_name(destination.name+'.part'); temporary.write_bytes(response.content); temporary.replace(destination); return
    raise RuntimeError('provider throttled; no source replacement permitted')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--task',required=True,choices=['ar','od','ot'])
    args=p.parse_args()
    raw=gzip.decompress((ROOT/'configs/v34r1-cloud-data-manifest.json.gz').read_bytes())
    manifest=json.loads(raw); target=ROOT/'data/v34r1/validation_manifest.json'
    target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(raw)
    items=[r for r in manifest[args.task] if r['split']=='test']
    if args.task=='ar':
        jobs=[(r['youtube_id'],name,Path(r['path'])/name,sha) for r in items for name,sha in r['file_sha256'].items()]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i,_ in enumerate(pool.map(ar_frame,jobs),1):
                if i%256==0: print('AR frozen source JPEGs',i,'/',len(jobs),flush=True)
    elif args.task=='od':
        annotation=target.parent/'coco/annotations/instances_val2017.json'
        source=mounted('annotations/instances_val2017.json')
        if source and hashlib.sha256(source.read_bytes()).hexdigest()==manifest['coco_annotations_sha256']:
            checked_copy(source,annotation,manifest['coco_annotations_sha256'])
        elif not annotation.is_file():
            annotation.parent.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(HTTPRangeFile(COCO+'/annotations/annotations_trainval2017.zip')) as archive:
                annotation.write_bytes(archive.read('annotations/instances_val2017.json'))
        verify_source(dict(path=str(annotation),file_sha256=manifest['coco_annotations_sha256']))
        def image(row):
            source=mounted('val2017/'+row['file_name']); destination=Path(row['path'])
            if source and hashlib.sha256(source.read_bytes()).hexdigest()==row['file_sha256']:
                checked_copy(source,destination,row['file_sha256'])
            else: http_file(COCO+'/val2017/'+row['file_name'],destination,row['file_sha256'])
        with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(image,items))
    else:
        reader=HTTPRangeFile(MOT)
        with zipfile.ZipFile(reader) as archive: index={i.filename:i for i in archive.infolist()}
        for row in items:
            prefix=next(name[:-len('seqinfo.ini')] for name in index if name.endswith(row['sequence']+'/seqinfo.ini'))
            jobs=[(index[prefix+name],Path(row['path'])/name,reader.length) for name in row['file_sha256']]
            with ThreadPoolExecutor(max_workers=8) as pool:
                for i,_ in enumerate(pool.map(fetch_member,jobs),1):
                    if i%64==0: print('MOT20 verified members',i,'/',len(jobs),flush=True)
    for row in items: verify_source(row)
    print(json.dumps(dict(task=args.task,records=len(items),manifest_sha256=hashlib.sha256(raw).hexdigest(),
        source_hashes_verified=True,replacement=False)),flush=True)


if __name__=='__main__': main()

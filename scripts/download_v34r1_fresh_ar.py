"""Fetch predeclared 32-frame source windows for a fresh official K400-val holdout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests

API='https://www.kaggle.com/api/v1/datasets.DatasetApiService/DownloadDataset'


def fetch(job):
    filename,destination=job
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        raw=destination.read_bytes()
        if raw.startswith(b'\xff\xd8'): return
        raise ValueError('existing file is not JPEG: '+str(destination))
    payload=dict(ownerSlug='jizeyong',datasetSlug='kinetics400-rgb256-f10-val-data',datasetVersionNumber=1,fileName=filename)
    for attempt in range(12):
        try:
            r=requests.post(API,json=payload,timeout=(20,60),allow_redirects=True)
            if r.status_code==429:
                # Respect provider throttling. A larger pool aggravated 429s;
                # leave completed JPEGs in place for an identical frozen resume.
                delay=min(60,max(10,float(r.headers.get('Retry-After',0)),2**attempt))
                time.sleep(delay); continue
            r.raise_for_status()
            raw=r.content
            if not raw.startswith(b'\xff\xd8'): raise ValueError('JPEG payload missing: '+filename)
            temporary=destination.with_name(destination.name+'.part')
            temporary.write_bytes(raw); temporary.replace(destination)
            return
        except (requests.RequestException,ValueError):
            if attempt==11: raise
            time.sleep(min(30,2**attempt))
    raise RuntimeError('request repeatedly rate-limited')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--screen-manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    selection=json.loads(args.selection.read_text(encoding='utf-8'))
    if selection['qps']!=[30,35,40,45] or selection['count']!=128: raise ValueError('selection protocol drift')
    args.out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for row in selection['records']:
        start=(row['frame_count']-32)//2
        files=[f"{row['youtube_id']}/img_{i+1:05d}.jpg" for i in range(start,start+32)]
        if files[::2]!=row['selected_16_paths']: raise ValueError('provider frame indices disagree')
        jobs.extend((name,args.out/name) for name in files[::2])
    print('Frozen fresh AR holdout: 128 official-val videos; exact 2048 selected JPEGs; unused intervening frames are not required',flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,_ in enumerate(pool.map(fetch,jobs),1):
            if i%256==0: print('Fresh K400 JPEGs',i,'/',len(jobs),flush=True)
    manifest=json.loads(args.screen_manifest.read_text(encoding='utf-8'))
    manifest['ar']=[r for r in manifest['ar'] if r['split']!='test']
    for row in selection['records']:
        path=args.out/row['youtube_id']
        files=sorted(path.glob('img_*.jpg'))
        if not {Path(p).name for p in row['selected_16_paths']}.issubset({f.name for f in files}):
            raise ValueError('fresh predeclared source frame set is incomplete')
        manifest['ar'].append(dict(id=row['youtube_id'],youtube_id=row['youtube_id'],label=row['label'],
            path=str(path.resolve()),kind='frames',split='test',provider_original_frames=row['frame_count'],
            provider_selected_paths=row['selected_16_paths'],provider_window_files=[f.name for f in files],
            file_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files}))
    manifest['ar_test_provenance']=selection
    manifest['ot']=[dict(sequence=sequence,split=split,path=str((args.screen_manifest.parent/'mot20'/sequence).resolve()),frames=256)
        for split,sequence in [('train','MOT20-02'),('cal','MOT20-05'),('dev','MOT20-03'),('test','MOT20-01')]]
    for row in manifest['ot']:
        directory=Path(row['path'])
        files=['seqinfo.ini','gt/gt.txt',*[f'img1/{i:06d}.jpg' for i in range(1,row['frames']+1)]]
        row['file_sha256']={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in files}
    manifest['coco_annotations_sha256']=hashlib.sha256((args.screen_manifest.parent/'coco/annotations/instances_val2017.json').read_bytes()).hexdigest()
    target=args.screen_manifest.parent/'validation_manifest.json'
    if target.exists(): raise ValueError('validation manifest is already frozen')
    target.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Fresh validation manifest:',target,flush=True)


if __name__=='__main__': main()

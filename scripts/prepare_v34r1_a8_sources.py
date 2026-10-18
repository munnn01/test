"""Kaggle-only raw-byte source freeze/restore; never runs a task model or GT metric."""
from concurrent.futures import ThreadPoolExecutor
import configparser
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile
import requests

from adaptive_vcm.v34r1_a4.cloud_record import require_sha256
from adaptive_vcm.v34r1_a6.ar_source import provider_window
from adaptive_vcm.v34r1_a8.contracts import native_plan
from prepare_v34r1_cloud_data import mounted
import prepare_v34r1_mot_fast as mot_transport

ROOT=Path(__file__).resolve().parents[1]
_DANCE_VALIDATED=set()


def protocol():
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'):raise ValueError('Kaggle-only source access')
    raw=(ROOT/'configs/v34r1-a8-expansion.json').read_bytes();cfg=json.loads(raw)
    if cfg['qps']!=[30,35,40,45] or cfg['codecs']!=['h264','h265'] or cfg['policy_retuning']:raise ValueError('fixed A8 protocol required')
    return cfg,hashlib.sha256(raw).hexdigest()


def safe_target(root,member):
    path=(Path(root)/member).resolve()
    if not path.is_relative_to(Path(root).resolve()):raise ValueError('source member escape')
    return path


def dance_zip(cfg):
    dest=ROOT/'data/v34r1_a8/dancetrack-val.zip';dest.parent.mkdir(parents=True,exist_ok=True)
    expected=cfg['ot']['dance_archive_sha256'];size=cfg['ot']['dance_archive_bytes']
    key=(str(dest),expected,size)
    if key in _DANCE_VALIDATED and dest.stat().st_size==size:return dest
    if dest.exists():
        digest=hashlib.sha256()
        with dest.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
        if dest.stat().st_size!=size or digest.hexdigest()!=expected:raise ValueError('DanceTrack archive length/SHA drift')
        _DANCE_VALIDATED.add(key)
        return dest
    for attempt in range(4):
        part=dest.with_name(dest.name+'.'+str(os.getpid())+'.part');digest=hashlib.sha256();count=0
        try:
            with requests.get(cfg['ot']['dance_archive_url'],stream=True,timeout=(30,180)) as response:
                response.raise_for_status()
                with part.open('wb') as f:
                    for chunk in response.iter_content(8*1024*1024):
                        if chunk:f.write(chunk);digest.update(chunk);count+=len(chunk)
            if count!=size or digest.hexdigest()!=expected:raise ValueError('pinned DanceTrack archive size/SHA mismatch')
            part.replace(dest);_DANCE_VALIDATED.add(key);return dest
        except requests.RequestException:
            if attempt==3:raise
            time.sleep(min(30,2**attempt))


def mot_members(record,cfg,destination,*,gt=False):
    raw=(ROOT/'configs/v34r1-a3-mot17-sources.json.gz').read_bytes();require_sha256(raw,cfg['ot']['mot_catalog_sha256']);catalog=json.loads(gzip.decompress(raw))
    raw=(ROOT/'configs/v34r1-a3-mot17-byte-manifest.json.gz').read_bytes();require_sha256(raw,cfg['ot']['mot_byte_manifest_sha256']);old=json.loads(gzip.decompress(raw))
    seq=next(r for r in catalog['sequences'] if r['name']==record['name']);prior=next(r for r in old['sequences'] if r['name']==record['name'])
    requested=['gt/gt.txt'] if gt else ['seqinfo.ini',*[f'img1/{i:06d}.jpg' for i in range(1,record['frames']+1)]]
    members={r['member']:r for r in seq['members']};jobs=[];mot_transport.MOT=catalog['url']
    for member in requested:
        e=members[member];info=zipfile.ZipInfo(seq['prefix']+member)
        for field,value in [('file_size',e['file_size']),('compress_size',e['compress_size']),('compress_type',e['compress_type']),('CRC',e['crc32']),('header_offset',e['header_offset'])]:setattr(info,field,value)
        jobs.append((info,safe_target(destination,member),catalog['archive_bytes']))
    def fetch(job):
        for attempt in range(6):
            try:return mot_transport.fetch_member(job)
            except requests.RequestException:
                if attempt==5:raise
                time.sleep(min(30,2**attempt))
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(fetch,jobs))
    result={}
    for member in requested:
        data=safe_target(destination,member).read_bytes();require_sha256(data,prior['files'][member]);result[member]=hashlib.sha256(data).hexdigest()
    return result


def dance_members(record,cfg,destination,*,gt=False):
    with zipfile.ZipFile(dance_zip(cfg)) as z:
        prefixes=[n[:-len('seqinfo.ini')] for n in z.namelist() if n.endswith(record['name']+'/seqinfo.ini')]
        if len(prefixes)!=1:raise ValueError('exact DanceTrack video directory required')
        prefix=prefixes[0];requested=['gt/gt.txt'] if gt else ['seqinfo.ini',*[f'img1/{i:08d}.jpg' for i in range(1,record['frames']+1)]]
        result={}
        for member in requested:
            data=z.read(prefix+member);path=safe_target(destination,member);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            result[member]=hashlib.sha256(data).hexdigest()
    return result


def restore_ot(record,cfg,destination,*,expected=None,gt=False):
    destination=Path(destination)
    hashes=(mot_members if record['dataset']=='MOT17' else dance_members)(record,cfg,destination,gt=gt)
    if expected is not None:
        table=expected['gt_files'] if gt else expected['files']
        if hashes!=table:raise ValueError('global frozen OT source/GT byte mismatch')
    if gt:return hashes
    ini=configparser.ConfigParser()
    if not ini.read(destination/'seqinfo.ini'):raise ValueError('missing native metadata')
    plan=native_plan(ini['Sequence'],record['frames'])
    if plan['name']!=record['name']:raise ValueError('native sequence name drift')
    if expected is not None and plan!=expected['native']:raise ValueError('native source metadata drift')
    return dict(record,files=hashes,native=plan)


def freeze_all(base):
    cfg,digest=protocol();base=Path(base);base.mkdir();ar=[];od=[];ot=[];all_selected={}
    from adaptive_vcm.data import read_image
    from adaptive_vcm.v34r1.frames import read_selected
    def ar_one(item):
        first=mounted(item['id']+'/img_00001.jpg')
        if first is None:raise FileNotFoundError('missing predeclared AR source; no substitution: '+item['id'])
        folder=first.parent;files=[p.name for p in folder.iterdir() if p.is_file()]
        window,selected=provider_window(files,item['frame_count_metadata']);hashes={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in window}
        row=dict(id=item['id'],label=item['label'],kind='frames',path=str(folder),split='test',provider_original_frames=item['frame_count_metadata'],
            provider_window_files=window,provider_selected_paths=[item['id']+'/'+n for n in selected],file_sha256=hashes)
        rgb,_=read_selected(row,frames=16,size=128);row['canonical_sha256']=hashlib.sha256(rgb.tobytes()).hexdigest()
        print('A8_AR_FREEZE',item['id'],flush=True);return row
    with ThreadPoolExecutor(max_workers=8) as pool:ar=list(pool.map(ar_one,cfg['ar']['records']))
    for row in ar:
        for name in row['provider_selected_paths']:
            h=row['file_sha256'][Path(name).name]
            if h in all_selected and all_selected[h]!=row['id']:raise ValueError('duplicate selected physical AR frame across video IDs')
            all_selected[h]=row['id']
    annotation=mounted('annotations/instances_val2017.json')
    if annotation is None:raise FileNotFoundError('COCO annotations unavailable')
    require_sha256(annotation.read_bytes(),cfg['od']['annotation_sha256'])
    def od_one(item):
        source=mounted('val2017/'+item['file_name'])
        if source is None:raise FileNotFoundError('missing declared COCO image; no substitution')
        row=dict(item,path=str(source),split='a8_test1000',file_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        rgb,_=read_image(row,320);row['canonical_sha256']=hashlib.sha256(rgb.tobytes()).hexdigest();return row
    with ThreadPoolExecutor(max_workers=8) as pool:od=list(pool.map(od_one,cfg['od']['records']))
    for field in ('file_sha256','canonical_sha256'):
        if len({r[field] for r in od})!=1000:raise ValueError('duplicate OD byte/canonical image')
    for item in cfg['ot']['records']:
        dest=ROOT/'data/v34r1_a8/freeze-ot'/item['name'];row=restore_ot(item,cfg,dest);row['gt_files']=restore_ot(item,cfg,dest,gt=True);ot.append(row)
        print('A8_OT_FREEZE',item['name'],len(row['files']),row['native'],flush=True)
    result=dict(schema='v34r1-A8-sources-1',selection_protocol_sha256=digest,qps=cfg['qps'],codecs=cfg['codecs'],ar=ar,od=od,ot=ot,
        physical_counts=dict(ar=1000,od=1000,ot=10),source_only=True,model_inferences=0,codec_trials=0,gt_metrics=0)
    data=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode();(base/'source-manifest.json').write_bytes(data)
    (base/'freeze-complete.json').write_text(json.dumps(dict(complete=True,manifest_sha256=hashlib.sha256(data).hexdigest(),counts=result['physical_counts'],source_only=True),indent=2)+'\n')
    print('A8_GLOBAL_SOURCE_FREEZE_COMPLETE',hashlib.sha256(data).hexdigest(),flush=True)


def restore_ar(row):
    folder=mounted(row['id']+'/img_00001.jpg')
    if folder is None:raise FileNotFoundError('declared mounted AR source missing')
    destination=ROOT/'data/v34r1_a8/ar'/row['id'];destination.mkdir(parents=True,exist_ok=False)
    restored=dict(row,path=str(destination))
    for name,h in row['file_sha256'].items():
        raw=(folder.parent/name).read_bytes();require_sha256(raw,h);safe_target(destination,name).write_bytes(raw)
    return restored


def restore_od(row):
    path=mounted('val2017/'+row['file_name'])
    if path is None:raise FileNotFoundError('declared mounted OD source missing')
    require_sha256(path.read_bytes(),row['file_sha256']);return dict(row,path=str(path))

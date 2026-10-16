"""Kaggle-only authenticated subset fetch; GPU caches never extract GT."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import time
import zipfile
import requests
import prepare_v34r1_mot_fast as transport
from prepare_v34r1_data import HTTPRangeFile
from adaptive_vcm.v34r1_a4.cloud_record import require_sha256

ROOT=Path(__file__).resolve().parents[1]


def restore(destination,names,*,metadata_only=False):
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'):raise ValueError('Kaggle-only source preparation')
    destination=Path(destination).resolve()
    if destination.exists():raise ValueError('fresh source extraction required')
    destination.mkdir(parents=True)
    protocol=json.loads((ROOT/'configs/v34r1-a6-ot21.json').read_text())
    raw=(ROOT/'configs/v34r1-a3-mot17-sources.json.gz').read_bytes();require_sha256(raw,protocol['source_catalog_sha256']);catalog=json.loads(gzip.decompress(raw))
    raw=(ROOT/'configs/v34r1-a3-mot17-byte-manifest.json.gz').read_bytes();require_sha256(raw,protocol['source_byte_manifest_sha256']);manifest=json.loads(gzip.decompress(raw))
    allowed={r['name'] for r in protocol['pilot_sequences']}
    if not names or len(set(names))!=len(names) or not set(names)<=allowed:raise ValueError('declared A6 DEV physical videos required')
    remote=HTTPRangeFile(catalog['url'])
    if remote.length!=catalog['archive_bytes']:raise ValueError('official archive length drift')
    transport.MOT=catalog['url'];frozen=[]
    for name in names:
        seq=next(r for r in catalog['sequences'] if r['name']==name);old=next(r for r in manifest['sequences'] if r['name']==name);jobs=[];entries=[]
        for e in seq['members']:
            member=e['member']
            keep=member in ('seqinfo.ini','gt/gt.txt') if metadata_only else member=='seqinfo.ini' or (member.startswith('img1/') and int(Path(member).stem)<=64)
            if not keep:continue
            target=(destination/name/member).resolve()
            if not target.is_relative_to(destination):raise ValueError('source member escape')
            info=zipfile.ZipInfo(seq['prefix']+member)
            for field,value in [('file_size',e['file_size']),('compress_size',e['compress_size']),('compress_type',e['compress_type']),('CRC',e['crc32']),('header_offset',e['header_offset'])]:setattr(info,field,value)
            jobs.append((info,target,catalog['archive_bytes']));entries.append(member)
        if len(entries)!=(2 if metadata_only else 65):raise ValueError('complete source64 or metadata-only members required')
        def fetch(job):
            for attempt in range(6):
                try:return transport.fetch_member(job)
                except requests.RequestException:
                    if attempt==5:raise
                    time.sleep(min(30,2**attempt))
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(fetch,jobs))
        hashes={}
        for member in entries:
            data=(destination/name/member).read_bytes();require_sha256(data,old['files'][member]);hashes[member]=hashlib.sha256(data).hexdigest()
        frozen.append(dict(name=name,split=old['split'],frames=64,files=hashes))
        print('A6_MOT17_READY',name,len(hashes),'metadata_only',metadata_only,flush=True)
    result=dict(source_url=catalog['url'],sequences=frozen,metadata_only=metadata_only,gt_extracted=metadata_only)
    (destination/'source-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

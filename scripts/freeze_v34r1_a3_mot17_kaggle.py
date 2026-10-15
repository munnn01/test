"""Fetch only frozen original MOT17 members on Kaggle, CRC/SHA check, private cache."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_v34r1_mot_fast as transport
from prepare_v34r1_data import HTTPRangeFile


def main():
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'): raise ValueError('Kaggle-only source preparation')
    work=Path('/kaggle/working'); cache=Path('/kaggle/temp/mot17-source-cache'); cache.mkdir()
    raw=gzip.decompress((ROOT/'configs/v34r1-a3-mot17-sources.json.gz').read_bytes())
    protocol=json.loads(raw)
    if protocol['qps']!=[30,35,40,45]: raise ValueError('source protocol QP drift')
    remote=HTTPRangeFile(protocol['url'])
    if remote.length!=protocol['archive_bytes']: raise ValueError('official archive byte length changed')
    transport.MOT=protocol['url']
    manifest=dict(schema='v34r1-A3-MOT17-byte-freeze-1',source_protocol_sha256=hashlib.sha256(raw).hexdigest(),
        source_url=protocol['url'],sequences=[],scope='source preparation only; zero model inference or codec trials')
    for sequence in protocol['sequences']:
        dest=cache/sequence['name']; jobs=[]
        for entry in sequence['members']:
            target=(dest/entry['member']).resolve()
            if not target.is_relative_to(cache): raise ValueError('frozen source path escape')
            info=zipfile.ZipInfo(sequence['prefix']+entry['member'])
            info.file_size=entry['file_size']; info.compress_size=entry['compress_size']; info.compress_type=entry['compress_type']
            info.CRC=entry['crc32']; info.header_offset=entry['header_offset']
            jobs.append((info,target,protocol['archive_bytes']))
        def fetch(job):
            for attempt in range(6):
                try: return transport.fetch_member(job)
                except requests.RequestException:
                    if attempt==5: raise
                    time.sleep(min(30,2**attempt))
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i,_ in enumerate(pool.map(fetch,jobs),1):
                if i%64==0: print('MOT17_SOURCE_PROGRESS',sequence['name'],i,'/',len(jobs),flush=True)
        if hashlib.sha256((dest/'seqinfo.ini').read_bytes()).hexdigest()!=sequence['seqinfo_sha256']:
            raise ValueError('original seqinfo hash differs from predeclared ZIP catalog')
        frozen=dict(name=sequence['name'],split=sequence['split'],frames=sequence['frames'],files={})
        for entry in sequence['members']:
            path=dest/entry['member']; frozen['files'][entry['member']]=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest['sequences'].append(frozen)
        print('MOT17_SOURCE_READY',sequence['name'],sequence['split'],sequence['frames'],flush=True)
    original_hashes=[]; owners={}; repeated_within_sequence=0
    for sequence in manifest['sequences']:
        for name,sha in sequence['files'].items():
            if not name.startswith('img1/'): continue
            original_hashes.append(sha)
            if sha in owners:
                if owners[sha]!=sequence['name']: raise ValueError('duplicate exact raw frame across physical sequences; no source replacement')
                repeated_within_sequence+=1
            owners[sha]=sequence['name']
    manifest_bytes=(json.dumps(manifest,indent=2)+'\n').encode()
    (work/'mot17-source-manifest.json').write_bytes(manifest_bytes)
    with zipfile.ZipFile(work/'mot17-source-cache.zip','w',compression=zipfile.ZIP_STORED) as archive:
        for sequence in manifest['sequences']:
            for member in sequence['files']:
                path=cache/sequence['name']/member
                archive.write(path,arcname=sequence['name']+'/'+member)
        archive.writestr('source-manifest.json',manifest_bytes)
    archive=work/'mot17-source-cache.zip'
    status=dict(complete=True,source_only=True,model_inferences=0,codec_trials=0,
        source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),cache_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        cache_bytes=archive.stat().st_size,frames=len(original_hashes),original_exact_repeats_within_sequence=repeated_within_sequence,
        sequences=[dict(name=r['name'],split=r['split'],frames=r['frames']) for r in manifest['sequences']])
    (work/'mot17-source-status.json').write_text(json.dumps(status,indent=2)+'\n')
    print('MOT17_SOURCE_COMPLETE',json.dumps(status),flush=True)


if __name__=='__main__': main()

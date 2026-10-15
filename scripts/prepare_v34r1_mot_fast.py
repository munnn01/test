"""Fetch selected official MOT20 members in parallel with ZIP CRC validation."""
from concurrent.futures import ThreadPoolExecutor
import binascii
import json
from pathlib import Path
import struct
import threading
import zipfile
import zlib
import requests
from prepare_v34r1_data import HTTPRangeFile,MOT

LOCAL=threading.local()


def fetch_member(job):
    info,destination,total=job
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        raw=destination.read_bytes()
        if len(raw)==info.file_size and binascii.crc32(raw)&0xffffffff==info.CRC: return
    if not hasattr(LOCAL,'session'): LOCAL.session=requests.Session()
    start=info.header_offset
    end=min(total-1,start+30+len(info.filename.encode('utf-8'))+len(info.extra)+info.compress_size+1023)
    for attempt in range(3):
        response=LOCAL.session.get(MOT,headers={'Range':f'bytes={start}-{end}'},timeout=(20,90))
        response.raise_for_status()
        if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{total}':
            raise ValueError('MOT archive range mismatch')
        raw=response.content
        header=struct.unpack(zipfile.structFileHeader,raw[:zipfile.sizeFileHeader])
        offset=zipfile.sizeFileHeader+header[zipfile._FH_FILENAME_LENGTH]+header[zipfile._FH_EXTRA_FIELD_LENGTH]
        if raw[:4]!=zipfile.stringFileHeader or offset+info.compress_size>len(raw):
            raise ValueError('local ZIP header length mismatch')
        compressed=raw[offset:offset+info.compress_size]
        if info.compress_type==zipfile.ZIP_STORED: decoded=compressed
        elif info.compress_type==zipfile.ZIP_DEFLATED: decoded=zlib.decompress(compressed,-15)
        else: raise ValueError('unsupported official MOT member compression')
        if len(decoded)!=info.file_size or binascii.crc32(decoded)&0xffffffff!=info.CRC:
            if attempt<2: continue
            raise ValueError('ZIP size or CRC mismatch')
        temporary=destination.with_name(destination.name+'.fast.part')
        temporary.write_bytes(decoded); temporary.replace(destination)
        return


def main():
    root=Path('data/v34r1/mot20').resolve()
    reader=HTTPRangeFile(MOT)
    with zipfile.ZipFile(reader) as archive:
        index={info.filename:info for info in archive.infolist()}
    for sequence in ('MOT20-01','MOT20-03','MOT20-05','MOT20-02'):
        prefix=next(name[:-len('seqinfo.ini')] for name in index if name.endswith(sequence+'/seqinfo.ini'))
        members=['seqinfo.ini','gt/gt.txt',*[f'img1/{i:06d}.jpg' for i in range(1,257)]]
        jobs=[(index[prefix+member],root/sequence/member,reader.length) for member in members]
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i,_ in enumerate(pool.map(fetch_member,jobs),1):
                if i%64==0: print('MOT20 CRC-verified',sequence,i,'/',len(jobs),flush=True)
        (root/sequence/'v34r1-download-complete.json').write_text(json.dumps({'frames':256,'source':MOT,'crc_verified':True}))
        print('Ready:',sequence,flush=True)


if __name__=='__main__': main()

"""Kaggle CPU data-only freeze of all512 sources before any A6 AR inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from adaptive_vcm.v34r1_a6.ar_source import provider_window,historical_parity
from adaptive_vcm.v34r1_a4.cloud_record import record_cloud_run,require_sha256
from prepare_v34r1_cloud_data import mounted
from run_v34r1_kaggle import run


def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);args=p.parse_args()
    import subprocess
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'):raise ValueError('Kaggle-only data freeze')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=args.commit:raise ValueError('immutable revision required')
    base=Path('/kaggle/working/a6-ar512-sources')
    with record_cloud_run(base,base.parent/'v34r1-a6-ar512-sources.tgz',dict(commit=args.commit,source_only=True,model_inferences=0,codec_trials=0,test_scored=False)):
        base.mkdir();run([sys.executable,'-m','unittest','tests.test_v34r1_a6_ar_source','-v'],log=base/'source-tests.log')
        raw=(ROOT/'configs/v34r1-a6-ar512-selection.json').read_bytes();protocol=json.loads(raw)
        if protocol['count']!=512 or protocol['frames']!=16 or protocol['stride']!=2:raise ValueError('fixed AR512 protocol required')
        annotations=(ROOT/'results/v34_r1/protocol/annotations.csv').read_bytes();require_sha256(annotations,protocol['annotation_sha256'])
        labels={r['youtube_id']:int(r['label']) for r in csv.DictReader(io.StringIO(annotations.decode('utf-8-sig')))}
        old=json.loads((ROOT/'results/v34r1/data-manifest.json').read_text());old_rows=[r for r in old['ar'] if r['split']=='test'];old_by_id={r['id']:r for r in old_rows}
        items=protocol['records'];ids=[r['youtube_id'] for r in items]
        if len(ids)!=512 or len(set(ids))!=512 or ids[:128]!=[r['id'] for r in old_rows]:raise ValueError('exact old128 prefix/new384 disjoint IDs required')
        def freeze(pair):
            index,item=pair;yt=item['youtube_id']
            if item['directory']!=yt or item['label']!=labels[yt]:raise ValueError('frozen identity/label mapping drift')
            first=mounted(yt+'/img_00001.jpg')
            if first is None:raise FileNotFoundError('frozen provider directory missing; never replace: '+yt)
            folder=first.parent;all_files=[p.name for p in folder.iterdir() if p.is_file()]
            if any(p.is_dir() for p in folder.iterdir()):raise ValueError('unexpected nested provider directory')
            window,selected=provider_window(all_files,item['frame_count_metadata']);hashes={};sizes={}
            for name in window:
                data=(folder/name).read_bytes();hashes[name]=hashlib.sha256(data).hexdigest();sizes[name]=len(data)
                from PIL import Image
                with Image.open(io.BytesIO(data)) as image:
                    image.verify()
            authenticated_old_files=0
            if yt in old_by_id:authenticated_old_files=historical_parity(hashes,[yt+'/'+n for n in selected],old_by_id[yt])
            result=dict(id=yt,youtube_id=yt,label=item['label'],path=str(ROOT/'data/v34r1_a6/ar512'/yt),kind='frames',split='test',
                subset='original128' if index<128 else 'additional384',provider_original_frames=item['frame_count_metadata'],provider_window_files=window,
                provider_selected_paths=[yt+'/'+n for n in selected],file_sha256=hashes,file_bytes=sizes,rank=item['rank'],
                historical_authenticated_window_file_count=authenticated_old_files)
            print('A6_AR_SOURCE',index+1,yt,len(hashes),flush=True);return result
        with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(freeze,enumerate(items)))
        owners={};within=0
        for r in records:
            local=set()
            for selected in r['provider_selected_paths']:
                sha=r['file_sha256'][Path(selected).name]
                if sha in owners and owners[sha]!=r['id']:raise ValueError('identical selected JPEG across physical video IDs; preserve failure without substitution')
                within+=sha in local;local.add(sha);owners[sha]=r['id']
        manifest=dict(schema='v34r1-A6-AR512-sources-1',frozen_utc=datetime.now(timezone.utc).isoformat(),selection_protocol_sha256=hashlib.sha256(raw).hexdigest(),
            qps=protocol['qps'],codecs=protocol['codecs'],ar=records,od=[],ot=[],source_count=512,selected_frames=8192,authenticated_window_files=16384,
            byte_identical_selected_frames_within_video=within,source_cadence=protocol['source_cadence'],model_inferences=0,codec_trials=0,test_scored=False)
        data=(json.dumps(manifest,indent=2)+'\n').encode();(base/'source-manifest.json').write_bytes(data)
        (base/'freeze-complete.json').write_text(json.dumps(dict(complete=True,source_manifest_sha256=hashlib.sha256(data).hexdigest(),
            original128_exact_source_byte_parity=True,physical_video_ids=512,selected_frames=8192,window_files=16384,model_inferences=0,codec_trials=0),indent=2)+'\n')
        print('A6_AR512_SOURCE_FREEZE_COMPLETE',hashlib.sha256(data).hexdigest(),flush=True)


if __name__=='__main__':main()

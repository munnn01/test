"""Aggregate sealed OD predictions on Kaggle; descriptive point estimates only."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import time

PIN='db6d8b5b162d6e0303fc9a94422c32648d34c98d'
SOURCE_SHA='c1418f2ca848dc1cd51c1a09a0ca7030f35f7ad5e76fe8d0215eb3015fe7679b'
SHARDS=(0,1,2)
ROOT=Path('/kaggle/temp/adaptive_preprocessing')

def validate_rows(rows,ids):
    from adaptive_vcm.v34r1_a8.contracts import audit_selected
    audit_selected(rows,['od/coco2017/'+str(i) for i in ids])
    lookup={}
    for r in rows:
        if (r['image_id'] not in ids or r['id']!='od/coco2017/'+str(r['image_id'])
            or not math.isclose(r['bpp'],r['bytes']*8/(320*320),rel_tol=0.,abs_tol=1e-12)
            or any(p['image_id']!=r['image_id'] for p in r['coco'])
            or (r['arm']=='anchor' and r['action']!=0)):
            raise ValueError('prediction/source identity, rate or identity-anchor drift')
        lookup[(r['image_id'],r['codec'],r['qp'],r['arm'])]=r
    for r in rows:
        if r['bytes']>lookup[(r['image_id'],r['codec'],r['qp'],'anchor')]['bytes']:
            raise ValueError('selected adaptive paid rate exceeds paired identity')
    return len(rows)

def read(p):return json.loads(p.read_bytes())
def sha(p):
    digest=hashlib.sha256()
    with p.open('rb') as stream:
        while block:=stream.read(4*1024*1024):digest.update(block)
    return digest.hexdigest()

def unpack(archive,target,expected_sha):
    if sha(archive)!=expected_sha or target.exists():raise ValueError('exact archive hash and fresh staging required')
    target.mkdir()
    with tarfile.open(archive) as tar:
        seen=set()
        for m in tar:
            dest=(target/m.name).resolve()
            if not m.isfile() or not dest.is_relative_to(target.resolve()) or dest in seen:raise ValueError('unsafe/duplicate input member')
            seen.add(dest);dest.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as src,dest.open('xb') as dst:
                while block:=src.read(4*1024*1024):dst.write(block)
    manifest=read(target/'bundle-manifest.json')
    expected={p['path']:p for p in manifest['files']}
    actual={p.relative_to(target).as_posix():p for p in target.rglob('*') if p.is_file() and p.name!='bundle-manifest.json'}
    if len(expected)!=len(manifest['files']) or set(expected)!=set(actual):raise ValueError('exact bundle inventory required')
    for name,path in actual.items():
        if path.stat().st_size!=expected[name]['bytes'] or sha(path)!=expected[name]['sha256']:raise ValueError('bundle member byte drift')
    if manifest['producer_commit']!=PIN or manifest['source_manifest_sha256']!=SOURCE_SHA or manifest['shards']!=list(SHARDS):raise ValueError('snapshot pin drift')
    return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--archive-sha256',required=True);p.add_argument('--annotation',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not ROOT.is_dir() or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=PIN:raise ValueError('exact pinned Kaggle checkout required')
    if not str(a.out).startswith('/kaggle/working/') or a.out.exists():raise ValueError('fresh Kaggle output required')
    sys.path.insert(0,str(ROOT))
    import numpy as np
    from adaptive_vcm.v34r1_a4.cloud_record import record_cloud_run
    from adaptive_vcm.v34r1_a8.merge import validate_receipt,verify_packets
    from adaptive_vcm.v34r1_a3.analyze import metric
    from adaptive_vcm.v34r1.metrics import assess_curve
    status=dict(producer_commit=PIN,shards=list(SHARDS),source_manifest_sha256=SOURCE_SHA,scope='OD750 from complete shards0/1/2; descriptive point estimates',new_model_inferences=0,new_encodes=0,new_decodes=0,retuned=False)
    with record_cloud_run(a.out,a.out.with_suffix('.tgz'),status):
        a.out.mkdir();started=time.perf_counter()
        manifest=unpack(a.archive,Path('/kaggle/temp/a8-od-inputs750'),a.archive_sha256)
        cfg=read(ROOT/'configs/v34r1-a8-expansion.json');central=ROOT/'results/v34r1_a8/source-freeze/source-manifest.json'
        if sha(central)!=SOURCE_SHA:raise ValueError('global source freeze drift')
        policy_raw=(ROOT/cfg['od']['policy_path']).read_bytes()
        if hashlib.sha256(policy_raw).hexdigest()!=cfg['od']['policy_sha256']:raise ValueError('frozen policy drift')
        policy=json.loads(policy_raw)['policy'];all_rows=[];receipts=[];all_ids=[]
        source_protocol_sha=sha(ROOT/'configs/v34r1-a8-expansion.json')
        for i in SHARDS:
            folder=Path('/kaggle/temp/a8-od-inputs750')/f'shard{i}';cloud=read(folder/'cloud-status.json');tasks=read(folder/'task-status.json');complete=read(folder/'complete.json')
            if validate_receipt(cloud,tasks,complete,PIN,SOURCE_SHA)!=i:raise ValueError('shard receipt mismatch')
            od=folder/'od';report=read(od/'evaluation-report.json');src_raw=(od/'source-manifest.json').read_bytes();src=json.loads(src_raw)
            expected=[r['id'] for j,r in enumerate(cfg['od']['records']) if j%4==i]
            if (len(src['images'])!=250 or {r['id'] for r in src['images']}!=set(expected)
                or src['role']!='a8_shard250' or src['source_id_protocol_sha256']!=source_protocol_sha
                or src['frozen_policy_sha256']!=cfg['od']['policy_sha256']):raise ValueError('declared250 source partition drift')
            for w in (0,1):
                worker=od/f'worker{w}';done=read(worker/'complete.json');wp=read(worker/'protocol.json')
                worker_ids=[r['id'] for j,r in enumerate(src['images']) if j%2==w]
                if (done['source_inferences']!=125 or done['pseudo_loss_forwards']!=0 or not done['reproducibility']['exact_packet_and_pixels']
                    or done['grid']['images']!=125 or done['grid']['logical_rows']!=21000 or set(done['grid']['source_ids'])!=set(worker_ids)
                    or wp['manifest_sha256']!=hashlib.sha256(src_raw).hexdigest()):raise ValueError('worker source/grid/repeat check failed')
            decision_raw=gzip.decompress((od/'selected-streams.json.gz').read_bytes());decision=json.loads(decision_raw);rows=decision['rows']
            if (hashlib.sha256(decision_raw).hexdigest()!=report['source_only_decisions_sha256']
                or decision['frozen_policy_sha256']!=report['frozen_policy_sha256'] or report['frozen_policy_sha256']!=cfg['od']['policy_sha256']
                or report['policy']!=policy or report['candidate_rows']!=42000 or report['selected_rows']!=4000 or report['complete'] is not True):raise ValueError('sealed decisions/policy/count drift')
            validate_rows(rows,expected);verify_packets(od,rows);all_rows.extend(rows);all_ids.extend(expected)
            receipts.append(dict(shard=i,images=len(expected),decision_sha256=report['source_only_decisions_sha256'],cloud=cloud))
        if len(all_ids)!=750 or len(set(all_ids))!=750:raise ValueError('exact750 distinct images required')
        validate_rows(all_rows,all_ids)
        # Access official GT only after all sealed payloads/bytes/IDs pass.
        if sha(a.annotation)!=cfg['od']['annotation_sha256']:raise ValueError('original COCO annotation drift')
        meta=read(a.annotation);gt=defaultdict(list);allowed=set(all_ids)
        for row in meta['annotations']:
            if row['image_id'] in allowed:gt[row['image_id']].append(row)
        reserved={r['id'] for r in cfg['od']['records'][:512]}
        groups=dict(available750=all_ids,established_reserved384_of512=[i for i in all_ids if i in reserved],additional366_of488=[i for i in all_ids if i not in reserved])
        if [len(v) for v in groups.values()]!=[750,384,366]:raise ValueError('declared subgroup count drift')
        results={};metric_calls=0;table=[]
        for name,ids in groups.items():
            allowed=set(ids);curves={c:{arm:dict(rate=[],quality=[]) for arm in ('anchor','adaptive')} for c in ('h264','h265')};details=[]
            for c in ('h264','h265'):
                for q in (30,35,40,45):
                    for arm in ('anchor','adaptive'):
                        rows=[r for r in all_rows if r['image_id'] in allowed and r['codec']==c and r['qp']==q and r['arm']==arm]
                        if len(rows)!=len(ids):raise ValueError('incomplete subset condition')
                        scored=metric([dict(p) for r in rows for p in r['coco']],sorted(ids),meta,gt);metric_calls+=1
                        rate=float(np.mean([r['bytes']*8/(320*320) for r in rows]));curves[c][arm]['rate'].append(rate);curves[c][arm]['quality'].append(scored['AP'])
                        details.append(dict(codec=c,qp=q,arm=arm,**scored));table.append(dict(subset=name,count=len(ids),codec=c,qp=q,arm=arm,paid_bpp=rate,**scored))
            results[name]=dict(count=len(ids),curves=curves,metrics=details,comparisons={c:assess_curve(curves[c]['anchor'],curves[c]['adaptive'],qps=(30,35,40,45)) for c in curves})
            print('OD_MERGE_SUBSET',name,len(ids),json.dumps(results[name]['comparisons']),flush=True)
        report=dict(complete=True,source_commit=PIN,source_manifest_sha256=SOURCE_SHA,input_archive_sha256=a.archive_sha256,input_manifest_sha256=sha(Path('/kaggle/temp/a8-od-inputs750/bundle-manifest.json')),
            merge_script_sha256=sha(Path(__file__)),shards=list(SHARDS),scope='Partial750/1000 fixed A8 images; pooled original COCO AP, no averaging shard AP or BD-rate',
            qps=[30,35,40,45],codecs=['h264','h265'],counts=dict(images=750,selected_rows=len(all_rows),candidate_rows=126000,official_AP_aggregations=metric_calls),
            subsets=results,receipts=receipts,frozen_policy_sha256=cfg['od']['policy_sha256'],od_provenance=cfg['od']['exposure_audit'],retuned=False,
            new_model_inferences=0,new_encodes=0,new_decodes=0,ci=None,threshold_statistically_confirmed=False,wall_seconds=time.perf_counter()-started)
        if metric_calls!=48:raise ValueError('all48 pooled conditions required')
        (a.out/'od-merged-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        with (a.out/'od-merged-curves.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        print('OD750_MERGE_COMPLETE',flush=True)

if __name__=='__main__':main()

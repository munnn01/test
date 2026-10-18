"""Kaggle-only complete-cohort metrics; never average nonlinear shard AP/HOTA."""
import argparse
from collections import defaultdict,Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from ..v34r1.metrics import assess_curve
from .contracts import QPS,CODECS,audit_selected

ROOT=Path(__file__).resolve().parents[2]


def validate_receipt(status,tasks,complete,expected_commit,source_sha):
    i=status.get('shard')
    if (status.get('complete') is not True or type(i) is not int or i not in range(4)
        or status.get('commit')!=expected_commit or status.get('source_manifest_sha256')!=source_sha):raise ValueError('complete pinned shard receipt required')
    if set(tasks)!=set(('ar','od','ot')) or any(t.get('complete') is not True for t in tasks.values()):raise ValueError('all task statuses must complete')
    if (complete.get('complete') is not True or complete.get('shard')!=i or complete.get('ar')!=250 or complete.get('od')!=250
        or complete.get('ot')!=sum(j%4==i for j in range(10)) or complete.get('tasks')!=tasks):raise ValueError('complete shard count/task receipt mismatch')
    return i


def check_partition(parts,expected):
    if set(parts)!=set(range(4)) or len(expected)!=len(set(expected)):raise ValueError('four exact declared source shards required')
    for i,ids in parts.items():
        if len(ids)!=len(set(ids)) or set(ids)!={r for j,r in enumerate(expected) if j%4==i}:raise ValueError('source shard identity overlap/missing/drift')
    return len(expected)


def pool_hota(metric,stats):
    converted={n:{k:np.asarray(v) if isinstance(v,list) else v for k,v in r.items()} for n,r in stats.items()}
    if not converted:raise ValueError('nonempty sequence statistics required')
    combined=metric.combine_sequences(converted)
    return {k:float(np.mean(combined[k])) for k in ('HOTA','DetA','AssA')}


def read(path):return json.loads(path.read_bytes())
def gz(path):return json.loads(gzip.decompress(path.read_bytes()))


def verify_packets(base,rows):
    checked={};base=Path(base).resolve()
    for row in rows:
        path=(base/row['packet_path']).resolve()
        if not path.is_relative_to(base):raise ValueError('selected packet path escape')
        if path not in checked:
            raw=path.read_bytes()
            checked[path]=(len(raw),hashlib.sha256(raw).hexdigest())
        if checked[path]!=(row['bytes'],row['packet_sha256']):raise ValueError('selected wire file mismatch')


def comparison(curves,ci=None):return {c:assess_curve(curves[c]['anchor'],curves[c]['adaptive'],qps=QPS,ci=(ci or {}).get(c)) for c in CODECS}


def paired_ci(make_curves,groups,*,draws,count,unit):
    rng=np.random.default_rng(303401);values={c:[] for c in CODECS};invalid={c:Counter() for c in CODECS}
    for _ in range(draws):
        sample=[i for group in groups for i in rng.choice(group,len(group),replace=True).tolist()]
        curves=make_curves(sample)
        for c,result in comparison(curves).items():
            if result['valid']:values[c].append(result['pchip_bd_rate_pct'])
            else:invalid[c][result['invalid_reason']]+=1
    result={};alpha=.05/6
    for c in CODECS:
        missing=draws-len(values[c]);lo=float(np.quantile(values[c]+[-np.inf]*missing,alpha,method='lower'));hi=float(np.quantile(values[c]+[np.inf]*missing,1-alpha,method='higher'))
        result[c]=dict(lo=lo if math.isfinite(lo) else None,hi=hi if math.isfinite(hi) else None,draws=draws,seed=303401,
            finite_fraction=len(values[c])/draws,invalid_reasons=dict(invalid[c]),familywise_endpoints=6,independent_unit=unit,independent_count=count,
            one_sided_confidence=1-alpha,invalid_draw_policy='worst-case-unbounded',resampling='paired within declared class/dataset strata')
    return result


def ar_metrics(rows,cfg,draws):
    ids=['ar/'+r['id'] for r in cfg['ar']['records']];audit_selected(rows,ids);lookup={(r['id'],r['codec'],r['qp'],r['arm']):r for r in rows}
    rate={};quality={}
    for c in CODECS:
        for arm in ('anchor','adaptive'):
            rate[(c,arm)]=np.array([[lookup[(i,c,q,arm)]['bpp'] for q in QPS] for i in ids]);quality[(c,arm)]=np.array([[lookup[(i,c,q,arm)]['correct'] for q in QPS] for i in ids])
    def curves(indices):return {c:{arm:dict(rate=rate[(c,arm)][indices].mean(0).tolist(),quality=quality[(c,arm)][indices].mean(0).tolist()) for arm in ('anchor','adaptive')} for c in CODECS}
    all_curves=curves(list(range(1000)));groups=defaultdict(list)
    for i,row in enumerate(cfg['ar']['records']):groups[row['label']].append(i)
    ci=paired_ci(curves,list(groups.values()),draws=draws,count=1000,unit='video') if draws else None
    return dict(count=1000,classes=400,curves=all_curves,comparisons=comparison(all_curves,ci),ci=ci)


def od_metrics(rows,cfg,annotation):
    from ..v34r1_a3.analyze import metric
    raw=annotation.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=cfg['od']['annotation_sha256']:raise ValueError('official COCO annotations drift')
    meta=json.loads(raw);ids=[r['id'] for r in cfg['od']['records']];audit_selected(rows,['od/coco2017/'+str(i) for i in ids]);gt=defaultdict(list)
    allowed_ids=set(ids)
    for a in meta['annotations']:
        if a['image_id'] in allowed_ids:gt[a['image_id']].append(a)
    groups={'full1000':ids,'established_reserved512':ids[:512],'additional488':ids[512:]};reports={}
    for name,group in groups.items():
        allowed=set(group);curves={c:{arm:dict(rate=[],quality=[]) for arm in ('anchor','adaptive')} for c in CODECS}
        for c in CODECS:
            for q in QPS:
                for arm in ('anchor','adaptive'):
                    subset=[r for r in rows if r['image_id'] in allowed and r['codec']==c and r['qp']==q and r['arm']==arm]
                    if len(subset)!=len(group):raise ValueError('missing OD subset condition')
                    curves[c][arm]['rate'].append(float(np.mean([r['bpp'] for r in subset])))
                    curves[c][arm]['quality'].append(metric([dict(p) for r in subset for p in r['coco']],sorted(group),meta,gt)['AP'])
        reports[name]=dict(count=len(group),curves=curves,comparisons=comparison(curves))
    return dict(subsets=reports,provenance=cfg['od']['exposure_audit'],independent_confirmation_scope='established_reserved512 only;475 reopened-partition additions are provenance-limited',
        ci=None,ci_status='image-cluster AP bootstrap requires separate compute; no confidence confirmation claimed')


def ot_metrics(videos,cfg,package,draws):
    ordered=[r['name'] for r in cfg['ot']['records']];metric=package.metrics.HOTA();reports={}
    def curves(indices):
        current=[ordered[i] for i in indices];result={c:{arm:dict(rate=[],quality=[]) for arm in ('anchor','adaptive')} for c in CODECS}
        for c in CODECS:
            for q in QPS:
                for arm in ('anchor','adaptive'):
                    tracker=f'{c}_{q}_{arm}';stats={str(i):videos[n]['raw'][tracker][n]['pedestrian']['HOTA'] for i,n in enumerate(current)}
                    result[c][arm]['rate'].append(float(np.mean([videos[n]['report']['curves'][c][arm]['rate'][QPS.index(q)] for n in current])))
                    result[c][arm]['quality'].append(pool_hota(metric,stats)['HOTA'])
        return result
    groups=defaultdict(list)
    for i,row in enumerate(cfg['ot']['records']):groups[row['dataset']].append(i)
    full=curves(list(range(10)));ci=paired_ci(curves,list(groups.values()),draws=draws,count=10,unit='sequence') if draws else None
    reports['full10']=dict(count=10,curves=full,comparisons=comparison(full,ci),ci=ci)
    for dataset,indices in groups.items():
        value=curves(indices);reports[dataset]=dict(count=len(indices),curves=value,comparisons=comparison(value))
    return dict(subsets=reports,per_video={n:videos[n]['report'] for n in ordered},scope='official pooled HOTA statistics; fixed3MOT17/7DanceTrack mixture; per-dataset outcomes separate')


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,nargs=4,required=True);p.add_argument('--annotation',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--draws',type=int,default=5000);a=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'):raise ValueError('Kaggle-only actual official full-cohort metrics/CI')
    if a.out.exists() or a.draws<0:raise ValueError('fresh output and nonnegative bootstrap count required')
    cfg=read(ROOT/'configs/v34r1-a8-expansion.json');source=ROOT/'results/v34r1_a8/source-freeze/source-manifest.json';source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    parts={};commits=set();evaluator_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for folder in a.inputs:
        status=read(folder/'cloud-status.json');tasks=read(folder/'task-status.json');complete=read(folder/'complete.json')
        i=validate_receipt(status,tasks,complete,evaluator_commit,source_sha)
        if i in parts:raise ValueError('duplicate shard receipt')
        parts[i]=folder;commits.add(status['commit'])
    if set(parts)!=set(range(4)) or commits!={evaluator_commit}:raise ValueError('four complete inputs must match pinned evaluator revision')
    ar=[];od=[];videos={};ar_ids={};od_ids={};ot_ids={}
    for i,folder in sorted(parts.items()):
        ai=[]
        for w in (0,1):
            base=folder/'ar'/f'worker{w}';rows=gz(base/'rows.json.gz');sealed=gz(base/'source-only-selections.json.gz');summary=read(base/'summary.json')
            raw=gzip.decompress((base/'source-only-selections.json.gz').read_bytes())
            if hashlib.sha256(raw).hexdigest()!=summary['source_only_decisions_sha256'] or [{k:v for k,v in r.items() if k!='correct'} for r in rows]!=sealed['rows']:raise ValueError('AR decision/GT sealing mismatch')
            labels={'ar/'+r['id']:r['label'] for r in cfg['ar']['records']}
            if any(r['correct']!=int(r['predicted_label']==labels[r['id']]) for r in rows):raise ValueError('AR correctness mismatch')
            verify_packets(base,rows);ai.extend(read(base/'protocol.json')['expected_ids']);ar.extend(rows)
        ar_ids[i]=ai
        base=folder/'od';decisions=gz(base/'selected-streams.json.gz');rows=decisions['rows'];od_report=read(base/'evaluation-report.json')
        if (hashlib.sha256(gzip.decompress((base/'selected-streams.json.gz').read_bytes())).hexdigest()!=od_report['source_only_decisions_sha256']
            or decisions['frozen_policy_sha256']!=cfg['od']['policy_sha256'] or od_report['frozen_policy_sha256']!=cfg['od']['policy_sha256']):raise ValueError('OD frozen policy/decision seal drift')
        verify_packets(base,rows);od.extend(rows);od_ids[i]=[r['id'] for r in read(base/'source-manifest.json')['images']]
        names=[]
        for report_file in (folder/'ot').glob('worker*/*/evaluation-report.json'):
            base=report_file.parent;report=read(report_file);name=report['sequence']
            if name in videos or not report['complete'] or report['source_frames']!=256 or report['candidate_rows']!=2688:raise ValueError('partial/duplicate OT video')
            decisions=gz(base/'source-only-selections.json.gz');raw=gzip.decompress((base/'source-only-selections.json.gz').read_bytes())
            if hashlib.sha256(raw).hexdigest()!=report['source_only_decisions_sha256']:raise ValueError('OT decisions drift')
            audit_selected(decisions['selected_rows'],[name+'/'+str(j) for j in range(1,257,16)]);verify_packets(base,decisions['selected_rows'])
            metric_raw=read(base/'trackeval-raw.json')['results']
            for c in CODECS:
                for q in QPS:
                    for arm in ('anchor','adaptive'):
                        stream=[r for r in decisions['selected_rows'] if r['codec']==c and r['qp']==q and r['arm']==arm]
                        paid=float(np.mean([r['bytes']*8/(16*960*960) for r in stream]));quality=float(np.mean(metric_raw[f'{c}_{q}_{arm}'][name]['pedestrian']['HOTA']['HOTA']))
                        if len(stream)!=16 or not math.isclose(paid,report['curves'][c][arm]['rate'][QPS.index(q)],abs_tol=1e-12) or quality!=report['curves'][c][arm]['quality'][QPS.index(q)]:raise ValueError('OT raw packet rates/official metrics disagree with report')
            videos[name]=dict(report=report,raw=metric_raw);names.append(name)
        ot_ids[i]=names
    check_partition(ar_ids,['ar/'+r['id'] for r in cfg['ar']['records']]);check_partition(od_ids,[r['id'] for r in cfg['od']['records']]);check_partition(ot_ids,[r['name'] for r in cfg['ot']['records']])
    from ..v32.tracking import load_trackeval
    package=load_trackeval(ROOT/'data/TrackEval');a.out.mkdir()
    result=dict(scope='complete fixed1000/1000/10, no retuning, no shard metric averaging',source_commit=next(iter(commits)),source_manifest_sha256=source_sha,
        qps=list(QPS),ar=ar_metrics(ar,cfg,a.draws),od=od_metrics(od,cfg,a.annotation),ot=ot_metrics(videos,cfg,package,a.draws),primary_codecs=cfg['primary_codecs'],
        od_provenance_limited=True,bootstrap_draws_ar_ot=a.draws,all_thresholds_confirmed=False)
    (a.out/'merged-report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('A8_COMPLETE_COHORT_MERGED',flush=True)


if __name__=='__main__':main()

"""Authenticated historical OD development data and cohort-local new labels."""
from collections import defaultdict
import gc
import gzip
import hashlib
import json
from pathlib import Path
import time
from .contracts import cohort_folds
from ..v34r1_a4.cloud_record import require_sha256
from ..v34r1_a4.marginal_ap import MarginalAP
from ..v34r1_a3.core import QPS,CODECS,audit_grid
from ..v34r1_a3.analyze import metric


def read(path):return json.loads(Path(path).read_text())
def packed(path):return json.loads(gzip.decompress(Path(path).read_bytes()))
def key(row):return (row['image_id'],row['codec'],row['qp'],row['profile'])


def load_development(root,protocol):
    if Path(root)!=Path('/kaggle/temp/adaptive_preprocessing'): raise ValueError('Kaggle-only actual prediction features/labels')
    paths={}
    for name,parent in protocol['parents'].items():
        folder=root/parent['path']; raw=(folder/'publication-manifest.json').read_bytes();require_sha256(raw,parent['publication_manifest_sha256'])
        for entry in json.loads(raw)['files']:
            p=(folder/entry['path']).resolve()
            if not p.is_relative_to(folder.resolve()) or p.stat().st_size!=entry['bytes']: raise ValueError('parent evidence path/size drift')
            require_sha256(p.read_bytes(),entry['sha256'])
        paths[name]=folder
    reservation_raw=(root/'configs/v34r1-a6-od-reservation.json').read_bytes();require_sha256(reservation_raw,protocol['reservation_sha256'])
    reservation=json.loads(reservation_raw);forbidden=[r['id'] for r in reservation['selected']+reservation['confirmation256']]
    folds=cohort_folds(protocol['cohorts'],forbidden=forbidden);cohort={i:n for n,ids in protocol['cohorts'].items() for i in ids}
    rows=[];sources={}
    for name in ('a3','fresh'):
        for worker in (0,1):
            folder=paths[name]/f'worker{worker}';rows.extend(packed(folder/'rows.json.gz'))
            for s in read(folder/'sources.json'):
                if s['image_id'] in sources: raise ValueError('overlapping historical OD images')
                sources[s['image_id']]=s
    if set(sources)!=set(cohort):raise ValueError('exact three development cohorts required')
    audit_grid(rows,cohort)
    old_labels=packed(paths['labels']/'marginal-ap-labels.json.gz')
    expected_old={key(r) for r in rows if cohort[r['image_id']]!='a4_fresh'}
    if len(old_labels)!=len(expected_old) or {key(r) for r in old_labels}!=expected_old:raise ValueError('complete historical teacher labels required')
    return rows,sources,old_labels,cohort,folds,paths


def fresh_labels(rows,protocol,paths,meta,gt,out):
    """Only the consumed64-image A4 cohort enters this teacher, never a pilot/TEST."""
    out.mkdir();started=time.perf_counter();ids=sorted(protocol['cohorts']['a4_fresh']);groups=defaultdict(list)
    for r in rows:
        if r['image_id'] in ids:groups[(r['image_id'],r['codec'],r['qp'])].append(r)
    for group in groups.values():group.sort(key=lambda r:r['profile'])
    parent=read(paths['fresh']/'fresh-dev-report.json');labels=[];checks=[]
    for c in CODECS:
        for qi,q in enumerate(QPS):
            anchors={i:groups[(i,c,q)][0] for i in ids};pred=[dict(p) for i in ids for p in anchors[i]['coco']]
            teacher=MarginalAP(meta,gt,ids,pred)
            if abs(teacher.anchor_ap-parent['results']['anchor']['curves'][c]['quality'][qi])>1e-12:raise ValueError('fresh cohort official anchor parity failed')
            cache={}
            for image in ids:
                for row in groups[(image,c,q)]:
                    prediction_sha=hashlib.sha256(json.dumps(row['coco'],sort_keys=True,separators=(',',':')).encode()).hexdigest();k=(image,prediction_sha)
                    if row['profile']==0:ap=teacher.anchor_ap
                    elif k in cache:ap=cache[k]
                    else:ap=teacher.ap_with(image,row['coco']);cache[k]=ap
                    if (image,row['profile']) in ((ids[0],14),(ids[-1],20)):
                        joint=[dict(p) for other in ids for p in (row if other==image else anchors[other])['coco']]
                        full=metric(joint,ids,meta,gt)['AP'];error=abs(full-ap)
                        if error>1e-12:raise ValueError('fresh teacher/full COCO replacement parity failed')
                        checks.append(dict(codec=c,qp=q,image_id=image,profile=row['profile'],absolute_ap_error=error))
                    labels.append(dict(image_id=image,codec=c,qp=q,profile=row['profile'],anchor_cohort_ap=teacher.anchor_ap,
                        replacement_cohort_ap=ap,scaled_ap_damage=64*(teacher.anchor_ap-ap),teacher_cohort='a4_fresh'))
            print('A6_OD_LABEL',c,q,len(labels),round(time.perf_counter()-started,2),flush=True);del teacher,cache;gc.collect()
    if len(labels)!=10752 or len(checks)!=16:raise ValueError('fresh teacher grid/parity incomplete')
    raw=gzip.compress(json.dumps(labels,separators=(',',':'),allow_nan=False).encode(),mtime=0);(out/'labels.json.gz').write_bytes(raw)
    (out/'label-report.json').write_text(json.dumps(dict(cohort_ids=ids,cohort_images=64,label_rows=len(labels),label_sha256=hashlib.sha256(raw).hexdigest(),
        full_official_parity_checks=checks,pilot_scored=False,test_scored=False,wall_seconds=time.perf_counter()-started),indent=2)+'\n')
    return labels

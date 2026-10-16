"""Kaggle-only cohort-disjoint OOF choice; final models lock before new pilot."""
import argparse
from collections import defaultdict,Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .od_inputs import load_development,fresh_labels,key
from ..v34r1_a4.risk import features,predict_forest,choose_predicted
from ..v34r1_a4.od_train import portable
from ..v34r1_a4.rd_fit import select_rd_fit
from ..v34r1_a4.cloud_record import require_sha256
from ..v34r1_a3.analyze import metric
from ..v34r1_a3.core import QPS,CODECS
from ..v34r1.metrics import assess_curve

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--annotation',type=Path,required=True);args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'):raise ValueError('Kaggle-only actual OD features/training/GT scoring')
    if args.out.exists():raise ValueError('immutable new crossfit output required')
    import sklearn
    from sklearn.ensemble import ExtraTreesRegressor
    raw=(ROOT/'configs/v34r1-a6-od-crossfit.json').read_bytes();protocol=json.loads(raw)
    if sklearn.__version__!=protocol['sklearn']:raise ValueError('pinned sklearn drift')
    rows,sources,old_labels,cohort,folds,paths=load_development(ROOT,protocol);args.out.mkdir();started=time.perf_counter()
    annotation_raw=args.annotation.read_bytes();require_sha256(annotation_raw,protocol['annotation_sha256']);meta=json.loads(annotation_raw);gt=defaultdict(list)
    for a in meta['annotations']:
        if a['image_id'] in cohort:gt[a['image_id']].append(a)
    labels=old_labels+fresh_labels(rows,protocol,paths,meta,gt,args.out/'fresh-labels');targets={key(r):r['scaled_ap_damage'] for r in labels}
    rows.sort(key=lambda r:(r['image_id'],CODECS.index(r['codec']),QPS.index(r['qp']),r['profile']))
    if len(targets)!=len(labels) or set(targets)!={key(r) for r in rows}:raise ValueError('complete image/cohort/teacher alignment required')
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[key(r)[:3]].append(i)
    x=[];y=[];names=None
    for i,r in enumerate(rows):
        feature_names,values=features(r,rows[groups[key(r)[:3]][0]],sources[r['image_id']])
        if names is not None and feature_names!=names:raise ValueError('fixed feature schema drift')
        names=feature_names;x.append(values);y.append(targets[key(r)])
        if (i+1)%4096==0:print('A6_OD_FEATURES',i+1,len(rows),flush=True)
    x=np.asarray(x,np.float32);y=np.asarray(y,float)
    if len(names)!=302 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('finite fixed302 feature/label table required')
    models_dir=args.out/'models';models_dir.mkdir();oof={};final=[];model_records=[];errors={}

    def fit(spec,mask,predict_mask,tag,train_ids,heldout_ids):
        model=ExtraTreesRegressor(**spec['parameters']);model.fit(x[mask],y[mask]);encoded=portable(model,names)
        native=model.predict(x[predict_mask]);pred=np.asarray(predict_forest(encoded,x[predict_mask]),float);parity=float(np.max(np.abs(native-pred)))
        if parity>1e-10:raise ValueError('native/portable forest parity failed')
        model_raw=(json.dumps(encoded,separators=(',',':'),allow_nan=False)+'\n').encode();path=models_dir/(tag+'.json');path.write_bytes(model_raw)
        record=dict(name=tag,family=spec['name'],file=path.relative_to(args.out).as_posix(),sha256=hashlib.sha256(model_raw).hexdigest(),
            train_image_ids=sorted(train_ids),heldout_image_ids=sorted(heldout_ids),native_portable_max_abs_error=parity)
        model_records.append(record);print('A6_OD_MODEL',tag,len(train_ids),len(heldout_ids),parity,flush=True);return pred,record

    all_ids=sorted(cohort);all_mask=np.ones(len(rows),bool)
    for spec in protocol['models']:
        values=np.full(len(rows),np.nan);errors[spec['name']]={}
        for fold in folds:
            train_ids=set(fold['train_ids']);heldout_ids=set(fold['heldout_ids'])
            train=np.asarray([r['image_id'] in train_ids for r in rows]);heldout=np.asarray([r['image_id'] in heldout_ids for r in rows])
            if np.any(train&heldout) or not np.all(train|heldout):raise ValueError('cohort masks overlap or omit rows')
            pred,_=fit(spec,train,heldout,spec['name']+'-without-'+fold['cohort'],train_ids,heldout_ids);values[heldout]=pred
            nonidentity=np.asarray([r['profile']!=0 for r in rows])[heldout];e=pred[nonidentity]-y[heldout][nonidentity]
            errors[spec['name']][fold['cohort']]=dict(images=64,correlated_nonidentity_rows=len(e),mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))))
        if not np.isfinite(values).all():raise ValueError('complete OOF predictions required')
        oof[spec['name']]=values
        _,record=fit(spec,all_mask,all_mask,spec['name']+'-final192',all_ids,[]);final.append(record)
    model_manifest=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=hashlib.sha256(raw).hexdigest(),
        all_models_before_oof_metric_selection=True,folds=folds,feature_names=names,models=model_records)
    (args.out/'trained-model-manifest.json').write_text(json.dumps(model_manifest,indent=2)+'\n')
    payload=[dict(image_id=r['image_id'],cohort=cohort[r['image_id']],codec=r['codec'],qp=r['qp'],profile=r['profile'],
        predicted_damage={n:float(v[i]) for n,v in oof.items()}) for i,r in enumerate(rows)]
    (args.out/'oof-predictions.json.gz').write_bytes(gzip.compress(json.dumps(payload,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    evaluation_cache={}

    def evaluate(name,threshold):
        chosen=[]
        for image in all_ids:
            for c in CODECS:
                for q in QPS:
                    indices=groups[(image,c,q)];group=[rows[i] for i in indices]
                    profile=0 if name is None else choose_predicted(group,[oof[name][i] for i in indices],threshold)
                    chosen.append(next(i for i in indices if rows[i]['profile']==profile))
        chosen_key=tuple(chosen)
        if chosen_key in evaluation_cache:return evaluation_cache[chosen_key]
        curves={c:dict(rate=[],quality=[]) for c in CODECS};selected=[];ratios=[];counts=Counter()
        for c in CODECS:
            for q in QPS:
                subset=[rows[i] for i in chosen if rows[i]['codec']==c and rows[i]['qp']==q]
                curves[c]['rate'].append(float(np.mean([r['bpp'] for r in subset])))
                curves[c]['quality'].append(metric([dict(p) for r in subset for p in r['coco']],all_ids,meta,gt)['AP'])
                for r in subset:
                    anchor=rows[groups[(r['image_id'],c,q)][0]];ratios.append(np.log(r['bytes']/anchor['bytes']));counts[r['profile']]+=1
                    selected.append(dict(image_id=r['image_id'],codec=c,qp=q,profile=r['profile']))
        result=dict(curves=curves,selected=selected,selection_counts=dict(counts),log_rate_ratio=float(np.mean(ratios)));evaluation_cache[chosen_key]=result;return result

    anchor=evaluate(None,0);results={};choices={};best=[]
    for order,spec in enumerate(protocol['models']):
        name=spec['name'];results[name]={};candidates=[]
        for threshold in protocol['thresholds']:
            result=dict(evaluate(name,threshold),threshold=threshold);result['bd']={c:assess_curve(anchor['curves'][c],result['curves'][c],qps=QPS) for c in CODECS}
            results[name][str(threshold)]=result;candidates.append(dict(threshold=threshold,quality=[v for c in CODECS for v in result['curves'][c]['quality']],log_rate_ratio=result['log_rate_ratio']))
        choice=select_rd_fit(anchor['curves'],candidates);choices[name]=choice
        if choice['selected'] is not None:best.append(dict(model=name,model_order=order,**choice['selected']))
    selected=min(best,key=lambda r:(r['utility'],r['model_order'],r['threshold'])) if best else None
    policy=dict(kind='identity') if selected is None else dict(kind='risk_forest',model=selected['model']+'-final192',threshold=selected['threshold'],
        model_sha256=next(r['sha256'] for r in final if r['family']==selected['model']))
    frozen=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),policy=policy,training_protocol_sha256=hashlib.sha256(raw).hexdigest(),
        fresh_dev_id_protocol_sha256=protocol['reservation_sha256'],oof_endpoint_scales=next(iter(choices.values()))['endpoint_scales'],
        selection_scope='development OOF192 only; final model fits192, OOF fits128; fresh64 is the prospective check',pilot_scored=False,test_scored=False)
    (args.out/'frozen-policy.json').write_text(json.dumps(frozen,indent=2,allow_nan=False)+'\n')
    report=dict(protocol=protocol,grid=dict(images=192,rows=len(rows),profiles=21,qps=list(QPS),codecs=list(CODECS)),models=model_records,
        prediction_error=errors,oof_anchor=anchor,oof_results=results,oof_choices=choices,selected=selected,frozen=frozen,
        labels_cohort_local=True,final_models_before_selection=True,new_task_model_inferences=0,new_encodes=0,pilot_scored=False,test_scored=False,
        wall_seconds=time.perf_counter()-started)
    (args.out/'crossfit-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('A6_OD_CROSSFIT_COMPLETE',json.dumps(dict(policy=policy,seconds=report['wall_seconds'],pilot_scored=False,test_scored=False)),flush=True)


if __name__=='__main__':main()

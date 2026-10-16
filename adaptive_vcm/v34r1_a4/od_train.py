"""Cloud-only encoder-risk learning; FIT training then joint development calibration."""
import argparse
from collections import defaultdict,Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .risk import features,predict_forest,choose_predicted
from .rd_fit import select_rd_fit
from .cloud_record import require_sha256
from ..v34r1_a3.core import QPS,CODECS,audit_grid
from ..v34r1_a3.analyze import metric
from ..v34r1.metrics import assess_curve

ROOT=Path(__file__).resolve().parents[2]


def portable(model,names):
    trees=[]
    for estimator in model.estimators_:
        tree=estimator.tree_
        trees.append(dict(left=tree.children_left.tolist(),right=tree.children_right.tolist(),feature=tree.feature.tolist(),
                          threshold=tree.threshold.tolist(),value=tree.value[:,0,0].tolist()))
    return dict(schema='v34r1-risk-forest-1',features=names,trees=trees)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--labels',type=Path,required=True); p.add_argument('--annotation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'): raise ValueError('Kaggle-only actual feature extraction/training/task scoring')
    if args.out.exists(): raise ValueError('immutable fresh model output required')
    import sklearn
    from sklearn.ensemble import ExtraTreesRegressor
    protocol_raw=(ROOT/'configs/v34r1-a4-od-risk.json').read_bytes(); protocol=json.loads(protocol_raw)
    fresh_ids_sha=require_sha256((ROOT/'configs/v34r1-a4-od-next-dev.json').read_bytes(),protocol['fresh_dev_id_protocol_sha256'])
    if sklearn.__version__!=protocol['sklearn']: raise ValueError('pinned sklearn version drift')
    source=ROOT/'results/v34r1_a3/od-dev'; raw=(args.labels/'marginal-ap-labels.json.gz').read_bytes()
    label_report=json.loads((args.labels/'label-report.json').read_text()); label_status=json.loads((args.labels/'cloud-status.json').read_text())
    if not label_status['complete'] or label_status['commit']!=protocol['label_commit'] or label_status['test_scored']: raise ValueError('complete frozen training labels required')
    parent_sha=hashlib.sha256((source/'publication-manifest.json').read_bytes()).hexdigest()
    if label_report['parent_publication_manifest_sha256']!=parent_sha or hashlib.sha256(raw).hexdigest()!=label_report['label_sha256']:
        raise ValueError('label/source evidence binding drift')
    labels=json.loads(gzip.decompress(raw)); target={}
    for row in labels:
        key=(row['image_id'],row['codec'],row['qp'],row['profile'])
        if key in target: raise ValueError('duplicate label')
        target[key]=row
    rows=[]; sources={}
    for w in (0,1):
        rows.extend(json.loads(gzip.decompress((source/f'worker{w}/rows.json.gz').read_bytes())))
        for value in json.loads((source/f'worker{w}/sources.json').read_text()):
            if value['image_id'] in sources: raise ValueError('overlapping source cohorts')
            sources[value['image_id']]=value
    manifest=json.loads((source/'source-manifest.json').read_text()); audit=audit_grid(rows,[v['id'] for v in manifest['dev']])
    rows.sort(key=lambda r:(r['image_id'],CODECS.index(r['codec']),QPS.index(r['qp']),r['profile']))
    keys=[(r['image_id'],r['codec'],r['qp'],r['profile']) for r in rows]
    if set(keys)!=set(target): raise ValueError('complete labels/predictions alignment required')
    groups=defaultdict(list)
    for i,row in enumerate(rows): groups[(row['image_id'],row['codec'],row['qp'])].append(i)
    ids={s:sorted(v['id'] for v in manifest['dev'] if v['split']==s) for s in ('fit','validation')}
    if len(ids['fit'])!=64 or len(ids['validation'])!=64 or set(ids['fit'])&set(ids['validation']): raise ValueError('image-disjoint 64FIT/64CAL required')
    started=time.perf_counter(); args.out.mkdir(); x=[]; y=[]; names=None
    for i,row in enumerate(rows):
        key=keys[i]; label=target[key]
        if label['split']!=row['split'] or row['image_id'] not in ids[row['split']]: raise ValueError('training split drift')
        anchor=rows[groups[key[:3]][0]]; feature_names,values=features(row,anchor,sources[row['image_id']])
        if names is not None and feature_names!=names: raise ValueError('unstable feature schema')
        names=feature_names; x.append(values); y.append(label['scaled_ap_damage'])
        if (i+1)%2048==0: print('A4_FEATURE_PROGRESS',i+1,len(rows),flush=True)
    x=np.asarray(x,dtype=np.float32); y=np.asarray(y,float)
    if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('nonfinite training arrays')
    train=np.asarray([r['split']=='fit' for r in rows]); nonidentity=np.asarray([r['profile']!=0 for r in rows])
    models={}; prediction={}; model_manifest=[]; error={}
    for index,spec in enumerate(protocol['models']):
        name=spec['name']; model=ExtraTreesRegressor(**spec['parameters']); model.fit(x[train],y[train])
        encoded=portable(model,names); portable_prediction=np.asarray(predict_forest(encoded,x),float); native=model.predict(x)
        parity=float(np.max(np.abs(portable_prediction-native)))
        if parity>1e-10: raise ValueError('portable JSON critic differs from sklearn prediction')
        model_raw=(json.dumps(encoded,separators=(',',':'),allow_nan=False)+'\n').encode()
        path=args.out/f'{name}.json'; path.write_bytes(model_raw)
        models[name]=encoded; prediction[name]=portable_prediction
        model_manifest.append(dict(name=name,file=path.name,sha256=hashlib.sha256(model_raw).hexdigest(),native_portable_max_abs_error=parity))
        print('A4_MODEL_TRAINED',name,parity,flush=True)
    # Freeze both models before any calibration error/metric is computed.
    (args.out/'trained-model-manifest.json').write_text(json.dumps(dict(frozen_utc=datetime.now(timezone.utc).isoformat(),fit_only=True,
        fit_image_ids=ids['fit'],feature_names=names,models=model_manifest,label_sha256=label_report['label_sha256']),indent=2)+'\n')
    for name,values in prediction.items():
        error[name]={}
        for split,mask in (('fit',train&nonidentity),('calibration',~train&nonidentity)):
            e=values[mask]-y[mask]; error[name][split]=dict(rows=int(mask.sum()),mae=float(np.mean(np.abs(e))),rmse=float(np.sqrt(np.mean(e*e))),
                scope='descriptive correlated profile/QP rows, no independent-row confidence claim')
    prediction_rows=[dict(image_id=r['image_id'],split=r['split'],codec=r['codec'],qp=r['qp'],profile=r['profile'],
        predicted_damage={name:float(values[i]) for name,values in prediction.items()}) for i,r in enumerate(rows)]
    (args.out/'risk-predictions.json.gz').write_bytes(gzip.compress(json.dumps(prediction_rows,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    annotation_raw=args.annotation.read_bytes()
    if hashlib.sha256(annotation_raw).hexdigest()!=protocol['annotation_sha256']: raise ValueError('official GT drift')
    meta=json.loads(annotation_raw); gt=defaultdict(list)
    for a in meta['annotations']:
        if a['image_id'] in sources: gt[a['image_id']].append(a)
    parent=json.loads((source/'dev-report.json').read_text()); calibration={}; candidates={}; choices={}; evaluation_cache={}

    def evaluate(model_name,threshold):
        chosen=[]
        for image in ids['validation']:
            for c in CODECS:
                for q in QPS:
                    indices=groups[(image,c,q)]; group=[rows[i] for i in indices]
                    profile=0 if model_name is None else choose_predicted(group,prediction[model_name][indices],threshold)
                    chosen.append(next(i for i in indices if rows[i]['profile']==profile))
        key=tuple(chosen)
        if key in evaluation_cache: return evaluation_cache[key]
        curves={c:dict(rate=[],quality=[]) for c in CODECS}; selected=[]; ratios=[]; counts=Counter()
        for c in CODECS:
            for q in QPS:
                subset=[rows[i] for i in chosen if rows[i]['codec']==c and rows[i]['qp']==q]
                pred=[dict(p) for r in subset for p in r['coco']]
                curves[c]['rate'].append(float(np.mean([r['bpp'] for r in subset])))
                curves[c]['quality'].append(metric(pred,ids['validation'],meta,gt)['AP'])
                for r in subset:
                    anchor=rows[groups[(r['image_id'],c,q)][0]]; ratios.append(np.log(r['bytes']/anchor['bytes'])); counts[r['profile']]+=1
                    selected.append(dict(image_id=r['image_id'],codec=c,qp=q,profile=r['profile']))
        result=dict(curves=curves,selected=selected,selection_counts=dict(counts),log_rate_ratio=float(np.mean(ratios)))
        evaluation_cache[key]=result; return result

    anchor=evaluate(None,0)
    old_anchor=parent['results']['validation']['anchor']['curves']
    if any(abs(x-y)>1e-12 for c in CODECS for key in ('rate','quality') for x,y in zip(anchor['curves'][c][key],old_anchor[c][key])):
        raise ValueError('A3 calibration anchor parity failed')
    best=[]
    for order,spec in enumerate(protocol['models']):
        name=spec['name']; candidates[name]=[]; calibration[name]={}
        for threshold in protocol['thresholds']:
            result=evaluate(name,threshold); result=dict(result,threshold=threshold)
            result['bd']={c:assess_curve(anchor['curves'][c],result['curves'][c],qps=QPS) for c in CODECS}
            calibration[name][str(threshold)]=result
            candidates[name].append(dict(threshold=threshold,quality=[v for c in CODECS for v in result['curves'][c]['quality']],log_rate_ratio=result['log_rate_ratio']))
        choice=select_rd_fit(anchor['curves'],candidates[name]); choices[name]=choice
        if choice['selected'] is not None: best.append(dict(model=name,model_order=order,**choice['selected']))
    selected=min(best,key=lambda r:(r['utility'],r['model_order'],r['threshold'])) if best else None
    policy=dict(kind='identity') if selected is None else dict(kind='risk_forest',model=selected['model'],threshold=selected['threshold'],
        model_sha256=next(v['sha256'] for v in model_manifest if v['name']==selected['model']))
    frozen=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),policy=policy,calibration_reused=True,independent_validation=False,
        training_protocol_sha256=hashlib.sha256(protocol_raw).hexdigest(),fresh_dev_id_protocol_sha256=fresh_ids_sha,
        calibration_scales=next(iter(choices.values()))['endpoint_scales'],fresh_dev_gate=protocol['fresh_dev_gate'],test_scored=False)
    (args.out/'frozen-policy.json').write_text(json.dumps(frozen,indent=2,allow_nan=False)+'\n')
    report=dict(scope='actual FIT64-trained encoder critic and reused64-image development calibration; no freshDEV/TEST inference or GT scoring',
        protocol=protocol,grid=audit,features=len(names),models=model_manifest,prediction_error=error,calibration_anchor=anchor,
        calibration=calibration,calibration_choices=choices,selected=selected,frozen=frozen,new_task_model_inferences=0,new_encodes=0,
        critic_trained=True,logical_critic_predictions=len(rows)*len(models),native_parity_predictions=len(rows)*len(models),
        task_models_frozen=True,test_scored=False,fresh_dev64_scored=False,
        label_sha256=label_report['label_sha256'],wall_seconds=time.perf_counter()-started)
    (args.out/'training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('A4_OD_TRAINING_COMPLETE',json.dumps(dict(policy=policy,seconds=report['wall_seconds'],test_scored=False)),flush=True)


if __name__=='__main__': main()

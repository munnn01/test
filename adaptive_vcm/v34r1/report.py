"""Aggregate every prospective TEST endpoint; missing/invalid measurements never pass."""
import argparse
import json
import hashlib
from pathlib import Path
from .metrics import assess_all
from . import QPS


def collect(runs):
    endpoints={}; configs=set(); manifests=set(); runtimes=set(); encoders=set(); task_models=set()
    for run in runs:
        run=Path(run)
        protocol=json.loads((run/'protocol.json').read_text(encoding='utf-8'))
        summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
        if protocol['split']!='test' or protocol.get('limit') is not None:
            raise ValueError('only complete prospective TEST runs may enter the target report')
        if protocol['config']['qps']!=list(QPS) or not summary['grid_audit']['complete']:
            raise ValueError('incomplete grid or QP drift')
        config_sha=hashlib.sha256(json.dumps(protocol['config'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if config_sha!=protocol['config_sha256']: raise ValueError('config content hash mismatch')
        task=summary['task']
        if protocol['checkpoint_sha256']!=protocol['config']['controller_sha256_by_task'][task]:
            raise ValueError('unexpected task-specific controller checkpoint')
        configs.add(protocol['config_sha256']); manifests.add(protocol['manifest_sha256'])
        runtimes.add(json.dumps(protocol['runtime_sha256'],sort_keys=True))
        encoders.add(protocol['ffmpeg_sha256']); task_models.add(json.dumps(protocol['task_models'],sort_keys=True))
        for codec in ('h264','h265'):
            key=summary['task']+'/'+codec
            if key in endpoints: raise ValueError('duplicate task-codec endpoint')
            endpoints[key]=summary['comparisons'][codec]['adaptive']
    if any(len(values)>1 for values in (configs,manifests,runtimes,encoders,task_models)):
        raise ValueError('different frozen config, manifest, runtime, encoder or task weights across TEST tasks')
    return dict(**assess_all(endpoints),scope='finite declared holdout; undefined or unmeasured endpoints are not demonstrated',
                config_sha256=next(iter(configs),None),manifest_sha256=next(iter(manifests),None))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(); result=collect(args.runs)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()

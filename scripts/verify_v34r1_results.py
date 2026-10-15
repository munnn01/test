"""CPU audit of published bytes, complete paired grids, AR curves and strict target status.

This does not claim to repeat GPU inference, COCO GT scoring or TrackEval GT scoring.
Use runner.py with the hashed source data to reproduce those measurements.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adaptive_vcm.v34r1.audit import audit_grid
from adaptive_vcm.v34r1.metrics import assess_curve
from adaptive_vcm.v34r1.report import collect


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'results/v34r1')
    args=p.parse_args(); root=args.root.resolve()
    manifest=json.loads((root/'publication-manifest.json').read_text(encoding='utf-8'))
    for relative,value in manifest['files'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root): raise ValueError('publication path escape')
        raw=path.read_bytes()
        if len(raw)!=value['bytes'] or hashlib.sha256(raw).hexdigest()!=value['sha256']:
            raise ValueError('publication hash mismatch: '+relative)
    for relative,value in manifest['compressed_rows'].items():
        raw=gzip.decompress((root/relative).read_bytes())
        if len(raw)!=value['unpacked_bytes'] or hashlib.sha256(raw).hexdigest()!=value['unpacked_sha256']:
            raise ValueError('paired row archive mismatch: '+relative)
    declared_report=json.loads((root/'target-report.json').read_text(encoding='utf-8'))
    data_sha256=hashlib.sha256((root/'data-manifest.json').read_bytes()).hexdigest()
    if data_sha256!=declared_report.get('manifest_sha256'):
        raise ValueError('publication data manifest does not match frozen TEST protocol/report')
    counts={}
    for task in ('ar','od','ot'):
        run=root/(task+'-test')
        protocol=json.loads((run/'protocol.json').read_text(encoding='utf-8'))
        if protocol['manifest_sha256']!=data_sha256:
            raise ValueError('publication data manifest differs from task protocol: '+task)
        summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
        rows=json.loads(gzip.decompress((run/'rows.json.gz').read_bytes()))
        audit=audit_grid(rows,source_ids=protocol['expected_source_ids'],arms=protocol['config']['arms'])
        if audit!=summary['grid_audit']: raise ValueError('grid summary mismatch')
        counts[task]=len(rows)
        if task=='ar':
            for codec in ('h264','h265'):
                for arm in ('anchor','adaptive'):
                    curve={'rate':[],'quality':[]}
                    for qp in (30,35,40,45):
                        selected=[r for r in rows if r['codec']==codec and r['qp']==qp and r['arm']==arm]
                        curve['rate'].append(float(np.mean([r['bpp'] for r in selected])))
                        curve['quality'].append(float(np.mean([r['correct'] for r in selected])))
                    if curve!=summary['curves'][codec][arm]: raise ValueError('raw AR curve mismatch')
        for codec in ('h264','h265'):
            actual=assess_curve(summary['curves'][codec]['anchor'],summary['curves'][codec]['adaptive'],
                                qps=protocol['config']['qps'],ci=summary['comparisons'][codec]['adaptive']['ci'])
            if actual!=summary['comparisons'][codec]['adaptive']: raise ValueError('BD-rate summary mismatch')
    actual=collect([root/(t+'-test') for t in ('ar','od','ot')])
    if actual!=declared_report: raise ValueError('target report mismatch')
    print(json.dumps(dict(verified=True,file_count=len(manifest['files']),paired_rows=counts,
        all_point_targets=actual['all_point_targets'],all_thresholds_confirmed=actual['all_thresholds_confirmed'],
        scope='CPU publication/grid/AR-curve/BD-rate audit; does not repeat task-model or GT inference'),indent=2))


if __name__=='__main__': main()

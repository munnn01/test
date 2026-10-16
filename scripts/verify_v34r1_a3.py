"""Read-only A3 evidence byte/grid/selection/BD audit; no models, media or GT inference."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import sys


def verify_files(root):
    root=Path(root).resolve(); manifest=json.loads((root/'publication-manifest.json').read_text(encoding='utf-8'))
    expected=set()
    for item in manifest['files']:
        relative=item['path']; pure=PurePosixPath(relative); candidate=root/relative; path=candidate.resolve()
        if (pure.is_absolute() or '..' in pure.parts or '\\' in relative or ':' in relative
            or not path.is_relative_to(root) or candidate.is_symlink() or relative in expected):
            raise ValueError('unsafe/duplicate evidence path')
        expected.add(relative)
        raw=path.read_bytes()
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('evidence byte/hash mismatch: '+relative)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual!=expected|{'publication-manifest.json'}: raise ValueError('missing or undeclared evidence files')
    return manifest


def verify_od(root):
    from adaptive_vcm.v34r1_a3.core import audit_grid,choose,fit_threshold
    from adaptive_vcm.v34r1.policy import select_measured
    from adaptive_vcm.v34r1.metrics import assess_curve
    root=Path(root); publication=verify_files(root)
    if publication.get('task')!='od' or publication.get('test_scored') is not False: raise ValueError('OD DEV evidence required')
    source_raw=(root/'source-manifest.json').read_bytes(); source=json.loads(source_raw)
    if source['source_id_protocol_sha256']!=publication['frozen_source_id_protocol_sha256']: raise ValueError('frozen source ID hash mismatch')
    rows=[]
    for worker in (0,1):
        folder=root/f'worker{worker}'; protocol=json.loads((folder/'protocol.json').read_text())
        if protocol['manifest_sha256']!=hashlib.sha256(source_raw).hexdigest(): raise ValueError('worker source hash mismatch')
        subset=json.loads(gzip.decompress((folder/'rows.json.gz').read_bytes()))
        if {r['image_id'] for r in subset}!=set(protocol['expected_ids']): raise ValueError('worker source membership mismatch')
        rows.extend(subset)
    audit=audit_grid(rows,[r['id'] for r in source['dev']]); groups=defaultdict(list)
    split_by_id={r['id']:r['split'] for r in source['dev']}
    for row in rows:
        if row['split']!=split_by_id[row['image_id']] or row['bytes']!=row['elementary_bytes']+row['metadata_bytes'] or row['bytes']<=0:
            raise ValueError('invalid source split or paid-byte accounting')
        groups[(row['image_id'],row['codec'],row['qp'])].append(row)
    for group in groups.values(): group.sort(key=lambda r:r['profile'])
    report=json.loads((root/'dev-report.json').read_text()); frozen_raw=(root/'fit-frozen-policies.json').read_bytes()
    if report['fit_policy_sha256']!=hashlib.sha256(frozen_raw).hexdigest(): raise ValueError('FIT policy hash mismatch')
    anchor_quality=[v for c in ('h264','h265') for v in report['results']['fit']['anchor']['curves'][c]['quality']]
    for fit in report['fit'].values():
        if fit_threshold(fit['candidates'],anchor_quality)!=fit['selected']: raise ValueError('FIT threshold decision mismatch')
    for split,results in report['results'].items():
        for name,result in results.items():
            if name=='fixed_profile_diagnostics': continue
            policy=result['policy']; chosen={(r['image_id'],r['codec'],r['qp']):r['profile'] for r in result['selected']}
            keys={k for k in groups if split_by_id[k[0]]==split}
            if set(chosen)!=keys or len(result['selected'])!=len(keys): raise ValueError('policy selection grid mismatch')
            for key in keys:
                group=groups[key]
                if policy['kind']=='fixed': expected=policy['profile']
                elif policy['kind']=='source_proxy': expected=choose(group,policy['proxy'],policy['threshold'])
                else:
                    subset=[r for r in group if policy['kind']=='exhaust_guard' or r['profile'] in group[0]['a2_shortlist']]
                    i,_=select_measured([r['bytes']/group[0]['bytes'] for r in subset],[[None,r['current_cost'],None] for r in subset],
                        [False,True,False],[0,.05,0],budget=1.,quality_weight=.05)
                    expected=subset[i]['profile']
                if chosen[key]!=expected: raise ValueError('source-only replay decision mismatch')
        variants=[v for k,v in results.items() if k!='fixed_profile_diagnostics']+list(results['fixed_profile_diagnostics'].values())
        for result in variants:
            for codec in ('h264','h265'):
                fresh=assess_curve(results['anchor']['curves'][codec],result['curves'][codec],qps=[30,35,40,45]); saved=result['bd'][codec]
                if any(fresh[k]!=saved[k] for k in ('valid','point_passes','invalid_reason')): raise ValueError('BD validity mismatch')
                if fresh['valid'] and not math.isclose(fresh['pchip_bd_rate_pct'],saved['pchip_bd_rate_pct'],abs_tol=1e-8,rel_tol=0):
                    raise ValueError('BD arithmetic mismatch')
    return dict(files=len(publication['files']),grid=audit,validation_gate=report['validation_both_codecs_below_minus15'],test_scored=False,
                scope='hashes, paid-byte sums, source selector decisions and BD arithmetic from measured COCO AP; no GT inference repeated')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True); args=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    print(json.dumps(verify_od(args.root),indent=2))


if __name__=='__main__': main()

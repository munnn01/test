"""Package real paired measurements and provenance; exclude raw media and weights."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adaptive_vcm.v34r1.report import collect


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'output/v34r1')
    p.add_argument('--dest',type=Path,default=ROOT/'results/v34r1')
    p.add_argument('--manifest',type=Path,default=ROOT/'data/v34r1/validation_manifest.json')
    p.add_argument('--dev-source',type=Path,help='separate original DEV directory when TEST runs execute on Kaggle')
    p.add_argument('--cloud-bundles',type=Path,help='verified task folders containing original cloud status and logs')
    args=p.parse_args()
    if args.dest.exists() and any(args.dest.iterdir()): raise ValueError('empty publication directory required')
    final=[args.source/(task+'-test') for task in ('ar','od','ot')]
    report=collect(final)
    manifest_bytes=args.manifest.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest()!=report['manifest_sha256']:
        raise ValueError('data manifest does not match frozen TEST protocol')
    args.dest.mkdir(parents=True,exist_ok=True)
    (args.dest/'target-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (args.dest/'data-manifest.json').write_bytes(manifest_bytes)
    runs=['ar-dev-A0','ar-dev-A1','ar-dev-A2','od-dev-A0','od-dev-A1','ot-dev-A1','ot-dev-A2','ar-test','od-test','ot-test']
    archives={}
    for name in runs:
        source=(args.dev_source if args.dev_source and '-dev-' in name else args.source)/name
        if not (source/'summary.json').is_file(): raise ValueError('missing complete run: '+name)
        dest=args.dest/name; dest.mkdir()
        for file in ('protocol.json','summary.json'): shutil.copyfile(source/file,dest/file)
        raw=(source/'rows.json').read_bytes()
        (dest/'rows.json.gz').write_bytes(gzip.compress(raw,mtime=0))
        archives[name+'/rows.json.gz']=dict(unpacked_bytes=len(raw),unpacked_sha256=hashlib.sha256(raw).hexdigest())
        for predictions in sorted((source/'tracking/trackers').glob('*/data/*.txt')):
            target=dest/predictions.relative_to(source).with_suffix('.txt.gz')
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(gzip.compress(predictions.read_bytes(),mtime=0))
        if args.cloud_bundles and name.endswith('-test'):
            task=name.split('-')[0]; bundle=args.cloud_bundles/task
            status=json.loads((bundle/'cloud-status.json').read_text(encoding='utf-8'))
            if status.get('complete') is not True or status.get('task')!=task:
                raise ValueError('incomplete or incorrect cloud provenance: '+task)
            cloud=dest/'cloud'; cloud.mkdir()
            for file in ('cloud-status.json','block-tests.log','source-restore.log','benchmark.log','import-audit.json',
                         'kaggle-execution.log','kernel-metadata.json','notebook.py'):
                shutil.copyfile(bundle/file,cloud/file)
    files={str(file.relative_to(args.dest)).replace('\\','/'):dict(bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest())
           for file in sorted(args.dest.rglob('*')) if file.is_file()}
    publication=dict(schema='v34r1-publication-1',files=files,compressed_rows=archives,
        scope='actual paired four-QP measurements; DEV is exploratory; final policy frozen before TEST; no raw media or task weights included',
        caveats=['K400 provider source cadence unverified','COCO64 finite sample','MOT20 internal heldout single sequence segment; no population sequence CI',
                 'early OT16frame smoke had a GT-prefix scoring defect and is excluded from comparative tables; fixed32frameDEV and256frameTEST use exact observed prefix'])
    (args.dest/'publication-manifest.json').write_text(json.dumps(publication,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(publication=str(args.dest),files=len(files),all_point_targets=report['all_point_targets']),indent=2))


if __name__=='__main__': main()

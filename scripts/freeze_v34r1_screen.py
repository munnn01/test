"""Freeze the downloaded AR/OD screening inventory before task inference."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import gzip
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_v34r1_data import rank,partition,download,COCO


def main():
    root=ROOT/'data/v34r1'
    catalog=json.loads((ROOT/'data/kineticscleaned/catalog.json').read_text(encoding='utf-8'))
    manifest=dict(schema='v34r1-screen-data-1',qps=[30,35,40,45],ar=[],od=[],ot=[],
        kinetics_scope='user-supplied official-train-path videos; controller development only; not confirmatory official K400 val')
    for row in catalog['files']:
        p=Path(row['name'])
        if p.suffix!='.mp4': continue
        identifier=p.parent.name+'/'+p.name
        dest=root/'kinetics'/p.parent.name/p.name
        if dest.is_file():
            manifest['ar'].append(dict(id=identifier,category=p.parent.name,path=str(dest),split=partition(identifier),
                file_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    manifest['ar'].sort(key=lambda r:rank(r['id']))
    meta=json.loads((root/'coco/annotations/instances_val2017.json').read_text(encoding='utf-8'))
    old_rows=json.loads(gzip.decompress((ROOT/'results/v34_od_coco/test_rows.json.gz').read_bytes()))
    forbidden={int(str(r['id']).split('/')[-1]) for r in old_rows}
    jobs=[]
    for split,count in [('train',64),('dev',16),('test',64)]:
        candidates=[r for r in meta['images'] if partition('coco2017/'+str(r['id']))==split and (split!='test' or r['id'] not in forbidden)]
        chosen=sorted(candidates,key=lambda r:rank(str(r['id'])))[:count]
        for row in chosen:
            dest=root/'coco/val2017'/row['file_name']
            manifest['od'].append({**row,'path':str(dest),'split':split})
            if not dest.exists(): jobs.append((COCO+'/val2017/'+row['file_name'],dest,None))
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(download,jobs))
    for row in manifest['od']: row['file_sha256']=hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()
    manifest['od_excluded_published_test_image_ids']=sorted(forbidden)
    target=root/'screen_manifest.json'
    if target.exists(): raise ValueError('screen manifest already frozen')
    target.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'manifest':str(target),'ar':len(manifest['ar']),'od':len(manifest['od']),
                      'new_coco_downloads':len(jobs),'old_od_test_excluded':len(forbidden)}))


if __name__=='__main__': main()

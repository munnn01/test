from collections import Counter
import json
from pathlib import Path
import requests

out = Path(__file__).resolve().parents[1]/'data/kineticscleaned'
out.mkdir(parents=True, exist_ok=True)
session = requests.Session()
url = 'https://www.kaggle.com/api/v1/datasets/list/qktttttttttt/kineticscleaned'
rows = []
token = None
for page in range(1000):
    params = {'pageSize': 1000}
    if token: params['pageToken'] = token
    response = session.get(url, params=params, timeout=45)
    response.raise_for_status()
    data = response.json()
    files = data.get('datasetFiles', [])
    rows.extend({'name': r['name'], 'bytes': r['totalBytes']} for r in files)
    token = data.get('nextPageToken') or data.get('nextPageTokenNullable')
    if page % 10 == 0 or not token:
        print(json.dumps({'page': page+1, 'files': len(rows), 'last': rows[-1]['name'] if rows else None}), flush=True)
    if not token: break
else:
    raise RuntimeError('Pagination exceeded limit')
assert len({r['name'] for r in rows}) == len(rows), 'repeated page or duplicate filename'
(out / 'catalog.json').write_text(json.dumps({'dataset': 'qktttttttttt/kineticscleaned', 'version': 4, 'files': rows}, indent=2), encoding='utf-8')
print(json.dumps({'total': len(rows), 'bytes': sum(r['bytes'] for r in rows), 'top_paths': Counter('/'.join(r['name'].split('/')[:4]) for r in rows).most_common(12)}, indent=2))

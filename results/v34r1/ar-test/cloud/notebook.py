import os, pathlib, subprocess, sys
repo=pathlib.Path('/kaggle/temp/adaptive_preprocessing')
repo.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['git','clone','-q','https://github.com/munnn01/test.git',str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'checkout','-q','b3f30bcd1c111b69311180f9a5259f0541eee817'],check=True)
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='b3f30bcd1c111b69311180f9a5259f0541eee817'
import json
import gzip,hashlib,zipfile
manifest=json.loads(gzip.decompress((repo/'configs/v34r1-cloud-data-manifest.json.gz').read_bytes()))
bases=[pathlib.Path('/kaggle/input/v34r1-frozen-ar-source-cache'),pathlib.Path('/kaggle/input/datasets/baooo25r/v34r1-frozen-ar-source-cache')]
cached=next((p for p in bases if p.is_dir()),None)
if cached is None: raise RuntimeError('private frozen source cache mount missing')
zip_path=cached/'source-cache.zip'
bundle=None
if zip_path.is_file():
    if hashlib.sha256(zip_path.read_bytes()).hexdigest()!='727e1b582943e77cdf509b0a27986af52bdb13a7d6dd046de2ad72fc03caac19': raise ValueError('source archive drift')
    bundle=zipfile.ZipFile(zip_path)
count=0
for row in manifest['ar']:
    if row['split']!='test': continue
    for name,sha in row['file_sha256'].items():
        relative=row['youtube_id']+'/'+name
        if bundle is not None: raw=bundle.read(relative)
        else:
            candidates=[cached/relative,cached/'source-cache'/relative]
            src=next((p for p in candidates if p.is_file()),None)
            if src is None: raise RuntimeError('frozen cache file missing: '+relative)
            raw=src.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sha: raise ValueError('frozen source hash drift: '+relative)
        target=pathlib.Path(row['path'])/name
        if not target.resolve().is_relative_to(repo.resolve()): raise ValueError('source path escape')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw); count+=1
if bundle is not None: bundle.close()
assert count==3053
print('PRIVATE_CACHE exact frozen sources verified:',count,flush=True)

subprocess.run([sys.executable,str(repo/'scripts/run_v34r1_kaggle.py'),'--task','ar','--commit','b3f30bcd1c111b69311180f9a5259f0541eee817'],check=True)

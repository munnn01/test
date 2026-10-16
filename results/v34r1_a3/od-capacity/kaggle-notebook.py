import pathlib,subprocess,sys
repo=pathlib.Path('/kaggle/temp/adaptive_preprocessing')
repo.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['git','clone','-q','https://github.com/munnn01/test.git',str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'checkout','-q','e97f4778ac68074d697007a5e81dbe772f374180'],check=True)
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='e97f4778ac68074d697007a5e81dbe772f374180'
subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps','numpy==2.1.3','scipy==1.16.3','pycocotools==2.0.11'],check=True)
subprocess.run([sys.executable,'-m','unittest','tests.test_v34r1_a3_core','tests.test_v34r1_a3_pseudo','tests.test_v34r1_a3_evidence','-v'],cwd=repo,check=True)
source=repo/'results/v34r1_a3/od-dev'
subprocess.run([sys.executable,str(repo/'scripts/verify_v34r1_a3.py'),'--root',str(source)],cwd=repo,check=True)
subprocess.run([sys.executable,'-u','-m','adaptive_vcm.v34r1_a3.od_capacity','--source',str(source),'--out','/kaggle/working/od-capacity'],cwd=repo,check=True)

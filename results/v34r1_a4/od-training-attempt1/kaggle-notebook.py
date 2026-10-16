import pathlib,subprocess,sys
repo=pathlib.Path('/kaggle/temp/adaptive_preprocessing'); repo.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['git','clone','-q','https://github.com/munnn01/test.git',str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'checkout','-q','d64a341e3f5cc874aebb064bed0948328c23c139'],check=True)
subprocess.run([sys.executable,'-u',str(repo/'scripts/run_v34r1_a4_od_train_kaggle.py'),'--commit','d64a341e3f5cc874aebb064bed0948328c23c139','--labels-sha256','f74d6b9002d0ea96a31c955142a1723ddf08b8c5cd36cb44ade35049b5fabca1'],cwd=repo,check=True)

import pathlib,subprocess,sys
repo=pathlib.Path('/kaggle/temp/adaptive_preprocessing'); repo.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['git','clone','-q','https://github.com/munnn01/test.git',str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'checkout','-q','e4f2062f02e6c53b99419c6f9e57a1aafcafb24e'],check=True)
subprocess.run([sys.executable,'-u',str(repo/'scripts/run_v34r1_a4_od_train_kaggle.py'),'--commit','e4f2062f02e6c53b99419c6f9e57a1aafcafb24e','--labels-sha256','f74d6b9002d0ea96a31c955142a1723ddf08b8c5cd36cb44ade35049b5fabca1'],cwd=repo,check=True)

import pathlib,subprocess,sys
repo=pathlib.Path('/kaggle/temp/adaptive_preprocessing'); repo.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['git','clone','-q','https://github.com/munnn01/test.git',str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'checkout','-q','ac42edc62344546d093f98c9cbe4c8090e488e53'],check=True)
subprocess.run([sys.executable,'-u',str(repo/'scripts/run_v34r1_a5_ot_kaggle.py'),'--commit','ac42edc62344546d093f98c9cbe4c8090e488e53'],cwd=repo,check=True)

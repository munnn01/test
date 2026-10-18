import pathlib,subprocess,sys
repo=pathlib.Path('/kaggle/temp/adaptive_preprocessing');repo.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['git','clone','-q','https://github.com/munnn01/test.git',str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'checkout','-q','5fd58a25e4db72b39d1187212331c5c6ef9b2f93'],check=True)
subprocess.run([sys.executable,'-u',str(repo/'scripts/run_v34r1_a6_od_crossfit_kaggle.py'),'--commit','5fd58a25e4db72b39d1187212331c5c6ef9b2f93'],cwd=repo,check=True)

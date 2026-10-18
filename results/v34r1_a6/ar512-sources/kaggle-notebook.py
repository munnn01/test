import pathlib,subprocess,sys
repo=pathlib.Path('/kaggle/temp/adaptive_preprocessing');repo.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['git','clone','-q','https://github.com/munnn01/test.git',str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'checkout','-q','c40ee4a2fd9937e4339026361a50678378e3dead'],check=True)
subprocess.run([sys.executable,'-u',str(repo/'scripts/freeze_v34r1_a6_ar512_kaggle.py'),'--commit','c40ee4a2fd9937e4339026361a50678378e3dead'],cwd=repo,check=True)

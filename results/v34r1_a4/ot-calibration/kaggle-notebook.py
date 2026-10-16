import pathlib,subprocess,sys
repo=pathlib.Path('/kaggle/temp/adaptive_preprocessing'); repo.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['git','clone','-q','https://github.com/munnn01/test.git',str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'checkout','-q','97cabc2a54269366168b9146e8c5e1b8da342fcf'],check=True)
subprocess.run([sys.executable,'-u',str(repo/'scripts/run_v34r1_a4_ot_kaggle.py'),'--commit','97cabc2a54269366168b9146e8c5e1b8da342fcf'],cwd=repo,check=True)

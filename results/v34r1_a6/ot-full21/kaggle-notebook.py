import pathlib,subprocess,sys
repo=pathlib.Path('/kaggle/temp/adaptive_preprocessing');repo.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['git','clone','-q','https://github.com/munnn01/test.git',str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'checkout','-q','c46327828ab252ce81dc8501d7202ced49908046'],check=True)
subprocess.run([sys.executable,'-u',str(repo/'scripts/run_v34r1_a6_ot_replay_kaggle.py'),'--commit','c46327828ab252ce81dc8501d7202ced49908046','--bundle-sha256','75b921531305e3d1daed70a319d7f31c3e20865bc0137dae427e75219ea6fe46'],cwd=repo,check=True)

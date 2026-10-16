import pathlib,subprocess,sys
repo=pathlib.Path('/kaggle/temp/adaptive_preprocessing'); repo.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['git','clone','-q','https://github.com/munnn01/test.git',str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'checkout','-q','5559fc7db276992828919d68ee68bb996eff7813'],check=True)
subprocess.run([sys.executable,'-u',str(repo/'scripts/run_v34r1_a4_od_labels_kaggle.py'),'--commit','5559fc7db276992828919d68ee68bb996eff7813'],cwd=repo,check=True)

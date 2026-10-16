import pathlib,subprocess,sys
repo=pathlib.Path('/kaggle/temp/adaptive_preprocessing')
repo.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['git','clone','-q','https://github.com/munnn01/test.git',str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'checkout','-q','749f8f876ccca875536d4c88565ccc075d8a9197'],check=True)
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='749f8f876ccca875536d4c88565ccc075d8a9197'
subprocess.run([sys.executable,str(repo/'scripts/run_v34r1_a3_ot_kaggle.py'),'--commit','749f8f876ccca875536d4c88565ccc075d8a9197'],check=True)

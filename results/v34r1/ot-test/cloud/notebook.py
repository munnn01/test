import os, pathlib, subprocess, sys
repo=pathlib.Path('/kaggle/temp/adaptive_preprocessing')
repo.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['git','clone','-q','https://github.com/munnn01/test.git',str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'checkout','-q','5a369016e4214b8339f89768534fed19cbcf194e'],check=True)
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='5a369016e4214b8339f89768534fed19cbcf194e'
subprocess.run([sys.executable,str(repo/'scripts/run_v34r1_kaggle.py'),'--task','ot','--commit','5a369016e4214b8339f89768534fed19cbcf194e'],check=True)

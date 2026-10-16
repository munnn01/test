import pathlib,subprocess,sys
repo=pathlib.Path('/kaggle/temp/adaptive_preprocessing')
repo.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['git','clone','-q','https://github.com/munnn01/test.git',str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'checkout','-q','33ef4181abb9afb6b829de6c6094a2e015b6346a'],check=True)
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='33ef4181abb9afb6b829de6c6094a2e015b6346a'
subprocess.run([sys.executable,str(repo/'scripts/run_v34r1_a3_kaggle.py'),'--commit','33ef4181abb9afb6b829de6c6094a2e015b6346a'],check=True)

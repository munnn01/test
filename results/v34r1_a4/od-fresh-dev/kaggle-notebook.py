import pathlib,subprocess,sys
repo=pathlib.Path('/kaggle/temp/adaptive_preprocessing'); repo.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['git','clone','-q','https://github.com/munnn01/test.git',str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'checkout','-q','186f29e26f6ba534699f0cf74410feb55753237e'],check=True)
subprocess.run([sys.executable,'-u',str(repo/'scripts/run_v34r1_a4_od_fresh_kaggle.py'),'--commit','186f29e26f6ba534699f0cf74410feb55753237e','--training-sha256','517cbaada33bbca88ff0492ceda9a9860d00a26cf269ae4ef0d62247b67855c3'],cwd=repo,check=True)

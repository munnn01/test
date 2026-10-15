import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from adaptive_vcm.v34r1.report import collect


class ReportTests(unittest.TestCase):
    def run_files(self,root,task,*,encoder='ffmpeg',split='test'):
        run=root/task; run.mkdir()
        cfg=dict(qps=[30,35,40,45],controller_sha256_by_task={t:t+'-weights' for t in ('ar','od','ot')})
        sha=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        protocol=dict(split=split,limit=None,config=cfg,config_sha256=sha,manifest_sha256='data',runtime_sha256={'runtime.py':'same'},
                      ffmpeg_sha256=encoder,task_models={'frozen_model':'same'},checkpoint_sha256=task+'-weights')
        passed=dict(valid=True,point_passes=True,threshold_confirmed=False)
        summary=dict(task=task,grid_audit={'complete':True},comparisons={c:{'adaptive':passed} for c in ('h264','h265')})
        (run/'protocol.json').write_text(json.dumps(protocol)); (run/'summary.json').write_text(json.dumps(summary))
        return run

    def test_six_points_are_required_and_point_success_is_not_ci_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); runs=[self.run_files(root,t) for t in ('ar','od','ot')]
            self.assertTrue(collect(runs)['all_point_targets'])
            self.assertFalse(collect(runs)['all_thresholds_confirmed'])
            self.assertFalse(collect(runs[:2])['all_point_targets'])

    def test_encoder_drift_and_dev_runs_cannot_enter_test_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            a=self.run_files(root,'ar'); b=self.run_files(root,'od',encoder='different')
            with self.assertRaises(ValueError): collect([a,b])
            c=self.run_files(root,'ot',split='dev')
            with self.assertRaises(ValueError): collect([c])

    def test_config_checkpoint_limit_and_duplicate_endpoints_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run=self.run_files(Path(directory),'ar'); path=run/'protocol.json'
            original=json.loads(path.read_text())
            for update in [dict(checkpoint_sha256='wrong'),dict(config_sha256='wrong'),dict(limit=1)]:
                value=deepcopy(original); value.update(update); path.write_text(json.dumps(value))
                with self.assertRaises(ValueError): collect([run])
            path.write_text(json.dumps(original))
            with self.assertRaises(ValueError): collect([run,run])

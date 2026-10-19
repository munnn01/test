"""Synthetic contracts only; no actual detector, codec or GT evaluation."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, r'D:\TEST')
path=Path(__file__).with_name('merge_od.py')
if not path.exists():path=Path(__file__).parent/'kaggle-jobs/v34r1-a8-od-merge750/merge_od.py'
spec=importlib.util.spec_from_file_location('od_merge750',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def sample():
    return [dict(id='od/coco2017/11',image_id=11,codec=c,qp=q,arm=arm,action=0 if arm=='anchor' else 1,
        bytes=80,elementary_bytes=40,metadata_bytes=40,bpp=80*8/(320*320),coco=[dict(image_id=11,category_id=1,bbox=[0,0,1,1],score=.5)])
        for c in ('h264','h265') for q in (30,35,40,45) for arm in ('anchor','adaptive')]

class Contracts(unittest.TestCase):
    def test_complete_four_qp_paired_grid_and_no_replacement(self):
        rows=sample();self.assertEqual(module.validate_rows(rows,[11]),16)
        for changed in (rows[:-1],rows+[deepcopy(rows[0])]):
            with self.assertRaises(ValueError):module.validate_rows(changed,[11])
        with self.assertRaises(ValueError):module.validate_rows(rows,[12])

    def test_foreign_prediction_id_or_unpaid_bpp_is_rejected(self):
        rows=sample();rows[0]['coco'][0]['image_id']=12
        with self.assertRaises(ValueError):module.validate_rows(rows,[11])
        rows=sample();rows[0]['bpp']*=2
        with self.assertRaises(ValueError):module.validate_rows(rows,[11])

    def test_anchor_edit_and_paid_budget_overrun_are_rejected(self):
        rows=sample();rows[0]['action']=2
        with self.assertRaises(ValueError):module.validate_rows(rows,[11])
        rows=sample();rows[1].update(bytes=81,elementary_bytes=41,bpp=81*8/(320*320))
        with self.assertRaises(ValueError):module.validate_rows(rows,[11])

if __name__=='__main__':unittest.main(verbosity=2)

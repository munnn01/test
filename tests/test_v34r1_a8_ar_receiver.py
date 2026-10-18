"""Synthetic receiver serialization must defer every GT label read."""
import unittest
from types import SimpleNamespace


class ForbiddenGT:
    @property
    def gt_label(self):raise AssertionError('GT accessed before source-only decisions were sealed')


class Receiver(unittest.TestCase):
    def test_deferred_output_never_accesses_label_and_legacy_output_is_identical(self):
        from adaptive_vcm.v34r1.runner import ar_receiver_result
        self.assertEqual(ar_receiver_result([.1,.8,.1],ForbiddenGT(),defer=True),dict(predicted_label=1))
        self.assertEqual(ar_receiver_result([.1,.8,.1],SimpleNamespace(gt_label=1)),dict(correct=1))
        self.assertEqual(ar_receiver_result([.1,.8,.1],SimpleNamespace(gt_label=0)),dict(correct=0))
        with self.assertRaises(AssertionError):ar_receiver_result([.1,.8,.1],ForbiddenGT())


if __name__=='__main__':unittest.main()

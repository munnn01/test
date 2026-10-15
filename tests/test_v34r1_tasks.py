"""Source-only OD cannot consult GT, including when record.task is od."""
import importlib.util
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch


class WithoutGT:
    rgb=np.zeros((1,16,16,3),np.uint8)
    task='od'
    mask=[False,True,False]
    source_features={'od':torch.ones(1,4)}
    source_predictions={'od':[{'boxes':np.array([[1.,1.,8.,8.]]),'labels':np.array([1]),'scores':np.array([.9])}]}
    @property
    def targets(self):
        raise AssertionError('source-only candidate scoring accessed ground truth')


def test_od_source_only_succeeds_without_any_ground_truth():
    assert importlib.util.find_spec('adaptive_vcm.v34r1.tasks') is not None, 'new source-only task adapter missing'
    from adaptive_vcm.v34r1.tasks import Heads
    head=Heads.__new__(Heads)
    head.device='cpu'; head.detector=object()
    predictions=[{'boxes':torch.tensor([[1.,1.,8.,8.]]),'labels':torch.tensor([1]),'scores':torch.tensor([.9])}]
    with patch('adaptive_vcm.v34r1.tasks.od_forward',return_value=(predictions,torch.ones(1,4))) as forward:
        costs=head.costs(WithoutGT.rgb,WithoutGT(),source_only=True)
    assert len(costs)==3 and np.isfinite(costs[1])
    assert len(forward.call_args.args)==3, 'targets were supplied to frozen detector source guard'


def test_non_source_training_path_still_uses_legacy_gt_supervision():
    assert importlib.util.find_spec('adaptive_vcm.v34r1.tasks') is not None
    from adaptive_vcm.v34r1.tasks import Heads
    head=Heads.__new__(Heads)
    record=SimpleNamespace(task='od')
    with patch('adaptive_vcm.v34.tasks.Heads.costs',return_value=[None,3.,None]) as base:
        assert head.costs(np.zeros((1,4,4,3),np.uint8),record,source_only=False)==[None,3.,None]
    assert base.call_args.kwargs['source_only'] is False


def test_ar_decision_guard_uses_clean_predictions_without_gt():
    from adaptive_vcm.v34r1.tasks import Heads
    head=Heads.__new__(Heads); head.device='cpu'; head.action=object(); head.cfg={'ar_top1_penalty':.1}
    record=SimpleNamespace(task='ar',mask=[True,False,False],source_predictions={'ar':np.array([.9,.1])},
                           source_features={'ar':torch.ones(1,4)})
    logits=torch.tensor([[0.,2.]])
    with patch('adaptive_vcm.v34r1.tasks.ar_forward',return_value=(logits,torch.ones(1,4))):
        result=head.costs(np.zeros((2,8,8,3),np.uint8),record,source_only=True)
    expected=.25*torch.nn.functional.kl_div(logits.log_softmax(-1),torch.tensor([[.9,.1]],dtype=torch.float64),reduction='batchmean')+.1
    assert result[0]==pytest.approx(float(expected))

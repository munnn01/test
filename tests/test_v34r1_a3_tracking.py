"""Synthetic causal tracker-state contracts for the source-HOTA guard."""
import importlib.util
import unittest
import numpy as np
from tests.test_v34r1_a3_hota import metric


def guard(fps=14,lookback=None):
    if importlib.util.find_spec('adaptive_vcm.v34r1_a3.tracking') is None:
        raise AssertionError('native-cadence source-HOTA guard is missing')
    from adaptive_vcm.v34r1_a3.tracking import CausalProxyGuard
    return CausalProxyGuard(metric(),fps=fps,lookback=lookback)


def pred(x):
    return dict(boxes=np.array([[x,10.,x+15,30.]]),scores=np.array([.95]),labels=np.array([1]))


class HOTAStateContracts(unittest.TestCase):
    def test_trials_share_committed_past_and_only_one_commit_advances_history(self):
        state=guard(); clean=[pred(i+10) for i in range(4)]
        state.begin('a',clean)
        good=state.trial(clean)
        empty=[dict(boxes=np.empty((0,4)),scores=np.empty(0),labels=np.empty(0,int)) for _ in clean]
        bad=state.trial(empty); again=state.trial(clean)
        self.assertEqual(good.proxy_costs,again.proxy_costs)
        self.assertLess(good.proxy_costs[0],bad.proxy_costs[0])
        self.assertEqual(len(state.committed_history),0)
        state.commit(good)
        self.assertEqual(len(state.committed_history),4)
        with self.assertRaises(ValueError): state.commit(bad)
        self.assertEqual(len(state.committed_history),4)
        self.assertEqual(state.source_tracker.tracker.max_time_lost,14)

    def test_native_state_resets_at_sequence_boundary_and_preserves_prefix_alignment(self):
        state=guard(30,32); clean=[pred(i+10) for i in range(4)]
        state.begin('a',clean); state.commit(state.trial(clean))
        following=[pred(i+14) for i in range(4)]
        state.begin('a',following); trial=state.trial(following)
        self.assertEqual(trial.frames[-1]['frame_number'],8)
        state.commit(trial)
        state.begin('b',clean); trial=state.trial(clean)
        self.assertEqual(trial.frames[-1]['frame_number'],4)
        self.assertEqual(len(state.source_history),4)
        self.assertEqual(len(state.committed_history),0)
        state.commit(trial)
        with self.assertRaises(ValueError): guard(0)


if __name__=='__main__': unittest.main()

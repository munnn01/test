import importlib.util
import copy
import numpy as np
import pytest


def prediction(x=10.):
    return dict(boxes=np.array([[x,10.,x+15,30.]]),scores=np.array([.95]),labels=np.array([1]))


def guard():
    assert importlib.util.find_spec('adaptive_vcm.v34r1.tracking') is not None, 'causal tracking guard missing'
    from adaptive_vcm.v34r1.tracking import CausalTrackingGuard
    return CausalTrackingGuard()


def test_candidate_trials_are_order_independent_and_only_selection_commits():
    state=guard()
    clean=[prediction(10+i) for i in range(8)]
    state.begin('seq-A',clean)
    good=state.trial(clean)
    empty=[dict(boxes=np.empty((0,4)),scores=np.empty(0),labels=np.empty(0,int)) for _ in clean]
    bad=state.trial(empty)
    again=state.trial(clean)
    assert good.cost==again.cost
    assert bad.cost>good.cost
    assert state.committed_windows==0
    state.commit(good)
    assert state.committed_windows==1
    with pytest.raises(ValueError): state.commit(bad)


def test_state_crosses_window_boundaries_and_resets_on_sequence_change():
    state=guard()
    frames=[prediction(10+i) for i in range(4)]
    state.begin('seq-A',frames); trial=state.trial(frames); state.commit(trial)
    state.begin('seq-A',[prediction(14+i) for i in range(4)])
    trial=state.trial([prediction(14+i) for i in range(4)])
    assert trial.frames[-1]['frame_number']==8
    state.commit(trial)
    state.begin('seq-B',frames)
    trial=state.trial(frames)
    assert trial.frames[-1]['frame_number']==4
    state.commit(trial)


def test_divergent_arms_keep_distinct_committed_prefixes():
    a,b=guard(),guard()
    clean=[prediction(10+i) for i in range(4)]
    empty=[dict(boxes=np.empty((0,4)),scores=np.empty(0),labels=np.empty(0,int)) for _ in clean]
    a.begin('seq',clean); a.commit(a.trial(clean))
    b.begin('seq',clean); b.commit(b.trial(empty))
    next_frames=[prediction(14+i) for i in range(4)]
    a.begin('seq',next_frames); b.begin('seq',next_frames)
    first,second=a.trial(next_frames),b.trial(next_frames)
    assert first.cost<second.cost
    assert first.frames[-1]['frame_number']==second.frames[-1]['frame_number']==8

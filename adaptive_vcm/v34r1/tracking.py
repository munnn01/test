"""Source-only tracker trials start from identical past state; only selection commits."""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from ..v34.tracking import ByteTracker,track_frames
from ..v32.tracking import iou_matrix


@dataclass
class Trial:
    cost: float
    tracker: object
    frames: list
    links: dict
    reverse: dict
    generation: int


def association_cost(frames,reference,links,reverse):
    if len(frames)!=len(reference): raise ValueError('tracking frame count differs')
    links,reverse=links.copy(),reverse.copy()
    misses=fp=switches=count=0
    for current,source in zip(frames,reference):
        a=np.asarray(source['boxes']).reshape(-1,4)
        b=np.asarray(current['boxes']).reshape(-1,4)
        overlap=iou_matrix(a,b)
        pairs=[]
        if overlap.size:
            ii,jj=linear_sum_assignment(np.where(overlap>=.5,1-overlap,1e6))
            pairs=[(int(i),int(j)) for i,j in zip(ii,jj) if overlap[i,j]>=.5]
        count+=len(a); misses+=len(a)-len(pairs); fp+=len(b)-len(pairs)
        for i,j in pairs:
            reference_id=int(source['track_ids'][i]); candidate_id=int(current['track_ids'][j])
            switches+=int(reference_id in links and links[reference_id]!=candidate_id)
            switches+=int(candidate_id in reverse and reverse[candidate_id]!=reference_id)
            links[reference_id]=candidate_id; reverse[candidate_id]=reference_id
    return (misses+fp+2*switches)/max(count,1),links,reverse


class CausalTrackingGuard:
    def __init__(self):
        self.sequence=None; self.generation=0; self.open=False; self.committed_windows=0

    def begin(self,sequence,source_predictions):
        if self.open: raise ValueError('commit a selected trial before advancing the source')
        if sequence!=self.sequence:
            self.sequence=sequence
            self.source_tracker=ByteTracker(); self.decoded_tracker=ByteTracker()
            self.links={}; self.reverse={}; self.committed_windows=0
        self.reference=track_frames(source_predictions,self.source_tracker)
        self.generation+=1; self.open=True

    def trial(self,predictions):
        if not self.open: raise ValueError('begin a source window first')
        # The wrapper holds the supervision module as an immutable API handle;
        # clone its mutable ByteTrack state, while sharing that module handle.
        tracker=deepcopy(self.decoded_tracker,{id(self.decoded_tracker.sv):self.decoded_tracker.sv})
        frames=track_frames(predictions,tracker)
        cost,links,reverse=association_cost(frames,self.reference,self.links,self.reverse)
        return Trial(cost,tracker,frames,links,reverse,self.generation)

    def commit(self,trial):
        if not self.open or trial.generation!=self.generation:
            raise ValueError('trial is stale or a selection was already committed')
        self.decoded_tracker=trial.tracker; self.links=trial.links; self.reverse=trial.reverse
        self.open=False; self.committed_windows+=1

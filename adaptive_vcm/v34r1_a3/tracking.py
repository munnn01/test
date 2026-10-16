"""Native-cadence causal state; all candidate HOTA scores share one observed past."""
from ..v34.tracking import ByteTracker
from ..v34r1.tracking import CausalTrackingGuard
from .hota_proxy import source_hota


class CausalProxyGuard(CausalTrackingGuard):
    def __init__(self,metric,*,fps=25,lookback=None):
        super().__init__()
        if type(fps) is not int or not 1<=fps<=120: raise ValueError('native integer frame rate required')
        if lookback is not None and (type(lookback) is not int or lookback<1): raise ValueError('positive causal lookback required')
        self.metric,self.fps,self.lookback=metric,fps,lookback
        self.source_history=[]; self.committed_history=[]

    def native_tracker(self):
        tracker=ByteTracker()
        tracker.tracker=tracker.sv.ByteTrack(track_activation_threshold=.25,lost_track_buffer=30,
            minimum_matching_threshold=.8,frame_rate=self.fps,minimum_consecutive_frames=1)
        return tracker

    def begin(self,sequence,source_predictions):
        if self.open: raise ValueError('commit a selected trial before advancing the source')
        if sequence!=self.sequence:
            self.sequence=sequence; self.source_tracker=self.native_tracker(); self.decoded_tracker=self.native_tracker()
            self.links={}; self.reverse={}; self.committed_windows=0
            self.source_history=[]; self.committed_history=[]
        super().begin(sequence,source_predictions)
        self.source_history.extend(self.reference)

    def trial(self,predictions):
        trial=super().trial(predictions)
        score=source_hota(self.source_history,self.committed_history+trial.frames,self.metric,lookback=self.lookback) if self.metric is not None else None
        trial.source_metrics=score
        trial.proxy_costs=[1-score[k] for k in ('HOTA','DetA','AssA')] if score is not None else None
        return trial

    def commit(self,trial):
        super().commit(trial)
        self.committed_history.extend(trial.frames)

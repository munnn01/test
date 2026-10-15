"""Ground-truth-free TEST OD source guard; original supervised training is retained."""
from copy import copy
import torch
import torch.nn.functional as F
from ..v34.tasks import Heads as LegacyHeads, detection_distance
from ..v33ap.tasks import od_forward, ar_forward, prediction, frame_indices, distance
from ..v32vc.train import tensor


class Heads(LegacyHeads):
    @torch.no_grad()
    def costs(self,rgb,record,source_only=False):
        penalty=getattr(self,'cfg',{}).get('ar_top1_penalty',0.)
        if source_only and record.mask[0] and penalty:
            other=copy(record); other.mask=list(record.mask); other.mask[0]=False
            costs=super().costs(rgb,other,source_only=True)
            logits,features=ar_forward(self.action,tensor(rgb,self.device))
            target=torch.as_tensor(record.source_predictions['ar'],device=self.device)[None]
            changed=int(logits.argmax(-1).item()!=target.argmax(-1).item())
            costs[0]=float(.25*F.kl_div(logits.log_softmax(-1),target,reduction='batchmean')+
                           .1*distance(features,record.source_features['ar']))+float(penalty)*changed
            return costs
        if not (source_only and record.task=='od' and record.mask[1]):
            return super().costs(rgb,record,source_only=source_only)
        # Delegate the other active tasks without entering the legacy OD GT-loss
        # branch. No target field is read, even if the record contains OD GT.
        other=copy(record)
        other.mask=list(record.mask)
        other.mask[1]=False
        costs=super().costs(rgb,other,source_only=True)
        indices=frame_indices(len(rgb))
        predictions,features=od_forward(self.detector,tensor(rgb,self.device),indices)
        reference=[record.source_predictions['od'][i] for i in indices]
        costs[1]=detection_distance(reference,[prediction(p) for p in predictions])+.1*float(distance(features,record.source_features['od']))
        return costs

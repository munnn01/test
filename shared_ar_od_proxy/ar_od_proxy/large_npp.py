"""Paper-sized pixel/U-Net extension with causal coarse temporal context."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .models import Condition,ConvGRU,RegionalVideoNPP,validate_video


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels, channels):
        super().__init__(nn.Conv2d(input_channels,channels,3,padding=1),nn.SiLU(),
                         nn.Conv2d(channels,channels,3,padding=1),nn.SiLU())


class LargeRegionalVideoNPP(RegionalVideoNPP):
    """The original branch plus an active multiscale branch, one common RGB output.

    All additional parameters belong to the forward path. Per-frame checkpointing
    recomputes activations during backward without detaching temporal hidden state.
    No teacher maps, future frames, spatial coordinate remapping or side data.
    """
    def __init__(self,width=12,strength=.12,filter_limit=.2,base_channels=29,checkpoint_frames=True):
        super().__init__(width,strength,filter_limit)
        if type(base_channels) is not int or base_channels<4:
            raise ValueError('base_channels must be an integer at least four')
        self.base_channels=base_channels
        self.checkpoint_frames=bool(checkpoint_frames)
        channels=[base_channels*2**i for i in range(5)]
        self.unet_stem=ConvBlock(3,channels[0])
        self.unet_down=nn.ModuleList(ConvBlock(a,b) for a,b in zip(channels,channels[1:]))
        self.unet_conditions=nn.ModuleList(Condition(c) for c in channels)
        self.unet_temporal_in=nn.Conv2d(channels[-1],channels[-2],1)
        self.unet_temporal=ConvGRU(channels[-2])
        self.unet_temporal_out=nn.Conv2d(channels[-2],channels[-1],1)
        self.unet_up=nn.ModuleList(ConvBlock(channels[i]+channels[i-1],channels[i-1])
                                  for i in range(4,0,-1))
        self.unet_rgb=nn.Conv2d(channels[0],3,3,padding=1)
        nn.init.zeros_(self.unet_rgb.weight);nn.init.zeros_(self.unet_rgb.bias)

    @classmethod
    def from_regional(cls,parent,base_channels=29,checkpoint_frames=True):
        model=cls(parent.spatial.out_channels,parent.strength,parent.filter_limit,
                  base_channels,checkpoint_frames).to(next(parent.parameters()).device)
        missing,unexpected=model.load_state_dict(parent.state_dict(),strict=False)
        expected={name for name in model.state_dict() if name.startswith('unet_')}
        if unexpected or set(missing)!=expected:
            raise ValueError('large initialization requires the exact small regional parent')
        return model

    def _frame(self,frame,hidden,codec,qp):
        features=[]
        z=self.unet_stem(frame)*self.unet_conditions[0](frame,codec,qp)
        features.append(z)
        for i,block in enumerate(self.unet_down,1):
            z=block(F.avg_pool2d(z,2))*self.unet_conditions[i](frame,codec,qp)
            features.append(z)
        hidden=self.unet_temporal(self.unet_temporal_in(z),hidden)
        z=z+self.unet_temporal_out(hidden)
        for block,skip in zip(self.unet_up,reversed(features[:-1])):
            z=F.interpolate(z,skip.shape[-2:],mode='bilinear',align_corners=False)
            z=block(torch.cat((z,skip),1))
        return self.unet_rgb(z),hidden

    def forward(self,x,codec,qp):
        validate_video(x)
        if min(x.shape[-2:])<32:
            raise ValueError('large NPP requires spatial dimensions at least32')
        base=super().forward(x,codec,qp)
        hidden,outputs=None,[]
        for index,frame in enumerate(x.unbind(1)):
            if self.checkpoint_frames and self.training and torch.is_grad_enabled():
                delta,hidden=checkpoint(self._frame,frame,hidden,codec,qp,use_reentrant=False)
            else:
                delta,hidden=self._frame(frame,hidden,codec,qp)
            outputs.append((base[:,index]+self.strength*delta.tanh()).clamp(0,1))
        return torch.stack(outputs,1)

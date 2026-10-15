"""Versioned geometry-preserving multiscale candidates; legacy profiles stay exact."""
from dataclasses import dataclass
import cv2
import numpy as np
from ..v34.protocol import ACTIONS
from ..preprocessing import validate_clip


@dataclass(frozen=True)
class Profile:
    name: str
    coefficients: tuple
    sigma_high: float = .8
    sigma_mid: float = 1.6
    background_chroma: float = 1.

    def __post_init__(self):
        a=np.asarray(self.coefficients,float)
        values=np.asarray([self.sigma_high,self.sigma_mid,self.background_chroma],float)
        if (not self.name or a.shape!=(6,) or not np.isfinite(a).all() or a.min() < -1 or a.max() > .25
            or not np.isfinite(values).all() or not 0<self.sigma_high<=self.sigma_mid<=8
            or not .5<=self.background_chroma<=1):
            raise ValueError('invalid bounded V34r1 profile')


PROFILES=tuple(Profile(f'legacy-{i}',tuple(a)) for i,a in enumerate(ACTIONS))+(
    Profile('protected-core-bg-lowpass', (0,0,0,0,-1,-1)),
    Profile('protected-core-bg-wide', (0,0,-.25,0,-1,-1),sigma_mid=3.2),
    Profile('protected-core-bg-coarse', (0,0,-.35,-.1,-1,-1),sigma_mid=4.8),
    Profile('mild-object-wide-background', (-.1,0,-.35,-.1,-1,-1),sigma_mid=3.2),
    Profile('uniform-moderate', (-.5,-.25,-.5,-.25,-.5,-.25)),
    Profile('protected-core-chroma', (0,0,-.2,0,-.75,-.25),background_chroma=.75),
    Profile('uniform-soft', (-.75,-.4,-.75,-.4,-.75,-.4)),
    Profile('uniform-lowpass-1p2', (-1,-1,-1,-1,-1,-1),sigma_mid=1.2),
)


def filter_clip(rgb,maps,profile):
    validate_clip(rgb)
    maps=np.asarray(maps,dtype=np.float32)
    if (maps.shape!=rgb.shape or not np.isfinite(maps).all() or np.any(maps<0)
        or not np.allclose(maps.sum(-1),1,atol=1e-5)):
        raise ValueError('finite normalized nonnegative region maps required')
    if not isinstance(profile,Profile): raise ValueError('versioned Profile required')
    a=np.asarray(profile.coefficients,np.float32)
    if not a.any() and profile.background_chroma==1: return rgb.copy()
    coeff=np.einsum('thwr,rb->thwb',maps,a.reshape(3,2))
    result=[]
    for i,frame in enumerate(rgb):
        x=frame.astype(np.float32)
        g1=cv2.GaussianBlur(x,(0,0),profile.sigma_high,borderType=cv2.BORDER_REPLICATE)
        g2=cv2.GaussianBlur(x,(0,0),profile.sigma_mid,borderType=cv2.BORDER_REPLICATE)
        y=np.rint(np.clip(x+coeff[i,:,:,0,None]*(x-g1)+coeff[i,:,:,1,None]*(g1-g2),0,255)).astype(np.uint8)
        if profile.background_chroma < 1:
            color=cv2.cvtColor(y,cv2.COLOR_RGB2YCrCb).astype(np.float32)
            color[:,:,1:]=128+(color[:,:,1:]-128)*profile.background_chroma
            attenuated=cv2.cvtColor(np.rint(color).clip(0,255).astype(np.uint8),cv2.COLOR_YCrCb2RGB)
            weight=maps[i,:,:,2,None]
            y=np.rint(y*(1-weight)+attenuated*weight).astype(np.uint8)
        result.append(y)
    return np.stack(result)

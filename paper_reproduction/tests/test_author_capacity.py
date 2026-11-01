"""Capacity diagnostic only; production initialization is unchanged."""
import torch
from reproduce.tri_task_models import PaperNPP
def test_original_nonresidual_architecture_can_represent_identity():
    model=PaperNPP(adaptive=False).eval()
    with torch.no_grad():
        for layer in (model.network.conv1,model.network.conv2):
            layer.weight.zero_();layer.bias.zero_()
            for i in range(3):layer.weight[i,i,0,0]=1
        final=model.network.UNet.o[-2]
        final.weight.zero_();final.bias.zero_()
        x=torch.rand(1,1,3,32,48)
        assert torch.allclose(model(x,35),x,atol=1e-7)

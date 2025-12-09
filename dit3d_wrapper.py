from dit3d import DiT
import torch.nn as nn

class DiT3DWrapper(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.model = DiT(in_channels=in_channels, out_channels=out_channels, **kwargs)

    def forward(self, x, t):
        return self.model(x, t)

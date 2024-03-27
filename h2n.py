from torch import nn
import torch
from torchvision.transforms import GaussianBlur

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, x):
        x = self.filter(x)
        return x
    
class HeightToNormal(nn.Module):
    def __init__(
            self,
            filter = None,
            transforms = GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
            strength: float = 1.0,
            invert_x: bool = False,
            invert_y: bool = False,
            invert_height: bool = False,
        ):
        super().__init__()
        self.filter = filter
        self.strength = strength
        self.invert_x = invert_x
        self.invert_y = invert_y
        self.invert_height = invert_height
        self.transforms = transforms

    def pre_process(self, height):
        height = -1.0 * height + 1.0 if not self.invert_height else height
        if self.transforms:
            height = self.transforms(height)
        return height

    def forward(self, height):
        x = self.pre_process(height)
        out = self.filter(x)   # height Bx1xHxW, out: Bx2xHxW
        dx = -1.0 * out[:, 0] if self.invert_x else out[:, 0]
        dy = -1.0 * out[:, 1] if self.invert_y else out[:, 1]
        normal = torch.stack([dx, dy, torch.ones(dx.shape)/self.strength], dim=1)
        normal = torch.nn.functional.normalize(normal, dim=1)
        return normal
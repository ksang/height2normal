from torch import nn
import torch
from torchvision.transforms import GaussianBlur
    
class HeightToNormal(nn.Module):
    def __init__(
            self,
            filter_type = "sobel",
            blur = GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1)),
            strength: float = 1.0,
            invert_x: bool = False,
            invert_y: bool = False,
            invert_height: bool = False,
        ):
        super().__init__()
        self.filter_kernel = self.get_filter_kernel(filter_type)
        self.strength = strength
        self.invert_x = invert_x
        self.invert_y = invert_y
        self.invert_height = invert_height
        self.blur = blur

    def get_filter_kernel(self, filter_type): # 2x1x3x3  
        if filter_type == "sobel":
            sobel_x = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]) / 2.0
            sobel_y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]) / 2.0
            return torch.stack([sobel_x, sobel_y]).unsqueeze(1)   
        elif filter_type == "scharr":
            scharr_x = torch.tensor([[47, 0, -47], [162, 0, -162], [47, 0.0, -47]]) / 255.0
            scharr_y = torch.tensor([[47, 162, 47], [0, 0, 0], [-47, -162, -47]]) / 255.0
            return torch.stack([scharr_x, scharr_y]).unsqueeze(1) 
        else:
            raise ValueError(f"Filter type {filter_type} not supported.")   
        
    def pre_process(self, height):
        height = -1.0 * height + 1.0 if not self.invert_height else height
        if self.blur:
            height = self.blur(height)
        return height

    def filter(self, x):
        x = self.pre_process(x)
        x = torch.nn.functional.conv2d(x, self.filter_kernel, bias=None, stride=1, padding=1, groups=1)   # Bx2xHxW
        return x
    
    def filter_grad(self, x):
        x = self.filter(x)
        Gx, Gy = x[:,0], x[:,1]
        return torch.sqrt(Gx**2 + Gy**2)

    # Expecting input height is a tensor in shape [B, C, H, W] and in value range [0, 1]
    def forward(self, height):
        x = self.filter(height)
        dx = -1.0 * x[:, 0] if self.invert_x else x[:, 0]
        dy = -1.0 * x[:, 1] if self.invert_y else x[:, 1]
        normal = torch.stack([dx, dy, torch.ones(dx.shape, device=dx.device)/self.strength], dim=1)
        normal = torch.nn.functional.normalize(normal, dim=1)
        return normal
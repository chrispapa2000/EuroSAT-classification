from typing import List

import torch
import torch.nn as nn
from einops import rearrange

from ViT.layers import Patchifier
from Swin.layers import SwinTransformerStage

class SwinTransformer(nn.Module):
    def __init__(self, dim: int, heads: int, stage_layers: List[int] = [2,2,6,2], num_classes: int = 10, h: int = 32, w: int = 32, c : int = 3, patch_size: int = 4, window_size: int = 4, act : nn.Module = nn.GELU):
        super().__init__()
        self.patchifier = Patchifier(dim, h, w, c, patch_size)
        
        self.H = h
        self.W = w
        self.C = c 
        self.window_size = window_size
        self.dim = dim
        self.patch_size = patch_size
        
        stages = []
        
        # -- stage 1 -- #
        h //= patch_size
        w //= patch_size
        stage = SwinTransformerStage(dim=dim, heads=heads, n_layers=stage_layers[0], h=h, w=w, window_size=window_size, act=act, downsampling_factor=1)
        stages.append(stage)
        
        for i in range(1,len(stage_layers)):
            # -- i-th stage -- #     
            n_layers = stage_layers[i]
            stage = SwinTransformerStage(dim=dim, heads=heads, n_layers=n_layers, h=h, w=w, window_size=window_size, act=act, downsampling_factor=2)
            stages.append(stage)
            h //= 2
            w //= 2
            dim *= 2
            
        self.stages = nn.ModuleList(stages)
        
        self.clf = nn.Linear(dim, num_classes) 
        
        
    def forward(self, x: torch.Tensor):
        B, C, H, W= x.shape
        assert H == self.H and W == self.W, f"Wrong input dimentions! Provide input with dimensions: (H,W)=({self.H}, {self.W})"
         
        # -- Patchify input images -- #
        x = self.patchifier(x)
        
        # -- Arange flattened patches into square grid -- #
        h, w = H//self.patch_size, W//self.patch_size
        x = rearrange(x, 'B (n1 n2) D -> B n1 n2 D', n1=h)
        
        # -- Apply Swin Transformer Stages -- #
        for stage in self.stages:
            x = stage(x)
        
        # -- Average Across Spatial dimensions -- #
        x_mean = x.mean(dim=(1,2)) 
        
        # -- Compute logits and return -- #
        logits = self.clf(x_mean)
        return logits
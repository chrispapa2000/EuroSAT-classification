import torch
import torch.nn as nn
from einops import rearrange

import math
from typing import Optional

from Swin.helper_functions import (
    shift,
    undo_shift,
    fold,
    unfold,
    get_window_grid,
    get_attn_mask_from_windows,
    get_rel_position_indices
)

from ViT.helper_functions import exists
from ViT.layers import MLP


class Attn(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim = dim
        self.heads = heads
        assert dim % heads == 0, "The number of heads needs to devide the hidden dim"
        
        self.norm = nn.LayerNorm(dim)
        
        self.w_q, self.w_k, self.w_v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
    
    def apply_mask(self, attn: torch.Tensor, mask: torch.Tensor):
        mask = mask.repeat_interleave(self.heads, dim=0) # repeat in batch dim by head number
        if len(attn.shape) > len(mask.shape): # clone mask for each element of the sequence
            mask = mask.repeat((1,attn.shape[-1])) # repeat by sequence length  
            mask = rearrange(mask, 'B (N1 N2) -> B N1 N2', N1=attn.shape[-1]) 
        attn = torch.where(mask==1, attn, -float('inf'))
        return attn
        
    def forward(self, x, mask: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None):
        """
        Args:
            x (input sequence): [B, N, D]
            mask (attention mask): [B, N] or [B, N, N], 0 for masked positions
            bias (relative positions bias embeddings) [N, N, N_heads], appropriate size for a single window; will be added to attention matrix
        Output:
            out (output sequence): [B, N, D]
            attn (attention weights): [B, N, N]
        """
        B = x.shape[0]
        x = self.norm(x)
        
        # compute q,k,v features
        q, k, v = [layer(x) for layer in [self.w_q, self.w_k, self.w_v]]
        q, k, v = [rearrange(seq, 'B N (H D) -> (B H) N D', H=self.heads) for seq in [q, k, v]]
        
        # compute attention weights and apply optional mask
        attn = torch.matmul(q, k.transpose(1,2)) / math.sqrt(self.dim)
        if bias is not None:
            bias = rearrange(bias[None,:,:,:].repeat((B, 1, 1, 1)), 'B n1 n2 (H D) -> (B H) n1 n2 D', H=self.heads).squeeze(-1)
            attn = attn + bias
        if mask is not None:
            attn = self.apply_mask(attn, mask)
        attn = attn.softmax(dim=-1) 
        
        # compute output and concat heads
        out = torch.matmul(attn, v)
        out = rearrange(out, '(B H) N D -> B N (H D)', H=self.heads)
        
        # rearrange attn as separate heads
        attn = rearrange(attn, '(B H) ... -> B H ...', H=self.heads)
        
        return out, attn

class SwinLayer(nn.Module):
    def __init__(self, dim: int, heads: int, h: int, w:int, window_size: int = 7, shifted: bool = False,  hidden_dim: Optional[int] = None, act : nn.Module = nn.GELU):
        super().__init__()
        hidden_dim = dim if not exists(hidden_dim) else hidden_dim
        
        self.H = h
        self.W = w
        self.window_size = window_size
        self.shifted = shifted

        self.multihead_attn = Attn(dim, heads)
        self.mlp = MLP(dim, hidden_dim, act)
        
        self.relative_positions = get_rel_position_indices(window_size)
        self.bias_embeddings = nn.Embedding(self.relative_positions.max()+1, heads)
    
           
    def forward(self, x, return_attn=False):
        """
        Args:
            x (input): [B, N, N, D]
            return_attn (bool, optional): whether to return attn weights. Defaults to False.

        Returns:
            x: (transformed input): [B, N, N, D]
            attn (attention weights, optionally)
        """
        B = x.shape[0]
        bias = self.bias_embeddings(self.relative_positions.to(x.device))
        
        # --- CASE: SHIFTED --- #
        if self.shifted:
            # shift 
            x = shift(x, self.window_size)

            # fold by window
            x = fold(x, self.window_size)

            # attention and residual connection
            window_grid = get_window_grid(self.H, self.window_size) # create grid of windows
            attn_mask = get_attn_mask_from_windows(window_grid, self.window_size).to(x.device) # calculate mask from window grid
            attn_mask = attn_mask.repeat((B,1,1)) # repeat mask by batch size  
            
            x_attn, attn = self.multihead_attn.forward(x, mask=attn_mask, bias=bias) # attention
            x = x + x_attn # residual
            
            # mlp and residual connection
            x_mlp = self.mlp(x)
            x = x + x_mlp
            
            # unfold 
            x = unfold(x, self.H, self.W, self.window_size)

            # undo shift
            x = undo_shift(x, self.window_size)
        
        # --- CASE: NOT SHIFTED --- #
        else:
            # fold by window
            x = fold(x, self.window_size)

            # attention and residual connection
            x_attn, attn = self.multihead_attn.forward(x, mask=None, bias=bias) # attention
            x = x + x_attn # residual

            # mlp and residual connection
            x_mlp = self.mlp(x)
            x = x + x_mlp

            # unfold 
            x = unfold(x, self.H, self.W, self.window_size)
        
        if return_attn:
            return x, attn
        return x
    
class PatchMergingLayer(nn.Module):
    def __init__(self, dim:int, downsampling_factor:int=2, expansion_factor=2):
        super().__init__()
        self.fc = nn.Linear((downsampling_factor**2)*dim, expansion_factor*dim)
        self.downsampling_factor = downsampling_factor
    
    def forward(self, x):
        """
        Args:
            x (input): [B N N D]
        Returns:
            x (input with merged patches (factor of 2)): [B N/2 N/2 2*D]
        """
        x = rearrange(x, 'B (n1 f1) (n2 f2) D -> B n1 n2 (f1 f2 D)', f1=self.downsampling_factor, f2=self.downsampling_factor)
        x = self.fc(x)
        return x       
    

class SwinTransformerStage(nn.Module):        
    def __init__(self, dim, heads, n_layers, h: int = 32, w: int = 32, window_size: int = 4, act : nn.Module = nn.GELU, downsampling_factor: int = 2):
        """
        Args:
            dim (input dim)
            heads (transformer heads)
            n_layers (transformer layers)
            h (input height)
            w (input width)
            window_size (window size for swin transformer blocks): Defaults to 4.
            act (activation): Defaults to nn.GELU.
            downsampling_factor (downsampling applied in patch merging layer. If equal to 1, no downsampling is applied. If downsampling is applied, the input dim is increased by a factor of 2): Defaults to 2.
        """
        super().__init__()
        
        self.patch_merger = nn.Identity() if downsampling_factor == 1 else PatchMergingLayer(dim, downsampling_factor, 2)
        
        dim *= downsampling_factor
        h, w = h//downsampling_factor, w//downsampling_factor
        
        layers = []
        shifted = False
        for _ in range(n_layers):
            layers.append(SwinLayer(dim=dim, heads=heads, h=h, w=w, window_size=window_size, shifted=shifted, hidden_dim=4*dim, act=act))
            shifted = not shifted # alternate between shifted and not shifted layers
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x (input): [B N N D]
        """
        x = self.patch_merger(x)
        for layer in self.layers:
            x = layer(x)
        
        return x
            
from typing import Optional
import math
import torch
import torch.nn as nn
from einops import rearrange

from ViT.helper_functions import exists

# ---- ViT modules / layers ----- #

class Patchifier(nn.Module):
    def __init__(self, input_dim: int, h: int = 32, w: int = 32, c : int = 3, patch_size: int = 4):
        super().__init__()
        
        assert h * w % patch_size**2 == 0, "The square of the patchification factor needs to devide the product H*W"
        self.patch_size = patch_size
        self.n = h*w // patch_size**2 
        self.input_dim = input_dim
        
        self.input_reshape = lambda x : rearrange(x, 'B C (H P1) (W P2) -> B (H W) (P1 P2 C)', P1=patch_size, P2=patch_size)
        self.input_proj = nn.Linear((patch_size**2)*c, input_dim) 
    
    def forward(self, x):
        """
        Args:
            x (batch of images): [B, C, H, W]
        output:
            x (patchified transformer input): [B N D]
        """
        x = self.input_reshape(x)
        x = self.input_proj(x)
        return x
    
class Attn(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim = dim
        self.heads = heads
        assert dim % heads == 0, "The number of heads needs to devide the hidden dim"
        
        self.norm = nn.LayerNorm(dim)
        
        self.w_q, self.w_k, self.w_v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        """
        Args:
            x (input sequence): [B, N, D]
        Output:
            out (output sequence): [B, N, D]
            attn (attention weights): [B, N, N]
        """
        x = self.norm(x)
        
        # compute q,k,v features
        q, k, v = [layer(x) for layer in [self.w_q, self.w_k, self.w_v]]
        q, k, v = [rearrange(seq, 'B N (H D) -> (B H) N D', H=self.heads) for seq in [q, k, v]]
        
        # compute attention weights
        attn = torch.matmul(q, k.transpose(1,2)) / math.sqrt(self.dim)
        attn = attn.softmax(dim=-1) 
        
        # compute output and concat heads
        out = torch.matmul(attn, v)
        out = rearrange(out, '(B H) N D -> B N (H D)', H=self.heads)
        
        # rearrange attn as separate heads
        attn = rearrange(attn, '(B H) ... -> B H ...', H=self.heads)
        
        return out, attn

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, act: nn.Module = nn.GELU):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.l1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), act())
        self.l2 = nn.Sequential(nn.Linear(hidden_dim, input_dim), act())
            
    def forward(self, x):
        x = self.norm(x)
        x = self.l1(x)
        x = self.l2(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, hidden_dim: Optional[int] = None, act : nn.Module = nn.GELU):
        super().__init__()
        hidden_dim = dim if not exists(hidden_dim) else hidden_dim
        
        self.multihead_attn = Attn(dim, heads)
        self.mlp = MLP(dim, hidden_dim, act)
        
    def forward(self, x, return_attn=False):
        # apply attention and residual connection
        x_attn, attn = self.multihead_attn(x)
        x = x + x_attn
        
        # apply mlp and residual connection
        x_mlp = self.mlp(x)
        x = x + x_mlp
        
        if return_attn:
            return x, attn
        return x
    
    
# ---- SWIN Transformer functions / modules ---- #

def shift(x, window_size):
    """
    Args:
        x (B H W D): input to be shifted by half window_size for shifted window attention
    Returns:
        shifted x
    """
    padder = nn.ConstantPad2d(padding=(0,window_size//2,0, window_size//2), value=0)
    
    x = rearrange(x, 'B H W D -> B D H W')
    padded_x = padder(x)
    rolled_x = torch.roll(padded_x, (-window_size//2,-window_size//2), dims=(2,3))
    
    all_indices = torch.arange(0, rolled_x.shape[-1])
    indices = torch.cat([all_indices[:-window_size], all_indices[-window_size//2:]])
    
    rolled_x = rolled_x[:,:,:,indices]
    rolled_x = rolled_x[:,:,indices,:]
    rolled_x = rearrange(rolled_x, 'B D H W -> B H W D')
    return rolled_x

def fold(x, window_size):
    return rearrange(x, 'B (H M1) (W M2) D -> (B H W) (M1 M2) D', M1=window_size, M2=window_size)

def get_window_grid(n, window_size):
    n_windows = ((n+window_size) // window_size) ** 2
    windows = [torch.full((window_size, window_size), i).unsqueeze(0) for i in range(n_windows)]
    windows = rearrange(torch.cat(windows, dim=0), '(a b) c d -> (a c) (b d)', a = math.floor(math.sqrt(n_windows)))
    rolled_windows = torch.roll(windows, (-window_size//2,-window_size//2), dims=(0,1))
    rolled_windows = rolled_windows[:,:-window_size]
    rolled_windows = rolled_windows[:-window_size, :]
    return rolled_windows

def get_attn_mask_from_windows(window_grid, window_size):
    window_grid_shifted = shift(window_grid[None,:,:,None], window_size)
    window_grid_folded = fold(window_grid_shifted, window_size)[:,:,0]
    attn_mask = window_grid_folded[:,None,:] == window_grid_folded[:,:,None].repeat((1,1,window_grid_folded.shape[-1]))
    return attn_mask
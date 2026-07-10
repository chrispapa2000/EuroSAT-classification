
import math
import torch
import torch.nn as nn
from einops import rearrange

# ---- SWIN Transformer functions ---- #

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

def undo_shift(x, window_size):
    """
    Args:
        x (B H W D): input to be unshifted by half window_size after window attention
    Returns:
        unshifted x
    """
    padder = nn.ConstantPad2d(padding=(window_size//2, 0, window_size//2, 0), value=0)
    
    x = rearrange(x, 'B H W D -> B D H W')
    padded_x = padder(x)
    rolled_x = torch.roll(padded_x, (window_size//2,window_size//2), dims=(2,3))
    # return rolled_x
    
    all_indices = torch.arange(0, rolled_x.shape[-1])
    indices = torch.cat([all_indices[:window_size//2], all_indices[window_size:]])
    
    rolled_x = rolled_x[:,:,:,indices]
    rolled_x = rolled_x[:,:,indices,:]
    rolled_x = rearrange(rolled_x, 'B D H W -> B H W D')
    return rolled_x

def fold(x, window_size):
    return rearrange(x, 'B (H M1) (W M2) D -> (B H W) (M1 M2) D', M1=window_size, M2=window_size)

def unfold(x, h, w, window_size):
    h //= window_size
    w //= window_size
    return rearrange(x, '(B H W) (M1 M2) D -> B (H M1) (W M2) D', H=h, W=w, M1=window_size)

def get_window_grid(n, window_size):
    """
    creates n x n grid separated into windows with size window_size x window_size. Each window is assigned a distinct integer. Cyclic rotation is applied as in Swin Transformer paper.
    """
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

def get_rel_position_indices(window_size):
    # relative position indices for one window
    x_coords, y_coords = torch.arange(window_size)[:, None, None], torch.arange(window_size)[None, :, None]
    window = torch.cat([
        x_coords.expand((window_size, window_size, 1)), 
        y_coords.expand((window_size, window_size, 1))], dim=-1)
    window_flat = window.flatten(0,1)
    window_flat = window_flat[None,:,:].repeat((window_flat.shape[0],1,1))
    rel_pos = window_flat - rearrange(window_flat, 'a b c -> b a c')
    rel_pos += window_size-1
    rel_pos[:,:,0] *= rel_pos.max()
    rel_pos = rel_pos.sum(dim=-1)
    return rel_pos
from typing import Optional
import torch
import torch.nn as nn

from ViT.layers import (
    Patchifier,
    TransformerLayer,
)

class ViT(nn.Module):
    def __init__(self, dim: int, heads: int, n_layers : int, num_classes: int = 10, hidden_dim: Optional[int] = None, h: int = 32, w: int = 32, c : int = 3, patch_size: int = 4, act : nn.Module = nn.GELU):
        super().__init__()
        self.patchifier = Patchifier(dim, h, w, c, patch_size)
        
        self.sequence_len = h * w // patch_size**2
        self.positional_encodings = nn.Embedding(self.sequence_len, dim)
        
        self.clf_embedding = nn.Parameter(torch.randn([1, dim]))
        
        
        layers = []
        for _ in range(n_layers):
            layers.append(TransformerLayer(dim, heads, hidden_dim, act))
        self.layers = nn.ModuleList(layers)
        
        self.clf = nn.Linear(dim, num_classes)
        
    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        # patchify input images
        x = self.patchifier(x)
        
        # apply positional encodings
        posemb = self.positional_encodings(torch.arange(self.sequence_len, device=x.device))
        x = x + posemb.unsqueeze(0)
        
        # prepend clf embedding
        clf_emb = self.clf_embedding[:, :].repeat((B,1))
        x = torch.cat([clf_emb.unsqueeze(1), x], dim=1)
        
        # pass through the transformer
        for layer in self.layers:
            x = layer(x, return_attn=False)
        
        # only keep the clf embeddings
        x = x[:, 0, :]
        
        # calculate logits
        out = self.clf(x)
        return out
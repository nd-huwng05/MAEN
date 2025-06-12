import torch
from torch import nn

class MAEN(nn.Module):
    def __init__(self, args):
        super().__init__()


    def forward_encoder(self, x):
        cls_token = []
        attn_map = []
        return cls_token, attn_map

    def forward_decoder(self, x):
        x_recon = x
        return x_recon

    def forward(self, x):
        cls_token, attn_map = self.forward_encoder(x)
        x_recon = self.forward_decoder(x)
        return cls_token, attn_map, x_recon
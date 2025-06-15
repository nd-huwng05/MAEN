from models.maen import MAEN
from functools import partial
import torch.nn as nn

def MAE_Net(args):
    model = MAEN(patch_size=16, embed_dim=256, depth=3, num_heads=4,
        decoder_embed_dim=128, decoder_depth=3, decoder_num_heads=4,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    return model
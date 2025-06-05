import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTransformer(nn.Module):
    def __init__(self, img_size=128, patch_size=16, embed_dim=256, num_layers=4, num_heads=8):
        super(VisionTransformer, self).__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches_per_dim = img_size // patch_size
        self.num_patches = self.num_patches_per_dim ** 2

        # Patch embedding: Conv2d chia ảnh thành patch và nhúng vào không gian D
        self.patch_embed = nn.Conv2d(in_channels=3, out_channels=embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

        # CLS token và Positional Encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder: tái tạo lại patch -> ảnh
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, patch_size * patch_size * 3),
            nn.Unflatten(1, (3, patch_size, patch_size))
        )

    def forward_encoder(self, x):
        B = x.size(0)

        # Step 1: Patch embedding
        patch_feature_map = self.patch_embed(x)               # [B, D, H', W']
        feature_map = patch_feature_map.clone()               # Giữ lại feature map gốc

        # Step 2: Flatten cho transformer
        x = patch_feature_map.flatten(2).transpose(1, 2)      # [B, T, D], T = H'*W'

        # Step 3: Thêm CLS token và Positional Encoding
        cls_tokens = self.cls_token.expand(B, -1, -1)         # [B, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)                 # [B, T+1, D]
        x = x + self.pos_embed[:, :x.size(1), :]              # [B, T+1, D]

        # Step 4: Transformer Encoder
        x = self.encoder(x)                                   # [B, T+1, D]

        # Step 5: Tách ra cls_token và patch tokens
        cls_token = x[:, 0]                                   # [B, D]
        patch_tokens = x[:, 1:]                               # [B, T, D]

        # Attention map giả lập (nếu cần hiển thị)
        attn_map = torch.rand(B, 4, self.num_patches, self.num_patches)  # [B, H, T, T]

        return cls_token, patch_tokens, attn_map, feature_map

    def forward_decoder(self, patch_tokens):
        # patch_tokens: [B, T, D]
        B, T, D = patch_tokens.shape
        side = int(T ** 0.5)
        assert side * side == T, "patch_tokens length must be a perfect square"

        x = patch_tokens.reshape(-1, D)                       # [B*T, D]
        x = self.decoder(x)                                   # [B*T, 3, P, P]

        # Gộp lại thành ảnh
        patch_recon = x.view(B, side, side, 3, self.patch_size, self.patch_size)  # [B, H, W, C, P, P]
        patch_recon = patch_recon.permute(0, 3, 1, 4, 2, 5).contiguous()           # [B, C, H, P, W, P]
        x_recon = patch_recon.view(B, 3, self.img_size, self.img_size)            # [B, 3, H', W']

        return x_recon

    def forward(self, x):
        cls_token, patch_tokens, attn_map, feature_map = self.forward_encoder(x)
        x_recon = self.forward_decoder(patch_tokens)
        return cls_token, attn_map, x_recon, feature_map

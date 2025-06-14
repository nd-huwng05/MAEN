import torch
from torch import nn
from models.base.cvt import ConvEmbed, Block
from einops import rearrange

class MAEN(nn.Module):
    def __init__(self, img_size=(512,512), patch_size=16, in_chans=1, out_chans=1,
                 embed_dim=256, depth=3, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 masking_method="random_masking"):
        super().__init__()
        self.masking = getattr(self, masking_method)
        self.patch_embed = ConvEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_size,
            padding=0,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        self.patch_size = patch_size
        self.num_patches = img_size[0] // patch_size * img_size[1] // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                  norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.out_chans = out_chans
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * out_chans, bias=True)
        self.decoder_pred_var = nn.Linear(decoder_embed_dim, patch_size ** 2 * out_chans, bias=True)

    def patchify(self, imgs, chans=None):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p

        x = imgs.reshape(shape=(imgs.shape[0], self.out_chans if chans is None else chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * (self.out_chans if chans is None else chans)))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        C = self.out_chans
        h = w = int(x.shape[1] ** 0.5)

        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, C))  # (N, h, w, p, p, C)
        x = torch.einsum('nhwpqc->nchpwq', x)             # (N, C, h, p, w, p)
        imgs = x.reshape(shape=(x.shape[0], C, h * p, w * p))  # (N, C, H, W)
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, D, H, W = x.shape  # batch, length, dim
        L = H * W
        x = rearrange(x, 'b c h w -> b (h w) c')
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        self.masked_H = H
        self.masked_W = int(W * (1. - mask_ratio))
        self.H = H
        self.W = W
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x, mask, ids_restore = self.masking(x, mask_ratio)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        for blk in self.blocks:
            x = blk(x, self.masked_H, self.masked_W)
        x = self.norm(x)
        cls_token = x[:, 0]
        return x, mask, ids_restore, cls_token

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)
        for blk in self.decoder_blocks:
            x = blk(x, self.H, self.W)
        x = self.decoder_norm(x)
        x = x[:, 1:, :]
        x_hat = self.decoder_pred(x)
        log_var = self.decoder_pred_var(x)
        return x_hat, log_var

    def forward(self, img, mask_ratio=0.5):
        x, mask, ids_restore, cls_token = self.forward_encoder(img, mask_ratio)
        x, log_var = self.forward_decoder(x, ids_restore)
        if self.training:
            log_var = self.unpatchify(log_var)
            x = self.unpatchify(x)
            return {'x_hat': x, 'log_var': log_var, 'z': cls_token,
                    'features': cls_token, 'mask': mask}
        else:
            img = self.patchify(img)
            return {'x_hat': x, 'log_var': log_var, 'img': img, 'mask': mask}
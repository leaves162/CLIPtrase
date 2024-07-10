from typing import List
import math
import torch
from torch import nn
from torch.nn import functional as F
from .origin_clip import VisionTransformer
from einops import rearrange

class FeatureExtractor(nn.Module):
    def __init__(
        self,
        visual_encoder: VisionTransformer,
        frozen_exclude=[],
    ):
        super().__init__()
        self.input_resolution = visual_encoder.input_resolution
        self.patch_size = visual_encoder.patch_size
        self.output_dim = visual_encoder.output_dim
        self.width = visual_encoder.width
        self.heads = visual_encoder.heads
        self.conv1 = visual_encoder.conv1
        self.class_embedding = visual_encoder.class_embedding
        self.positional_embedding = visual_encoder.positional_embedding
        self.ln_pre = visual_encoder.ln_pre
        self.ln_post = visual_encoder.ln_post
        self.proj = visual_encoder.proj

        self.resblocks = visual_encoder.transformer.resblocks

        self.frozen_exclude = frozen_exclude
        self._freeze(self.frozen_exclude)

    def forward(self, x: torch.Tensor):
        B, nc, w, h = x.shape
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        self.h,self.w = x.shape[-2:]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        if x.shape[1] != self.positional_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, w, h).to(x.dtype)
        else:
            x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        attn_weights = []

        for idx,blk in enumerate(self.resblocks,start=1):
            if idx==11:
                x, attn_i = blk(x, self_out_weight = True)
            elif idx==12:
                x, attn_i = blk(x, self_qk_weight = True)
            else:
                x, attn_i = blk(x)
            attn_weights.append(attn_i)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj
        attn_weights = torch.stack(attn_weights, dim=0)
        return x, attn_weights

    def _freeze(self, frozen_exclude):
        if "all" in frozen_exclude:
            return
        for name, param in self.named_parameters():
            if not any([exclude in name for exclude in frozen_exclude]):
                param.requires_grad = False
    
    def interpolate_pos_encoding(self, x, w, h):
        # h，w pos插值
        npatch = x.shape[1] - 1
        N = self.positional_embedding.shape[0] - 1
        if npatch == N and w == h:
            return self.positional_embedding
        class_pos_embed = self.positional_embedding[[0]]
        patch_pos_embed = self.positional_embedding[1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

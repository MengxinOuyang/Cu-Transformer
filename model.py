"""
Cu-Transformer: Integrated Prediction Model for Copper Converter Blowing

Architecture:
  - PatchEmbed (Swin-style): 8-channel input (3 RGB + 5 production params)
  - Stage 1-3: RepViT blocks (CVPR 2024) with PatchMerging between stages
  - Auxiliary branch (after Stage 3): RepViT block -> GAP -> period classification
  - Stage 4: SHViT block (single-head vision transformer) for high-level semantics
  - LiteMLA: multi-scale linear attention for feature refinement
  - Main head: regression for Cu/Fe/S composition + time-to-endpoint

Reference:
  SwinTransformer: https://github.com/microsoft/Swin-Transformer
  RepViT: https://github.com/THU-MIG/RepViT (CVPR 2024)
  SHViT: https://github.com/ysjsimon/SHViT
  LiteMLA (EfficientViT): https://github.com/mit-han-lab/efficientvit (ICCV 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from SwinTransformer import PatchEmbed, PatchMerging
from SHViTBlock import SHViTBlock
from LiteMLA import LiteMLA
from repvit import RepViTBlock


class CuTransformer(nn.Module):
    """
    Cu-Transformer: Multi-task model for simultaneous prediction of
    blowing endpoint time and melt composition in copper converters.

    Args:
        img_size: input image resolution (default 224)
        patch_size: patch embedding kernel size (default 4)
        in_c: image channels (default 3 for RGB)
        num_classes: number of blowing periods (default 4: B1, B2, S1, S2)
        embed_dim: base embedding dimension (default 96)
        num_extra_features: number of production parameters (default 5)
    """

    def __init__(self, img_size=224, patch_size=4, in_c=3, num_classes=4,
                 embed_dim=96, num_extra_features=5):
        super(CuTransformer, self).__init__()

        # Patch embedding: 3 image channels + 5 production params = 8 input channels
        self.patch_embed = PatchEmbed(
            in_c=in_c + num_extra_features,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        # ---- Stage 1: RepViT blocks (embed_dim=96, spatial=56x56) ----
        dim1 = embed_dim
        self.stage1 = nn.Sequential(
            RepViTBlock(inp=dim1, oup=dim1, kernel_size=3, stride=1,
                        hidden_dim=dim1 * 2),
            RepViTBlock(inp=dim1, oup=dim1, kernel_size=3, stride=1,
                        hidden_dim=dim1 * 2),
        )
        self.merge1 = PatchMerging(dim=dim1, c2=dim1 * 2)

        # ---- Stage 2: RepViT blocks (dim=192, spatial=28x28) ----
        dim2 = embed_dim * 2
        self.stage2 = nn.Sequential(
            RepViTBlock(inp=dim2, oup=dim2, kernel_size=3, stride=1,
                        hidden_dim=dim2 * 2),
            RepViTBlock(inp=dim2, oup=dim2, kernel_size=3, stride=1,
                        hidden_dim=dim2 * 2),
        )
        self.merge2 = PatchMerging(dim=dim2, c2=dim2 * 2)

        # ---- Stage 3: RepViT blocks (dim=384, spatial=14x14) ----
        dim3 = embed_dim * 4
        self.stage3 = nn.Sequential(
            RepViTBlock(inp=dim3, oup=dim3, kernel_size=3, stride=1,
                        hidden_dim=dim3 * 2),
            RepViTBlock(inp=dim3, oup=dim3, kernel_size=3, stride=1,
                        hidden_dim=dim3 * 2),
            RepViTBlock(inp=dim3, oup=dim3, kernel_size=3, stride=1,
                        hidden_dim=dim3 * 2),
            RepViTBlock(inp=dim3, oup=dim3, kernel_size=3, stride=1,
                        hidden_dim=dim3 * 2),
            RepViTBlock(inp=dim3, oup=dim3, kernel_size=3, stride=1,
                        hidden_dim=dim3 * 2),
            RepViTBlock(inp=dim3, oup=dim3, kernel_size=3, stride=1,
                        hidden_dim=dim3 * 2),
        )

        # ---- Auxiliary branch: blowing-period classification ----
        self.aux_repvit = RepViTBlock(inp=dim3, oup=dim3, kernel_size=3,
                                       stride=1, hidden_dim=dim3 * 2)
        self.aux_norm = nn.LayerNorm(dim3)
        self.aux_head = nn.Linear(dim3, num_classes)

        self.merge3 = PatchMerging(dim=dim3, c2=dim3 * 2)

        # ---- Stage 4: SHViT block (dim=768, spatial=7x7) ----
        dim4 = embed_dim * 8
        self.stage4_shvit = SHViTBlock(dim=dim4, qk_dim=16, pdim=32, type='s')

        # ---- LiteMLA: multi-scale linear attention ----
        self.lite_mla = LiteMLA(in_channels=dim4, out_channels=dim4, scales=(5,))

        # ---- Output heads ----
        self.norm = nn.LayerNorm(dim4)
        self.head = nn.Linear(dim4, num_classes)  # 4 outputs: Cu, Fe, S, time

    def forward(self, img, extra_features):
        """
        Args:
            img: (B, 3, H, W) melt cooling sample images
            extra_features: (B, 5) production parameters
        Returns:
            final_output: (B, 4) predicted [Cu%, Fe%, S%, time-to-endpoint]
            aux_output: (B, num_classes) period classification logits
        """
        B, _, H, W = img.shape

        # Concatenate image and production parameters as additional channels
        extra_features = extra_features.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)
        x = torch.cat([img, extra_features], dim=1)

        # Stages 1-3 with PatchMerging
        x = self.patch_embed(x)
        x = self.stage1(x)
        x = self.merge1(x)
        x = self.stage2(x)
        x = self.merge2(x)
        x = self.stage3(x)

        # ---- Auxiliary branch: period classification ----
        aux_feat = self.aux_repvit(x)
        aux_pooled = aux_feat.mean(dim=[2, 3])         # global average pool
        aux_pooled = self.aux_norm(aux_pooled)
        aux_logits = self.aux_head(aux_pooled)          # (B, num_classes)

        # ---- Main branch: composition + time regression ----
        x = self.merge3(x)
        x = self.stage4_shvit(x)
        x = self.lite_mla(x)
        x = x.mean(dim=[2, 3])                          # global average pool
        x = self.norm(x)
        final_output = self.head(x)                      # (B, 4)

        # Composition constraint: Cu + Fe + S = 100%, each >= 0
        composition = final_output[:, :3]
        composition = F.relu(composition)                # ensure non-negative
        epsilon = 1e-6
        comp_sum = composition.sum(dim=1, keepdim=True) + epsilon
        composition = composition / comp_sum * 100       # normalize to 100%

        final_output = torch.cat([composition, final_output[:, 3:]], dim=1)

        return final_output, aux_logits

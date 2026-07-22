import torch
from timm.models.vision_transformer import trunc_normal_
try:
    from timm.layers import SqueezeExcite
except ImportError:
    from timm.models.layers import SqueezeExcite

'''

这篇论文提出了一种高效的单头视觉Transformer（SHViT）设计，以解决多头注意力机制中的计算冗余问题，
并优化了Transformer在资源受限设备上的性能。
Vision Transformers (ViTs) 已在许多计算机视觉任务中展现出优异的表现，
但其多头自注意力机制（Multi-Head Self-Attention, MHSA）带来了较大的计算复杂度和冗余。
当前高效的ViTs模型通常使用4×4的patch embedding和多头注意力来处理特征，但这些模型在内存访问和计算方面仍存在冗余。

本文提出了SHViT的设计：分析了传统ViTs设计中的冗余，提出通过更大的步幅（如16×16 patch embedding）来减少空间冗余，
并通过单头自注意力（SHSA）代替多头机制，减少冗余计算。

SHViTBlock模块的原理及作用：
原理：SHViTBlock模块的核心在于其单头自注意力（SHSA）机制。与传统的多头自注意力机制不同，
SHSA仅对部分输入通道进行自注意力计算，而剩余通道保持不变。这样可以减少冗余的计算，同时保留全局和局部信息。

作用：通过SHSA模块，模型在确保速度和内存效率的前提下，依然能够捕捉到全局和局部的特征。
SHViTBlock通过局部卷积层和单头自注意力的结合，有效地聚合局部和全局信息，降低计算复杂度，提高推理速度。
SHViT通过创新的单头自注意力机制，显著减少了计算冗余，并实现了在多种硬件平台上的高速高效推理，尤其适用于资源受限的设备。

SHViTBlock即插即用模块适用于计算机视觉CV所有任务.
'''
class GroupNorm(torch.nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m
class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m
class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0)
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x
class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self
class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x
class SHSA(torch.nn.Module):
    """Single-Head Self-Attention"""

    def __init__(self, dim, qk_dim, pdim):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        self.pre_norm = GroupNorm(pdim)

        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            dim, dim, bn_weight_init=0))

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim=1))

        return x
class SHViTBlock(torch.nn.Module):
    def __init__(self, dim, qk_dim=16, pdim=32, type='s'):
        super().__init__()
        if type == "s":  # for later stages
            self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0))
            self.mixer = Residual(SHSA(dim, qk_dim, pdim))
            self.ffn = Residual(FFN(dim, int(dim * 2)))
        elif type == "i":  # for early stages
            self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0))
            self.mixer = torch.nn.Identity()
            self.ffn = Residual(FFN(dim, int(dim * 2)))

    def forward(self, x):
        return self.ffn(self.mixer(self.conv(x)))

if __name__ == "__main__":
    # 创建一个简单的输入特征图
    input = torch.randn(1,64, 32, 32)
    # 创建一个SHViTBlock实例
    SHViTBlock = SHViTBlock(64,type='s')  #注意：前期阶段使用类型i,后期阶段使用类型s
    # 将输入特征图传递给 SHViTBlock模块
    output = SHViTBlock(input)
    # 打印输入和输出的尺寸
    print(f"input  shape: {input.shape}")
    print(f"output shape: {output.shape}")
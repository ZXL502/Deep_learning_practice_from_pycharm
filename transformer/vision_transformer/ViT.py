"""
original code from Jimmy
"""
from functools import partial
from collections import OrderedDict
import torch
from torch import nn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class patch_embed(nn.Module):
    def __init__(self, image_size = 224, patch_size = 16, in_c = 3, em_dim = 768, norm_layer = None):
        super(patch_embed, self).__init__()
        image_size = (image_size,image_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patch = patches_resolution[0] * patches_resolution[1]

        self.proj = nn.Conv2d(in_c, em_dim, kernel_size=patch_size,stride=patch_size)
        self.norm = norm_layer(em_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape()
        assert H == self.image_size[0] & W == self.image_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).Flatten(2).transpose(1,2) # B,H*W, C
        if self.norm is not None:
            x = self.norm(x)
        return x


class MLP(nn.Module):
    def __init__(self, ff_input, ff_hiden, ff_output, act_layer = nn.GELU, drop = 0.):
        super(MLP, self).__init__()
        self.fc = nn.Linear(ff_input, ff_hiden)
        self.act = act_layer()
        self.fc1 = nn.Linear(ff_hiden,ff_output)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.drop(x)
        return x


class Position_wise(nn.Module):
    pass


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_head = num_heads
        per_dim = dim//self.num_head
        self.dim = per_dim
        self.qkv = nn.Linear(dim, 3 * dim)
        self.bias = qkv_bias
        self.scale = qk_scale
        self.atten_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)


    def forward(self, x):
        # batch_size, num_patches + 1. total_embed_dim
        B, N, C = x.shape()
        # B, N, 3, 8, per_embed_dim
        # 3, B, 8, N, per_embed_dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        x_ = (q @ k.transpose[2, 3]) * self.scale
        x_ = x_.softmax(dim= -1)
        x_ = self.atten_drop(x_)

        # [batch_size, num_heads, num_patches + 1, per_embed_dim]
        # [batch_size, num_patches + 1, total_embed_dim]
        x = (x_ @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm
                 ):
        super(Block, self).__init__()
        self.norm = norm_layer
        self.attention = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop_ratio, drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm1 = norm_layer
        mlp_hidden = (dim * mlp_ratio)
        self.mlp = MLP(ff_input=dim, ff_hiden=mlp_hidden, ff_output= dim, act_layer=act_layer,drop=drop_ratio)

    def forward(self, x):
        x = x+ self.drop_path(self.attention(self.norm(x)))
        x = x+ self.drop_path(self.mlp(self.norm1(x)))
        return x


class Visiontransformer(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None,  drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=patch_embed, norm_layer=None,
                 act_layer=None
                 ):
        super(Visiontransformer, self).__init__()
        self.number_class = num_classes
        self.number_feature = self.embed_dim = embed_dim
        self.num_pateches = img_size**2 // patch_size**2
        self.patch_embed = embed_layer(image_size=img_size, patch_size=patch_size,
                                        in_c=in_c, em_dim=embed_dim, norm_layer=norm_layer)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embed = nn.Parameter(torch.zeros(1, 1+ self.num_pateches, embed_dim ))
        self.pos_drop = nn.Dropout(drop_ratio)
        act_layer = act_layer or nn.GELU
        norm_layer = norm_layer or partial(nn.LayerNorm, eps = 1e-6)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]

        self.block = nn.Sequential(
             *[Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
          norm_layer=norm_layer, act_layer=act_layer) for i in range(depth)]
         )
        self.norm_layer = norm_layer(embed_dim)

        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, self.number_class) if self.number_class > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.position_embed, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)

        self.apply(_init_vit_weights)

    def forward_future(self, x):
        # x: B, C, H, W
        x = self.patch_embed(x) # B, 196, 768
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.position_embed)
        x = self.block(x)
        x = self.norm_layer(x)
        return self.pre_logits(x[:, 0])

    def forward(self,x):
        x = self.forward_future(x)
        x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = Visiontransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model








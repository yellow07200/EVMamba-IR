# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
        use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=192, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x.unsqueeze(0).permute(0,2,1,3,4))
        return x

class vivi_PatchEmbed(nn.Module):
    # include both position and temporal patch embedding processes
    def __init__(
            self, 
            img_size=192, 
            patch_size=16, 
            # depth=24, 
            embed_dim=192, 
            channels=1, #3, 
            drop_rate=0.,
            kernel_size=1, 
            num_frames=8, 
        ):
        # factory_kwargs = {"device": device, "dtype": dtype} # follow MambaLMHeadModel
        super().__init__()
        self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            kernel_size=kernel_size,
            in_chans=channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        x = self.patch_embed(x) 
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C) 

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  
        x = torch.cat((cls_token, x), dim=1) 
        x = x + self.pos_embed # 


        cls_tokens = x[:B, :1, :] 
        x = x[:, 1:] 
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        x = x + self.temporal_pos_embedding 
        x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T) 
        x = torch.cat((cls_tokens, x), dim=1) 
        x = self.pos_drop(x) 
        return x

class vivi_mamba(nn.Module):  
    def __init__(
            self, 
            depth=1, 
            embed_dim=192, 
            num_classes=1000,
            drop_path_rate=0.1,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True, 
            residual_in_fp32=True,
            bimamba=True,
            device=None,
            dtype=None,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
        ):
        factory_kwargs = {"device": device, "dtype": dtype} # follow MambaLMHeadModel
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        # self.checkpoint_num = checkpoint_num
        # print(f'Use checkpoint: {use_checkpoint}')
        # print(f'Checkpoint number: {checkpoint_num}')

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # original init
        self.apply(segm_init_weights)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}
    
    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None):
        # mamba impl
        residual = None # yellow residual = None
        hidden_states = x
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params,
                    use_checkpoint=True
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False, ##TO DO yellow
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states

    def forward(self, x, inference_params=None):
        x = self.forward_features(x, inference_params)  
        return x

class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
    
class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(
            dim, 2*dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x= self.norm(x)

        return x


class vivi_block(nn.Module): 
    def __init__(
            self, 
            img_size=192, 
            patch_size=16, 
            depth=24, 
            embed_dim=96, 
            channels=3, #3, 
            num_classes=1000,
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True, 
            residual_in_fp32=True,
            bimamba=True,
            # video
            kernel_size=1, 
            num_frames=8, 
            fc_drop_rate=0., 
            device=None,
            dtype=None,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
            # convolution layers
            in_channels = 3,
            out_channels = 64,
            kernel_size_conv = 3,
            stride =2,
            concate = False,
        ):
        factory_kwargs = {"device": device, "dtype": dtype} # follow MambaLMHeadModel
        super().__init__()

        self.img_size = img_size
        self.channels = channels
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.patch_size = patch_size

        self.concate = concate

        # spatial & temporal embeddings
        self.pt_embed = vivi_PatchEmbed(
            img_size=self.img_size, 
            patch_size=self.patch_size,  # 4
            embed_dim=self.embed_dim, 
            channels=self.channels, #3, 
            drop_rate=0.,
            kernel_size=1, 
            num_frames=self.num_frames)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )


        self.depth_wise_conv = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.channels,
            bias=False,
        )

        self.pm0 = PatchMerging2D(dim=96)
        self.v0 = vivi_mamba(depth=2, embed_dim=96, fused_add_norm=False)
        
        self.pm1 = PatchMerging2D(dim=192)
        self.v1 = vivi_mamba(depth=2, embed_dim=192, fused_add_norm=False)

        self.pm2 = PatchMerging2D(dim=384)
        self.v2 = vivi_mamba(depth=2, embed_dim=384, fused_add_norm=False)
        self.v3 = vivi_mamba(depth=2, embed_dim=768, fused_add_norm=False)
        
        self.silu=nn.SiLU()
        self.sigmoid=nn.Sigmoid()
        self.relu = nn.ReLU()
        
        self.dv2 = vivi_mamba(depth=2, embed_dim=384, fused_add_norm=True)
        self.dv1 = vivi_mamba(depth=2, embed_dim=192, fused_add_norm=True)
        self.dv0 = vivi_mamba(depth=2, embed_dim=96, fused_add_norm=True)

        self.px4=PatchExpand(dim=768)
        self.px3=PatchExpand(dim=384)
        self.px2=PatchExpand(dim=192)
        
        self.up1 = PatchExpand(dim=96)
        self.up2 = PatchExpand(dim=48)
        self.conv_out = nn.Conv2d(in_channels=24 , out_channels=1 , kernel_size=3, stride=1, padding=1)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}
    
    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None): 
        if x.shape[0]<self.num_frames:
            ln = self.num_frames-x.shape[0] # x: [4, 3, 256, 256]
            ad = torch.zeros((ln, self.channels, self.img_size, self.img_size)).to('cuda')
            x = torch.cat((x,ad),0)
        # import pdb;pdb.set_trace() 
        x = self.pt_embed(x) # [1, 16385, 96]

        # vivi_mamva implementation
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        residual = None
        hidden_states = [] 

        hw1 = int(self.img_size/4)

        # encoder1
        x11 = self.v0(x[:,1:,:]) 
        x12=rearrange(x11, '1 (b h w) c-> b h w c', b=self.num_frames, h=hw1, w=hw1) 
        x1 = self.pm0(x12) 
        

        # encoder2
        hw2 = int(self.img_size/8)
        x21 = rearrange(x1, 'b h w c-> 1 (b h w) c') 
        x22 = self.v1(x21) 
        x23=rearrange(x22, '1 (b h w) c-> b h w c', b=self.num_frames, h=hw2, w=hw2) 
        x2 = self.pm1(x23) 
        

        # encoder3, /16
        hw3 = int(self.img_size/16)
        x31 = rearrange(x2, 'b h w c-> 1 (b h w) c')
        x32 = self.v2(x31) 
        x33=rearrange(x32, '1 (b h w) c-> b h w c', b=self.num_frames, h=hw3, w=hw3) 
        x3 = self.pm2(x33) 
        

        # encoder4, /32
        hw4 = int(self.img_size/32)
        x41 = rearrange(x3, 'b h w c-> 1 (b h w) c') 
        x42 = self.v3(x41) 
        x4=rearrange(x42, '1 (b h w) c-> b h w c', b=self.num_frames, h=hw4, w=hw4) 
        
        xr = rearrange(x[:,1:, :], '1 (b h w) c-> b h w c', b=self.num_frames, h=self.img_size/4, w=self.img_size/4)
        hidden_states.append([xr, x12, x23, x33, x4])

        xd4_p = self.px4(x4) 
        xd30 = hidden_states[0][3] + xd4_p
        xd31 = rearrange(xd30, 'b h w c-> 1 (b h w) c') 
        xd32 = self.dv2(xd31) 
        xd3=rearrange(xd32, '1 (b h w) c-> b h w c', b=self.num_frames, h=hw3, w=hw3) 

        xd3_p = self.px3(xd3) 
        xd20 = hidden_states[0][2] + xd3_p
        xd21 = rearrange(xd20, 'b h w c-> 1 (b h w) c')
        xd22 = self.dv1(xd21) 
        xd2=rearrange(xd22, '1 (b h w) c-> b h w c', b=self.num_frames, h=hw2, w=hw2) 

        xd2_p = self.px2(xd2) 
        xd10 = hidden_states[0][1] + xd2_p
        if self.concate:
            xd10 = torch.cat([hidden_states[0][1], xd2_p], 3) 
            xd10 = self.concat_linear1(xd10) 
        xd11 = rearrange(xd10, 'b h w c-> 1 (b h w) c')
        xd12 = self.dv0(xd11) 
        xd1=rearrange(xd12, '1 (b h w) c-> b h w c', b=self.num_frames, h=hw1, w=hw1)

        B,H,W,C = xd1.shape
        

        ## expand_X2 @ 2
        xp1 = self.up1(xd1) 
        xp2 = self.up2(xp1) 

        xp2 = xp2.permute(0, 3, 1, 2)  
        xp3 = self.conv_out(xp2)
        xp3 = self.sigmoid(xp3)

        return xp3
    
    def forward(self, x, inference_params=None):
        x = self.forward_features(x, inference_params) 
        return x



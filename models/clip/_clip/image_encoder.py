import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Union, Any, List, Iterable, Optional

from .blocks import LayerNorm, Transformer, Bottleneck, AttentionPool2d


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """
    def __init__(
        self,
        layers: Tuple[int, int, int, int],
        output_dim: int,
        input_resolution: int = 224,
        width: int = 64,
        heads: int = 8,
        features_only: bool = False,
        out_indices: Optional[Iterable[int]] = None,
        reduction: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        input_resolution = (input_resolution, input_resolution) if isinstance(input_resolution, int) else input_resolution
        assert isinstance(input_resolution, tuple) and len(input_resolution) == 2, f"input_resolution should be a tuple of length 2, but got {input_resolution}"
        self.input_resolution = input_resolution
        self.downsampling_rate = 32  # the rate at which the input is downsampled by the network

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=1 if reduction <= 16 else 2)

        self.features_only = features_only
        if features_only:
            self.out_indices = out_indices if out_indices is not None else range(5)
            self.out_indices = [idx + 5 if idx < 0 else idx for idx in self.out_indices]  # map negative indices to positive indices
            self.out_indices = sorted(set(self.out_indices))  # remove duplicates and sort
            assert min(self.out_indices) >= 0 and max(self.out_indices) <= 4, f"out_indices={self.out_indices} is invalid for a ResNet with 5 stages"
            self.channels = width * 32  # the ResNet feature dimension
        else:
            self.out_indices = None
            embed_dim = width * 32  # the ResNet feature dimension
            self.attnpool = AttentionPool2d((input_resolution[0] // 32) * (input_resolution[1] // 32), embed_dim, heads, output_dim)
            self.channels = output_dim

        self.reduction = self.downsampling_rate // 2 if reduction <= 16 else self.downsampling_rate
        self.clip_embed_dim = output_dim

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def _stem(self, x: Tensor) -> Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x: Tensor) -> Union[Tensor, List[Tensor]]:
        x = x.type(self.conv1.weight.dtype)
        x = self._stem(x)

        feats = [x] if self.features_only and 0 in self.out_indices else []

        x = self.layer1(x)
        if self.features_only and 1 in self.out_indices:
            feats.append(x)

        x = self.layer2(x)
        if self.features_only and 2 in self.out_indices:
            feats.append(x)

        x = self.layer3(x)
        if self.features_only and 3 in self.out_indices:
            feats.append(x)

        x = self.layer4(x)
        if self.features_only and 4 in self.out_indices:
            feats.append(x)

        if self.features_only:
            if len(self.out_indices) == 1:
                return feats[0]
            else:
                return feats
        else:
            x = self.attnpool(x)
            return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        output_dim: int,
        width: int,
        layers: int,
        heads: int,
        features_only: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        input_resolution = (input_resolution, input_resolution) if isinstance(input_resolution, int) else input_resolution
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        assert isinstance(input_resolution, tuple) and len(input_resolution) == 2, f"input_resolution should be a tuple of length 2, but got {input_resolution}"
        assert isinstance(patch_size, tuple) and len(patch_size) == 2, f"patch_size should be a tuple of length 2, but got {patch_size}"
        assert patch_size[0] == patch_size[1], f"ViT only supports square patches, patch_size={patch_size} is invalid."
        assert input_resolution[0] % patch_size[0] == 0 and input_resolution[1] % patch_size[1] == 0, f"input_resolution {input_resolution} should be divisible by patch_size {patch_size}"
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.downsampling_rate = patch_size[0]

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.num_patches_h = int(input_resolution[0] // patch_size[0])
        self.num_patches_w = int(input_resolution[1] // patch_size[1])
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.num_patches_h * self.num_patches_w + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)

        self.features_only = features_only  # if True, return the final patches instead of the CLS token
        if features_only:
            self.channels = width
        else:
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
            self.channels = output_dim

        self.reduction = patch_size[0]
        self.clip_embed_dim = output_dim

    def adjust_pos_embed(self, h: int, w: int) -> None:
        """
        Permanently adjust the size of the positional embedding matrix.

        Args:
            h: the height of the original input image.
            w: the width of the original input image.
        """
        assert h % self.patch_size[0] == 0 and w % self.patch_size[1] == 0, f"input_resolution {h, w} should be divisible by patch_size {self.patch_size}"
        if self.input_resolution[0] != h or self.input_resolution[1] != w:
            new_num_patches_h = int(h // self.patch_size[0])
            new_num_patches_w = int(w // self.patch_size[1])
            positional_embedding = rearrange(self.positional_embedding[1:, :], "(h w) c -> c h w", h=self.num_patches_h, w=self.num_patches_w).unsqueeze(0)  # add batch dimension
            positional_embedding = F.interpolate(positional_embedding, size=(new_num_patches_h, new_num_patches_w), mode="bicubic", ).squeeze(0)  # remove batch dimension
            positional_embedding = rearrange(positional_embedding, "c h w -> (h w) c")
            self.positional_embedding = nn.Parameter(torch.cat([self.positional_embedding[:1, :], positional_embedding], dim=0))
            self.input_resolution = (h, w)
            self.num_patches_h = new_num_patches_h
            self.num_patches_w = new_num_patches_w

    def _interpolate_pos_embed(self, h: int, w: int) -> Tensor:
        """
        Interpolate the positional embedding matrix to match the size of the input image.

        Args:
            h: the required number of patches along the height dimension.
            w: the required number of patches along the width dimension.
        """
        if h == self.num_patches_h and w == self.num_patches_w:
            return self.positional_embedding
        else:
            positional_embedding = rearrange(self.positional_embedding[1:, :], "(h w) c -> c h w", h=self.num_patches_h, w=self.num_patches_w).unsqueeze(0)  # add batch dimension
            positional_embedding = F.interpolate(positional_embedding, size=(h, w), mode="bicubic").squeeze(0)  # remove batch dimension
            positional_embedding = rearrange(positional_embedding, "c h w -> (h w) c")
            positional_embedding = torch.cat([self.positional_embedding[:1, :], positional_embedding], dim=0)
            return positional_embedding

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x) # shape = [*, width, grid, grid]
        num_patches_h, num_patches_w = x.shape[-2:]

        positional_embedding = self._interpolate_pos_embed(num_patches_h, num_patches_w).to(x.dtype)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
                self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), 
                x
            ], dim=1)
        x = x + positional_embedding
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND. N: batch size, L: sequence length, D: feature dimension
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)

        if self.features_only:
            x = x[:, 1:, :]  # remove the CLS token
            x = rearrange(x, "n (h w) c -> n c h w", h=num_patches_h, w=num_patches_w)
        else:
            x = x[:, 0, :]
            x = x @ self.proj
        return x


# 模块内新增一个类：ConvNeXtEncoder
class ConvNeXtEncoder(nn.Module):
    """
    A lightweight wrapper over timm ConvNeXt that aligns with the interface used by CLIP-EBC:
    - features_only: return feature map(s) instead of pooled embedding
    - channels: channels of the selected output feature map
    - reduction: spatial downsampling factor of the selected output
    - clip_embed_dim: target CLIP embedding dim (so the outside projection can map to this dim)
    """
    def __init__(
        self,
        variant: str = "convnext_base",
        output_dim: int = 512,
        input_resolution: Union[int, Tuple[int, int]] = 224,
        features_only: bool = True,
        reduction: int = 32,
        out_indices: Optional[Iterable[int]] = None,
        pretrained: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        try:
            import timm
        except Exception as e:
            raise ImportError("ConvNeXtEncoder requires `timm`. Please install: pip install timm") from e

        input_resolution = (input_resolution, input_resolution) if isinstance(input_resolution, int) else input_resolution
        self.input_resolution = input_resolution
        self.features_only = features_only
        # 构建所有 stage 的输出，后续按 reduction 选其中一个
        # 使用 'backbone' 命名以匹配 encoder_keywords
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # 4 个 stage
        )

        # 根据目标 reduction 选择输出 stage（16 -> 倒数第二个，32 -> 最后一个）
        # timm 的 feature_info 可提供每个 stage 的下采样倍率
        reductions = []
        try:
            # 部分 timm 版本提供 reduction() / reductions 属性
            reductions = getattr(self.backbone.feature_info, "reductions", None) or self.backbone.feature_info.reduction()
        except Exception:
            # 兜底：常见 convnext 下采样序列 [4, 8, 16, 32]
            reductions = [4, 8, 16, 32]
        target = 16 if reduction <= 16 else 32
        # 选择最后一个 <= target 的 stage
        cand_idx = [i for i, r in enumerate(reductions) if r <= target]
        # 使用更语义化的命名
        self._feature_stage_idx = cand_idx[-1] if cand_idx else len(reductions) - 1
        self.reduction = reductions[self._feature_stage_idx]

        # 通道数
        try:
            chs = getattr(self.backbone.feature_info, "channels", None)
            if callable(chs):
                chs = chs()
            if isinstance(chs, list):
                self.channels = chs[self._feature_stage_idx]
            else:
                # 兜底
                self.channels = self.backbone.num_features
        except Exception:
            self.channels = getattr(self.backbone, "num_features", 1024)

        # 对齐外部 CLIP 投影维度
        self.clip_embed_dim = output_dim

    def forward(self, x: Tensor) -> Union[Tensor, List[Tensor]]:
        x = x.type(next(self.backbone.parameters()).dtype)
        feats = self.backbone(x)  # list of 4 stage outputs
        x = feats[self._feature_stage_idx]
        return x

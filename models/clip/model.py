import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import os
import math
from typing import List, Tuple, Union, Optional

from . import _clip
from ..utils import _init_weights, make_resnet_layers, Bottleneck, BasicBlock, ConvNeXtBlock, make_convnext_layers
from .utils import format_blood_indicator

curr_dir = os.path.abspath(os.path.dirname(__file__))


# resnet50: reduction, channels, embed_dim = 32, 2048, 1024
# resnet101: reduction, channels, embed_dim = 32, 2048, 512
# resnet50x4: reduction, channels, embed_dim = 32, 2560, 640
# resnet50x16: reduction, channels, embed_dim = 32, 3072, 768
# resnet50x64: reduction, channels, embed_dim = 32, 4096, 1024
# vit_b_32: reduction, channels, embed_dim = 32, 768, 512
# vit_b_16: reduction, channels, embed_dim = 16, 768, 512
# vit_l_14: reduction, channels, embed_dim = 14, 1024, 768
# vit_l_14_336px: reduction, channels, embed_dim = 14, 1024, 768

resnet_backbones = ["resnet50", "resnet101", "resnet50x4", "resnet50x16", "resnet50x64"]
vit_backbones = ["vit_b_16", "vit_b_32", "vit_l_14", "vit_l_14_336px"]
convnext_backbones = ["convnext_b"]  # 新增

class CLIP_EBC(nn.Module):
    def __init__(
        self,
        backbone: str,
        bins: List[Tuple[float, float]],
        anchor_points: List[float],
        reduction: Optional[int] = None,
        freeze_text_encoder: bool = True,
        prompt_type: str = "number",
        input_size: Optional[int] = None,
        num_vpt: Optional[int] = None,
        deep_vpt: Optional[bool] = None,
        vpt_drop: Optional[float] = None,
        decoder_block: Optional[nn.Module] = None,
        decoder_cfg: Optional[List[Union[str, int]]] = None,
        indicator_name: str = "diabp",  # 新增参数
        indicator_unit: str = "",    # 新增参数
        # === new: phrase/prompt hyperparameters ===
        phrase_config_path: Optional[str] = None,
        prompt_template: Optional[str] = None,
        name_syns_mode: str = "pro",
        phrase_pick_mode: str = "random",
        phrase_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert backbone in resnet_backbones + vit_backbones + convnext_backbones, f"Backbone should be in {resnet_backbones + vit_backbones + convnext_backbones}, got {backbone}"
        self.backbone = backbone
        self.indicator_name = indicator_name
        self.indicator_unit = indicator_unit
        # === store new hyperparams ===
        self.phrase_config_path = phrase_config_path
        self.prompt_template = prompt_template
        self.name_syns_mode = name_syns_mode
        self.phrase_pick_mode = phrase_pick_mode
        self.phrase_seed = phrase_seed

        # Image encoder
        if backbone in resnet_backbones or backbone in convnext_backbones:
            self.image_encoder = getattr(_clip, f"{backbone}_img")(features_only=True, out_indices=(-1,), reduction=reduction)
        else:
            assert input_size is not None, "Expected input_size to be an integer, got None."
            assert num_vpt is not None, "Expected num_vpt to be an integer, got None."
            assert deep_vpt is not None, "Expected deep_vpt to be a boolean, got None."
            assert vpt_drop is not None, "Expected vpt_drop to be a float, got None."

            self.image_encoder = getattr(_clip, f"{backbone}_img")(features_only=True, input_size=input_size)
            self.image_encoder_depth = len(self.image_encoder.transformer.resblocks)

            # Use VPT. Freeze the image encoder.
            for param in self.image_encoder.parameters():
                param.requires_grad = False

            self.num_vpt = num_vpt
            self.deep_vpt = deep_vpt

            patch_size = self.image_encoder.patch_size[0]
            val = math.sqrt(6. / float(3 * patch_size + self.image_encoder.channels))

            for idx in range(self.image_encoder_depth if self.deep_vpt else 1):
                setattr(self, f"vpt_{idx}", nn.Parameter(torch.empty(self.num_vpt, self.image_encoder.channels)))
                nn.init.uniform_(getattr(self, f"vpt_{idx}"), -val, val)
                setattr(self, f"vpt_drop_{idx}", nn.Dropout(vpt_drop) if vpt_drop > 0 else nn.Identity())

        self.encoder_reduction = self.image_encoder.reduction
        self.reduction = self.encoder_reduction if reduction is None else reduction
        self.channels = self.image_encoder.channels
        self.clip_embed_dim = self.image_encoder.clip_embed_dim

        if decoder_cfg is not None:
            assert decoder_block is not None, "Expected decoder_block to be a nn.Module, got None."
            if self.backbone in convnext_backbones:
                # 使用 ConvNeXt 风格解码器
                self.image_decoder = make_convnext_layers(decoder_block, decoder_cfg, in_channels=self.channels)
            else:
                # ResNet/Vit 分支维持原逻辑（vit 这边此前也套用了 resnet 风格 block）
                self.image_decoder = make_resnet_layers(decoder_block, decoder_cfg, in_channels=self.channels, expansion=1, dilation=1)
            self.image_decoder.apply(_init_weights)
            self.channels = decoder_cfg[-1]
        else:
            self.image_decoder = nn.Identity()

        if self.channels != self.clip_embed_dim:
            self.projection = nn.Conv2d(in_channels=self.channels, out_channels=self.clip_embed_dim, kernel_size=1)
            self.projection.apply(_init_weights)
        else:
            self.projection = nn.Identity()

        # Text encoder
        assert prompt_type in ["number", "word"], f"Expected prompt_type to be 'number' or 'word', got {prompt_type}"
        self.prompt_type = prompt_type
        self.text_encoder = getattr(_clip, f"{backbone}_txt")()
        self.freeze_text_encoder = freeze_text_encoder
        if self.freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.bins = bins
        self.anchor_points = torch.tensor(anchor_points, dtype=torch.float32, requires_grad=False).view(1, -1)
        
        # 移除固定的文本提示生成，改为动态生成
        # self._get_text_prompts()
        # self._tokenize_text_prompts()
        # if self.freeze_text_encoder:
        #     self._extract_text_features()
        # else:
        #     self.text_features = None
        
        # 初始化时不生成固定的文本特征
        self.text_features = None
        self.text_prompts = None

        # 添加全局池化层
        # self.global_pool = nn.AdaptiveAvgPool2d(1)

        self._get_text_prompts()
        self._tokenize_text_prompts()

        if self.freeze_text_encoder:
            self._extract_text_features()
        else:
            self.text_features = None

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)

        # 添加可学习温度参数用于Log-Mean-Exp池化
        # 每个类别（bin）一个温度参数，初始化为1.0
        self.temperature = nn.Parameter(torch.ones(len(bins)), requires_grad=True)

        # === 新增：单调“分布->数值”映射（替代固定线性期望的锚点） ===
        self.num_bins = len(bins)
        anchors_np = np.array(anchor_points, dtype=np.float32)
        v0_init = float(anchors_np[0])
        diffs = np.diff(anchors_np)  # 长度 N-1，初始为各bin间距
        # 将初始间距映射为softplus的逆，保证softplus(deltas) ≈ diffs
        delta_init = np.log(np.exp(diffs) - 1.0 + 1e-8).astype(np.float32)

        self.calib_v0 = nn.Parameter(torch.tensor(v0_init, dtype=torch.float32))               # 标定起点 v[0]
        self.calib_deltas = nn.Parameter(torch.tensor(delta_init, dtype=torch.float32))        # 长度 N-1，经softplus后为正

        # === 新增：轻量残差回归头（并联） ===
        self.residual_pool = nn.AdaptiveAvgPool2d(1)
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.clip_embed_dim, self.clip_embed_dim//4),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.clip_embed_dim//4, 1),
        )


    def _get_text_prompts(self) -> None:
        """
        动态生成血液指标相关的文本提示
        每次调用都会生成新的随机表述
        """
        # 将bins转换为范围
        indicator_ranges = []
        for i, (low, high) in enumerate(self.bins):
            indicator_ranges.append((low, high))
        
        self.text_prompts = format_blood_indicator(
            indicator_ranges, 
            self.indicator_name, 
            self.indicator_unit,
            self.prompt_type,
            phrase_config_path=self.phrase_config_path,
            template=self.prompt_template,
            name_syns_mode=self.name_syns_mode,
            pick_mode=self.phrase_pick_mode,
            seed=self.phrase_seed,
        )

    def refresh_text_prompts(self) -> None:
        """
        刷新文本提示，生成新的随机表述
        可以在每个epoch开始时调用
        """
        self._get_text_prompts()
        # print(f"Refreshed text prompts: {self.text_prompts}")
        
        # 如果文本编码器被冻结，预先提取文本特征
        if self.freeze_text_encoder:
            with torch.no_grad():
                tokenized_prompts = _clip.tokenize(self.text_prompts)
                # 将tokenized_prompts移动到与text_encoder相同的设备
                device = next(self.text_encoder.parameters()).device
                tokenized_prompts = tokenized_prompts.to(device)
                self.text_features = self.text_encoder(tokenized_prompts)
        else:
            self.text_features = None

    def _tokenize_text_prompts(self) -> None:
        self.text_prompts = _clip.tokenize(self.text_prompts)

    def _extract_text_features(self) -> None:
        with torch.no_grad():
            self.text_features = self.text_encoder(self.text_prompts)

    def _prepare_vpt(self, layer: int, batch_size: int, device: torch.device) -> Tensor:
        if not self.deep_vpt:
            assert layer == 0, f"Expected layer to be 0 when using Shallow Visual Prompt Tuning, got {layer}"

        vpt = getattr(self, f"vpt_{layer}").to(device)
        vpt = vpt.unsqueeze(0).expand(batch_size, -1, -1)
        vpt = getattr(self, f"vpt_drop_{layer}")(vpt)
        vpt = vpt.permute(1, 0, 2)  # (num_vpt, batch_size, hidden_dim)
        assert vpt.shape[1] == batch_size, f"Expected the VPT to have the shape [L_vis B C], got {vpt.shape}."
        return vpt

    def _forward_vpt(self, x: Tensor) -> Tuple[Tensor]:
        device = x.device
        batch_size, _, height, width = x.shape
        num_h_patches, num_w_patches = height // self.image_encoder.patch_size[0], width // self.image_encoder.patch_size[1]

        image_features = self.image_encoder.conv1(x)
        image_features = image_features.reshape(batch_size, image_features.shape[1], -1)
        image_features = image_features.permute(0, 2, 1)  # (B, num_patches, C)
        image_features = torch.cat([
            self.image_encoder.class_embedding + torch.zeros(batch_size, 1, image_features.shape[-1], dtype=image_features.dtype, device=device),
            image_features,
        ], dim=1)  # (B, num_patches + 1, C)

        pos_embedding = self.image_encoder._interpolate_pos_embed(num_h_patches, num_w_patches)
        image_features = image_features + pos_embedding
        image_features = self.image_encoder.ln_pre(image_features)
        image_features = image_features.permute(1, 0, 2)  # (num_patches + 1, B, C)
        assert image_features.shape[0] == num_h_patches * num_w_patches + 1 and image_features.shape[1] == batch_size, f"Expected image_features to have shape [num_patches + 1, B, C], got {image_features.shape}."

        vpt = self._prepare_vpt(0, batch_size, device)
        for idx in range(self.image_encoder_depth):
            # assemble
            image_features = torch.cat([
                image_features[:1, :, :],  # CLS token
                vpt,
                image_features[1:, :, :],
            ], dim=0)

            # transformer
            image_features = self.image_encoder.transformer.resblocks[idx](image_features)

            # disassemble
            if idx < self.image_encoder_depth - 1:
                if self.deep_vpt:
                    vpt = self._prepare_vpt(idx + 1, batch_size, device)
                else:
                    vpt = image_features[1: (self.num_vpt + 1), :, :]

            image_features = torch.cat([
                image_features[:1, :, :],  # CLS token
                image_features[(self.num_vpt + 1):, :, :],
            ], dim=0)
            
        image_features = image_features.permute(1, 0, 2)  # (B, num_patches + 1, C)
        image_features = self.image_encoder.ln_post(image_features)
        image_features = image_features[:, 1:, :].permute(0, 2, 1)  # (B, C, num_patches)
        image_features = image_features.reshape(batch_size, -1, num_h_patches, num_w_patches)
        return image_features

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        device = x.device

        # x = self.image_encoder(x) if self.backbone in resnet_backbones else self._forward_vpt(x)
        x = self.image_encoder(x)
        if self.reduction != self.encoder_reduction:
            x = F.interpolate(x, scale_factor=self.encoder_reduction / self.reduction, mode="bilinear")
        x = self.image_decoder(x)
        x = self.projection(x)

        image_features = x.permute(0, 2, 3, 1)  # shape (B, H, W, C)
        
        # 动态生成文本特征
        if self.text_prompts is None:
            # 如果还没有生成过文本提示，先生成一次
            self.refresh_text_prompts()
        
        if self.text_features is None:
            # 如果文本编码器没有被冻结，每次都重新编码
            tokenized_prompts = _clip.tokenize(self.text_prompts).to(device)
            text_features = self.text_encoder(tokenized_prompts)
        else:
            # 如果文本编码器被冻结，使用预先提取的特征
            text_features = self.text_features.to(device)

        # 特征归一化
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # 计算相似度
        # 对 exp(logit_scale) 做上限裁剪，避免 logits 过尖锐导致不稳定
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * image_features @ text_features.t()  # (B, H, W, N), logits per image
        logits = logits.permute(0, 3, 1, 2)  # (B, N, H, W)

        B, N, H, W = logits.shape
        logits_flat = logits.view(B, N, -1)
        # Log-Mean-Exp池化：从 (B, N, H, W) 到 (B, N)
        # 应用可学习温度参数的Log-Mean-Exp池化，并对温度做范围约束以稳定训练
        # 等价于: τ_n * (logsumexp(logits_n / τ_n) - log(H*W))
        temperature = self.temperature.to(device).clamp(min=0.5, max=5.0).view(1, -1, 1)  # (1, N, 1)
        scaled_logits = logits_flat / temperature  # (B, N, H*W)
        global_logits = temperature.squeeze(-1) * (torch.logsumexp(scaled_logits, dim=2) - math.log(H * W))  # (B, N)

        # 计算分布
        probs = F.softmax(global_logits, dim=1)  # (B, N)

        # === 修改：用“可学习单调标定向量 v”替代固定 anchor_points 的线性期望 ===
        # 构造单调递增的标定向量 v: v[0] + cumsum(softplus(deltas))
        v0 = self.calib_v0.to(device)                                  # 标定起点
        deltas_pos = F.softplus(self.calib_deltas.to(device))          # (N-1), 保证 >0
        cumsum = torch.cumsum(deltas_pos, dim=0)                       # (N-1)
        v = torch.cat([v0.view(1), (v0 + cumsum)], dim=0)              # (N,), 单调递增
        # 按分布做加权得到“标定后的数值”
        value_from_dist = torch.sum(probs * v.view(1, -1), dim=1)      # (B,)

        # === 新增：并联轻量残差回归头，修正极端/长尾样本 ===
        pooled = self.residual_pool(x).squeeze(-1).squeeze(-1)         # (B, C)
        residual = self.residual_mlp(pooled).squeeze(-1)               # (B,)

        pred_value = value_from_dist + residual                        # (B,)

        if self.training:
            return global_logits, pred_value
        else:
            return pred_value


def _clip_ebc(
    backbone: str,
    bins: List[Tuple[float, float]],
    anchor_points: List[float],
    reduction: Optional[int] = None,
    freeze_text_encoder: bool = True,
    prompt_type: str = "number",
    input_size: Optional[int] = None,
    num_vpt: Optional[int] = None,
    deep_vpt: Optional[bool] = None,
    vpt_drop: Optional[float] = None,
    decoder_block: Optional[nn.Module] = None,
    decoder_cfg: Optional[List[Union[str, int]]] = None,
    indicator_name: str = "diabp",
    indicator_unit: str = "",
    # === new: phrase/prompt hyperparameters ===
    phrase_config_path: Optional[str] = None,
    prompt_template: Optional[str] = None,
    name_syns_mode: str = "pro",
    phrase_pick_mode: str = "random",
    phrase_seed: Optional[int] = None,
) -> CLIP_EBC:
    if backbone in resnet_backbones:
        decoder_block = Bottleneck
        if decoder_cfg is None:
            if backbone == "resnet50":
                decoder_cfg = [2048]
            elif backbone == "resnet50x4":
                decoder_cfg = [1280]
            elif backbone == "resnet50x16":
                decoder_cfg = [1536]
            elif backbone == "resnet50x64":
                decoder_cfg = [2048]
            else:  # backbone == "resnet101"
                decoder_cfg = [2048, 1024]
    elif backbone in convnext_backbones:
        # ConvNeXt 分支：使用 ConvNeXt 风格块
        decoder_block = ConvNeXtBlock
        if decoder_cfg is None:
            # ConvNeXt-B 通常最后 stage 通道为 1024（若你的实现不同，请显式传入 decoder_cfg 覆盖）
            decoder_cfg = [1024]
    else:
        decoder_block = BasicBlock
        if decoder_cfg is None:
            if backbone == "vit_b_16":
                decoder_cfg = [768]
            elif backbone == "vit_b_32":
                decoder_cfg = [768]
            else:  # backbone == "vit_l_14"
                decoder_cfg = [1024]

    return CLIP_EBC(
        backbone=backbone,
        bins=bins,
        anchor_points=anchor_points,
        reduction=reduction,
        freeze_text_encoder=freeze_text_encoder,
        prompt_type=prompt_type,
        input_size=input_size,
        num_vpt=num_vpt,
        deep_vpt=deep_vpt,
        vpt_drop=vpt_drop,
        decoder_block=decoder_block,
        decoder_cfg=decoder_cfg,
        indicator_name=indicator_name,
        indicator_unit=indicator_unit,
        # === new: phrase/prompt hyperparameters ===
        phrase_config_path=phrase_config_path,
        prompt_template=prompt_template,
        name_syns_mode=name_syns_mode,
        phrase_pick_mode=phrase_pick_mode,
        phrase_seed=phrase_seed,
    )

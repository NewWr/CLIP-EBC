weight_path = "/opt/DM/OCT/CLIP_Code/CLIP-EBC/compare_weights/open_clip_pytorch_model.bin"

import torch
import os
from collections import OrderedDict
import re

def load_weights(weight_path):
    """加载权重文件"""
    if not os.path.exists(weight_path):
        print(f"❌ 权重文件不存在: {weight_path}")
        return None
    
    try:
        weights = torch.load(weight_path, map_location='cpu', weights_only=False)
        print(f"✅ 成功加载: {weight_path}")
        return weights
    except Exception as e:
        print(f"❌ 加载权重文件失败 {weight_path}: {e}")
        return None

def _extract_state_dict(weights):
    """从多种打包格式中提取 state_dict"""
    if isinstance(weights, dict):
        # 常见字段优先
        for key in ["state_dict", "model"]:
            if key in weights and isinstance(weights[key], dict):
                return weights[key]
        # 有些直接就是参数字典
        if all(isinstance(v, torch.Tensor) for v in weights.values()):
            return weights
        # 兜底：尝试在一层嵌套里找
        for k, v in weights.items():
            if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                return v
    # 若已是 OrderedDict/参数映射
    if isinstance(weights, (dict, OrderedDict)):
        return weights
    print("⚠️ 未能从权重文件中提取有效的 state_dict。")
    return None

def _strip_prefix(key):
    """去除常见前缀，例如 'module.'"""
    if key.startswith("module."):
        return key[len("module."):]
    return key

def _is_image_key(k):
    """判定是否为图像编码器相关参数"""
    # OpenCLIP/CLIP 通常以 visual.*/vision.* 作为视觉分支
    prefixes = ("visual.", "vision.", "image_encoder.", "backbone.")
    return k.startswith(prefixes)

def _is_text_key(k):
    """判定是否为文本编码器相关参数"""
    # 文本分支常见键：transformer.* / token_embedding.* / positional_embedding / ln_final.* / text_projection
    if k.startswith(("text.", "transformer.", "token_embedding.")):
        return True
    if k in ("positional_embedding", "text_projection", "ln_final.weight", "ln_final.bias"):
        return True
    if k.startswith("ln_final."):
        return True
    return False

def _split_state_dict(state_dict):
    """按视觉/文本拆分权重"""
    img_sd = OrderedDict()
    txt_sd = OrderedDict()
    others = []

    for k, v in state_dict.items():
        ck = _strip_prefix(k)
        if _is_image_key(ck):
            img_sd[ck] = v
        elif _is_text_key(ck):
            txt_sd[ck] = v
        else:
            # 例如 logit_scale 等全局参数，这里忽略
            others.append(ck)

    return img_sd, txt_sd, others

def convert_to_backbone_format(state_dict):
    """将权重keys从visual.trunk.格式转换为backbone.格式
    
    目标格式示例:
    - backbone.stem_0
    - backbone.stem_1  
    - backbone.stages_0.blocks.0.gamma
    - backbone.stages_0.blocks.0.conv_dw.weight
    - backbone.stages_0.blocks.0.norm.weight
    - backbone.stages_0.blocks.0.mlp.fc1.weight
    
    Args:
        state_dict: 原始权重字典
    
    Returns:
        转换后的权重字典
    """
    converted_dict = OrderedDict()
    
    for key, tensor in state_dict.items():
        new_key = key
        
        # 移除visual前缀，保留trunk部分进行转换
        if new_key.startswith('visual.'):
            new_key = new_key[len('visual.'):]
        
        # 将trunk前缀替换为backbone
        if new_key.startswith('trunk.'):
            new_key = new_key.replace('trunk.', 'backbone.', 1)
        
        # 处理stem层：trunk.stem.0 -> backbone.stem_0
        if 'backbone.stem.' in new_key:
            new_key = new_key.replace('backbone.stem.', 'backbone.stem_')
        
        # 处理stages层：trunk.stages.0 -> backbone.stages_0
        # 使用正则表达式匹配 backbone.stages.{数字}
        new_key = re.sub(r'backbone\.stages\.(\d+)', r'backbone.stages_\1', new_key)
        
        # 处理downsample层：如果存在downsample，需要根据具体情况调整
        # 这里假设downsample层也遵循类似的命名规则
        if 'downsample.' in new_key:
            # 可能需要根据实际情况调整downsample的映射规则
            pass
        
        converted_dict[new_key] = tensor
    
    return converted_dict

def save_converted_weights(state_dict, output_path, description=""):
    """保存转换后的权重文件
    
    Args:
        state_dict: 要保存的权重字典
        output_path: 输出文件路径
        description: 描述信息
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存权重
        torch.save(state_dict, output_path)
        print(f"✅ 已保存{description}权重: {output_path}")
        print(f"   参数层数: {len(state_dict)}，参数总量: {_num_params(state_dict)}")
        
        # 显示前几个key作为示例
        print(f"   示例keys: {list(state_dict.keys())[:5]}")
        
    except Exception as e:
        print(f"❌ 保存权重文件失败 {output_path}: {e}")

def _guess_convnext_variant(weights):
    """从权重中的配置尝试推断 ConvNeXt 变体，返回我们项目中使用的命名后缀：
    - base/convnext_base -> convnext_b
    - large/convnext_large -> convnext_l
    - small/convnext_small -> convnext_s
    - tiny/convnext_tiny -> convnext_t
    默认回退为 convnext_b
    """
    candidates = []
    if isinstance(weights, dict):
        for topk in ["model_cfg", "config", "cfg", "args"]:
            if topk in weights and isinstance(weights[topk], dict):
                candidates.append(weights[topk])
    # 递归查找字符串值里包含 convnext 的线索
    def _find_variant(d):
        try:
            for k, v in d.items():
                if isinstance(v, dict):
                    val = _find_variant(v)
                    if val:
                        return val
                elif isinstance(v, str):
                    s = v.lower()
                    if "convnext" in s:
                        return s
        except Exception:
            return None
        return None

    found = None
    for c in candidates:
        found = _find_variant(c)
        if found:
            break

    if not found:
        # 还可以在 state_dict 键名上猜测（若视觉分支中含有 convnext 结构特征，也较困难）
        return "convnext_b"

    if "tiny" in found:
        return "convnext_t"
    if "small" in found or re.search(r"\bconvnext_s\b", found):
        return "convnext_s"
    if "large" in found or re.search(r"\bconvnext_l\b", found):
        return "convnext_l"
    # base/default
    return "convnext_b"

def _num_params(sd):
    try:
        return sum(int(v.numel()) for v in sd.values())
    except Exception:
        return 0

def main():
    """主函数 - 加载权重并转换为backbone格式"""
    print("🔍 开始权重转换为backbone格式...")
    print("="*60)
    
    # 加载权重文件
    print("📂 加载权重文件:")
    weights = load_weights(weight_path)
    if weights is None:
        return

    # 提取 state_dict
    state_dict = _extract_state_dict(weights)
    if state_dict is None:
        return

    # 拆分为 image/text 两部分
    print("✂️  拆分权重为 Image Encoder 与 Text Encoder ...")
    img_sd, txt_sd, others = _split_state_dict(state_dict)
    print(f"🖼️  Image Encoder 参数层数: {len(img_sd)}，参数总量: {_num_params(img_sd)}")
    print(f"📝 Text Encoder  参数层数: {len(txt_sd)}，参数总量: {_num_params(txt_sd)}")
    if others:
        print(f"ℹ️  其余未归类参数（将被忽略）数量: {len(others)}，例如: {others[:5]} ...")

    # 转换Image Encoder为backbone格式
    print("\n🔄 转换Image Encoder为backbone格式...")
    converted_img_sd = convert_to_backbone_format(img_sd)
    
    # 推断 ConvNeXt 变体用于命名
    variant = _guess_convnext_variant(weights)
    out_dir = "/opt/DM/OCT/CLIP_Code/CLIP-EBC/models/clip/_clip/weights"
    os.makedirs(out_dir, exist_ok=True)
    
    # 保存原始格式的权重
    img_out = os.path.join(out_dir, f"clip_image_encoder_{variant}.pth")
    txt_out = os.path.join(out_dir, f"clip_text_encoder_{variant}.pth")
    
    # 保存backbone格式的权重
    backbone_img_out = os.path.join(out_dir, f"backbone_convnext_{variant}.pth")
    
    print("\n💾 保存权重文件:")
    
    # 保存原始格式
    save_converted_weights(img_sd, img_out, "原始Image Encoder")
    save_converted_weights(txt_sd, txt_out, "Text Encoder")
    
    # 保存backbone格式
    save_converted_weights(converted_img_sd, backbone_img_out, "backbone格式")
    
    print("\n📊 转换对比:")
    print(f"原始Image Encoder keys示例: {list(img_sd.keys())[:3]}")
    print(f"backbone格式keys示例: {list(converted_img_sd.keys())[:3]}")
    
    # 验证转换结果
    print("\n🔍 转换结果验证:")
    backbone_keys = list(converted_img_sd.keys())
    stem_keys = [k for k in backbone_keys if 'stem_' in k]
    stage_keys = [k for k in backbone_keys if 'stages_' in k][:5]
    
    print(f"Stem层keys: {stem_keys}")
    print(f"Stage层keys示例: {stage_keys}")
    
    print("="*60)
    print("🎉 权重转换完成！")
    print(f"📁 输出目录: {out_dir}")
    print(f"📄 原始格式: {os.path.basename(img_out)}, {os.path.basename(txt_out)}")
    print(f"📄 backbone格式: {os.path.basename(backbone_img_out)}")

if __name__ == "__main__":
    main()
weight_path1 = "/opt/DM/OCT/CLIP_Code/CLIP-EBC/models/clip/_clip/weights/clip_text_encoder_vit_b_16.pth"
weight_path2 = "/opt/DM/OCT/CLIP_Code/RETFound_MAE/hug_model/RETFound_mae_natureCFP.pth"

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

def normalize_layer_name(layer_name):
    """标准化层名称，处理不同命名约定"""
    # 移除常见的前缀
    prefixes_to_remove = [
        'model.', 'module.', 'encoder.', 'visual.', 'vision_model.',
        'backbone.', 'feature_extractor.', 'base_model.'
    ]
    
    normalized = layer_name
    for prefix in prefixes_to_remove:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    
    # 标准化常见的层名称模式
    replacements = {
        'transformer.resblocks': 'blocks',
        'transformer.layers': 'blocks',
        'layers': 'blocks',
        'resblocks': 'blocks',
        'attn.': 'attention.',
        'self_attn.': 'attention.',
        'multihead_attn.': 'attention.',
        'ln_': 'norm',
        'layernorm': 'norm',
        'layer_norm': 'norm',
        'mlp.c_fc': 'mlp.fc1',
        'mlp.c_proj': 'mlp.fc2',
        'mlp.linear1': 'mlp.fc1',
        'mlp.linear2': 'mlp.fc2'
    }
    
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    
    return normalized

def extract_encoder_architecture(weights, name):
    """提取encoder部分的架构信息"""
    if weights is None:
        return None
    
    # 处理不同的权重文件格式
    if isinstance(weights, dict):
        if 'state_dict' in weights:
            state_dict = weights['state_dict']
        elif 'model' in weights:
            state_dict = weights['model']
        else:
            state_dict = weights
    else:
        state_dict = weights
    
    encoder_architecture = OrderedDict()
    
    # 定义encoder相关的关键词
    encoder_keywords = [
        'encoder', 'visual', 'vision', 'backbone', 'feature',
        'transformer', 'blocks', 'layers', 'resblocks',
        'attention', 'attn', 'mlp', 'norm', 'ln_', 'layernorm',
        'patch_embed', 'pos_embed', 'cls_token', 'positional_embedding'
    ]
    
    if isinstance(state_dict, dict):
        for key, value in state_dict.items():
            # 检查是否为encoder相关的层
            key_lower = key.lower()
            is_encoder_layer = any(keyword in key_lower for keyword in encoder_keywords)
            
            # 排除明显的非encoder层（如分类头、解码器等）
            exclude_keywords = ['head', 'classifier', 'fc', 'decoder', 'text']
            is_excluded = any(exclude in key_lower for exclude in exclude_keywords)
            
            if is_encoder_layer and not is_excluded:
                if hasattr(value, 'shape'):
                    normalized_key = normalize_layer_name(key)
                    encoder_architecture[normalized_key] = tuple(value.shape)
    
    print(f"📊 {name} encoder部分包含 {len(encoder_architecture)} 个参数层")
    return encoder_architecture

def find_matching_layers(arch1, arch2):
    """寻找两个架构中可能匹配的层"""
    matches = []
    unmatched_1 = set(arch1.keys())
    unmatched_2 = set(arch2.keys())
    
    # 精确匹配
    for key1 in list(unmatched_1):
        if key1 in unmatched_2:
            matches.append((key1, key1, 'exact'))
            unmatched_1.remove(key1)
            unmatched_2.remove(key1)
    
    # 模糊匹配 - 基于层的功能和位置
    for key1 in list(unmatched_1):
        best_match = None
        best_score = 0
        
        for key2 in unmatched_2:
            # 计算相似度分数
            score = 0
            
            # 提取数字（层索引）
            nums1 = re.findall(r'\d+', key1)
            nums2 = re.findall(r'\d+', key2)
            if nums1 and nums2 and nums1 == nums2:
                score += 3
            
            # 检查关键词匹配
            key_words1 = set(re.split(r'[._]', key1.lower()))
            key_words2 = set(re.split(r'[._]', key2.lower()))
            common_words = key_words1.intersection(key_words2)
            score += len(common_words)
            
            # 形状匹配
            if arch1[key1] == arch2[key2]:
                score += 5
            
            if score > best_score and score >= 3:  # 最低匹配阈值
                best_score = score
                best_match = key2
        
        if best_match:
            matches.append((key1, best_match, 'fuzzy'))
            unmatched_1.remove(key1)
            unmatched_2.remove(best_match)
    
    return matches, unmatched_1, unmatched_2

def compare_encoder_architectures(arch1, arch2):
    """详细比较两个encoder架构"""
    print("\n" + "="*60)
    print("🔍 Encoder架构对比分析")
    print("="*60)
    
    if arch1 is None or arch2 is None:
        print("❌ 无法进行比较：权重文件加载失败")
        return False
    
    print(f"\n📈 统计信息:")
    print(f"   权重文件1 encoder层数: {len(arch1)}")
    print(f"   权重文件2 encoder层数: {len(arch2)}")
    
    # 寻找匹配的层
    matches, unmatched_1, unmatched_2 = find_matching_layers(arch1, arch2)
    
    print(f"   匹配的层数: {len(matches)}")
    print(f"   权重1中未匹配: {len(unmatched_1)}")
    print(f"   权重2中未匹配: {len(unmatched_2)}")
    
    # 显示匹配的层
    if matches:
        print(f"\n✅ 匹配的层 ({len(matches)}个):")
        exact_matches = [m for m in matches if m[2] == 'exact']
        fuzzy_matches = [m for m in matches if m[2] == 'fuzzy']
        
        if exact_matches:
            print(f"\n   精确匹配 ({len(exact_matches)}个):")
            for i, (key1, key2, _) in enumerate(exact_matches[:10], 1):  # 只显示前10个
                shape1, shape2 = arch1[key1], arch2[key2]
                status = "✅" if shape1 == shape2 else "❌"
                print(f"   {i:2d}. {key1} -> {key2} {status}")
                if shape1 != shape2:
                    print(f"       形状: {shape1} vs {shape2}")
            if len(exact_matches) > 10:
                print(f"       ... 还有 {len(exact_matches) - 10} 个精确匹配")
        
        if fuzzy_matches:
            print(f"\n   模糊匹配 ({len(fuzzy_matches)}个):")
            for i, (key1, key2, _) in enumerate(fuzzy_matches[:5], 1):  # 只显示前5个
                shape1, shape2 = arch1[key1], arch2[key2]
                status = "✅" if shape1 == shape2 else "❌"
                print(f"   {i:2d}. {key1} -> {key2} {status}")
                if shape1 != shape2:
                    print(f"       形状: {shape1} vs {shape2}")
            if len(fuzzy_matches) > 5:
                print(f"       ... 还有 {len(fuzzy_matches) - 5} 个模糊匹配")
    
    # 显示未匹配的层
    if unmatched_1:
        print(f"\n🚨 权重文件1中未匹配的层 ({len(unmatched_1)}个):")
        for i, key in enumerate(sorted(unmatched_1)[:10], 1):
            print(f"   {i:2d}. {key} -> {arch1[key]}")
        if len(unmatched_1) > 10:
            print(f"       ... 还有 {len(unmatched_1) - 10} 个未匹配层")
    
    if unmatched_2:
        print(f"\n🚨 权重文件2中未匹配的层 ({len(unmatched_2)}个):")
        for i, key in enumerate(sorted(unmatched_2)[:10], 1):
            print(f"   {i:2d}. {key} -> {arch2[key]}")
        if len(unmatched_2) > 10:
            print(f"       ... 还有 {len(unmatched_2) - 10} 个未匹配层")
    
    # 检查形状不匹配
    shape_mismatches = [(k1, k2) for k1, k2, _ in matches if arch1[k1] != arch2[k2]]
    
    # 最终结论
    print(f"\n" + "="*60)
    print(f"🎯 Encoder架构对比结论")
    print(f"="*60)
    
    total_layers = max(len(arch1), len(arch2))
    match_ratio = len(matches) / total_layers if total_layers > 0 else 0
    
    print(f"匹配度: {match_ratio:.2%} ({len(matches)}/{total_layers})")
    
    if match_ratio >= 0.8 and len(shape_mismatches) == 0:
        print("✅ Encoder架构高度相似")
        conclusion = "highly_similar"
    elif match_ratio >= 0.6:
        print("⚠️ Encoder架构部分相似")
        conclusion = "partially_similar"
    else:
        print("❌ Encoder架构差异较大")
        conclusion = "different"
    
    if shape_mismatches:
        print(f"⚠️ 发现 {len(shape_mismatches)} 个形状不匹配")
    
    return conclusion

def main():
    """主函数 - 专注于encoder架构对比"""
    print("🔍 开始Encoder权重架构对比分析...")
    print("="*60)
    
    # 加载权重文件
    print("📂 加载权重文件:")
    weights1 = load_weights(weight_path1)
    weights2 = load_weights(weight_path2)
    
    # 提取encoder架构信息
    print("\n🏗️ 提取Encoder架构信息:")
    arch1 = extract_encoder_architecture(weights1, "权重文件1")
    arch2 = extract_encoder_architecture(weights2, "权重文件2")
    
    # 进行encoder架构对比
    result = compare_encoder_architectures(arch1, arch2)
    
    return result

if __name__ == "__main__":
    similarity = main()
    print(f"\n🏁 程序执行完成，Encoder架构相似度: {similarity}")
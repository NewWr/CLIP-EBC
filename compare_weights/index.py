weight_path1 = "/opt/DM/OCT/CLIP_Code/CLIP-EBC/models/clip/_clip/weights/clip_text_encoder_vit_b_16.pth"
weight_path2 = "/opt/DM/OCT/CLIP_Code/CLIP-EBC/compare_weights/open_clip_pytorch_model.bin"

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

def convert_to_timm_convnext_keys(state_dict):
    """将权重文件的keys转换为timm ConvNext标准格式"""
    if state_dict is None:
        return None
    
    # 处理不同的权重文件格式
    if isinstance(state_dict, dict):
        if 'state_dict' in state_dict:
            original_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            original_dict = state_dict['model']
        else:
            original_dict = state_dict
    else:
        original_dict = state_dict
    
    converted_dict = OrderedDict()
    
    # timm ConvNext标准命名映射规则
    key_mappings = {
        # Stem层映射
        r'^(visual\.|encoder\.|backbone\.|model\.)?stem\.(\d+)\.': r'stem.\2.',
        r'^(visual\.|encoder\.|backbone\.|model\.)?patch_embed\.(\w+)': r'stem.0.\2',
        r'^(visual\.|encoder\.|backbone\.|model\.)?conv1\.': r'stem.0.',
        
        # Stage层映射 (stages.0, stages.1, stages.2, stages.3)
        r'^(visual\.|encoder\.|backbone\.|model\.)?stages\.(\d+)\.(\d+)\.': r'stages.\2.\3.',
        r'^(visual\.|encoder\.|backbone\.|model\.)?layer(\d+)\.(\d+)\.': lambda m: f'stages.{int(m.group(2))-1}.{m.group(3)}.',
        r'^(visual\.|encoder\.|backbone\.|model\.)?blocks\.(\d+)\.(\d+)\.': r'stages.\2.\3.',
        
        # ConvNext Block内部组件映射
        r'\.(dwconv|depthwise_conv)\.': r'.conv_dw.',
        r'\.(pwconv1|pointwise_conv1)\.': r'.mlp.fc1.',
        r'\.(pwconv2|pointwise_conv2)\.': r'.mlp.fc2.',
        r'\.norm\.': r'.norm.',
        r'\.ln\.': r'.norm.',
        r'\.layer_norm\.': r'.norm.',
        r'\.layernorm\.': r'.norm.',
        r'\.gamma': r'.gamma',
        r'\.layer_scale': r'.gamma',
        
        # 分类头映射
        r'^(visual\.|encoder\.|backbone\.|model\.)?head\.(\w+)': r'head.\2',
        r'^(visual\.|encoder\.|backbone\.|model\.)?classifier\.(\w+)': r'head.\2',
        r'^(visual\.|encoder\.|backbone\.|model\.)?fc\.(\w+)': r'head.\2',
        
        # Norm层映射
        r'^(visual\.|encoder\.|backbone\.|model\.)?norm\.(\w+)': r'norm.\2',
        r'^(visual\.|encoder\.|backbone\.|model\.)?ln_final\.(\w+)': r'norm.\2',
        r'^(visual\.|encoder\.|backbone\.|model\.)?final_norm\.(\w+)': r'norm.\2',
    }
    
    print(f"🔄 开始转换权重keys为timm ConvNext格式...")
    converted_count = 0
    unchanged_count = 0
    
    for original_key, value in original_dict.items():
        converted_key = original_key
        conversion_applied = False
        
        # 应用映射规则
        for pattern, replacement in key_mappings.items():
            if callable(replacement):
                # 处理lambda函数替换
                match = re.search(pattern, converted_key)
                if match:
                    converted_key = re.sub(pattern, replacement(match), converted_key)
                    conversion_applied = True
                    break
            else:
                # 处理字符串替换
                if re.search(pattern, converted_key):
                    converted_key = re.sub(pattern, replacement, converted_key)
                    conversion_applied = True
                    break
        
        # 清理多余的前缀
        prefixes_to_remove = [
            'visual.', 'encoder.', 'backbone.', 'model.', 'vision_model.',
            'feature_extractor.', 'base_model.'
        ]
        
        for prefix in prefixes_to_remove:
            if converted_key.startswith(prefix):
                converted_key = converted_key[len(prefix):]
                conversion_applied = True
                break
        
        converted_dict[converted_key] = value
        
        if conversion_applied:
            converted_count += 1
            if converted_count <= 10:  # 只显示前10个转换示例
                print(f"  ✅ {original_key} -> {converted_key}")
        else:
            unchanged_count += 1
    
    if converted_count > 10:
        print(f"  ... 还有 {converted_count - 10} 个keys被转换")
    
    print(f"\n📊 转换统计:")
    print(f"  总keys数量: {len(original_dict)}")
    print(f"  已转换: {converted_count}")
    print(f"  未改变: {unchanged_count}")
    
    return converted_dict

def save_converted_weights(converted_dict, original_path, suffix="_timm_convnext"):
    """保存转换后的权重文件"""
    if converted_dict is None:
        print("❌ 无法保存：转换后的权重为空")
        return None
    
    # 生成新的文件路径
    base_path = os.path.splitext(original_path)[0]
    extension = os.path.splitext(original_path)[1]
    new_path = f"{base_path}{suffix}{extension}"
    
    try:
        torch.save(converted_dict, new_path)
        print(f"✅ 转换后的权重已保存到: {new_path}")
        return new_path
    except Exception as e:
        print(f"❌ 保存权重文件失败: {e}")
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
    """主函数 - 专注于encoder架构对比和权重转换"""
    print("🔍 开始Encoder权重架构对比分析和timm ConvNext格式转换...")
    print("="*60)
    
    # 加载权重文件
    print("📂 加载权重文件:")
    weights1 = load_weights(weight_path1)
    weights2 = load_weights(weight_path2)
    
    # 转换权重keys为timm ConvNext格式
    print("\n🔄 转换权重keys为timm ConvNext格式:")
    if weights1 is not None:
        print("\n处理权重文件1:")
        converted_weights1 = convert_to_timm_convnext_keys(weights1)
        if converted_weights1 is not None:
            save_converted_weights(converted_weights1, weight_path1)
    
    if weights2 is not None:
        print("\n处理权重文件2:")
        converted_weights2 = convert_to_timm_convnext_keys(weights2)
        if converted_weights2 is not None:
            save_converted_weights(converted_weights2, weight_path2)
    
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import os, json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from typing import List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

current_dir = os.path.abspath(os.path.dirname(__file__))

from datasets import UKBDataset
from models import get_model
from utils import get_config
from datasets.transforms import Resize


class GradCAM:
    """GradCAM可视化类"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()
    
    def hook_layers(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, class_idx=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        if isinstance(output, tuple):
            output = output[0] if self.model.training else output[1]
        
        # For regression models, we use the output directly
        if class_idx is None:
            score = output.mean()  # Use mean of output for regression
        else:
            score = output[0, class_idx]
        
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        return cam.detach().cpu().numpy()


class LayerCAM:
    """LayerCAM可视化类"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()
    
    def hook_layers(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, class_idx=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        if isinstance(output, tuple):
            output = output[0] if self.model.training else output[1]
        
        # For regression models, we use the output directly
        if class_idx is None:
            score = output.mean()  # Use mean of output for regression
        else:
            score = output[0, class_idx]
        
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Generate LayerCAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Element-wise multiplication and sum across channels
        cam = torch.sum(gradients * activations, dim=0)  # [H, W]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        return cam.detach().cpu().numpy()


class GradCAMPlusPlus:
    """Grad-CAM++可视化类"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()
    
    def hook_layers(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, class_idx=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        if isinstance(output, tuple):
            output = output[0] if self.model.training else output[1]
        
        # For regression models, we use the output directly
        if class_idx is None:
            score = output.mean()  # Use mean of output for regression
        else:
            score = output[0, class_idx]
        
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Generate Grad-CAM++
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Calculate alpha weights
        alpha_num = gradients.pow(2)
        alpha_denom = 2.0 * gradients.pow(2) + activations.mul(gradients.pow(3)).sum(dim=(1, 2), keepdim=True)
        alpha = alpha_num.div(alpha_denom + 1e-7)
        
        # Calculate weights
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        return cam.detach().cpu().numpy()


class AttentionVisualization:
    """注意力可视化类（适用于ViT模型）"""
    def __init__(self, model):
        self.model = model
        self.attention_maps = []
        self.hook_attention_layers()
    
    def hook_attention_layers(self):
        def attention_hook(module, input, output):
            # output[1] contains attention weights
            if len(output) > 1 and output[1] is not None:
                self.attention_maps.append(output[1])
        
        # Hook attention layers in ViT
        if hasattr(self.model, 'image_encoder') and hasattr(self.model.image_encoder, 'transformer'):
            for layer in self.model.image_encoder.transformer.resblocks:
                if hasattr(layer, 'attn'):
                    layer.attn.register_forward_hook(attention_hook)
    
    def generate_attention_map(self, input_image, layer_idx=-1, head_idx=0):
        self.model.eval()
        self.attention_maps = []
        
        with torch.no_grad():
            _ = self.model(input_image)
        
        if not self.attention_maps:
            return None
        
        # Get attention from specified layer
        attention = self.attention_maps[layer_idx][0]  # [num_heads, seq_len, seq_len]
        attention = attention[head_idx]  # [seq_len, seq_len]
        
        # Remove CLS token attention
        attention = attention[1:, 1:]  # Remove first row and column (CLS token)
        
        # Reshape to spatial dimensions
        seq_len = attention.shape[0]
        spatial_dim = int(np.sqrt(seq_len))
        attention_map = attention.mean(dim=0).reshape(spatial_dim, spatial_dim)
        
        return attention_map.cpu().numpy()


def get_target_layer(model, backbone):
    """获取目标层用于可视化"""
    if "resnet" in backbone:
        # For ResNet-based CLIP models
        if hasattr(model, 'image_encoder'):
            return model.image_encoder.layer4[-1]  # Last layer of ResNet
        else:
            return model.backbone.encoder.layer4[-1]
    elif "vit" in backbone:
        # For ViT-based CLIP models
        if hasattr(model, 'image_encoder'):
            return model.image_encoder.transformer.resblocks[-1]  # Last transformer block
        else:
            return model.backbone.encoder.blocks[-1]
    else:
        # Fallback
        return list(model.modules())[-2]


def preprocess_image(image_path, input_size=224):
    """预处理图像"""
    image = Image.open(image_path)
    
    # 处理不同通道的图像
    if image.mode == 'L':  # 灰度图像
        image = image.convert('RGB')  # 转换为3通道
    elif image.mode == 'RGBA':  # 带透明通道的图像
        # 创建白色背景
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])  # 使用alpha通道作为mask
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 转换为tensor并调整维度 (H, W, C) -> (C, H, W)
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    
    # 使用自定义Resize变换（需要传递空的label）
    empty_label = torch.empty(0, 2)
    transform = Resize((input_size, input_size))
    image_tensor, _ = transform(image_tensor, empty_label)
    
    # 标准化（使用ImageNet的均值和标准差）
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    # 添加batch维度
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, image


def overlay_heatmap(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """将热力图叠加到原图像上"""
    # Convert PIL image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed


def visualize_model_attention(args):
    """主要的可视化函数"""
    print(f"Visualizing model attention for {args.target_column} prediction.")
    device = torch.device(args.device)
    
    # Configure model parameters
    if args.regression:
        bins, anchor_points = None, None
    else:
        with open(os.path.join(current_dir, "configs", f"reduction_{args.reduction}.json"), "r") as f:
            config = json.load(f)[str(args.truncation)]
        
        dataset_key = args.target_column
        config = config[dataset_key]
        bins = config["bins"][args.granularity]
        anchor_points = config["anchor_points"][args.granularity]["average"] if args.anchor_points == "average" else config["anchor_points"][args.granularity]["middle"]
        bins = [(float(b[0]), float(b[1])) for b in bins]
        anchor_points = [float(p) for p in anchor_points]

    # Initialize model
    model = get_model(
        backbone=args.model,
        input_size=args.input_size, 
        reduction=args.reduction,
        bins=bins,
        anchor_points=anchor_points,
        prompt_type=args.prompt_type,
        num_vpt=args.num_vpt,
        vpt_drop=args.vpt_drop,
        deep_vpt=not args.shallow_vpt
    )
    
    # Load model weights
    state_dict = torch.load(args.weight_path, map_location="cpu")
    state_dict = state_dict if "best" in os.path.basename(args.weight_path) else state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(args.image_path, args.input_size)
    image_tensor = image_tensor.to(device)
    
    # Get target layer
    target_layer = get_target_layer(model, args.model)
    
    # Create output directory
    output_dir = os.path.join(current_dir, "visualization")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    
    # Generate visualizations
    visualizations = {}
    
    if 'gradcam' in args.methods:
        print("Generating GradCAM...")
        gradcam = GradCAM(model, target_layer)
        cam_map = gradcam.generate_cam(image_tensor)
        visualizations['gradcam'] = cam_map
    
    if 'layercam' in args.methods:
        print("Generating LayerCAM...")
        layercam = LayerCAM(model, target_layer)
        layer_map = layercam.generate_cam(image_tensor)
        visualizations['layercam'] = layer_map
    
    if 'gradcam++' in args.methods:
        print("Generating Grad-CAM++...")
        gradcam_pp = GradCAMPlusPlus(model, target_layer)
        gradcam_pp_map = gradcam_pp.generate_cam(image_tensor)
        visualizations['gradcam++'] = gradcam_pp_map
    
    if 'attention' in args.methods and 'vit' in args.model:
        print("Generating Attention Map...")
        attention_vis = AttentionVisualization(model)
        attention_map = attention_vis.generate_attention_map(image_tensor)
        if attention_map is not None:
            visualizations['attention'] = attention_map
    
    # Get model prediction
    with torch.no_grad():
        prediction = model(image_tensor)
        if isinstance(prediction, tuple):
            prediction = prediction[1] if not model.training else prediction[0]
        
        if args.regression:
            pred_value = prediction.item() if prediction.dim() == 0 else prediction.mean().item()
        else:
            pred_value = prediction.sum().item()
    
    # Create visualization plots
    num_methods = len(visualizations)
    if num_methods == 0:
        print("No valid visualization methods selected.")
        return
    
    fig, axes = plt.subplots(2, num_methods, figsize=(5 * num_methods, 10))
    if num_methods == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (method, heatmap) in enumerate(visualizations.items()):
        # Original heatmap
        axes[0, i].imshow(heatmap, cmap='jet')
        axes[0, i].set_title(f'{method.upper()} Heatmap')
        axes[0, i].axis('off')
        
        # Overlayed image
        overlayed = overlay_heatmap(original_image, heatmap)
        axes[1, i].imshow(overlayed)
        axes[1, i].set_title(f'{method.upper()} Overlay')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Model Attention Visualization\nPredicted {args.target_column}: {pred_value:.3f}', fontsize=16)
    plt.tight_layout()
    
    # Save visualization
    output_path = os.path.join(output_dir, f"{base_name}_{args.target_column}_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to: {output_path}")
    print(f"Predicted {args.target_column}: {pred_value:.3f}")


def main():
    parser = ArgumentParser(description="Visualize CLIP-EBC model attention on UKB images.")
    
    # Model parameters
    parser.add_argument("--model", type=str, default="clip_resnet50", help="The model to visualize.")
    parser.add_argument("--input_size", type=int, default=224, help="The size of the input image.")
    parser.add_argument("--reduction", type=int, default=8, choices=[8, 16, 32], help="The reduction factor of the model.")
    parser.add_argument("--regression", action="store_true", help="Use blockwise regression instead of classification.")
    parser.add_argument("--truncation", type=int, default=15, help="The truncation of the count.")
    parser.add_argument("--anchor_points", type=str, default="average", choices=["average", "middle"], help="The representative count values of bins.")
    parser.add_argument("--prompt_type", type=str, default="word", choices=["word", "number"], help="The prompt type for CLIP.")
    parser.add_argument("--granularity", type=str, default="fine", choices=["fine", "dynamic", "coarse"], help="The granularity of bins.")
    parser.add_argument("--num_vpt", type=int, default=32, help="The number of visual prompt tokens.")
    parser.add_argument("--vpt_drop", type=float, default=0.0, help="The dropout rate for visual prompt tokens.")
    parser.add_argument("--shallow_vpt", action="store_true", help="Use shallow visual prompt tokens.")
    parser.add_argument("--weight_path", type=str, required=True, help="The path to the weights of the model.")
    
    # Target parameters
    parser.add_argument('--target_column', default='diabp', type=str,
                        choices=['diabp', 'sysbp', 'hba1c', 'fbg', 'ldl', 'hdl'],
                        help='target column for regression')
    
    # Visualization parameters
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image to visualize.")
    parser.add_argument("--methods", nargs='+', default=['gradcam', 'layercam'], 
                        choices=['gradcam', 'layercam', 'gradcam++', 'attention'],
                        help="Visualization methods to use.")
    parser.add_argument("--device", type=str, default="cuda", help="The device to use for visualization.")
    
    args = parser.parse_args()
    args.model = args.model.lower()
    
    # Set default parameters based on model type
    if args.regression:
        args.truncation = None
        args.anchor_points = None
        args.bins = None
        args.prompt_type = None
        args.granularity = None

    if "clip_vit" not in args.model:
        args.num_vpt = None
        args.vpt_drop = None
        args.shallow_vpt = None
    
    if "clip" not in args.model:
        args.prompt_type = None
    
    # Check if image path exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image path {args.image_path} does not exist.")
        return
    
    visualize_model_attention(args)


if __name__ == "__main__":
    main()

# Example usage:
# python visualize_ukb.py --model clip_resnet50 --target_column diabp --weight_path ./checkpoints/ukb/clip_resnet50_word_224_8_15_fine_1.0_mae/best_mae.pth --image_path /path/to/your/image.jpg --methods gradcam layercam gradcam++ --device cuda:0 --regression
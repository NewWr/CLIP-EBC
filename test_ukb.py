import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import os, json
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

current_dir = os.path.abspath(os.path.dirname(__file__))

from datasets import UKBDataset, Resize2Multiple
from models import get_model
from utils import get_config, calculate_errors, sliding_window_predict
from datasets.transforms import Resize

parser = ArgumentParser(description="Test a trained model on the UKB dataset.")

# Parameters for model
parser.add_argument("--model", type=str, default="clip_resnet50", help="The model to test.")
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
parser.add_argument("--weight_path", type=str, default="/opt/DM/OCT/CLIP_Code/CLIP-EBC/checkpoints/ukb/clip_resnet50_word_224_8_15_fine_1.0_mae/best_mae_0.pth", help="The pat to the weights of the model.")

# Parameters for UKB dataset
parser.add_argument('--dataset_type', type=str, default='ukb', choices=['crowd', 'ukb'], 
                   help='Dataset type: crowd counting or UKB regression')
parser.add_argument('--data_path', default='/mnt/e/UKB/StandardData/preprocessed_images/', type=str,
                        help='dataset path')
parser.add_argument('--excel_path', default='/mnt/e/UKB/StandardData/metadata/ukb_test_set.xlsx', type=str,
                        help='path to Excel metadata file for test set')
parser.add_argument('--target_column', default='diabp', type=str,
                        choices=['diabp', 'sysbp', 'hba1c', 'fbg', 'ldl', 'hdl'],
                        help='target column for regression')

# Parameters for evaluation
parser.add_argument("--sliding_window", action="store_true", help="Use sliding window strategy for evaluation.")
parser.add_argument("--stride", type=int, default=None, help="The stride for sliding window strategy.")
parser.add_argument("--window_size", type=int, default=None, help="The window size for in prediction.")
parser.add_argument("--resize_to_multiple", action="store_true", help="Resize the image to the nearest multiple of the input size.")
parser.add_argument("--zero_pad_to_multiple", action="store_true", help="Zero pad the image to the nearest multiple of the input size.")

parser.add_argument("--device", type=str, default="cuda", help="The device to use for evaluation.")
parser.add_argument("--num_workers", type=int, default=4, help="The number of workers for the data loader.")
parser.add_argument("--batch_size", type=int, default=8, help="The batch size for evaluation.")


def calculate_regression_metrics(predictions, targets):
    """计算回归相关的评估指标"""
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    # 相关系数
    pearson_corr, pearson_p = pearsonr(predictions, targets)
    spearman_corr, spearman_p = spearmanr(predictions, targets)
    # 平均绝对百分比误差 (MAPE)
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    # 平均百分比误差 (MPE)
    mpe = np.mean((targets - predictions) / (targets + 1e-8)) * 100
    # 标准化均方根误差 (NRMSE)
    nrmse = rmse / (np.max(targets) - np.min(targets)) * 100
    # 解释方差分数
    explained_var = 1 - np.var(targets - predictions) / np.var(targets)
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2_score': float(r2),
        'pearson_correlation': float(pearson_corr),
        'pearson_p_value': float(pearson_p),
        'spearman_correlation': float(spearman_corr),
        'spearman_p_value': float(spearman_p),
        'mape': float(mape),
        'mpe': float(mpe),
        'nrmse': float(nrmse),
        'explained_variance': float(explained_var)
    }


def evaluate_ukb(
    model,
    data_loader: DataLoader,
    device: torch.device,
    sliding_window: bool = False,
    window_size: int = None,
    stride: int = None,
    regression: bool = True
):
    """参考eval.py的评估函数结构"""
    model.eval()
    pred_values, target_values, filenames = [], [], []
    
    if sliding_window:
        assert window_size is not None, f"Window size must be provided when sliding_window is True, but got {window_size}"
        assert stride is not None, f"Stride must be provided when sliding_window is True, but got {stride}"

    for batch_data in tqdm(data_loader, desc="Evaluating"):
        if len(batch_data) == 3:  # 包含filename
            images, targets, batch_filenames = batch_data
            filenames.extend(batch_filenames)
        else:  # 不包含filename
            images, targets = batch_data
            filenames.extend([f"sample_{i}" for i in range(len(targets))])
        
        images = images.to(device)
        target_values.extend(targets.cpu().numpy().tolist())

        with torch.set_grad_enabled(False):
            if sliding_window:
                pred_value = sliding_window_predict(model, images, window_size, stride)
            else:
                pred_value = model(images)
                
            if regression:
                # 回归任务：直接输出预测值
                if pred_value.dim() > 1:
                    pred_value = pred_value.squeeze()
                pred_values.extend(pred_value.cpu().numpy().tolist())
            else:
                # 分类任务：对密度图求和
                if pred_value.dim() == 4:  # [B, C, H, W]
                    pred_value = pred_value.sum(dim=(1, 2, 3))
                elif pred_value.dim() == 3:  # [B, H, W]
                    pred_value = pred_value.sum(dim=(1, 2))
                pred_values.extend(pred_value.cpu().numpy().tolist())

    pred_values = np.array(pred_values)
    target_values = np.array(target_values)
    
    assert len(pred_values) == len(target_values), f"Length of predictions and ground truths should be equal, but got {len(pred_values)} and {len(target_values)}"
    
    return pred_values, target_values, filenames


def main(args: ArgumentParser):
    print(f"Testing a trained model on the UKB dataset ({args.target_column}).")
    device = torch.device(args.device)
    _ = get_config(vars(args).copy(), mute=False)
    
    # Configure bins and anchor points for regression or classification
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

    args.bins = bins
    args.anchor_points = anchor_points

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

    # 创建数据集实例
    dataset = UKBDataset(
        excel_path=args.excel_path,
        data_root=args.data_path,
        target_column=args.target_column,
        split="test",  # 或者 "all"
        transforms=None,
        return_filename=False,
        num_crops=1,
    )

    # 定义collate函数
    def ukb_collate_fn(batch):
        if len(batch[0]) == 3:  # images, targets, filenames
            images, targets, filenames = zip(*batch)
            images = torch.cat(images, 0)
            targets = torch.cat(targets, 0)
            return images, targets, filenames
        else:  # images, targets
            images, targets = zip(*batch)
            images = torch.cat(images, 0)
            targets = torch.cat(targets, 0)
            return images, targets
    
    # 然后在DataLoader中使用
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=ukb_collate_fn
    )

    print(f"Loaded {len(dataset)} test samples for evaluation.")

    # 使用参考eval.py的评估函数
    predictions, targets, filenames = evaluate_ukb(
        model=model,
        data_loader=data_loader,
        device=device,
        sliding_window=args.sliding_window,
        window_size=args.window_size,
        stride=args.stride,
        regression=args.regression
    )

    # 计算详细的回归指标
    metrics = calculate_regression_metrics(predictions, targets)
    
    print(f"\n=== Evaluation Results for {args.target_column} ===")
    print(f"Number of samples: {len(predictions)}")
    print(f"Target range: [{targets.min():.2f}, {targets.max():.2f}]")
    print(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print("\n--- Basic Metrics ---")
    print(f"MAE (Mean Absolute Error): {metrics['mae']:.4f}")
    print(f"MSE (Mean Squared Error): {metrics['mse']:.4f}")
    print(f"RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print("\n--- Correlation Metrics ---")
    print(f"Pearson Correlation: {metrics['pearson_correlation']:.4f} (p-value: {metrics['pearson_p_value']:.4e})")
    print(f"Spearman Correlation: {metrics['spearman_correlation']:.4f} (p-value: {metrics['spearman_p_value']:.4e})")
    print("\n--- Percentage-based Metrics ---")
    print(f"MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%")
    print(f"MPE (Mean Percentage Error): {metrics['mpe']:.2f}%")
    print(f"NRMSE (Normalized RMSE): {metrics['nrmse']:.2f}%")
    print(f"Explained Variance: {metrics['explained_variance']:.4f}")
    
    # Save results
    result_dir = os.path.join(current_dir, "ukb_test_results")
    os.makedirs(result_dir, exist_ok=True)
    
    weights_dir, weights_name = os.path.split(args.weight_path)
    model_name = os.path.split(weights_dir)[-1]
    result_path = os.path.join(result_dir, f"{model_name}_{weights_name.split('.')[0]}_{args.target_column}.csv")

    # Save detailed results to CSV
    results_df = pd.DataFrame({
        'filename': filenames,
        'prediction': predictions,
        'target': targets,
        'absolute_error': np.abs(predictions - targets),
        'squared_error': (predictions - targets) ** 2,
        'percentage_error': (targets - predictions) / (targets + 1e-8) * 100
    })
    results_df.to_csv(result_path, index=False)
    
    # Save comprehensive summary metrics
    summary_path = os.path.join(result_dir, f"{model_name}_{weights_name.split('.')[0]}_{args.target_column}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"UKB Test Results for {args.target_column}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Weight path: {args.weight_path}\n")
        f.write(f"Number of test samples: {len(predictions)}\n")
        f.write(f"Target range: [{targets.min():.2f}, {targets.max():.2f}]\n")
        f.write(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]\n")
        f.write(f"\nBasic Metrics:\n")
        f.write(f"MAE: {metrics['mae']:.4f}\n")
        f.write(f"MSE: {metrics['mse']:.4f}\n")
        f.write(f"RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"R² Score: {metrics['r2_score']:.4f}\n")
        f.write(f"\nCorrelation Metrics:\n")
        f.write(f"Pearson Correlation: {metrics['pearson_correlation']:.4f} (p-value: {metrics['pearson_p_value']:.4e})\n")
        f.write(f"Spearman Correlation: {metrics['spearman_correlation']:.4f} (p-value: {metrics['spearman_p_value']:.4e})\n")
        f.write(f"\nPercentage-based Metrics:\n")
        f.write(f"MAPE: {metrics['mape']:.2f}%\n")
        f.write(f"MPE: {metrics['mpe']:.2f}%\n")
        f.write(f"NRMSE: {metrics['nrmse']:.2f}%\n")
        f.write(f"Explained Variance: {metrics['explained_variance']:.4f}\n")
    
    print(f"\nResults saved to:")
    print(f"Detailed results: {result_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    args.model = args.model.lower()

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

    if args.sliding_window:
        args.window_size = args.input_size if args.window_size is None else args.window_size
        args.stride = args.input_size if args.stride is None else args.stride
        assert not (args.zero_pad_to_multiple and args.resize_to_multiple), "Cannot use both zero pad and resize to multiple."
    else:
        args.window_size = None
        args.stride = None
        args.zero_pad_to_multiple = False
        args.resize_to_multiple = False
    
    main(args)

# Example usage:
# python test_ukb.py --model clip_resnet50 --target_column diabp --weight_path ./checkpoints/ukb/clip_resnet50_word_224_8_11_fine_1.0_mae/best_mae.pth --device cuda:0 --regression

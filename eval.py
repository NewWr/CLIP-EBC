import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm

from utils import calculate_errors, sliding_window_predict


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    sliding_window: bool = False,
    window_size: Optional[int] = None,
    stride: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    pred_values, target_values = [], []
    
    if sliding_window:
        assert window_size is not None, f"Window size must be provided when sliding_window is True, but got {window_size}"
        assert stride is not None, f"Stride must be provided when sliding_window is True, but got {stride}"

    for image, blood_indicator in tqdm(data_loader):
        image = image.to(device)
        target_values.append(blood_indicator.cpu().numpy().tolist())

        with torch.set_grad_enabled(False):
            if sliding_window:
                pred_value = sliding_window_predict(model, image, window_size, stride)
            else:
                pred_value = model(image)

            pred_values.append(pred_value.cpu().numpy().tolist())

    pred_values = np.array([item for sublist in pred_values for item in sublist])
    target_values = np.array([item for sublist in target_values for item in sublist])
    assert len(pred_values) == len(target_values), f"Length of predictions and ground truths should be equal, but got {len(pred_values)} and {len(target_values)}"
    return calculate_errors(pred_values, target_values)

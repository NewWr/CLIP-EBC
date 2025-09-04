import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple


from utils import barrier, reduce_mean, update_loss_info


def train(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    grad_scaler: GradScaler,
    device: torch.device,
    rank: int,
    nprocs: int,
) -> Tuple[nn.Module, Optimizer, GradScaler, Dict[str, float]]:
    model.train()
    model.refresh_text_prompts()
    info = None
    data_iter = tqdm(data_loader) if rank == 0 else data_loader
    ddp = nprocs > 1
    regression = (model.module.bins is None) if ddp else (model.bins is None)

    for image, blood_indicator in data_iter:
        image = image.to(device)
        blood_indicator = blood_indicator.to(device)
        
        with torch.set_grad_enabled(True):
            if grad_scaler is not None:
                with torch.amp.autocast(device_type="cuda", enabled=grad_scaler.is_enabled()):
                    if not regression:
                        pred_class, pred_value = model(image)
                        loss, loss_info = loss_fn(pred_class, pred_value, blood_indicator)
                    else:
                        pred_value = model(image)
                        loss, loss_info = loss_fn(pred_value, blood_indicator)
            else:
                if not regression:
                    pred_class, pred_value = model(image)
                    loss, loss_info = loss_fn(pred_class, pred_value, blood_indicator)
                else:
                    pred_value = model(image)
                    loss, loss_info = loss_fn(pred_value, blood_indicator)

        optimizer.zero_grad()
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_info = {k: reduce_mean(v.detach(), nprocs).item() if ddp else v.detach().item() for k, v in loss_info.items()}
        info = update_loss_info(info, loss_info)

        barrier(ddp)

    return model, optimizer, grad_scaler, {k: np.mean(v) for k, v in info.items()}

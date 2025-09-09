import torch
from torch import nn, Tensor
from typing import Any, List, Tuple, Dict, Optional
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss实现，用于处理类别不平衡问题
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean',
                 label_smoothing: float = 0.0, class_weights: Optional[List[float]] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
        
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            label_smoothing=self.label_smoothing,
            weight=self.class_weights
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BloodIndicatorLoss(nn.Module):
    """
    血液指标预测损失函数（简化版）
    结合分类和回归损失，用于眼底彩照预测血液指标
    """
    def __init__(
        self,
        indicator_ranges: List[Tuple[float, float]],
        weight_regression: float = 1.0,
        weight_classification: float = 0.5,
        regression_loss: str = "huber",
        use_uncertainty: bool = False,
        bins: Optional[List[Tuple[float, float]]] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 1.0,
        label_smoothing: float = 0.1,
        class_weights: Optional[List[float]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        
        # 统一到 self.bins
        ranges = indicator_ranges if bins is None else bins
        assert len(ranges) > 0, f"Expected at least one bin, got {ranges}"
        assert all([len(r) == 2 for r in ranges]), f"Expected all bins to be of length 2, got {ranges}"
        assert all([r[0] <= r[1] for r in ranges]), f"Expected all bins to be in increasing order, got {ranges}"
        self.bins = ranges

        # 按 bins 进行分类监督：类别数 = len(bins)
        self.num_classes = len(self.bins)
        self.weight_regression = weight_regression
        self.weight_classification = weight_classification
        self.use_uncertainty = use_uncertainty

        # 分类损失：Focal Loss + 标签平滑/类别权重
        self.focal_loss_fn = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction="mean",
            label_smoothing=label_smoothing,
            class_weights=class_weights
        )
        
        # 回归损失
        regression_loss = regression_loss.lower()
        assert regression_loss in ["mse", "mae", "huber"], f"Expected regression_loss to be one of ['mse', 'mae', 'huber'], got {regression_loss}"
        
        if regression_loss == "mse":
            self.regression_fn = nn.MSELoss(reduction="mean")
        elif regression_loss == "mae":
            self.regression_fn = nn.L1Loss(reduction="mean")
        else:  # huber
            self.regression_fn = nn.HuberLoss(reduction="mean", delta=1.0)
            
        # 不确定性损失（可选）
        if self.use_uncertainty:
            self.uncertainty_fn = nn.MSELoss(reduction="mean")
    
    def _classify_indicator(self, values: Tensor) -> Tensor:
        """
        将连续的血液指标值按 bins 离散为类别标签
        Args:
            values: [batch_size] 血液指标值
        Returns:
            class_labels: [batch_size] 分类标签（0..len(bins)-1）
        """
        device = values.device
        batch_size = values.size(0)
        class_labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        for idx, (low, high) in enumerate(self.bins):
            mask = (values >= low) & (values <= high)
            class_labels[mask] = idx

        first_low = self.bins[0][0]
        last_high = self.bins[-1][1]
        class_labels[values < first_low] = 0
        class_labels[values > last_high] = self.num_classes - 1
        return class_labels
    
    def forward(
        self, 
        pred_class: Tensor,  # [batch_size, num_classes] 分类预测 logits
        pred_value: Tensor,  # [batch_size] 回归预测值
        target_value: Tensor,  # [batch_size] 目标血液指标值
        pred_uncertainty: Optional[Tensor] = None  # [batch_size] 预测不确定性（可选）
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        前向传播计算损失（简化版）
        """
        assert pred_class.size(0) == pred_value.size(0) == target_value.size(0), \
            f"Batch sizes must match: {pred_class.size(0)}, {pred_value.size(0)}, {target_value.size(0)}"
        
        device = pred_class.device
        target_class = self._classify_indicator(target_value)  # [B]
        classification_loss = self.focal_loss_fn(pred_class, target_class)
        regression_loss = self.regression_fn(pred_value, target_value)

        uncertainty_loss = torch.tensor(0.0, device=device)
        if self.use_uncertainty and pred_uncertainty is not None:
            prediction_error = torch.abs(pred_value - target_value)
            uncertainty_loss = self.uncertainty_fn(pred_uncertainty, prediction_error)

        total_loss = (
            self.weight_classification * classification_loss + 
            self.weight_regression * regression_loss
        )
        if self.use_uncertainty:
            total_loss += 0.1 * uncertainty_loss

        loss_info = {
            "total_loss": total_loss.detach(),
            "classification_loss": classification_loss.detach(),
            "regression_loss": regression_loss.detach(),
        }
        if self.use_uncertainty:
            loss_info["uncertainty_loss"] = uncertainty_loss.detach()
        return total_loss, loss_info
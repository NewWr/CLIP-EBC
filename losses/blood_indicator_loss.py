import torch
from torch import nn, Tensor
from typing import Any, List, Tuple, Dict, Optional


class BloodIndicatorLoss(nn.Module):
    """
    血液指标预测损失函数，基于DACELoss的设计思想
    结合分类和回归损失，用于眼底彩照预测血液指标
    """
    def __init__(
        self,
        indicator_ranges: List[Tuple[float, float]],  # 兼容旧配置：将被视为 bins
        weight_regression: float = 1.0,
        weight_classification: float = 1.0,
        regression_loss: str = "mse",  # "mse", "mae", "huber"
        use_uncertainty: bool = False,  # 是否使用不确定性估计
        bins: Optional[List[Tuple[float, float]]] = None,  # 新增：按bins进行分类监督
        **kwargs: Any
    ) -> None:
        super().__init__()
        
        # 统一到 self.bins（优先使用 bins，否则回退到 indicator_ranges）
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
        
        # 分类损失
        self.cross_entropy_fn = nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.1)
        
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

        # 先按 bins 内部命中
        for idx, (low, high) in enumerate(self.bins):
            mask = (values >= low) & (values <= high)
            class_labels[mask] = idx

        # 对越界样本进行边界归类
        first_low = self.bins[0][0]
        last_high = self.bins[-1][1]
        class_labels[values < first_low] = 0
        class_labels[values > last_high] = self.num_classes - 1

        return class_labels
    
    def _adaptive_weight(self, pred_values: Tensor, target_values: Tensor) -> Tensor:
        """
        根据预测误差自适应调整损失权重
        """
        error = torch.abs(pred_values - target_values)
        # 使用sigmoid函数将误差映射到权重
        weights = torch.sigmoid(error)
        return weights
    
    def forward(
        self, 
        pred_class: Tensor,  # [batch_size, num_classes] 分类预测
        pred_value: Tensor,  # [batch_size] 回归预测值
        target_value: Tensor,  # [batch_size] 目标血液指标值
        pred_uncertainty: Optional[Tensor] = None  # [batch_size] 预测不确定性（可选）
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        前向传播计算损失
        """
        assert pred_class.size(0) == pred_value.size(0) == target_value.size(0), \
            f"Batch sizes must match: {pred_class.size(0)}, {pred_value.size(0)}, {target_value.size(0)}"
        
        # 生成分类标签
        target_class = self._classify_indicator(target_value)
        # 分类损失
        classification_loss = self.cross_entropy_fn(pred_class, target_class)
        # 回归损失
        regression_loss = self.regression_fn(pred_value, target_value)
        # 自适应权重（可选）
        adaptive_weights = self._adaptive_weight(pred_value, target_value)
        weighted_regression_loss = (regression_loss * adaptive_weights.mean())
        # 不确定性损失（可选）
        uncertainty_loss = torch.tensor(0.0, device=pred_value.device)
        if self.use_uncertainty and pred_uncertainty is not None:
            # 不确定性应该与预测误差相关
            prediction_error = torch.abs(pred_value - target_value)
            uncertainty_loss = self.uncertainty_fn(pred_uncertainty, prediction_error)
        # 总损失
        total_loss = (
            self.weight_classification * classification_loss + 
            self.weight_regression * weighted_regression_loss
        )
        if self.use_uncertainty:
            total_loss += 0.1 * uncertainty_loss  # 不确定性损失权重较小
        # 损失信息
        loss_info = {
            "total_loss": total_loss.detach(),
            "classification_loss": classification_loss.detach(),
            "regression_loss": regression_loss.detach(),
            "weighted_regression_loss": weighted_regression_loss.detach(),
        }
        if self.use_uncertainty:
            loss_info["uncertainty_loss"] = uncertainty_loss.detach()
        return total_loss, loss_info


class MultiIndicatorLoss(nn.Module):
    """
    多血液指标预测损失函数
    用于同时预测多个血液指标的场景
    """
    def __init__(
        self,
        indicator_configs: Dict[str, Dict],  # 每个指标的配置
        indicator_weights: Optional[Dict[str, float]] = None,  # 各指标的权重
        **kwargs: Any
    ) -> None:
        super().__init__()
        
        self.indicator_names = list(indicator_configs.keys())
        self.indicator_weights = indicator_weights or {name: 1.0 for name in self.indicator_names}
        
        # 为每个指标创建单独的损失函数
        self.indicator_losses = nn.ModuleDict()
        for name, config in indicator_configs.items():
            self.indicator_losses[name] = BloodIndicatorLoss(**config, **kwargs)
    
    def forward(
        self,
        predictions: Dict[str, Tuple[Tensor, Tensor]],  # {indicator_name: (pred_class, pred_value)}
        targets: Dict[str, Tensor],  # {indicator_name: target_value}
        uncertainties: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        多指标损失计算
        """
        total_loss = torch.tensor(0.0, device=next(iter(targets.values())).device)
        all_loss_info = {}
        
        for name in self.indicator_names:
            if name in predictions and name in targets:
                pred_class, pred_value = predictions[name]
                target_value = targets[name]
                pred_uncertainty = uncertainties.get(name) if uncertainties else None
                
                loss, loss_info = self.indicator_losses[name](
                    pred_class, pred_value, target_value, pred_uncertainty
                )
                
                weighted_loss = loss * self.indicator_weights[name]
                total_loss += weighted_loss
                
                # 记录各指标的损失信息
                all_loss_info[f"{name}_loss"] = weighted_loss.detach()
                for key, value in loss_info.items():
                    all_loss_info[f"{name}_{key}"] = value
        
        all_loss_info["total_loss"] = total_loss.detach()
        
        return total_loss, all_loss_info
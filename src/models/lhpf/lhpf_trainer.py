import logging
import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection
from src.metrics import MR, minADE, minFDE
from src.metrics.prediction_avg_ade import PredAvgADE
from src.metrics.prediction_avg_fde import PredAvgFDE
from src.optim.warmup_cos_lr import WarmupCosLR
from scipy.optimize import linear_sum_assignment

from .loss.esdf_collision_loss import ESDFCollisionLoss

logger = logging.getLogger(__name__)


class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        use_collision_loss=True,
        use_contrast_loss=False,
        use_compliant_loss=False,
        regulate_yaw=False,
        objective_aggregate_mode: str = "mean",
        compliant_loss_weight: float = 1,
    ) -> None:
        """
        Initializes the class.

        :param model: pytorch model
        :param objectives: list of learning objectives used for supervision at each step
        :param metrics: list of planning metrics computed at each step
        :param batch_size: batch_size taken from dataloader config
        :param optimizer: config for instantiating optimizer. Can be 'None' for older models.
        :param lr_scheduler: config for instantiating lr_scheduler. Can be 'None' for older models and when an lr_scheduler is not being used.
        :param warm_up_lr_scheduler: config for instantiating warm up lr scheduler. Can be 'None' for older models and when a warm up lr_scheduler is not being used.
        :param objective_aggregate_mode: how should different objectives be combined, can be 'sum', 'mean', and 'max'.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.objective_aggregate_mode = objective_aggregate_mode
        self.history_steps = model.history_steps
        self.use_collision_loss = use_collision_loss
        self.use_contrast_loss = use_contrast_loss
        self.regulate_yaw = regulate_yaw
        self.use_compliant_loss = use_compliant_loss
        if self.use_compliant_loss:
            self.compliant_loss_weight = compliant_loss_weight
            self.compliant_length = self.model.compliant_length
            self.compliant_loss_decay_weights = self.model.compliant_loss_decay_weights.unsqueeze(-1)
            
        self.radius = model.radius
        self.num_modes = model.num_modes
        self.mode_interval = self.radius / self.num_modes

        if use_collision_loss:
            self.collision_loss = ESDFCollisionLoss()

    def on_fit_start(self) -> None:
        metrics_collection = MetricCollection(
            [
                minADE().to(self.device),
                minFDE().to(self.device),
                MR(miss_threshold=2).to(self.device),
                PredAvgADE().to(self.device),
                PredAvgFDE().to(self.device),
            ]
        )
        self.metrics = {
            "train": metrics_collection.clone(prefix="train/"),
            "val": metrics_collection.clone(prefix="val/"),
        }

    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str
    ) -> torch.Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        features, targets, scenarios = batch
        # features["feature"].data["agent"]["target"] = features["feature"].data["agent"]["target"][..., self.history_steps:, :]
        res = self.forward(features["feature"].data)

        losses = self._compute_objectives(res, features["feature"].data)
        
        features["feature"].data["agent"]["target"] = features["feature"].data["agent"]["target"][..., self.history_steps:, :]
        metrics = self._compute_metrics(res, features["feature"].data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix)

        return losses["loss"] if self.training else 0.0

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        bs, _, T, _ = res["prediction"].shape

        if self.use_contrast_loss:
            train_num = (bs // 3) * 2 if self.training else bs
        else:
            train_num = bs

        trajectory, probability, prediction = (
            res["trajectory_all_time"][:train_num],
            res["probability_all_time"][:train_num],
            res["prediction"][:train_num],
        )
        ref_free_trajectory = res.get("ref_free_trajectory", None)

        targets_pos = data["agent"]["target"][:train_num]
        valid_mask = data["agent"]["valid_mask"][:train_num, :, :]
        targets_vel = data["agent"]["velocity"][:train_num, -1, :]

        target = torch.cat(
            [
                targets_pos[..., :2],
                torch.stack(
                    [targets_pos[..., 2].cos(), targets_pos[..., 2].sin()], dim=-1
                ),
                targets_vel,
            ],
            dim=-1,
        )

        bs, TH, R, M, TF, _ = trajectory.shape
        ego_target_all, ego_target_all_mask = generate_target(position=target[:, 0], 
                                                        mask=data["agent"]["valid_mask"][:,-1, 0],
                                                        num_historical_steps=20,
                                                        num_future_steps=TF, targets_pos=targets_pos)  #[(N1,...Nb),H,F,2],[(N1,...Nb),H,F]
        ego_target_all = ego_target_all.index_select(1, res["select_indices"]-1)
        ego_target_all_mask = ego_target_all_mask.index_select(1, res["select_indices"]-1)

        # planning loss
        ego_reg_loss, ego_cls_loss, collision_loss, best_trajectory_curr = self.get_planning_loss(
            data, trajectory[:, -1], probability[:, -1], ego_target_all_mask[:, -1], ego_target_all[:, -1], train_num
        )

        # history prediction reg loss
        his_prediction_reg_loss, his_prediction_cls_loss, best_trajectory_his = self.get_his_prediction_loss(
            data, trajectory[:, :-1], probability[:, :-1], ego_target_all_mask[:, :-1], ego_target_all[:, :-1], train_num
        )
        
        if self.use_compliant_loss and R > 0:
        # # compliant loss between history prediction and planning
            # current_best_pred = best_trajectory_curr
            # his_best_pred = best_trajectory_his[-1]
            compliant_loss = self.compliant_loss_weight * self.get_his_current_compliant_loss(trajectory)
        else:
            compliant_loss = ego_reg_loss.new_zeros(1)

        if ref_free_trajectory is not None:
            ego_ref_free_reg_loss = F.smooth_l1_loss(
                ref_free_trajectory[:train_num],
                ego_target_all[:, -1, ..., :ref_free_trajectory.shape[-1]],
                reduction="none",
            ).sum(-1)
            ego_ref_free_reg_loss = (
                ego_ref_free_reg_loss * ego_target_all_mask[:train_num, -1]
            ).sum() / ego_target_all_mask[:train_num, -1].sum()
        else:
            ego_ref_free_reg_loss = ego_reg_loss.new_zeros(1)

        # prediction loss
        prediction_loss = self.get_prediction_loss(
            prediction, valid_mask[:, -1, 1:, self.history_steps:], target[:, 1:, self.history_steps:]
        )

        if self.training and self.use_contrast_loss:
            contrastive_loss = self._compute_contrastive_loss(
                res["hidden"], data["data_n_valid_mask"]
            )
        else:
            contrastive_loss = prediction_loss.new_zeros(1)

        loss = (
            ego_reg_loss
            + ego_cls_loss
            + prediction_loss
            + contrastive_loss
            + collision_loss
            + ego_ref_free_reg_loss
            + his_prediction_reg_loss
            + his_prediction_cls_loss
            + compliant_loss
        )

        return {
            "loss": loss,
            "reg_loss": ego_reg_loss.item(),
            "cls_loss": ego_cls_loss.item(),
            "ref_free_reg_loss": ego_ref_free_reg_loss.item(),
            "collision_loss": collision_loss.item(),
            "prediction_loss": prediction_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "his_prediction_reg_loss": his_prediction_reg_loss.item() if self.model.use_history_attention else 0,
            "his_prediction_cls_loss": his_prediction_cls_loss.item() if self.model.use_history_attention else 0,
            "compliant_loss": compliant_loss.item() if isinstance(compliant_loss, torch.Tensor) else 0
        }

    def get_prediction_loss(self, prediction, valid_mask, target):
        """
        prediction: (bs, A-1, T, 6)
        valid_mask: (bs, A-1, T)
        target: (bs, A-1, 6)
        """
        prediction_loss = F.smooth_l1_loss(
            prediction[valid_mask], target[valid_mask], reduction="none"
        ).sum(-1)
        prediction_loss = prediction_loss.sum() / valid_mask.sum()

        return prediction_loss

    def get_planning_loss(self, data, trajectory, probability, valid_mask, target, bs):
        """
        trajectory: (bs, R, M, T, 4)
        valid_mask: (bs, T)
        """
        valid_mask = valid_mask[:bs]
        num_valid_points = valid_mask.sum(-1)
        endpoint_index = (num_valid_points / 10).long().clamp_(min=0, max=7)  # max 8s
        r_padding_mask = ~data["reference_line"]["valid_mask"][:bs, -1].any(-1)  # (bs, R)
        future_projection = data["reference_line"]["future_projection"][:bs][
            torch.arange(bs), :, endpoint_index
        ]

        target_r_index = torch.argmin(
            future_projection[..., 1] + 1e6 * r_padding_mask, dim=-1
        )
        target_m_index = (
            future_projection[torch.arange(bs), target_r_index, 0] / self.mode_interval
        ).long()
        target_m_index.clamp_(min=0, max=self.num_modes - 1)

        target_label = torch.zeros_like(probability)
        target_label[torch.arange(bs), target_r_index, target_m_index] = 1

        best_trajectory = trajectory[torch.arange(bs), target_r_index, target_m_index]

        if self.use_collision_loss:
            collision_loss = self.collision_loss(
                best_trajectory, data["cost_maps"][:bs, :, :, 0].float()
            )
        else:
            collision_loss = trajectory.new_zeros(1)

        reg_loss = F.smooth_l1_loss(best_trajectory, target, reduction="none").sum(-1)
        reg_loss = (reg_loss * valid_mask).sum() / valid_mask.sum()

        probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

        cls_loss = F.cross_entropy(
            probability.reshape(bs, -1), target_label.reshape(bs, -1).detach()
        )

        if self.regulate_yaw:
            heading_vec_norm = torch.norm(best_trajectory[..., 2:4], dim=-1)
            yaw_regularization_loss = F.l1_loss(
                heading_vec_norm, heading_vec_norm.new_ones(heading_vec_norm.shape)
            )
            reg_loss += yaw_regularization_loss

        return reg_loss, cls_loss, collision_loss, best_trajectory

    def get_his_current_compliant_loss(self, trajectory):

        # compliant_loss = F.smooth_l1_loss(best_current_pred[:, :30], \
        #     best_his_pred[:, 10:40], reduction="none").sum(-1) * self.compliant_loss_decay_weights.to(best_current_pred.device)
        # current_pred = trajectory[:, -1, :, :, :self.compliant_length]
        # his_pred = trajectory[:, -2, :, :, 10 : self.compliant_length + 10]
        current_pred = trajectory[:, -1]
        his_pred = trajectory[:, -2]
        matches, compliant_loss = bidirectional_matching(current_pred, his_pred, self.compliant_loss_decay_weights, self.compliant_length)

        return compliant_loss

    def get_his_prediction_loss(self, data, trajectory, probability, valid_mask, target, bs):
        """
        trajectory: (bs, Th, R, M, Tf, 4)
        valid_mask: (bs, T)
        """

        bs, Th, R, M, Tf, _ = trajectory.shape
        
        if Th==0 :
            return 0, 0

        reg_loss_all = 0
        cls_loss_all = 0
        best_trajectory_list = []

        for i in range(Th):

            valid_mask_once = valid_mask[:bs, i]
            trajectory_once = trajectory[:, i]
            probability_once = probability[:, i]
            target_once = target[:, i]

            num_valid_points = valid_mask_once.sum(-1)
            endpoint_index = (num_valid_points / 10).long().clamp_(min=0, max=7)  # max 8s
            r_padding_mask = ~data["reference_line"]["valid_mask"][:bs, i].any(-1)  # (bs, R)
            future_projection = data["reference_line"]["future_projection"][:bs][
                torch.arange(bs), :, endpoint_index
            ]

            target_r_index = torch.argmin(
                future_projection[..., 1] + 1e6 * r_padding_mask, dim=-1
            )
            target_m_index = (
                future_projection[torch.arange(bs), target_r_index, 0] / self.mode_interval
            ).long()
            target_m_index.clamp_(min=0, max=self.num_modes - 1)

            target_label = torch.zeros_like(probability_once)
            target_label[torch.arange(bs), target_r_index, target_m_index] = 1

            best_trajectory = trajectory_once[torch.arange(bs), target_r_index, target_m_index]

            reg_loss = F.smooth_l1_loss(best_trajectory, target_once, reduction="none").sum(-1)
            reg_loss = (reg_loss * valid_mask_once).sum() / valid_mask_once.sum()

            probability_once.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

            cls_loss = F.cross_entropy(
                probability_once.reshape(bs, -1), target_label.reshape(bs, -1).detach()
            )

            if self.regulate_yaw:
                heading_vec_norm = torch.norm(best_trajectory[..., 2:4], dim=-1)
                yaw_regularization_loss = F.l1_loss(
                    heading_vec_norm, heading_vec_norm.new_ones(heading_vec_norm.shape)
                )
                reg_loss += yaw_regularization_loss
            
            reg_loss_all += reg_loss
            cls_loss_all += cls_loss
            best_trajectory_list.append(best_trajectory)
        
        reg_loss = reg_loss_all / Th
        cls_loss = cls_loss_all / Th

        return reg_loss, cls_loss, best_trajectory_list

    def _compute_contrastive_loss(
        self, hidden, valid_mask, normalize=True, tempreture=0.1
    ):
        """
        Compute triplet loss

        Args:
            hidden: (3*bs, D)
        """
        if normalize:
            hidden = F.normalize(hidden, dim=1, p=2)

        if not valid_mask.any():
            return hidden.new_zeros(1)

        x_a, x_p, x_n = hidden.chunk(3, dim=0)

        x_a = x_a[valid_mask]
        x_p = x_p[valid_mask]
        x_n = x_n[valid_mask]

        logits_ap = (x_a * x_p).sum(dim=1) / tempreture
        logits_an = (x_a * x_n).sum(dim=1) / tempreture
        labels = x_a.new_zeros(x_a.size(0)).long()

        triplet_contrastive_loss = F.cross_entropy(
            torch.stack([logits_ap, logits_an], dim=1), labels
        )
        return triplet_contrastive_loss

    def _compute_metrics(self, res, data, prefix) -> Dict[str, torch.Tensor]:
        """
        Computes a set of planning metrics given the model's predictions and targets.

        :param predictions: model's predictions
        :param targets: ground truth targets
        :return: dictionary of metrics names and values
        """
        # get top 6 modes
        trajectory, probability = res["trajectory"], res["probability"]
        r_padding_mask = ~data["reference_line"]["valid_mask"][:, -1].any(-1)
        probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

        bs, R, M, T, _ = trajectory.shape
        trajectory = trajectory.reshape(bs, R * M, T, -1)
        probability = probability.reshape(bs, R * M)
        top_k_prob, top_k_index = probability.topk(6, dim=-1)
        top_k_traj = trajectory[torch.arange(bs)[:, None], top_k_index]

        outputs = {
            "trajectory": top_k_traj[..., :2],
            "probability": top_k_prob,
            "prediction": res["prediction"][..., :2],
            "prediction_target": data["agent"]["target"][:, 1:],
            "valid_mask": data["agent"]["valid_mask"][:, -1, 1:, self.history_steps :],
        }
        target = data["agent"]["target"][:, 0]

        metrics = self.metrics[prefix](outputs, target)
        return metrics

    def _log_step(
        self,
        loss,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ) -> None:
        """
        Logs the artifacts from a training/validation/test step.

        :param loss: scalar loss value
        :type objectives: [type]
        :param metrics: dictionary of metrics names and values
        :param prefix: prefix prepended at each artifact's name
        :param loss_name: name given to the loss for logging
        """
        self.log(
            f"loss/{prefix}_{loss_name}",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True if prefix == "train" else False,
        )

        for key, value in objectives.items():
            self.log(
                f"objectives/{prefix}_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if metrics is not None:
            self.log_dict(
                metrics,
                prog_bar=(prefix == "val"),
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

    def training_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "val")

    def test_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "test")

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(features)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        # Get optimizer
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        # Get lr_scheduler
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return [optimizer], [scheduler]

    # def on_before_optimizer_step(self, optimizer) -> None:
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print("unused param", name)
def generate_target(position: torch.Tensor, mask: torch.Tensor, num_historical_steps: int, num_future_steps: int, targets_pos: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
    target_traj = [position[:, i+1:i+1+num_future_steps] for i in range(num_historical_steps)]
    target_traj = torch.stack(target_traj, dim=1)
    target_mask = [mask[:,i+1:i+1+num_future_steps] for i in range(num_historical_steps)]
    target_mask = torch.stack(target_mask, dim=1)
    for i in range(num_historical_steps):
        target_traj[:,i,:,:2] = target_traj[:,i,:,:2] - target_traj[:,i,0,:2].unsqueeze(1)
        heading = targets_pos[:,0,i+1:i+1+num_future_steps,2] - targets_pos[:,0,i,2].unsqueeze(1)
        target_traj[:,i,:,2:4] = torch.stack([heading.cos(), heading.sin()], dim=-1)
    return target_traj, target_mask

def greedy_linear_sum_assignment(cost_matrix):
    """
    一个简化的、基于贪心算法的匹配方法。
    注意：这不保证找到最优解。
    
    参数:
    - cost_matrix: 一个形状为 (n, m) 的成本矩阵，其中 n 是行数，m 是列数。
    
    返回:
    - rows: 匹配行的列表。
    - cols: 匹配列的列表。
    """
    n, m = cost_matrix.shape
    matched_rows = set()
    matched_cols = set()
    rows = []
    cols = []
    
    while len(matched_rows) < n and len(matched_cols) < m:
        # 找到当前成本矩阵中的最小值，首先将成本矩阵重塑为一维张量
        min_val, min_idx = torch.min(cost_matrix.view(-1), dim=0)
        # 根据一维张量中的索引计算原始成本矩阵的行和列索引
        i, j = divmod(min_idx.item(), m)
        
        # 如果行和列都还没有被匹配，则选择它们作为一个匹配对
        if i not in matched_rows and j not in matched_cols:
            matched_rows.add(i)
            matched_cols.add(j)
            rows.append(i)
            cols.append(j)
        
        # 将已经选择的行和列的成本设置为一个很大的值，以确保它们不会再次被选中
        cost_matrix[i, :] = torch.inf
        cost_matrix[:, j] = torch.inf
    
    return rows, cols

def calculate_cost_matrix(tracks_current, tracks_next):
    # 使用torch.cdist计算成本矩阵
    cost_matrix = torch.cdist(tracks_current[:, :, :, -1, :2], tracks_next[:, :, :, 0, :2])
    return cost_matrix

def huber_loss(delta, c=1.0):
    # 使用torch.where计算Huber损失
    return torch.where(torch.abs(delta) < c, 0.5 * delta**2, c * (torch.abs(delta) - 0.5 * c))

def bidirectional_matching(tracks_current, tracks_next, compliant_loss_decay_weights, compliant_length):
    match_deltas = []  # 存储所有匹配对的delta值
    
    batch_size, reference_line_num, mode_num, traj_length, traj_dim = tracks_current.shape
    
    # 初始化总一致性损失
    total_consistency_loss = torch.tensor(0.0, device=tracks_current.device)
    cost_matrix_forward = calculate_cost_matrix(tracks_current, tracks_next).cpu().detach().numpy()
    cost_matrix_backward = calculate_cost_matrix(tracks_next, tracks_current).cpu().detach().numpy()
    compliant_loss_decay_weights = compliant_loss_decay_weights.to(tracks_current.device)
    for b in range(batch_size):
        for ref_line in range(reference_line_num):
            # tracks_current_ref = tracks_current[b, ref_line, :]
            # tracks_next_ref = tracks_next[b, ref_line, :]
            
            # 将tensor转换为numpy进行匈牙利算法计算，这部分当前不易并行化
            forward_matches = linear_sum_assignment(cost_matrix_forward[b, ref_line])
            backward_matches = linear_sum_assignment(cost_matrix_backward[b, ref_line])
            
            forward_matches = list(zip(forward_matches[0], forward_matches[1]))
            backward_matches = list(zip(backward_matches[1], backward_matches[0]))
            
            bidirectional_matches = [match for match in forward_matches if match in backward_matches]
            # 计算每个匹配对的成本并进行排序
            cost_matrix_forward_now = cost_matrix_forward[b, ref_line]
            match_costs = [(match, cost_matrix_forward_now[match[0], match[1]]) for match in bidirectional_matches]
            # print("match_costs are", match_costs)
            sorted_matches = sorted(match_costs, key=lambda x: x[1])

            # 选取成本最低的前3个匹配对
            top_matches = sorted_matches[:3]            
            match_deltas = match_deltas + top_matches

            # for i, j in bidirectional_matches:
            for match, _ in top_matches:
                i, j = match
                # delta = tracks_next[b, ref_line, j, 10: compliant_length + 10, :] - tracks_current[b, ref_line, i, :compliant_length, :]
                delta_x = tracks_next[b, ref_line, j, 10: compliant_length + 10, 0] - tracks_current[b, ref_line, i, :compliant_length, 0]
                delta_y = tracks_next[b, ref_line, j, 10: compliant_length + 10, 1] - tracks_current[b, ref_line, i, :compliant_length, 1]
                total_consistency_loss += (compliant_loss_decay_weights * (huber_loss(delta_x) + huber_loss(delta_y))).sum()
                # total_consistency_loss += (compliant_loss_decay_weights * huber_loss(delta)).sum()
                # total_consistency_loss += (huber_loss(delta)).sum()

    if len(match_deltas) > 0:
        total_consistency_loss /= len(match_deltas)

    return len(match_deltas), total_consistency_loss
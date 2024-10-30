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
from .loss.esdf_collision_loss import ESDFCollisionLoss
from src.utils.utils import load_checkpoint
from .plantf_model import PlantfModel

logger = logging.getLogger(__name__)


class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        use_collision_loss=False,
        use_contrast_loss=False,
        regulate_yaw=False,
        objective_aggregate_mode: str = "mean",
        base_model_path: str = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
    
        # 实现一个load base model的功能, base model是pluto model的path
        weights = load_checkpoint(base_model_path)
        # 实例化一个pluto model，用dim=128，state_channel=6，polygon_channel=6，history_channel=9，history_steps=21，future_steps=80，encoder_depth=4，decoder_depth=4，drop_path=0.2，dropout=0.1，num_heads=8，num_modes=6，use_ego_history=False，state_attn_encoder=True，state_dropout=0.75，use_hidden_proj=False，cat_x=False，ref_free_traj=False
        base_model = PlantfModel(dim=self.model.dim, state_channel=self.model.state_channel, \
                                    polygon_channel=self.model.polygon_channel, history_channel=self.model.history_channel, \
                                        history_steps=self.model.history_steps, future_steps=self.model.future_steps, encoder_depth=self.model.encoder_depth, \
                                        drop_path=self.model.drop_path, num_heads=self.model.num_heads, \
                                        num_modes=self.model.num_modes, use_ego_history=self.model.use_ego_history, state_attn_encoder=self.model.state_attn_encoder, \
                                        state_dropout=self.model.state_dropout)
        base_model.load_state_dict(weights)
        # Freeze the parameters of the base model layers
        # for param in self.base_model.parameters():
        #     param.requires_grad = False
        # Load the pretrained base_model weight to model according to the name
        for name, param in self.model.named_parameters():
            if name in base_model.state_dict():
                param.data.copy_(base_model.state_dict()[name])
                # print("load weight from base model: ", name)
                # and freeze the parameters
                param.requires_grad = False

        self.model.temporal_aggregate_layer.load_state_dict(base_model.trajectory_decoder.state_dict())
        # make the parameters of the temporal_aggregate_layer trainable
        for param in self.model.temporal_aggregate_layer.parameters():
            param.requires_grad = True

        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.history_steps = model.history_steps
        self.use_collision_loss = use_collision_loss
        self.use_contrast_loss = use_contrast_loss
        self.regulate_yaw = regulate_yaw

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
        features, _, _ = batch
        data = features["feature"].data
        list_data = self.generate_his_features(data)
        temporal_m = []
        for time_shift in range(len(self.model.select_indices)):
            # if time_shift is the last time step, then output the scene_enc,\
            #  pos, key_padding_mask, agent_pos, agent_heading, bs, A, prediction, ref_free_traj, and temporal_m
            if time_shift == len(self.model.select_indices) - 1:
                with torch.no_grad():
                    scene_enc, pos, key_padding_mask, agent_pos, agent_heading, bs, A, \
                        prediction, pred_embedding  = self.model.forward_base(list_data[time_shift], last_one=True)
            else:
                with torch.no_grad():
                    pred_embedding = self.model.forward_base(list_data[time_shift])
            temporal_m.append(pred_embedding)

        res = self.model.forward_training(features["feature"].data, scene_enc, pos, key_padding_mask,\
                                           agent_pos, agent_heading, bs, A, temporal_m, prediction)

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
            res["trajectory"][:train_num],
            res["probability"][:train_num],
            res["prediction"][:train_num],
        )
        ref_free_trajectory = res.get("ref_free_trajectory", None)

        targets_pos = data["agent"]["target"][:train_num, :, -T:]
        valid_mask = data["agent"]["valid_mask"][:train_num, :, -T:]
        targets_vel = data["agent"]["velocity"][:train_num, :, -T:]

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

        # planning loss
        ego_reg_loss, ego_cls_loss, collision_loss, best_trajectory = self.get_planning_loss(
            data, trajectory, probability, valid_mask[:, 0], target[:, 0], train_num
        )

        prediction_loss = ego_reg_loss.new_zeros(1)

        if self.training and self.use_contrast_loss:
            contrastive_loss = self._compute_contrastive_loss(
                res["hidden"], data["data_n_valid_mask"]
            )
        else:
            contrastive_loss = ego_reg_loss.new_zeros(1)

        loss = (
            ego_reg_loss
            + ego_cls_loss
            + prediction_loss
            + contrastive_loss
            + collision_loss
        )

        return {
            "loss": loss,
            "reg_loss": ego_reg_loss.item(),
            "cls_loss": ego_cls_loss.item(),
            "collision_loss": collision_loss.item(),
            "prediction_loss": prediction_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
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
        bs, M, T, D = trajectory.shape
        ade = torch.norm(trajectory[..., :2] - target[:, None, :, :2], dim=-1)
        best_mode = torch.argmin(ade.sum(-1), dim=-1)
        best_trajectory = trajectory[torch.arange(trajectory.shape[0]), best_mode]
        ego_reg_loss = F.smooth_l1_loss(best_trajectory, target[..., :D], reduction="none").sum(-1)
        ego_reg_loss = (ego_reg_loss * valid_mask).sum() / valid_mask.sum()
        ego_cls_loss = F.cross_entropy(probability, best_mode.detach())

        if self.use_collision_loss:
            collision_loss = self.collision_loss(
                best_trajectory, data["cost_maps"][:bs, :, :, 0].float()
            )
        else:
            collision_loss = trajectory.new_zeros(1)

        if self.regulate_yaw:
            heading_vec_norm = torch.norm(best_trajectory[..., 2:4], dim=-1)
            yaw_regularization_loss = F.l1_loss(
                heading_vec_norm, heading_vec_norm.new_ones(heading_vec_norm.shape)
            )
            reg_loss += yaw_regularization_loss

        return ego_reg_loss, ego_cls_loss, collision_loss, best_trajectory

    def get_his_prediction_loss(self, data, trajectory, probability, valid_mask, target, bs):
        """
        trajectory: (bs, Th, R, M, Tf, 4)
        valid_mask: (bs, T)
        """

        bs, Th, M, Tf, _ = trajectory.shape
        
        if Th==0 :
            return 0, 0

        reg_loss_all = 0
        cls_loss_all = 0
        best_trajectory_list = []

        for i in range(Th):

            valid_mask_once = valid_mask[:, i]
            trajectory_once = trajectory[:, i]
            probability_once = probability[:, i]
            target_once = target[:, i]

            ade = torch.norm(trajectory_once[..., :2] - target_once[:, None, :, :2], dim=-1)
            best_mode = torch.argmin(ade.sum(-1), dim=-1)
            best_trajectory = trajectory_once[torch.arange(trajectory_once.shape[0]), best_mode]
            cls_loss = F.cross_entropy(probability_once, best_mode.detach())

            reg_loss = F.smooth_l1_loss(best_trajectory, target_once, reduction="none").sum(-1)
            reg_loss = (reg_loss * valid_mask_once).sum() / valid_mask_once.sum()

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

    def _compute_metrics(self, res, data, prefix) -> Dict[str, torch.Tensor]:
        """
        Computes a set of planning metrics given the model's predictions and targets.

        :param predictions: model's predictions
        :param targets: ground truth targets
        :return: dictionary of metrics names and values
        """
        # get top 6 modes
        trajectory, probability = res["trajectory"], res["probability"]
        # r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
        # probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

        # bs, R, M, T, _ = trajectory.shape
        # trajectory = trajectory.reshape(bs, R * M, T, -1)
        # probability = probability.reshape(bs, R * M)
        # top_k_prob, top_k_index = probability.topk(6, dim=-1)
        # top_k_traj = trajectory[torch.arange(bs)[:, None], top_k_index]

        outputs = {
            "trajectory": trajectory[..., :2],
            "probability": probability,
            "prediction": res["prediction"][..., :2],
            "prediction_target": data["agent"]["target"][:, 1:],
            "valid_mask": data["agent"]["valid_mask"][:, 1:, self.history_steps :],
        }
        target = data["agent"]["target"][:, 0]

        metrics = self.metrics[prefix](outputs, target)
        return metrics

    def _log_step(
        self,
        loss: torch.Tensor,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ) -> None:
        self.log(
            f"loss/{prefix}_{loss_name}",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
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

    def generate_his_features(self, data):
        used_T = len(self.model.select_indices)
    
        list_data = []
        # reverse data["current_states"], do not use ::-1
        data["current_states"] = data["current_states"].flip(1)
        for i, index in enumerate(self.model.select_indices):
            cur_state = data["agent"]["position"][:, 0, index]
            center_angle = data["agent"]["heading"][:, 0, index]
            center_xy, center_angle = cur_state[:, :2].clone().unsqueeze(1).unsqueeze(1), center_angle.clone().unsqueeze(1).unsqueeze(1)

            rotate_mat = torch.stack([
                torch.cos(center_angle), -torch.sin(center_angle),
                torch.sin(center_angle), torch.cos(center_angle)
            ], dim=1).reshape(-1, 2, 2).unsqueeze(1)

            data_dict = {
                "agent": {
                    "position": torch.matmul(data["agent"]["position"] - center_xy, rotate_mat),
                    "velocity": torch.matmul(data["agent"]["velocity"], rotate_mat),
                    "heading": data["agent"]["heading"] - center_angle,
                    "shape": data["agent"]["shape"],
                    "category": data["agent"]["category"],
                    "valid_mask": data["agent"]["valid_mask"],
                },
                "map": {
                    "point_position": torch.matmul(data["map"]["point_position"] - center_xy.unsqueeze(1), rotate_mat.unsqueeze(1)),
                    "point_vector": torch.matmul(data["map"]["point_vector"], rotate_mat.unsqueeze(1)),
                    "point_orientation": data["map"]["point_orientation"] - center_angle.unsqueeze(1),
                    "polygon_position": torch.matmul(data["map"]["polygon_position"] - center_xy.squeeze(1), rotate_mat.squeeze(1)),
                    "polygon_orientation": data["map"]["polygon_orientation"] - center_angle.squeeze(1),
                    "polygon_center": torch.cat((torch.matmul(data["map"]["polygon_center"][..., :2] - \
                                        center_xy.squeeze(1), rotate_mat.squeeze(1)), data["map"]["polygon_center"][..., 2].unsqueeze(-1) - \
                                            center_angle), dim=-1),
                    "polygon_type": data["map"]["polygon_type"],
                    "polygon_on_route": data["map"]["polygon_on_route"],
                    "polygon_tl_status": data["map"]["polygon_tl_status"],
                    "polygon_has_speed_limit": data["map"]["polygon_has_speed_limit"],
                    "polygon_speed_limit": data["map"]["polygon_speed_limit"],
                    "valid_mask": data["map"]["valid_mask"],
                },
                "static_objects": {
                    "position": torch.matmul(data["static_objects"]["position"] - center_xy.squeeze(1), rotate_mat.squeeze(1)),
                    "heading": data["static_objects"]["heading"] - center_angle.squeeze(1),
                    "category": data["static_objects"]["category"],
                    "valid_mask": data["static_objects"]["valid_mask"],
                    "shape": data["static_objects"]["shape"],
                },
                "reference_line": {
                    "position": torch.matmul(data["reference_line"]["position"] - center_xy, rotate_mat),
                    "vector": torch.matmul(data["reference_line"]["vector"], rotate_mat),
                    "orientation": data["reference_line"]["orientation"] - center_angle,
                    "valid_mask": data["reference_line"]["valid_mask"],
                },
                "current_state": data["current_state"],
            }
            # current_state includes x,y, heading, rear_axle_velocity_2d.x, rear_axle_acceleration_2d.x, steering_angle, yaw_rate
            data_dict["current_state"] = data["current_states"][:, index-1]
            # data_dict["current_state"][:, :3] = 0

            list_data.append(data_dict)

        # 还原原始数据
        # data.update(original_data)

        return list_data

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



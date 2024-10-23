import torch
import torch.nn as nn
import numpy as np
import pickle
from copy import deepcopy
import math

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import EgoTrajectoryTargetBuilder

from src.feature_builders.pluto_feature_builder import PlutoFeatureBuilder

from .layers.fourier_embedding import FourierEmbedding
from .layers.transformer import TransformerEncoderLayer
from .modules.agent_encoder import AgentEncoder
from .modules.agent_predictor import AgentPredictor
from .modules.map_encoder import MapEncoder
from .modules.static_objects_encoder import StaticObjectsEncoder
from .modules.planning_decoder import PlanningDecoder
from .modules.tesla_aggregate import TemporalAggregate
from .layers.mlp_layer import MLPLayer
from .pluto_model import PlutoModel
from src.utils.utils import load_checkpoint

trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)

class PlanningModel(TorchModuleWrapper):
    def __init__(self, dim=128, state_channel=6, polygon_channel=6, history_channel=9, history_steps=21, future_steps=80,
                 encoder_depth=4, decoder_depth=4, drop_path=0.2, dropout=0.1, num_heads=8, num_modes=6, use_ego_history=False,
                 state_attn_encoder=True, state_dropout=0.75, use_hidden_proj=False, cat_x=False, ref_free_traj=False,
                 use_history_attention=False, use_history_attention_intervals=1, cluster_results_file_path=None,
                 feature_builder: PlutoFeatureBuilder = PlutoFeatureBuilder()) -> None:
        super().__init__(feature_builders=[feature_builder],
                         target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
                         future_trajectory_sampling=trajectory_sampling)

        self.dim = dim
        self.state_channel = state_channel
        self.polygon_channel = polygon_channel
        self.history_channel = history_channel
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.drop_path = drop_path
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_modes = num_modes
        self.use_ego_history = use_ego_history
        self.state_attn_encoder = state_attn_encoder
        self.state_dropout = state_dropout
        self.use_hidden_proj = use_hidden_proj
        self.cat_x = cat_x
        self.ref_free_traj = ref_free_traj
        self.radius = feature_builder.radius

        self.use_history_attention = use_history_attention
        self.use_history_attention_intervals = use_history_attention_intervals
        self.cluster_results_file_path = cluster_results_file_path

        # Temporal Learning-based Aggregation Layer
        self.temporal_aggregate_layer = TemporalAggregate(
            num_mode=num_modes,
            decoder_depth=decoder_depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4,
            dropout=dropout,
            cat_x=cat_x,
            future_steps=future_steps,
        )

        # # Load the pretrained weight 
        # self.temporal_aggregate_layer.load_state_dict(self.base_model.planning_decoder.state_dict())

        # # Freeze the parameters of the base model layers
        # for param in self.base_model.parameters():
        #     param.requires_grad = False

        # Make sure the parameters of the temporal aggregation layers are trainable
        # for param in self.temporal_aggregate_layer.parameters():
        #     param.requires_grad = True

        self.select_indices = self._init_select_indices()
        self.compliant_length = torch.tensor([70], dtype=torch.long)
        self.compliant_loss_decay_weights = torch.nn.Parameter(0.9 ** torch.arange(int(self.compliant_length)), requires_grad=False).unsqueeze(-1)

        # self.agent_predictor = AgentPredictor(dim, future_steps)
        # self.agent_predictor.load_state_dict(deepcopy(self.base_model.agent_predictor.state_dict()))
        # for param in self.agent_predictor.parameters():
        #     param.requires_grad = True
            
        # if use_hidden_proj:
        #     self.hidden_proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        #     self.hidden_proj.load_state_dict(deepcopy(self.base_model.hidden_proj.state_dict()))
        #     for param in self.hidden_proj.parameters():
        #         param.requires_grad = True
                
        # if self.ref_free_traj:
        #     self.ref_free_decoder = MLPLayer(dim, 2 * dim, future_steps * 4)
        #     self.ref_free_decoder.load_state_dict(deepcopy(self.base_model.ref_free_decoder.state_dict()))
        #     for param in self.ref_free_decoder.parameters():
        #         param.requires_grad = True

        # self.apply(self._init_weights)

    def _init_select_indices(self):
        if self.use_history_attention:
            indices = list(range(self.history_steps - 1, 0, -self.use_history_attention_intervals))
            indices.reverse()
            return torch.tensor(indices, dtype=torch.long)
        else:
            return torch.tensor([19], dtype=torch.long)

    def _load_cluster_results(self, file_path):
        with open(file_path, "rb") as file:
            cluster_results = pickle.load(file)
        return cluster_results['cluster_centers']

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def generate_his_features(self, data):
        used_T = len(self.select_indices)
        repeat_dims = (1, used_T, 1, 1, 1)
        
        # 备份原始数据
        original_data = {
            "agent": {
                "position": data["agent"]["position"].clone(),
                "velocity": data["agent"]["velocity"].clone(),
                "heading": data["agent"]["heading"].clone(),
                "shape": data["agent"]["shape"].clone(),
                "category": data["agent"]["category"].clone(),
                "valid_mask": data["agent"]["valid_mask"].clone(),
            },
            "map": {
                "point_position": data["map"]["point_position"].clone(),
                "point_vector": data["map"]["point_vector"].clone(),
                "point_orientation": data["map"]["point_orientation"].clone(),
                "polygon_center": data["map"]["polygon_center"].clone(),
                "polygon_position": data["map"]["polygon_position"].clone(),
                "polygon_orientation": data["map"]["polygon_orientation"].clone(),
                "valid_mask": data["map"]["valid_mask"].clone(),
            },
            "static_objects": {
                "position": data["static_objects"]["position"].clone(),
                "heading": data["static_objects"]["heading"].clone(),
                "category": data["static_objects"]["category"].clone(),
                "valid_mask": data["static_objects"]["valid_mask"].clone(),
                "shape": data["static_objects"]["shape"].clone(),
            },
            "reference_line": {
                "position": data["reference_line"]["position"].clone(),
                "vector": data["reference_line"]["vector"].clone(),
                "orientation": data["reference_line"]["orientation"].clone(),
                "valid_mask": data["reference_line"]["valid_mask"].clone(),
            },
            "current_state": data["current_state"].clone(),
        }

        # 直接在原数据上进行操作
        data["agent"]["position"] = data["agent"]["position"].unsqueeze(1).repeat(*repeat_dims)
        data["agent"]["velocity"] = data["agent"]["velocity"].unsqueeze(1).repeat(*repeat_dims)
        data["agent"]["heading"] = data["agent"]["heading"].unsqueeze(1).repeat(1, used_T, 1, 1)
        data["agent"]["shape"] = data["agent"]["shape"].unsqueeze(1).repeat(*repeat_dims)
        data["agent"]["category"] = data["agent"]["category"].unsqueeze(1).repeat(1, used_T, 1)
        data["agent"]["valid_mask"] = data["agent"]["valid_mask"].unsqueeze(1).repeat(1, used_T, 1, 1)

        data["map"]["point_position"] = data["map"]["point_position"].unsqueeze(1).repeat(1, used_T, 1, 1, 1, 1)
        data["map"]["point_vector"] = data["map"]["point_vector"].unsqueeze(1).repeat(1, used_T, 1, 1, 1, 1)
        data["map"]["point_orientation"] = data["map"]["point_orientation"].unsqueeze(1).repeat(1, used_T, 1, 1, 1)
        data["map"]["polygon_center"] = data["map"]["polygon_center"].unsqueeze(1).repeat(1, used_T, 1, 1)
        data["map"]["polygon_position"] = data["map"]["polygon_position"].unsqueeze(1).repeat(1, used_T, 1, 1)
        data["map"]["polygon_orientation"] = data["map"]["polygon_orientation"].unsqueeze(1).repeat(1, used_T, 1)
        data["map"]["valid_mask"] = data["map"]["valid_mask"].unsqueeze(1).repeat(1, used_T, 1, 1)

        data["static_objects"]["position"] = data["static_objects"]["position"].unsqueeze(1).repeat(1, used_T, 1, 1)
        data["static_objects"]["heading"] = data["static_objects"]["heading"].unsqueeze(1).repeat(1, used_T, 1)
        data["static_objects"]["category"] = data["static_objects"]["category"].unsqueeze(1).repeat(1, used_T, 1)
        data["static_objects"]["valid_mask"] = data["static_objects"]["valid_mask"].unsqueeze(1).repeat(1, used_T, 1)
        data["static_objects"]["shape"] = data["static_objects"]["shape"].unsqueeze(1).repeat(1, used_T, 1, 1)

        data["reference_line"]["position"] = data["reference_line"]["position"].unsqueeze(1).repeat(1, used_T, 1, 1, 1)
        data["reference_line"]["vector"] = data["reference_line"]["vector"].unsqueeze(1).repeat(1, used_T, 1, 1, 1)
        data["reference_line"]["orientation"] = data["reference_line"]["orientation"].unsqueeze(1).repeat(1, used_T, 1, 1)
        data["reference_line"]["valid_mask"] = data["reference_line"]["valid_mask"].unsqueeze(1).repeat(1, used_T, 1, 1)

        data["current_state"] = data["current_state"].unsqueeze(1).repeat(1, used_T, 1)

        for i, index in enumerate(self.select_indices):
            cur_state = data["agent"]["position"][:, i, 0, index]
            center_angle = data["agent"]["heading"][:, i, 0, index]
            center_xy, center_angle = cur_state[:, :2].clone().unsqueeze(1).unsqueeze(1), center_angle.clone().unsqueeze(1).unsqueeze(1)

            rotate_mat = torch.stack([
                torch.cos(center_angle), -torch.sin(center_angle),
                torch.sin(center_angle), torch.cos(center_angle)
            ], dim=1).reshape(-1, 2, 2).unsqueeze(1)

            data["current_state"][:, i, :3] = 0
            data["agent"]["position"][:, i] = torch.matmul(data["agent"]["position"][:, i] - center_xy, rotate_mat)
            data["agent"]["velocity"][:, i] = torch.matmul(data["agent"]["velocity"][:, i], rotate_mat)
            data["agent"]["heading"][:, i] -= center_angle

            data["map"]["point_position"][:, i] = torch.matmul(data["map"]["point_position"][:, i] - center_xy.unsqueeze(1), rotate_mat.unsqueeze(1))
            data["map"]["point_vector"][:, i] = torch.matmul(data["map"]["point_vector"][:, i], rotate_mat.unsqueeze(1))
            data["map"]["point_orientation"][:, i] -= center_angle.unsqueeze(1)

            data["map"]["polygon_center"][:, i, ..., :2] = torch.matmul(data["map"]["polygon_center"][:, i, ..., :2] - center_xy.squeeze(1), rotate_mat.squeeze(1))
            data["map"]["polygon_center"][:, i, ..., 2] -= center_angle.squeeze(1)
            data["map"]["polygon_position"][:, i] = torch.matmul(data["map"]["polygon_position"][:, i] - center_xy.squeeze(1), rotate_mat.squeeze(1))
            data["map"]["polygon_orientation"][:, i] -= center_angle.squeeze(1)

            data["static_objects"]["position"][:, i] = torch.matmul(data["static_objects"]["position"][:, i] - center_xy.squeeze(1), rotate_mat.squeeze(1))
            data["static_objects"]["heading"][:, i] -= center_angle.squeeze(1)

        # output a list_data store the data for each time step
        list_data = []
        for i in range(used_T):
            data_dict = {
                "agent": {
                    "position": data["agent"]["position"][:, i],
                    "velocity": data["agent"]["velocity"][:, i],
                    "heading": data["agent"]["heading"][:, i],
                    "shape": data["agent"]["shape"][:, i],
                    "category": data["agent"]["category"][:, i],
                    "valid_mask": data["agent"]["valid_mask"][:, i],
                },
                "map": {
                    "point_position": data["map"]["point_position"][:, i],
                    "point_vector": data["map"]["point_vector"][:, i],
                    "point_orientation": data["map"]["point_orientation"][:, i],
                    "polygon_center": data["map"]["polygon_center"][:, i],
                    "polygon_position": data["map"]["polygon_position"][:, i],
                    "polygon_orientation": data["map"]["polygon_orientation"][:, i],
                    "polygon_type": data["map"]["polygon_type"],
                    "polygon_on_route": data["map"]["polygon_on_route"],
                    "polygon_tl_status": data["map"]["polygon_tl_status"],
                    "polygon_has_speed_limit": data["map"]["polygon_has_speed_limit"],
                    "polygon_speed_limit": data["map"]["polygon_speed_limit"],
                    "valid_mask": data["map"]["valid_mask"][:, i],
                },
                "static_objects": {
                    "position": data["static_objects"]["position"][:, i],
                    "heading": data["static_objects"]["heading"][:, i],
                    "category": data["static_objects"]["category"][:, i],
                    "valid_mask": data["static_objects"]["valid_mask"][:, i],
                    "shape": data["static_objects"]["shape"][:, i],
                },
                "reference_line": {
                    "position": data["reference_line"]["position"][:, i],
                    "vector": data["reference_line"]["vector"][:, i],
                    "orientation": data["reference_line"]["orientation"][:, i],
                    "valid_mask": data["reference_line"]["valid_mask"][:, i],
                },
                "current_state": data["current_state"][:, i],
            }
            list_data.append(data_dict)

        # 还原原始数据
        recursive_update(data, original_data)

        return list_data
    
    def forward(self, data):
        list_data = self.generate_his_features(data)
        self.select_indices = self.select_indices.to(data["agent"]["position"].device)
        x_agent = self.agent_encoder(data, self.select_indices)
        x_polygon = self.map_encoder(data)
        x_static, static_pos, static_key_padding = self.static_objects_encoder(data)

        bs, used_T, A = x_agent.shape[:3]
        agent_pos = data["agent"]["position"][:, :, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, :, :self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=2)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=2)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1)

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        x = torch.cat([x_agent, x_polygon, x_static], dim=2)
        pos = torch.cat([pos, static_pos], dim=2)
        pos_embed = self.pos_emb(pos)

        key_padding_mask = torch.cat([key_padding_mask, static_key_padding], dim=-1)
        x = x + pos_embed
        x = x.reshape(bs * used_T, -1, self.dim)
        key_padding_mask = key_padding_mask.reshape(bs * used_T, -1)
        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask, return_attn_weights=False)
        x = self.norm(x)

        x = x.reshape(bs, used_T, -1, self.dim)
        prediction = self.agent_predictor(x[:, -1, 1:A])

        ref_line_available = data["reference_line"]["position"].shape[2] > 0
        if ref_line_available:
            trajectory, probability = self.planning_decoder(data, {"enc_emb": x, "enc_key_padding_mask": key_padding_mask})
            if len(trajectory.shape) == 6:
                bs, TH, R, M, TF, _ = trajectory.shape
                probability_now = probability[:, -1]
                trajectory_now = trajectory[:, -1]
            else:
                bs, R, M, TF, _ = trajectory.shape
                TH = 1
                probability_now = probability
                trajectory_now = trajectory
            out = {
                "trajectory": trajectory_now.reshape(bs, R, M, TF, -1),
                "probability": probability_now.reshape(bs, R, M),
                "prediction": prediction,
                "trajectory_all_time": trajectory.reshape(bs, TH, R, M, TF, -1),
                "probability_all_time": probability.reshape(bs, TH, R, M),
                "select_indices": self.select_indices,
            }
        else:
            out = {
                "trajectory": None,
                "probability": None,
                "prediction": prediction,
            }

        if self.use_hidden_proj:
            out["hidden"] = self.hidden_proj(x[:, -1, 0])

        if self.ref_free_traj:
            ref_free_traj = self.ref_free_decoder(x[:, -1, 0]).reshape(bs, self.future_steps, 4)
            out["ref_free_trajectory"] = ref_free_traj

        if not self.training:
            if self.ref_free_traj:
                ref_free_traj_angle = torch.arctan2(ref_free_traj[..., 3], ref_free_traj[..., 2])
                ref_free_traj = torch.cat([ref_free_traj[..., :2], ref_free_traj_angle.unsqueeze(-1)], dim=-1)
                out["output_ref_free_trajectory"] = ref_free_traj

            output_prediction = torch.cat([
                prediction[..., :2] + agent_pos[:, -1, 1:A, None],
                torch.atan2(prediction[..., 3], prediction[..., 2]).unsqueeze(-1) + agent_heading[:, -1, 1:A, None, None],
                prediction[..., 4:6],
            ], dim=-1)
            out["output_prediction"] = output_prediction

            if out["trajectory"] is not None:
                r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
                out["probability"].masked_fill_(r_padding_mask[:, -1].unsqueeze(-1), -1e6)

                angle = torch.atan2(out["trajectory"][..., 3], out["trajectory"][..., 2])
                out_trajectory = torch.cat([out["trajectory"][..., :2], angle.unsqueeze(-1)], dim=-1)

                bs, R, M, T, _ = out_trajectory.shape
                flattened_probability = out["probability"].reshape(bs, R * M)
                best_trajectory = out_trajectory.reshape(bs, R * M, T, -1)[torch.arange(bs), flattened_probability.argmax(-1)]

                out["output_trajectory"] = best_trajectory
                out["candidate_trajectories"] = out_trajectory
            else:
                out["output_trajectory"] = out["output_ref_free_trajectory"]
                out["candidate_trajectories"] = torch.zeros(1, 0, 0, self.future_steps, 3)

    def forward_training(self, data, scene_enc, pos, key_padding_mask, \
                         agent_pos, agent_heading, bs, A, temporal_m, prediction, ref_free_traj):
        
        # list_data = self.generate_his_features(data)
        # temporal_m = []
        # for time_shift in range(len(self.select_indices)):
        #     scene_enc, pos, key_padding_mask, agent_pos, agent_heading, bs, A = self.base_model.encode(list_data[time_shift])
        #     _, pred_embedding, _ = self.base_model.decode_for_agg(scene_enc, pos, key_padding_mask, agent_pos, agent_heading, bs, A, list_data[time_shift])
        #     temporal_m.append(pred_embedding)

        temporal_m = torch.stack(temporal_m, dim=1)
        # sum to agreegate the temporal information
        # temporal_m = temporal_m.sum(dim=1)
        # prediction = self.agent_predictor(scene_enc[:, 1:A]) 
        ref_line_available = data["reference_line"]["position"].shape[1] > 0
    
        if ref_line_available:
            trajectory, probability = self.temporal_aggregate_layer(
                data, {"enc_emb": scene_enc, "enc_key_padding_mask": key_padding_mask}, temporal_m=temporal_m
            )
        else:
            trajectory, probability = None, None

        out = {
            "trajectory": trajectory,
            "probability": probability,  # (bs, R, M)
            "prediction": prediction,  # (bs, A-1, T, 2)
        }

        # if self.use_hidden_proj:
        #     out["hidden"] = self.hidden_proj(scene_enc[:, 0])

        if self.ref_free_traj:
            # ref_free_traj = self.ref_free_decoder(scene_enc[:, 0]).reshape(
            #     bs, self.future_steps, 4
            # )
            out["ref_free_trajectory"] = ref_free_traj

        if not self.training:
            if self.ref_free_traj:
                ref_free_traj_angle = torch.arctan2(
                    ref_free_traj[..., 3], ref_free_traj[..., 2]
                )
                ref_free_traj = torch.cat(
                    [ref_free_traj[..., :2], ref_free_traj_angle.unsqueeze(-1)], dim=-1
                )
                out["output_ref_free_trajectory"] = ref_free_traj

            output_prediction = torch.cat(
                [
                    prediction[..., :2] + agent_pos[:, 1:A, None],
                    torch.atan2(prediction[..., 3], prediction[..., 2]).unsqueeze(-1)
                    + agent_heading[:, 1:A, None, None],
                    prediction[..., 4:6],
                ],
                dim=-1,
            )
            out["output_prediction"] = output_prediction

            if trajectory is not None:
                r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
                probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

                angle = torch.atan2(trajectory[..., 3], trajectory[..., 2])
                out_trajectory = torch.cat(
                    [trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
                )

                bs, R, M, T, _ = out_trajectory.shape
                flattened_probability = probability.reshape(bs, R * M)
                best_trajectory = out_trajectory.reshape(bs, R * M, T, -1)[
                    torch.arange(bs), flattened_probability.argmax(-1)
                ]

                out["output_trajectory"] = best_trajectory
                out["candidate_trajectories"] = out_trajectory
            else:
                out["output_trajectory"] = out["output_ref_free_trajectory"]
                out["probability"] = torch.zeros(1, 0, 0)
                out["candidate_trajectories"] = torch.zeros(
                    1, 0, 0, self.future_steps, 3
                )

        return out

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
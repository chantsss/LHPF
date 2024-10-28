# from copy import deepcopy
import math
import pickle
import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from src.feature_builders.pluto_feature_builder import PlutoFeatureBuilder

from .layers.fourier_embedding import FourierEmbedding
from .layers.transformer import TransformerEncoderLayer
from .modules.agent_encoder import AgentEncoder
from .modules.agent_predictor import AgentPredictor
from .modules.map_encoder import MapEncoder
from .modules.static_objects_encoder import StaticObjectsEncoder
from .modules.planning_decoder import PlanningDecoder
from .layers.mlp_layer import MLPLayer
from .modules.tesla_aggregate import TemporalAggregate
from .pluto_model import PlutoModel
from src.utils.utils import load_checkpoint

# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)


class PlanningModel(TorchModuleWrapper):
    def __init__(self, dim=128, state_channel=6, polygon_channel=6, history_channel=9, history_steps=21, future_steps=80,
                 encoder_depth=4, decoder_depth=4, drop_path=0.2, dropout=0.1, num_heads=8, num_modes=6, use_ego_history=False,
                 state_attn_encoder=True, state_dropout=0.75, use_hidden_proj=False, cat_x=False, ref_free_traj=False,
                 use_history_attention=False, use_history_attention_intervals=1, cluster_results_file_path=None,
                 feature_builder: PlutoFeatureBuilder = PlutoFeatureBuilder(), base_model_path=None) -> None:
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
        self.select_indices = self._init_select_indices()
        self.compliant_length = torch.tensor([70], dtype=torch.long)
        self.compliant_loss_decay_weights = torch.nn.Parameter(0.9 ** torch.arange(int(self.compliant_length)), requires_grad=False).unsqueeze(-1)

        self.pos_emb = FourierEmbedding(3, dim, 64)
        cluster_results = self._load_cluster_results(cluster_results_file_path)

        self.agent_encoder = AgentEncoder(state_channel=state_channel, history_channel=history_channel, dim=dim,
                                          hist_steps=history_steps, drop_path=drop_path, use_ego_history=use_ego_history,
                                          state_attn_encoder=state_attn_encoder, state_dropout=state_dropout)

        self.map_encoder = MapEncoder(dim=dim, polygon_channel=polygon_channel, use_lane_boundary=True)
        self.static_objects_encoder = StaticObjectsEncoder(dim=dim)

        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in torch.linspace(0, drop_path, encoder_depth)
        )
        self.norm = nn.LayerNorm(dim)

        self.agent_predictor = AgentPredictor(dim=dim, future_steps=future_steps)
        self.planning_decoder = PlanningDecoder(
            num_mode=num_modes,
            decoder_depth=decoder_depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4,
            dropout=dropout,
            cat_x=cat_x,
            future_steps=future_steps,
        )

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

        if use_hidden_proj:
            self.hidden_proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

        if self.ref_free_traj:
            self.ref_free_decoder = MLPLayer(dim, 2 * dim, future_steps * 4)

        self.apply(self._init_weights)

        if base_model_path is not None:
            weights = load_checkpoint(base_model_path)
            base_model = PlutoModel(dim=self.dim, state_channel=self.state_channel, \
                                        polygon_channel=self.polygon_channel, history_channel=self.history_channel, \
                                            history_steps=self.history_steps, future_steps=self.future_steps, encoder_depth=self.encoder_depth, \
                                            decoder_depth=self.decoder_depth, drop_path=self.drop_path, dropout=self.dropout, num_heads=self.num_heads, \
                                            num_modes=self.num_modes, use_ego_history=self.use_ego_history, state_attn_encoder=self.state_attn_encoder, \
                                            state_dropout=self.state_dropout, use_hidden_proj=self.use_hidden_proj, cat_x=self.cat_x, \
                                            ref_free_traj=self.ref_free_traj)
            base_model.load_state_dict(weights)
            for name, param in self.named_parameters():
                if name in base_model.state_dict():
                    # print("Loading weights for", name)
                    param.data.copy_(base_model.state_dict()[name])
                    # and freeze the parameters
                    param.requires_grad = False
            print("Base model loaded successfully")
            del base_model

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
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def load_weights(self, weights_path):
        """
        从权重文件加载模型权重。

        :param weights_path: 权重文件的路径
        """
        try:
            weights = torch.load(weights_path)
            if 'state_dict' not in weights:
                raise KeyError("权重文件不包含 'state_dict'")
            self.load_state_dict(weights['state_dict'])
            print(f"模型权重已从 {weights_path} 加载成功")
        except Exception as e:
            print(f"加载模型权重时出错: {e}")

    def forward(self, data):
        x, pos, key_padding_mask, agent_pos, agent_heading, bs, A = self.encode(data)
        out = self.decode(x, pos, key_padding_mask, agent_pos, agent_heading, bs, A, data)

        list_data = self.generate_his_features(data)
        temporal_m = []
        for time_shift in range(len(self.select_indices)):
            # if time_shift is the last time step, then output the scene_enc,\
            #  pos, key_padding_mask, agent_pos, agent_heading, bs, A, prediction, ref_free_traj, and temporal_m
            if time_shift == len(self.select_indices) - 1:
                with torch.no_grad():
                    scene_enc, pos, key_padding_mask, agent_pos, agent_heading, bs, A, \
                        prediction, ref_free_traj, pred_embedding  = self.forward_base(list_data[time_shift], last_one=True)
            else:
                with torch.no_grad():
                    pred_embedding = self.forward_base(list_data[time_shift])
            temporal_m.append(pred_embedding)

        out = self.forward_training(data, scene_enc, pos, key_padding_mask,\
                                           agent_pos, agent_heading, bs, A, temporal_m, prediction, ref_free_traj)

        return out

    def forward_base(self, data, last_one=False):
        # 编码阶段
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]

        bs, A = agent_pos.shape[0:2]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1)

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)
        x_static, static_pos, static_key_padding = self.static_objects_encoder(data)

        x = torch.cat([x_agent, x_polygon, x_static], dim=1)

        pos = torch.cat([pos, static_pos], dim=1)
        pos_embed = self.pos_emb(pos)

        key_padding_mask = torch.cat([key_padding_mask, static_key_padding], dim=-1)
        x = x + pos_embed

        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask, return_attn_weights=False)
        x = self.norm(x)

        # 解码阶段
        prediction = self.agent_predictor(x[:, 1:A])

        ref_line_available = data["reference_line"]["position"].shape[1] > 0

        if ref_line_available:
            _, _, pred_embedding = self.planning_decoder(
                data, {"enc_emb": x, "enc_key_padding_mask": key_padding_mask}
            )
        else:
            pred_embedding = None

        ref_free_traj = None
        if self.ref_free_traj:
            ref_free_traj = self.ref_free_decoder(x[:, 0]).reshape(
                bs, self.future_steps, 4
            )
        if not last_one:
                # return prediction, pred_embedding, ref_free_traj
            return pred_embedding
        else:
            # return scene_enc, pos, key_padding_mask, agent_pos, agent_heading, bs, A, prediction, ref_free_traj, pred_embedding
            return x, pos, key_padding_mask, agent_pos, agent_heading, bs, A, prediction, ref_free_traj, pred_embedding

    def encode(self, data):
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]

        bs, A = agent_pos.shape[0:2]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1)

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)
        x_static, static_pos, static_key_padding = self.static_objects_encoder(data)

        x = torch.cat([x_agent, x_polygon, x_static], dim=1)

        pos = torch.cat([pos, static_pos], dim=1)
        pos_embed = self.pos_emb(pos)

        key_padding_mask = torch.cat([key_padding_mask, static_key_padding], dim=-1)
        x = x + pos_embed

        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask, return_attn_weights=False)
        x = self.norm(x)

        return x, pos, key_padding_mask, agent_pos, agent_heading, bs, A

    def decode(self, x, pos, key_padding_mask, agent_pos, agent_heading, bs, A, data):
        prediction = self.agent_predictor(x[:, 1:A])

        ref_line_available = data["reference_line"]["position"].shape[1] > 0

        if ref_line_available:
            trajectory, probability, q = self.planning_decoder(
                data, {"enc_emb": x, "enc_key_padding_mask": key_padding_mask}
            )
        else:
            trajectory, probability = None, None

        out = {
            "trajectory": trajectory,
            "probability": probability,  # (bs, R, M)
            "prediction": prediction,  # (bs, A-1, T, 2)
        }

        if self.use_hidden_proj:
            out["hidden"] = self.hidden_proj(x[:, 0])

        if self.ref_free_traj:
            ref_free_traj = self.ref_free_decoder(x[:, 0]).reshape(
                bs, self.future_steps, 4
            )
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

    def forward_training(self, data, scene_enc, pos, key_padding_mask, \
                         agent_pos, agent_heading, bs, A, temporal_m, prediction, ref_free_traj, train_pre_ref_free_traj=False):
        
        # list_data = self.generate_his_features(data)
        # temporal_m = []
        # for time_shift in range(len(self.select_indices)):
        #     scene_enc, pos, key_padding_mask, agent_pos, agent_heading, bs, A = self.base_model.encode(list_data[time_shift])
        #     _, pred_embedding, _ = self.base_model.decode_for_agg(scene_enc, pos, key_padding_mask, agent_pos, agent_heading, bs, A, list_data[time_shift])
        #     temporal_m.append(pred_embedding)
        if len (temporal_m) == 0:
            temporal_m = None   
        else:
            temporal_m = torch.stack(temporal_m, dim=1)
        # sum to agreegate the temporal information
        # temporal_m = temporal_m.sum(dim=1)
        if train_pre_ref_free_traj:
            prediction = self.agent_predictor(scene_enc[:, 1:A]) 
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

        if self.use_hidden_proj:
            out["hidden"] = self.hidden_proj(scene_enc[:, 0])

        if self.ref_free_traj:
            if train_pre_ref_free_traj:
                ref_free_traj = self.ref_free_decoder(scene_enc[:, 0]).reshape(
                    bs, self.future_steps, 4
                )
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

    def generate_his_features(self, data):
        used_T = len(self.select_indices)
    
        list_data = []
        for i, index in enumerate(self.select_indices):
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
            data_dict["current_state"][:, :3] = 0
            # if not the last time step, tranform the steering_angle
            if i != len(self.select_indices) - 1:
                data_dict["current_state"][:, 5] -= center_angle.squeeze()

            list_data.append(data_dict)

        # 还原原始数据
        # data.update(original_data)

        return list_data
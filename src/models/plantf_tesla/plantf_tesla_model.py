import math
import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from src.feature_builders.pluto_feature_builder import PlutoFeatureBuilder

from .layers.common_layers import build_mlp
from .layers.transformer_encoder_layer import TransformerEncoderLayer
from .modules.agent_encoder import AgentEncoder
from .modules.map_encoder import MapEncoder
from .modules.trajectory_decoder import TrajectoryDecoder
from .modules.agent_predictor import AgentPredictor
from .layers.embedding import FourierEmbedding
from .modules.tesla_aggregate import TemporalAggregate
from .modules.static_objects_encoder import StaticObjectsEncoder

# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)


class PlanningModel(TorchModuleWrapper):
    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        drop_path=0.2,
        num_heads=8,
        num_modes=6,
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        use_history_attention=False,
        use_history_attention_intervals=1,
        use_his_offest=False,
        base_model_path=None,
        feature_builder: PlutoFeatureBuilder = PlutoFeatureBuilder(),
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.dim = dim
        self.state_channel = state_channel
        self.polygon_channel = polygon_channel
        self.history_channel = history_channel
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.encoder_depth = encoder_depth
        self.drop_path = drop_path
        self.num_heads = num_heads
        self.num_modes = num_modes
        self.use_ego_history = use_ego_history
        self.state_attn_encoder = state_attn_encoder
        self.state_dropout = state_dropout
        self.radius = feature_builder.radius
        self.use_history_attention = use_history_attention
        self.use_history_attention_intervals = use_history_attention_intervals
        self.select_indices = self._init_select_indices()
        # self.compliant_length = torch.tensor([70], dtype=torch.long)
        # self.compliant_loss_decay_weights = torch.nn.Parameter(0.9 ** torch.arange(int(self.compliant_length)), requires_grad=False).unsqueeze(-1)


        self.use_history_attention = use_history_attention
        self.use_history_attention_intervals = use_history_attention_intervals
        self.use_his_offest = use_his_offest
        if self.use_history_attention:
            self.select_indices = list(range(self.history_steps - 1, 0, -self.use_history_attention_intervals))
            self.select_indices.reverse() 
            self.select_indices = torch.tensor(self.select_indices, dtype=torch.long)
        else:
            self.select_indices = torch.tensor([19], dtype=torch.long)


        self.pos_emb = FourierEmbedding(3, dim, 64)
        self.agent_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
        )

        self.map_encoder = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
            use_lane_boundary=True,
        )

        self.static_objects_encoder = StaticObjectsEncoder(dim=dim)
        
        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.norm = nn.LayerNorm(dim)

        self.trajectory_decoder = TrajectoryDecoder(
            embed_dim=dim,
            num_modes=num_modes,
            future_steps=future_steps,
            out_channels=6,
        )
        self.agent_predictor = AgentPredictor(dim=dim, future_steps=future_steps)

        self.temporal_aggregate_layer = TemporalAggregate(
            embed_dim=dim,
            num_modes=num_modes,
            future_steps=future_steps,
            out_channels=6,
        )

        self.apply(self._init_weights)

        if base_model_path is not None:
            from src.utils.utils import load_checkpoint
            from .plantf_model import PlantfModel
            weights = load_checkpoint(base_model_path)
            base_model = PlantfModel(dim=self.dim, state_channel=self.state_channel, \
                                        polygon_channel=self.polygon_channel, history_channel=self.history_channel, \
                                            history_steps=self.history_steps, future_steps=self.future_steps, encoder_depth=self.encoder_depth, \
                                            drop_path=self.drop_path, num_heads=self.num_heads, \
                                            num_modes=self.num_modes, use_ego_history=self.use_ego_history, state_attn_encoder=self.state_attn_encoder, \
                                            state_dropout=self.state_dropout)
            
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
                        prediction, pred_embedding  = self.forward_base(list_data[time_shift], last_one=True)
            else:
                with torch.no_grad():
                    pred_embedding = self.forward_base(list_data[time_shift])
            temporal_m.append(pred_embedding)

        out = self.forward_training(data, scene_enc, pos, key_padding_mask,\
                                           agent_pos, agent_heading, bs, A, temporal_m, prediction)

        return out
    
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
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        return x, pos, key_padding_mask, agent_pos, agent_heading, bs, A

    def decode(self, x, pos, key_padding_mask, agent_pos, agent_heading, bs, A, data):
        trajectory, probability, _ = self.trajectory_decoder(x[:, 0])
        prediction = self.agent_predictor(x[:, 1:A])

        out = {
            "trajectory": trajectory,
            "probability": probability,
            "prediction": prediction,
        }

        if not self.training:
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

            angle = torch.atan2(trajectory[..., 3], trajectory[..., 2])
            out_trajectory = torch.cat(
                [trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
            )
            best_mode = probability.argmax(dim=-1)
            best_trajectory = trajectory[torch.arange(bs), best_mode]
            out["output_trajectory"] = best_trajectory
            out["candidate_trajectories"] = out_trajectory

        return out

    def generate_his_features(self, data):
        used_T = len(self.select_indices)
    
        list_data = []
        data["current_states"] = data["current_states"].flip(1)
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
            data_dict["current_state"] = data["current_states"][:, index-1]
            list_data.append(data_dict)

        # 还原原始数据
        # data.update(original_data)

        return list_data

    def forward_training(self, data, scene_enc, pos, key_padding_mask, \
                         agent_pos, agent_heading, bs, A, temporal_m, prediction):

        if None in temporal_m:
            # print('temporal_m has None')
            temporal_m = None   
        else:
            temporal_m = torch.stack(temporal_m, dim=1)

        trajectory, probability = self.temporal_aggregate_layer(scene_enc[:, 0], temporal_m)

        # prediction = self.agent_predictor(scene_enc[:, 1:A]).view(bs, -1, self.future_steps, 2)

        out = {
            "trajectory": trajectory,
            "probability": probability,
            "prediction": prediction,
        }

        if not self.training:
            best_mode = probability.argmax(dim=-1)
            output_trajectory = trajectory[torch.arange(bs), best_mode]
            angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
            out["output_trajectory"] = torch.cat(
                [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
            )

        return out

    def forward_base(self, data, last_one=False):
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
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        pred_embedding = x[:, 0]
        
        prediction = self.agent_predictor(x[:, 1:A])

        if not last_one:
                # return prediction, pred_embedding, ref_free_traj
            return pred_embedding
        else:
            # trajectory, probability = self.trajectory_decoder(pred_embedding)
            # return scene_enc, pos, key_padding_mask, agent_pos, agent_heading, bs, A, prediction, ref_free_traj, pred_embedding
            return x, pos, key_padding_mask, agent_pos, agent_heading, bs, A, prediction, pred_embedding


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
from .modules.static_objects_encoder import StaticObjectsEncoder
from .modules.trajectory_decoder import TrajectoryDecoder
from .modules.agent_predictor import AgentPredictor
from .layers.embedding import FourierEmbedding

# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)


class PlantfModel(TorchModuleWrapper):
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
        feature_builder: PlutoFeatureBuilder = PlutoFeatureBuilder(),
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps

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

        if not self.use_his_offest:
            self.trajectory_decoder = TrajectoryDecoder(
                embed_dim=dim,
                num_modes=num_modes,
                future_steps=future_steps,
                out_channels=6,
            )
        else:
            self.trajectory_decoder_curr = TrajectoryDecoder(
                embed_dim=dim,
                num_modes=num_modes,
                future_steps=future_steps,
                out_channels=6,
            )
            self.trajectory_decoder_his = TrajectoryDecoder(
                embed_dim=dim,
                num_modes=num_modes,
                future_steps=future_steps,
                out_channels=6,
            )
        self.agent_predictor = AgentPredictor(dim=dim, future_steps=future_steps)

        self.apply(self._init_weights)

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
        if self.use_history_attention:
            data = self.generate_his_features(data)
            self.select_indices = self.select_indices.to(data["agent"]["position"].device)
            x_agent = self.agent_encoder(data, self.select_indices)
            bs, used_T, A = x_agent.shape[:3] 
            x_polygon = self.map_encoder(data, used_T)
            x_static, static_pos, static_key_padding = self.static_objects_encoder(data)

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
            x = x.reshape(bs*used_T, -1, self.dim)
            key_padding_mask = key_padding_mask.reshape(bs*used_T, -1)
            for blk in self.encoder_blocks:
                x = blk(x, key_padding_mask=key_padding_mask)
            x = self.norm(x)

            x = x.reshape(bs, used_T, -1, self.dim)
            prediction = self.agent_predictor(x[:, -1, 1:A])

            if not self.use_his_offest:    

                trajectory, probability, _ = self.trajectory_decoder(x[:, :, 0])
                probability= probability[:, -1]
                trajectory = trajectory[:, -1]

            else:
                trajectory_his, probability_his, _ = self.trajectory_decoder_his(x[:, -2, 0])
                trajectory_now, probability_now, _ = self.trajectory_decoder_curr(x[:, -1, 0] + x[:, -2, 0])
                # trajectory_now += trajectory_his
                trajectory = torch.concat((trajectory_his.unsqueeze(1), trajectory_now.unsqueeze(1)), dim=1)
                probability = torch.concat((probability_his.unsqueeze(1), probability_now.unsqueeze(1)), dim=1)
            out = {
                "trajectory": trajectory_now,
                "probability": probability_now,  # (bs, R, M)
                "prediction": prediction,  # (bs, A-1, T, 2)
                "trajectory_all_time": trajectory,
                "probability_all_time": probability,
                "select_indices": self.select_indices,
            }
            if not self.training:
                output_prediction = torch.cat(
                    [
                        prediction[..., :2] + agent_pos[:, -1, 1:A, None],
                        torch.atan2(prediction[..., 3], prediction[..., 2]).unsqueeze(-1)
                        + agent_heading[:, -1, 1:A,  None, None],
                        prediction[..., 4:6],
                    ],
                    dim=-1,
                )
                out["output_prediction"] = output_prediction
                angle = torch.atan2(trajectory[:, -1, ..., 3], trajectory[:, -1, ..., 2])
                out_trajectory = torch.cat(
                    [trajectory[:, -1, ..., :2], angle.unsqueeze(-1)], dim=-1
                )
                best_mode = probability[:, -1].argmax(dim=-1)
                best_trajectory = trajectory[:, -1][torch.arange(bs), best_mode]
                out["output_trajectory"] = best_trajectory
                out["candidate_trajectories"] = out_trajectory

        else:
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
        # current_states = data["agent"]["position"][:, 0, self.select_indices]
        # center_angles = data["agent"]["heading"][:, 0, self.select_indices]
        # center_xy, center_angle = current_states[:2].copy(), center_angles.copy()

        data["agent"]["position"] = data["agent"]["position"].unsqueeze(1).repeat(1, used_T, 1, 1, 1)
        data["agent"]["velocity"] = data["agent"]["velocity"].unsqueeze(1).repeat(1, used_T, 1, 1, 1)
        data["agent"]["heading"] = data["agent"]["heading"].unsqueeze(1).repeat(1, used_T, 1, 1)
        data["agent"]["shape"] = data["agent"]["shape"].unsqueeze(1).repeat(1, used_T, 1, 1, 1)
        data["agent"]["category"] = data["agent"]["category"].unsqueeze(1).repeat(1, used_T, 1)
        data["agent"]["valid_mask"] = data["agent"]["valid_mask"].unsqueeze(1).repeat(1, used_T, 1, 1)

        data["map"]["point_position"] = data["map"]["point_position"].unsqueeze(1).repeat(1, used_T, 1, 1, 1, 1)
        data["map"]["point_vector"] = data["map"]["point_vector"].unsqueeze(1).repeat(1, used_T, 1, 1, 1, 1)
        data["map"]["point_orientation"] = data["map"]["point_orientation"].unsqueeze(1).repeat(1, used_T, 1, 1, 1)
        data["current_state"] = data["current_state"].unsqueeze(1).repeat(1, used_T, 1)
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

        for i, index in enumerate(self.select_indices):

            cur_state = data["agent"]["position"][:, i, 0, index]
            center_angle = data["agent"]["heading"][:, i, 0, index]
            center_xy, center_angle = cur_state[:, :2].clone().unsqueeze(1).unsqueeze(1),\
                                        center_angle.clone().unsqueeze(1).unsqueeze(1)

            rotate_mat = torch.stack([
                torch.cos(center_angle), -torch.sin(center_angle),
                torch.sin(center_angle), torch.cos(center_angle)
            ], dim=1).reshape(-1, 2, 2).unsqueeze(1)

            data["current_state"][:, i, :3] = 0
            data["agent"]["position"][:, i] = torch.matmul(
                data["agent"]["position"][:, i] - center_xy, rotate_mat
            )
            data["agent"]["velocity"][:, i] = torch.matmul(data["agent"]["velocity"][:, i], rotate_mat)
            data["agent"]["heading"][:, i] -= center_angle

            data["map"]["point_position"][:, i] = torch.matmul(
                data["map"]["point_position"][:, i] - center_xy.unsqueeze(1), rotate_mat.unsqueeze(1)
            )
            data["map"]["point_vector"][:, i] = torch.matmul(data["map"]["point_vector"][:, i], rotate_mat.unsqueeze(1))
            data["map"]["point_orientation"][:, i] -= center_angle.unsqueeze(1)

            data["map"]["polygon_center"][:, i, ..., :2] = torch.matmul(
                data["map"]["polygon_center"][:, i, ..., :2] - center_xy.squeeze(1), rotate_mat.squeeze(1)
            )
            data["map"]["polygon_center"][:, i, ..., 2] = data["map"]["polygon_center"][:, i, ..., 2] - center_angle.squeeze(1)
            data["map"]["polygon_position"][:, i] = torch.matmul(
                data["map"]["polygon_position"][:, i] - center_xy.squeeze(1), rotate_mat.squeeze(1)
            )
            data["map"]["polygon_orientation"][:, i] -= center_angle.squeeze(1)

            data["static_objects"]["position"][:, i] = torch.matmul(
                data["static_objects"]["position"][:, i] - center_xy.squeeze(1), rotate_mat.squeeze(1)
            )
            data["static_objects"]["heading"][:, i] -= center_angle.squeeze(1)


        return data

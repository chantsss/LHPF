from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..layers.embedding import PointsEncoder
from ..layers.fourier_embedding import FourierEmbedding
from ..layers.mlp_layer import MLPLayer

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_time = self.dim // 2
        embeddings = torch.exp(torch.arange(half_time, device=device) * -(math.log(10000) / (half_time - 1)))
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout) -> None:
        super().__init__()
        self.dim = dim

        self.r2r_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.m2m_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        m_pos: Optional[Tensor] = None,
    ):
        """
        tgt: (bs, R, M, dim)
        tgt_key_padding_mask: (bs, R)
        """
        bs, R, M, D = tgt.shape

        tgt = tgt.transpose(1, 2).reshape(bs * M, R, D)
        tgt2 = self.norm1(tgt)
        tgt2 = self.r2r_attn(
            tgt2, tgt2, tgt2, key_padding_mask=tgt_key_padding_mask.repeat(M, 1)
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt_tmp = tgt.reshape(bs, M, R, D).transpose(1, 2).reshape(bs * R, M, D)
        tgt_valid_mask = ~tgt_key_padding_mask.reshape(-1)
        tgt_valid = tgt_tmp[tgt_valid_mask]
        tgt2_valid = self.norm2(tgt_valid)
        tgt2_valid, _ = self.m2m_attn(tgt2_valid + m_pos, tgt2_valid + m_pos, tgt2_valid)
        tgt_valid = tgt_valid + self.dropout2(tgt2_valid)
        tgt = torch.zeros_like(tgt_tmp)
        tgt[tgt_valid_mask] = tgt_valid

        tgt = tgt.reshape(bs, R, M, D).view(bs, R * M, D)
        tgt2 = self.norm3(tgt)
        tgt2 = self.cross_attn(
            tgt2, memory, memory, key_padding_mask=memory_key_padding_mask
        )[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm4(tgt)
        tgt2 = self.ffn(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        tgt = tgt.reshape(bs, R, M, D)

        return tgt

class PlanningDecoder(nn.Module):
    def __init__(self, num_mode, decoder_depth, dim, num_heads, mlp_ratio, dropout, future_steps, yaw_constraint=False, cat_x=False, use_history_attention=False, used_T=1, cluster_modes=None) -> None:
        super().__init__()

        self.num_mode = num_mode
        self.future_steps = future_steps
        self.yaw_constraint = yaw_constraint
        self.cat_x = cat_x
        self.use_history_attention = use_history_attention
        self.used_T = used_T

        self.decoder_blocks_his = nn.ModuleList(
            [
                DecoderLayer(dim, num_heads, mlp_ratio, dropout)
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_blocks_now = nn.ModuleList(
            [
                DecoderLayer(dim, num_heads, mlp_ratio, dropout)
                for _ in range(decoder_depth)
            ]
        )

        self.r_pos_emb = FourierEmbedding(3, dim, 64)
        self.r_encoder = PointsEncoder(6, dim)

        self.q_proj = nn.Linear(2 * dim, dim)
        self.q_proj_now = nn.Linear(3 * dim, dim)


        self.m_emb = nn.Parameter(torch.Tensor(1, 1, num_mode, dim))
        # self.m_pos = nn.Parameter(torch.Tensor(1, num_mode, dim))
        self.m_pos_to_dim = MLPLayer(3, dim, dim)
        self.m_pos = nn.Parameter(torch.Tensor(cluster_modes).unsqueeze(0), requires_grad=True)

        self.used_T_list = nn.Parameter(torch.tensor(range(self.used_T), dtype=torch.long), requires_grad=False)
        self.h_emb = nn.Embedding(num_embeddings=self.used_T, embedding_dim=dim)
    
        # self.h2h_attn = nn.MultiheadAttention(
        #     dim, num_heads, dropout=dropout, batch_first=True
        # )
        # self.dropout3 = nn.Dropout(dropout)
        # self.his_proj = nn.Linear(dim, dim)
        # self.norm4 = nn.LayerNorm(dim)
        
        if self.cat_x:
            self.cat_x_proj = nn.Linear(2 * dim, dim)

        self.loc_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.yaw_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.vel_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.pi_head = MLPLayer(dim, dim, 1)

        self.loc_head_now = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.yaw_head_now = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.vel_head_now = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.pi_head_now = MLPLayer(dim, dim, 1)

        nn.init.normal_(self.m_emb, mean=0.0, std=0.01)

    def forward(self, data, enc_data):
        enc_emb = enc_data["enc_emb"]
        bs, used_T = enc_emb.shape[:2]
        enc_key_padding_mask = enc_data["enc_key_padding_mask"].reshape(bs, 2, -1)

        r_position = data["reference_line"]["position"]
        r_vector = data["reference_line"]["vector"]
        r_orientation = data["reference_line"]["orientation"]
        r_valid_mask = data["reference_line"]["valid_mask"]
        r_key_padding_mask = ~r_valid_mask.any(-1)

        r_feature = torch.cat([r_position - r_position[..., 0:1, :2], r_vector, torch.stack([r_orientation.cos(), r_orientation.sin()], dim=-1)], dim=-1)

        bs, used_T, R, P, C = r_feature.shape
        r_valid_mask = r_valid_mask.view(bs * used_T * R, P)
        r_feature = r_feature.reshape(bs * used_T * R, P, C)
        r_emb = self.r_encoder(r_feature, r_valid_mask).view(bs, used_T, R, -1)

        r_pos = torch.cat([r_position[:, :, :, 0], r_orientation[:, :, :, 0, None]], dim=-1)
        r_emb = r_emb + self.r_pos_emb(r_pos)

        r_emb = r_emb.unsqueeze(-2).repeat(1, 1, 1, self.num_mode, 1)
        m_emb = self.m_emb.unsqueeze(1).repeat(bs, used_T, R, 1, 1)
        m_pos = self.m_pos_to_dim(self.m_pos)
        
        h_emb = self.h_emb(self.used_T_list) # 5, 256
        # else:
        #     r_emb = r_emb.repeat(1, 1, 1, 1)
        #     m_emb = self.m_emb.repeat(bs, R, 1, 1)
        # q = self.q_proj(torch.cat([r_emb, m_emb], dim=-1))
        q = self.q_proj(torch.cat([r_emb[:, -2], m_emb[:, -2]], dim=-1)) + h_emb[-2]
        for blk in self.decoder_blocks_his:
            q = blk(
                q,
                enc_emb[:, -2],
                tgt_key_padding_mask=r_key_padding_mask[:, -2],
                memory_key_padding_mask=enc_key_padding_mask[:, -2],
                m_pos=m_pos,
            )
            assert torch.isfinite(q).all()
        
        q_now = self.q_proj_now(torch.cat([r_emb[:, -1], m_emb[:, -1], q.detach()], dim=-1)) + h_emb[-1]
        for blk in self.decoder_blocks_now:
            q_now = blk(
                q_now,
                enc_emb[:, -1],
                tgt_key_padding_mask=r_key_padding_mask[:, -1],
                memory_key_padding_mask=enc_key_padding_mask[:, -1],
                m_pos=m_pos,
            )
            assert torch.isfinite(q_now).all()

        # bs, used_T, R, M, D = q.shape

        # if used_T != 1:
        #     q = q.reshape(bs, used_T, R, M, D).permute(0, 2, 3, 1, 4).reshape(bs * R * M , used_T, D)
        #     tgt2 = self.norm4(q)
        #     # attn_mask = torch.triu(torch.ones(used_T, used_T), diagonal=1).bool().to(q.device)
        #     # tgt2 = self.h2h_attn(tgt2 + h_emb.unsqueeze(0), tgt2 + h_emb.unsqueeze(0), tgt2, attn_mask=attn_mask)[0]
        #     tgt2[:, :-1] = self.his_proj(tgt2[:, -2]).unsqueeze(1) + tgt2[:, :-1]
        #     q = q + self.dropout3(tgt2)

        # if self.cat_x:
        #     x = enc_emb[:, -1].unsqueeze(1).unsqueeze(2).repeat(1, R, self.num_mode, 1)
        #     q = self.cat_x_proj(torch.cat([q, x], dim=-1))

        if self.use_history_attention:
            loc = self.loc_head(q).view(bs, 1, R, self.num_mode, self.future_steps, 2)
            yaw = self.yaw_head(q).view(bs, 1, R, self.num_mode, self.future_steps, 2)
            vel = self.vel_head(q).view(bs, 1, R, self.num_mode, self.future_steps, 2)
            pi = self.pi_head(q).squeeze(-1).unsqueeze(1)
            loc_now = self.loc_head_now(q_now).view(bs, 1, R, self.num_mode, self.future_steps, 2)
            yaw_now = self.yaw_head_now(q_now).view(bs, 1, R, self.num_mode, self.future_steps, 2)
            vel_now = self.vel_head_now(q_now).view(bs, 1, R, self.num_mode, self.future_steps, 2)
            pi_now = self.pi_head_now(q_now).squeeze(-1).unsqueeze(1)
            loc = torch.cat([loc, loc_now], dim=1)
            yaw = torch.cat([yaw, yaw_now], dim=1)
            vel = torch.cat([vel, vel_now], dim=1)
            pi = torch.cat([pi, pi_now], dim=1)
        else:
            loc = self.loc_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
            yaw = self.yaw_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
            vel = self.vel_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
            pi = self.pi_head(q).squeeze(-1)

        traj = torch.cat([loc, yaw, vel], dim=-1)
        return traj, pi

    def _compute_heads(self, q, q_now, bs, R):
        loc = self.loc_head(q).view(bs, 1, R, self.num_mode, self.future_steps, 2)
        yaw = self.yaw_head(q).view(bs, 1, R, self.num_mode, self.future_steps, 2)
        vel = self.vel_head(q).view(bs, 1, R, self.num_mode, self.future_steps, 2)
        pi = self.pi_head(q).squeeze(-1).unsqueeze(1)
        loc_now = self.loc_head_now(q_now).view(bs, 1, R, self.num_mode, self.future_steps, 2)
        yaw_now = self.yaw_head_now(q_now).view(bs, 1, R, self.num_mode, self.future_steps, 2)
        vel_now = self.vel_head_now(q_now).view(bs, 1, R, self.num_mode, self.future_steps, 2)
        pi_now = self.pi_head_now(q_now).squeeze(-1).unsqueeze(1)
        loc = torch.cat([loc, loc_now], dim=1)
        yaw = torch.cat([yaw, yaw_now], dim=1)
        vel = torch.cat([vel, vel_now], dim=1)
        pi = torch.cat([pi, pi_now], dim=1)
        return loc, yaw, vel, pi
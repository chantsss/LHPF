import torch
import torch.nn as nn
import torch.optim as optim
from src.models.lhpf.modules.planning_decoder import PlanningDecoder

def generate_data(bs, T, R, P, C, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    enc_emb = torch.randn(bs, T, R, 64)
    enc_key_padding_mask = torch.zeros(bs, T, R, dtype=torch.bool)
    data = {
        "reference_line": {
            "position": torch.randn(bs, T, R, P, 2),
            "vector": torch.randn(bs, T, R, P, 2),
            "orientation": torch.randn(bs, T, R, P),
            "valid_mask": torch.ones(bs, T, R, P, dtype=torch.bool)
        }
    }
    enc_data = {
        "enc_emb": enc_emb,
        "enc_key_padding_mask": enc_key_padding_mask
    }
    target_traj = torch.randn(bs, T, R, 3, 30, 6)
    target_pi = torch.randint(0, 3, (bs, T, R, 3))
    return data, enc_data, target_traj, target_pi

def train_model(planning_decoder, num_epochs=1000):
    bs, T, R, P, C = 2, 2, 10, 6, 3
    criterion_traj = nn.MSELoss()
    criterion_pi = nn.CrossEntropyLoss()
    optimizer = optim.Adam(planning_decoder.parameters(), lr=0.001)

    # 生成训练数据和验证数据，使用相同的随机种子确保相关性
    train_data, train_enc_data, train_target_traj, train_target_pi = generate_data(bs, T, R, P, C, seed=42)
    val_data, val_enc_data, val_target_traj, val_target_pi = generate_data(bs, T, R, P, C, seed=42)
    
    for epoch in range(num_epochs):
        # 训练阶段
        planning_decoder.train()
        optimizer.zero_grad()
        train_traj, train_pi = planning_decoder(train_data, train_enc_data)

        # 调整 pi 和 target_pi 的形状
        train_pi = train_pi.view(-1).float()
        train_target_pi = train_target_pi.view(-1).float()

        train_loss_traj = criterion_traj(train_traj, train_target_traj)
        train_loss_pi = criterion_pi(train_pi, train_target_pi)
        train_loss = train_loss_traj + train_loss_pi

        train_loss.backward()
        optimizer.step()

        # 验证阶段
        planning_decoder.eval()
        with torch.no_grad():
            val_traj, val_pi = planning_decoder(val_data, val_enc_data)

            # 调整 pi 和 target_pi 的形状
            val_pi = val_pi.view(-1).float()
            val_target_pi = val_target_pi.view(-1).float()

            val_loss_traj = criterion_traj(val_traj, val_target_traj)
            val_loss_pi = criterion_pi(val_pi, val_target_pi)
            val_loss = val_loss_traj + val_loss_pi

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

if __name__ == "__main__":
    planning_decoder = PlanningDecoder(
        num_mode=3,
        decoder_depth=2,
        dim=64,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        future_steps=30,
        yaw_constraint=False,
        cat_x=False,
        use_history_attention=True,
        used_T=1,
        cluster_modes=[0.1, 0.2, 0.3]
    )
    train_model(planning_decoder)
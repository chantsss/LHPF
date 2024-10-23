import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class TrajectoryComfortLoss(nn.Module):
    def __init__(self, 
                 min_lon_accel: float = -4.05, 
                 max_lon_accel: float = 2.40, 
                 max_abs_lat_accel: float = 4.89, 
                 max_abs_yaw_accel: float = 1.93, 
                 max_abs_yaw_rate: float = 0.95, 
                 max_abs_lon_jerk: float = 4.13, 
                 max_abs_mag_jerk: float = 8.37,
                 max_speed: float = 50.0, 
                 loss_clip_value: float = 10.0) -> None:
        super().__init__()
        self.min_lon_accel = min_lon_accel
        self.max_lon_accel = max_lon_accel
        self.max_abs_lat_accel = max_abs_lat_accel
        self.max_abs_yaw_accel = max_abs_yaw_accel
        self.max_abs_yaw_rate = max_abs_yaw_rate
        self.max_abs_lon_jerk = max_abs_lon_jerk
        self.max_abs_mag_jerk = max_abs_mag_jerk
        self.max_speed = max_speed
        self.loss_clip_value = loss_clip_value

    def forward(self, trajectory: Tensor, target_trajectory: Tensor) -> tuple:
        """
        trajectory: (bs, T, 6) - [x, y, cos(yaw), sin(yaw), speed_x, speed_y]
        target_trajectory: (bs, T, 6) - [x, y, cos(yaw), sin(yaw), speed_x, speed_y]
        """
        dt = 0.1  # Assuming a constant time step of 0.1 seconds

        def calculate_metrics(traj):
            # 提取速度分量
            speed_x = traj[:, :, 4]
            speed_y = traj[:, :, 5]
            speed = torch.sqrt(speed_x**2 + speed_y**2)

            # 计算纵向加速度
            lon_accel = (speed[:, 1:] - speed[:, :-1]) / dt

            # 计算横向加速度
            yaw = torch.atan2(traj[:, :, 3], traj[:, :, 2])
            lat_accel = speed[:, :-1] * (yaw[:, 1:] - yaw[:, :-1]) / dt

            # 计算偏航率和偏航加速度
            yaw_rate = (yaw[:, 1:] - yaw[:, :-1]) / dt
            yaw_accel = (yaw_rate[:, 1:] - yaw_rate[:, :-1]) / dt

            # 计算纵向跃度
            lon_jerk = (lon_accel[:, 1:] - lon_accel[:, :-1]) / dt

            # 计算幅度跃度
            mag_jerk = torch.sqrt(lon_jerk**2 + (lat_accel[:, 1:] - lat_accel[:, :-1])**2 / dt**2)

            return speed, lon_accel, lat_accel, yaw_accel, yaw_rate, lon_jerk, mag_jerk

        # 计算trajectory的各项指标
        speed, lon_accel, lat_accel, yaw_accel, yaw_rate, lon_jerk, mag_jerk = calculate_metrics(trajectory)

        # 计算target_trajectory的各项指标
        target_speed, target_lon_accel, target_lat_accel, target_yaw_accel, target_yaw_rate, target_lon_jerk, target_mag_jerk = calculate_metrics(target_trajectory)

        # 计算损失
        lon_accel_loss = F.relu(lon_accel - self.max_lon_accel)**2 + F.relu(self.min_lon_accel - lon_accel)**2
        lat_accel_loss = F.relu(torch.abs(lat_accel) - self.max_abs_lat_accel)**2
        yaw_accel_loss = F.relu(torch.abs(yaw_accel) - self.max_abs_yaw_accel)**2
        yaw_rate_loss = F.relu(torch.abs(yaw_rate) - self.max_abs_yaw_rate)**2
        lon_jerk_loss = F.relu(torch.abs(lon_jerk) - self.max_abs_lon_jerk)**2
        mag_jerk_loss = F.relu(torch.abs(mag_jerk) - self.max_abs_mag_jerk)**2
        speed_loss = F.relu(speed - self.max_speed)**2

        # 计算target_trajectory的损失
        target_lon_accel_loss = F.relu(target_lon_accel - self.max_lon_accel)**2 + F.relu(self.min_lon_accel - target_lon_accel)**2
        target_lat_accel_loss = F.relu(torch.abs(target_lat_accel) - self.max_abs_lat_accel)**2
        target_yaw_accel_loss = F.relu(torch.abs(target_yaw_accel) - self.max_abs_yaw_accel)**2
        target_yaw_rate_loss = F.relu(torch.abs(target_yaw_rate) - self.max_abs_yaw_rate)**2
        target_lon_jerk_loss = F.relu(torch.abs(target_lon_jerk) - self.max_abs_lon_jerk)**2
        target_mag_jerk_loss = F.relu(torch.abs(target_mag_jerk) - self.max_abs_mag_jerk)**2
        target_speed_loss = F.relu(target_speed - self.max_speed)**2

        # 计算每个损失与对应的目标损失之间的Smooth L1损失
        final_lon_accel_loss = F.smooth_l1_loss(lon_accel_loss, target_lon_accel_loss)
        final_lat_accel_loss = F.smooth_l1_loss(lat_accel_loss, target_lat_accel_loss)
        final_yaw_accel_loss = F.smooth_l1_loss(yaw_accel_loss, target_yaw_accel_loss)
        final_yaw_rate_loss = F.smooth_l1_loss(yaw_rate_loss, target_yaw_rate_loss)
        final_lon_jerk_loss = F.smooth_l1_loss(lon_jerk_loss, target_lon_jerk_loss)
        final_mag_jerk_loss = F.smooth_l1_loss(mag_jerk_loss, target_mag_jerk_loss)
        final_speed_loss = F.smooth_l1_loss(speed_loss, target_speed_loss)

        # 总的舒适度损失
        comfort_loss = (final_lon_accel_loss + final_lat_accel_loss + final_yaw_accel_loss + final_yaw_rate_loss + final_lon_jerk_loss + final_mag_jerk_loss)

        return comfort_loss, final_speed_loss
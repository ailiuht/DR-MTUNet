import math
import torch.nn as nn
from kornia.losses import ssim_loss
import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d


# DICE
def dice_loss(pred, target, smooth=1e-5):
    if pred.shape[1] > 1:
        target = F.one_hot(target.squeeze(1), num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    else:
        pred = torch.sigmoid(pred)
        target = target.float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2.0 * intersection + smooth) / (union + smooth)).mean()


def shape_parameter_loss(shape_pred, shape_gt):
    return F.smooth_l1_loss(shape_pred, shape_gt)


def tversky_focal_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.7,  # 假阳性惩罚系数
        beta: float = 0.3,  # 假阴性惩罚系数
        gamma: float = 2.0,  # 困难样本聚焦系数
        smooth: float = 1e-6
) -> torch.Tensor:
    """
    多类别Tversky-Focal混合损失函数
    适用于pred为各类别概率的情况（未经过argmax处理）

    参数：
    - pred: 预测概率 (B, C, H, W)
    - target: 真实标签 (B, 1, H, W) 或 (B, H, W)
    - alpha: 假阳性惩罚权重 [0,1]
    - beta: 假阴性惩罚权重 [0,1]
    - gamma: Focal系数（>0时增强困难样本关注）
    """
    # 转换目标为one-hot编码
    if target.dim() == 4:
        target = target.squeeze(1)  # (B, H, W)

    target_oh = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()  # (B, C, H, W)



    # 计算各通道的TP/FP/FN
    tp = (pred * target_oh).sum(dim=(2, 3))  # (B, C)
    fp = (pred * (1 - target_oh)).sum(dim=(2, 3))  # (B, C)
    fn = ((1 - pred) * target_oh).sum(dim=(2, 3))  # (B, C)

    # Tversky系数（逐类别计算）
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)  # (B, C)

    # Focal加权
    focal_weight = (1 - tversky).pow(gamma)  # 困难样本加权

    # 损失计算
    return (focal_weight * (1 - tversky)).mean()  # 最终损失值

#  主函数
class MultiTaskLoss(nn.Module):
    def __init__(self, num_classes=2, gamma=0.2, delta=0.4, device="cpu"):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        # 分割损失Dice Loss 初始时分割损失占主导
        self.alpha = nn.Parameter(torch.tensor(0.2), requires_grad=False)
        # 边缘损失权重较低
        self.beta = nn.Parameter(torch.tensor(0.2), requires_grad=False)
        # 参数损失
        self.register_buffer('gamma', torch.tensor(gamma))
        self.register_buffer('delta', torch.tensor(delta))
        # 跟踪当前批次的像素级（Pixel-level）和区域级（Region-level）IoU的指数移动平均值。
        # 更新逻辑：在 update_weights 方法中通过EMA（指数移动平均）更新，用于动态调整 alpha 和 beta
        # 注册一个不需要梯度更新但需要持久化的张量
        self.register_buffer('current_rln_iou', torch.tensor(0.1))
        # 假阳性控制参数
        self.tversky_alpha = 0.6  # 增大此值增强FP惩罚
        self.tversky_beta = 0.4
        self.focal_gamma = 3

    def update_weights(self, rln_iou, phase_name, epoch, total_epochs):
        if phase_name == 'recon':
            # 重建阶段：gamma=0.8, delta=0.9（张量操作）
            self.alpha.copy_(torch.tensor(0.2))
            self.beta.copy_(torch.tensor(0.2))
            self.gamma.copy_(torch.tensor(0.2))  # 使用 copy_() 赋值
            self.delta.copy_(torch.tensor(0.6))
            return

        # === 分割阶段（full）的动态调整 ===
        # 指数平滑更新IoU
        if not math.isnan(rln_iou):
            self.current_rln_iou = 0.6 * self.current_rln_iou + 0.4 * rln_iou
        avg_iou = self.current_rln_iou

        # 计算训练进度因子
        epoch_ratio = epoch / total_epochs

        if avg_iou < 0.6:  # 效果不好阶段
            if epoch_ratio < 0.3:  # 早期阶段
                # 使用与原始张量相同的设备和数据类型
                self.gamma.copy_(self.gamma.new_tensor(0.5))  # 参数损失权重
                self.delta.copy_(self.delta.new_tensor(0.5))  # 重建损失权重
                self.alpha.copy_(self.alpha.new_tensor(0.8))  # 分割损失权重
                self.beta.copy_(self.beta.new_tensor(0.2))  # 边缘损失权重
            else:
                # 显式创建计算图分离的张量
                alpha_val = 0.9 + 0.4 * epoch_ratio
                beta_val = 0.1 + 0.3 * epoch_ratio

                self.alpha.copy_(self.alpha.new_tensor(alpha_val))
                self.delta.copy_(self.delta.new_tensor(0.3))
                self.gamma.copy_(self.gamma.new_tensor(0.2))
                self.beta.copy_(self.beta.new_tensor(beta_val))
        else:  # 效果良好阶段
            # 确保所有运算在相同设备上
            smooth_factor = torch.sigmoid(
                torch.tensor((epoch_ratio - 0.5) * 10,
                             device=self.alpha.device,  # 使用已有参数设备
                             dtype=self.alpha.dtype)  # 使用已有参数类型
            ).detach()  # 显式断开计算图

            if epoch_ratio < 0.6:
                # 组合式参数更新（保持设备一致）
                self.alpha.copy_(self.alpha.new_tensor(0.7 + 0.2 * smooth_factor.item()))
                self.beta.copy_(self.beta.new_tensor(0.3 + 0.4 * smooth_factor.item()))
                self.delta.copy_(self.delta.new_tensor(0.5))
                self.gamma.copy_(self.gamma.new_tensor(0.3))
            else:
                # 动态参数计算（标量运算后转换）
                beta_val = 0.7 + 0.3 * smooth_factor.item()
                alpha_val = 0.5 + 0.3 * smooth_factor.item()

                self.beta.copy_(self.beta.new_tensor(beta_val))
                self.alpha.copy_(self.alpha.new_tensor(alpha_val))
                self.delta.copy_(self.delta.new_tensor(0.3))
                self.gamma.copy_(self.gamma.new_tensor(0.2))

            # 自动衰减辅助损失权重
            new_delta = max(0.1, 0.5 * (1 - epoch_ratio))
            new_gamma = max(0.05, 0.3 * (1 - epoch_ratio))
            self.delta.copy_(torch.tensor(new_delta))
            self.gamma.copy_(torch.tensor(new_gamma))

        # 数值稳定性约束
        self.alpha.data.clamp_(0.0, 1.0)
        self.beta.data.clamp_(0.0, 1.0)
        self.gamma.data.clamp_(0.0, 1.0)
        self.delta.data.clamp_(0.0, 1.0)
    # pred,              # 分割预测结果
    # target,            # 分割真实标签
    # edge_maps,         # 真实边缘图

    # shape_pred,        # 形状预测结果
    # shape_gt,         # 形状真实标签

    # recon_pred,        # 重建图像预测结果
    # recon_gt):         # 原始输入图像（重建目标）


    def multi_scale_edge_loss(self,
            pred_softmax: torch.Tensor,
            edges: torch.Tensor,
            scales: list = [1, 2, 4],
            scale_weights: list = None,
            sigma_ratio: float = 1.0,
            eps: float = 1e-8
    ) -> torch.Tensor:
        """
        多尺度边缘匹配损失函数（优化版）

        参数：
        - pred_softmax: 模型输出的概率图，形状为(B, 3, H, W)
        - edges: 真实边缘图，形状为(B, 2, H,W)，第0通道mask1边缘，第1通道mask2边缘
        - scales: 尺度参数列表，控制高斯模糊强度，默认[1,2,4]
        - scale_weights: 各尺度损失的权重，默认None表示等权重
        - sigma_ratio: 高斯核标准差系数，计算公式sigma = scale * sigma_ratio
        - eps: 数值稳定项，防止除零

        返回：
        - loss: 多尺度边缘匹配损失值
        """

        B, _, H, W = edges.shape

        # ==================== 生成多尺度边缘目标 ====================
        edge_targets = []
        for s in scales:
            # 动态计算高斯核参数
            sigma = max(s * sigma_ratio, 0.5)  # 确保sigma不小于0.5
            kernel_size = max(int(3 * sigma * 2) // 2 * 2 + 1, 3)

            # 应用高斯模糊生成多尺度边缘
            blurred = gaussian_blur2d(
                edges,
                kernel_size=(kernel_size, kernel_size),
                sigma=(sigma, sigma),
                border_type='reflect'
            )
            edge_targets.append(blurred)

        # ==================== 准备预测边缘 ====================
        # 提取mask1和mask2的概率 (B,2,H,W)
        pred_edges = pred_softmax[:, 1:, :, :]

        # ==================== 损失计算 ====================
        total_loss = 0.0
        num_scales = len(scales)

        # 设置默认权重（等权重）
        if scale_weights is None:
            scale_weights = [1.0 / num_scales] * num_scales
        else:
            # 归一化权重
            scale_weights = torch.tensor(scale_weights) / sum(scale_weights)

        # 生成边缘掩码（排除非边缘区域）
        edge_mask = (edges.sum(dim=1, keepdim=True) > 0).float()  # (B,1,H,W)

        for i, target in enumerate(edge_targets):
            # 计算余弦相似度（仅关注边缘区域）
            similarity = F.cosine_similarity(pred_edges, target, dim=1, eps=eps)  # (B,H,W)

            # 应用边缘掩码并计算损失
            l1_loss = F.l1_loss(pred_edges, target, reduction='none') * edge_mask
            masked_loss = (1 - similarity.unsqueeze(1)) * edge_mask # (B,1,H,W)
            # scale_loss = masked_loss.sum() / (edge_mask.sum() + eps)
            scale_loss = (masked_loss + l1_loss).sum() / (edge_mask.sum() + eps)
            # 加权累加
            total_loss += scale_weights[i] * scale_loss

        return total_loss



    def ellipse_loss(self,
            pred: torch.Tensor,
            gt: torch.Tensor,
            img_size: tuple = (256, 256),
            weights=(0.2, 0.5, 0.3),  # 降低角度权重
            iou_weight = 0.5 , # 降低几何损失比例
            axis_penalty = 0.5 , # 减少轴长顺序惩罚
            theta_samples: int = 180
    ) -> torch.Tensor:
        """
        鲁棒的椭圆参数损失函数，支持批量计算

        Args:
            pred: 预测参数 [batch_size, 10]，前5为label1，后5为label2
            gt: 真实参数 [batch_size, 10]
            img_size: (width, height) 图像尺寸
            weights: 各参数损失权重 (中心坐标，轴长，角度)
            iou_weight: 几何损失权重(0=纯参数，1=纯几何)
            axis_penalty: 长轴<短轴的惩罚系数
            theta_samples: IoU计算的采样点数

        Returns:
            综合损失值 (标量张量)
        """
        # 输入校验
        assert pred.shape == gt.shape, "预测和真实值形状必须一致"
        device = pred.device
        img_w, img_h = img_size
        max_axis = math.sqrt(img_w ** 2 + img_h ** 2)  # 最大可能轴长

        # --------------------- 内部工具函数 ---------------------
        def normalize_params(params: torch.Tensor) -> torch.Tensor:
            """带约束的参数归一化"""
            params_norm = params.clone()
            # 中心坐标约束到图像范围并归一化
            params_norm[:, 0] = torch.clamp(params[:, 0], 0, img_w) / img_w  # cx
            params_norm[:, 1] = torch.clamp(params[:, 1], 0, img_h) / img_h  # cy
            # 轴长使用Softplus确保正数
            params_norm[:, 2] = torch.nn.functional.softplus(params[:, 2]) / max_axis  # major
            params_norm[:, 3] = torch.nn.functional.softplus(params[:, 3]) / max_axis  # minor
            # 角度处理负值和周期性
            angle = params[:, 4] % 360  # 转换到0-360范围
            params_norm[:, 4] = angle / 360  # 归一化到[0,1]
            return params_norm

        def compute_single_loss(p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            """单个椭圆损失计算"""
            # 有效性过滤
            valid_mask = (g[:, 2] > 1e-4) & (g[:, 3] > 1e-4)
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=device)

            p_valid, g_valid = p[valid_mask], g[valid_mask]

            # 中心坐标损失（边缘增强）
            center_diff = torch.abs(p_valid[:, :2] - g_valid[:, :2])
            edge_mask = (g_valid[:, :2] < 0.1) | (g_valid[:, :2] > 0.9)
            center_loss = torch.mean(center_diff * (1.3 ** edge_mask)) * weights[0]

            # 轴长损失（对数+相对）
            epsilon = 1e-6
            p_axis = torch.clamp(p_valid[:, 2:4], min=epsilon)
            g_axis = torch.clamp(g_valid[:, 2:4], min=epsilon)
            axis_loss = (
                                0.6 * torch.mean(torch.abs(torch.log(p_axis) - torch.log(g_axis))) +
                                0.4 * torch.mean(torch.abs(p_axis - g_axis) / (g_axis + epsilon))
                        ) * weights[1]

            # 角度损失
            angle_diff = torch.abs(p_valid[:, 4] - g_valid[:, 4]) % 1.0
            angle_loss = torch.mean(torch.min(angle_diff, 1 - angle_diff)) * weights[2]

            # 几何IoU损失
            iou = compute_iou(p_valid, g_valid)
            geom_loss = (1 - torch.mean(iou)) * iou_weight

            # 轴长约束惩罚
            major, minor = p_valid[:, 2], p_valid[:, 3]
            axis_pen = torch.mean(torch.relu(minor - major)) * axis_penalty

            # 无效预测惩罚
            invalid_ratio = 1 - valid_mask.float().mean()
            return center_loss + axis_loss + angle_loss + geom_loss + axis_pen + invalid_ratio

        def compute_iou(p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            """快速IoU近似计算"""
            # 生成采样点
            theta = torch.linspace(0, 2 * math.pi, theta_samples, device=device)

            # 坐标转换
            def ellipse_points(params):
                cx = params[:, 0] * img_w
                cy = params[:, 1] * img_h
                a = torch.clamp(params[:, 2], min=1e-3) * max_axis
                b = torch.clamp(params[:, 3], min=1e-3) * max_axis
                angle = params[:, 4] * 2 * math.pi

                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
                cos_a = torch.cos(angle).unsqueeze(1)
                sin_a = torch.sin(angle).unsqueeze(1)

                x = a.unsqueeze(1) * cos_t * cos_a - b.unsqueeze(1) * sin_t * sin_a + cx.unsqueeze(1)
                y = a.unsqueeze(1) * cos_t * sin_a + b.unsqueeze(1) * sin_t * cos_a + cy.unsqueeze(1)
                return torch.stack([x, y], dim=-1)

            # 外包矩形计算
            def minmax_rect(points):
                x_min = points[..., 0].min(dim=1).values
                y_min = points[..., 1].min(dim=1).values
                x_max = points[..., 0].max(dim=1).values
                y_max = points[..., 1].max(dim=1).values
                return torch.stack([x_min, y_min, x_max, y_max], dim=1)

            # 交并比计算
            p_rect = minmax_rect(ellipse_points(p))
            g_rect = minmax_rect(ellipse_points(g))

            inter_left = torch.max(p_rect[:, 0], g_rect[:, 0])
            inter_right = torch.min(p_rect[:, 2], g_rect[:, 2])
            inter_top = torch.max(p_rect[:, 1], g_rect[:, 1])
            inter_bottom = torch.min(p_rect[:, 3], g_rect[:, 3])

            inter_w = torch.clamp(inter_right - inter_left, min=0)
            inter_h = torch.clamp(inter_bottom - inter_top, min=0)
            inter_area = inter_w * inter_h

            p_area = (p_rect[:, 2] - p_rect[:, 0]) * (p_rect[:, 3] - p_rect[:, 1])
            g_area = (g_rect[:, 2] - g_rect[:, 0]) * (g_rect[:, 3] - g_rect[:, 1])
            union = p_area + g_area - inter_area

            return torch.clamp(inter_area / (union + 1e-6), min=0, max=1)

        # --------------------- 主流程 ---------------------
        # 参数归一化
        pred_norm = normalize_params(pred[:, :5])
        gt_norm = normalize_params(gt[:, :5])

        # 分别计算两个椭圆的损失
        loss1 = compute_single_loss(pred_norm, gt_norm)

        # 最终损失（含NaN保护）
        total_loss = loss1
        return torch.where(torch.isnan(total_loss), torch.tensor(2, device=device), total_loss)

    # def forward(self, pred, target, edge_maps, shape_gt, recon_pred, recon_gt):
    def forward(self, pred, target, edge_maps, shape_pred, shape_gt, recon_pred, recon_gt):
        seg_loss = tversky_focal_loss(
            pred=pred,
            target=target,
            alpha=self.tversky_alpha,
            beta=self.tversky_beta,
            gamma=self.focal_gamma
        )


        # 损失函数通过多尺度的高斯模糊处理真实边缘，计算预测边缘与各尺度目标之间的余弦相似度，加权求和得到最终损失
        edge_loss = self.multi_scale_edge_loss(
            pred_softmax=pred,
            edges=edge_maps,
            scales=[1, 2, 3],
            scale_weights=[0.6, 0.3, 0.1],  # 更关注细边缘
            sigma_ratio=1.2
        )

        # kl_loss = F.kl_div(F.log_softmax(shape_pred, dim=1),
        #                    F.softmax(shape_gt, dim=1),
        #                    reduction='batchmean')

        shape_loss = self.ellipse_loss(shape_pred, shape_gt)


        recon_l1 = F.l1_loss(recon_pred, recon_gt)
        recon_ssim = 1 - ssim_loss(recon_pred, recon_gt, window_size=5)
        recon_loss = 0.5 * recon_l1 + 0.5 * recon_ssim

        return (self.alpha * seg_loss +
                self.beta * edge_loss +
                self.gamma * shape_loss +
                self.delta * recon_loss)
        # return (self.alpha * seg_loss +
        #         self.beta * edge_loss +
        #         self.delta * recon_loss)


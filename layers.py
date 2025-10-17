import torch
import torch.nn as nn
import torch.nn.functional as F
# ---------------------- 核心模块 ----------------------

#
# class ConvBlock(nn.Module):
#     """基础卷积块"""
#
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.PReLU(),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.PReLU()
#         )
#
#     def forward(self, x):
#         return self.conv(x)


class ConvBlock(nn.Module):
    """混合空洞卷积与注意力机制（修复通道不匹配）"""

    def __init__(self, in_ch, out_ch):
        super().__init__()

        # 添加通道对齐模块
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity(),
            nn.BatchNorm2d(out_ch) if in_ch != out_ch else nn.Identity()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),

            # 多尺度空洞卷积
            nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

        # 通道注意力（保持通道一致）
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, max(out_ch // 8, 1), 1),  # 确保最小有1通道
            nn.GELU(),
            nn.Conv2d(max(out_ch // 8, 1), out_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道对齐
        identity = self.shortcut(x)

        # 特征提取
        x = self.conv(identity)  # 使用对齐后的特征

        # 注意力计算
        attn = self.attn(x)

        # 残差连接
        return identity + x * attn


class DynamicResidualGate(nn.Module):
    """动态残差门控跳跃连接"""

    def __init__(self, in_ch, shape_dim=5):
        super().__init__()
        # 形状参数编码器
        self.shape_encoder = nn.Sequential(
            nn.Linear(shape_dim, in_ch),
            nn.PReLU()
        )
        # 多尺度残差金字塔
        self.res_pyramid = nn.ModuleList([
            nn.Conv2d(in_ch, in_ch, 3, padding=d, dilation=d)
            for d in [1, 2, 3]
        ])
        # 注意力门控
        self.attn = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x_enc, shape_params):
        # 形状特征编码
        batch_size = x_enc.shape[0]
        shape_feat = self.shape_encoder(shape_params)  # (B, C)
        shape_feat = shape_feat.view(batch_size, -1, 1, 1)  # (B, C, 1, 1)
        shape_feat = shape_feat.expand_as(x_enc)

        # 多尺度残差融合
        res_features = [branch(x_enc) for branch in self.res_pyramid]
        res_out = torch.stack(res_features).mean(dim=0)

        # 动态门控融合
        gate = self.attn(torch.cat([res_out, shape_feat], dim=1))
        return x_enc + gate * res_out


# 分割头通过 深度可分离卷积、通道注意力机制 、 多尺度特征融合和残差链接 提升分割精度，同时保持较低的参数量
class EnhancedSegHead(nn.Module):
    def __init__(self, in_ch=64, seg_ch=2, mid_ch=128):
        super().__init__()
        self.res_conv = nn.Conv2d(in_ch, seg_ch, 1)
        # 深度可分离卷积组
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, groups=in_ch),
            nn.Conv2d(mid_ch, mid_ch, 1),
            nn.BatchNorm2d(mid_ch),
            nn.PReLU()
        )
        # SE Block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_ch, mid_ch//16, 1),
            nn.PReLU(),
            nn.Conv2d(mid_ch//16, mid_ch, 1),
            nn.Sigmoid()
        )

        # 多尺度融合
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_ch, seg_ch, 3, padding=2, dilation=2),
            nn.Conv2d(seg_ch, seg_ch, 1)
        )
        self.activation = nn.Softmax(dim=1) if seg_ch>1 else nn.Sigmoid()
    def forward(self, x):
        identity = self.res_conv(x)
        x = self.depthwise(x)         # (B, mid_ch, H, W)
        weights = self.se(x)         # (B, mid_ch, 1, 1)
        x = x * weights              # 通道加权
        x = self.fusion(x)           # 多尺度融合
        out = x + identity
        return self.activation(out)


# 重建头
class EnhancedReconHead(nn.Module):
    # upscale_factor 上采样倍数（例如：将低分辨率图像放大2倍）
    def __init__(self, in_ch=64, recon_ch=3, upscale_factor=2):
        super().__init__()
        self.upscale = upscale_factor

        # 特征增强模块
        self.pre_upsample = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 4, 3, padding=1),  # 扩展通道供PixelShuffle使用
            nn.PReLU(),
            nn.BatchNorm2d(in_ch * 4)
        )

        # 多尺度残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_ch * 4, dilation=1),
            ResidualBlock(in_ch * 4, dilation=2),
            ResidualBlock(in_ch * 4, dilation=3)
        ])

        # 空间-通道协同注意力
        self.sc_attention = SCAttention(in_ch * 4)

        # 上采样核心
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        # 重建输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_ch * 4 // (upscale_factor ** 2), 64, 3, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, recon_ch, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 初始上采样准备
        x = self.pre_upsample(x)

        # 多尺度残差学习
        residual = x
        for block in self.res_blocks:
            x = block(x) + residual
            residual = x

        # 注意力增强
        x = self.sc_attention(x)

        # PixelShuffle上采样
        x = self.pixel_shuffle(x)

        # 最终重建
        return self.final_conv(x)


class ResidualBlock(nn.Module):
    """多尺度残差块"""

    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3,
                      padding=dilation, dilation=dilation),
            nn.InstanceNorm2d(channels),
            nn.PReLU(),

            nn.Conv2d(channels, channels, 3,
                      padding=dilation, dilation=dilation),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)


class SCAttention(nn.Module):
    """空间-通道协同注意力"""

    def __init__(self, in_ch):
        super().__init__()
        # 通道
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // 16, 1),
            nn.PReLU(),
            nn.Conv2d(in_ch // 16, in_ch, 1),
            nn.Sigmoid()
        )
        # 空间
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_ch, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_att = self.channel_att(x)
        spatial_att = self.spatial_att(x)
        return x * channel_att * spatial_att


class EllipseParamPredictor(nn.Module):
    """从分割结果预测椭圆参数（每个椭圆5个参数：中心x,y, 长轴, 短轴, 旋转角）"""

    def __init__(self, in_chs=2, hidden_dim=64):
        super().__init__()
        # 特征提取
        self.conv = nn.Sequential(
            nn.Conv2d(in_chs, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        # 参数回归
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 5)
        )
        # 参数范围约束
        self.sigmoid = nn.Sigmoid()

    def forward(self, seg_masks):
        """
        输入: seg_masks (B, 2, H, W) - 两个任务的分割结果
        输出: ellipse_params (B, 10) - 归一化的椭圆参数
        """
        features = self.conv(seg_masks)
        params = self.fc(features)

        # 参数归一化到[0,1]范围
        return self.sigmoid(params)

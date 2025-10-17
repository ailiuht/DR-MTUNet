import torch
from torch import nn


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
class UNetModel(torch.nn.Module):
    # init_features = 32 初始特征通道数
    # ↑ 增大：增强模型表达能力，但增加显存消耗
    # ↓ 减小：降低计算量，可能损失细节特征
    # 小数据集：16 - 32
    # 常规任务：32 - 64
    # 高分辨率图像：64 - 128

    def __init__(self, in_features=3, out_features=1, init_features=32):
        super(UNetModel, self).__init__()
        features = init_features

        self.encode_layer1 = torch.nn.Sequential(

            torch.nn.Conv2d(in_channels=in_features, out_channels=features, kernel_size=3, padding=1, stride=1),
            # 对每个通道的特征进行归一化（均值0，方差1），加速训练收敛。
            # 缓解梯度消失/爆炸问题，允许使用更大的学习率。
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU(),
            # 3x3, padding=1, stride=1	平衡计算量和特征保留	深层感受野有限	大多数分割任务（默认）
            # 5x5, padding=2, stride=1	单层感受野更大	参数量增加178%	小目标密集场景（细胞分割）
            # 3x3, padding=0, stride=2	显式下采样	边缘信息丢失	实时性要求高的任务
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU()
        )
        # 下采样,选取最大值
        # MaxPool	保留显著特征，抗噪声能力强	丢失位置细节
        # AvgPool	保留平滑信息	弱化高频特征（如边缘）
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encode_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features, out_channels=features*2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features*2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*2, out_channels=features*2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU()
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*2, out_channels=features*4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU()
        )
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.ReLU(),
        )
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)


        # 替换为dilated convolution（空洞卷积），扩大感受野。
        # 在ResNet等网络中，瓶颈层通常通过1x1卷积压缩通道数，减少计算量。而UNet的瓶颈层设计有所不同：
        #
        # 通道扩展：UNet的瓶颈层增加通道数（如256→512），目的是增强深层特征的表达能力。
        # 相似性：两者均位于网络中部，是信息流的关键节点，且对模型性能有决定性影响。
        self.encode_decode_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*16, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 16),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=features*16, out_channels=features*16, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 16),
            torch.nn.ReLU()
        )

        # 上采样方法，如转置卷积、双线性插值、PixelShuffle等
        # 方法	训练参数	计算速度	细节保留	适用场景
        # 转置卷积	有	中等	    高★	    高精度分割（医学图像）   插入零值
        # 双线性插值	无	快	    中	    实时推理或轻量化模型     基于相邻4个像素的加权平均插值，权重与距离成反比
        # 最近邻插值	无	最快	    低	    边缘敏感的二值分割       直接复制最近邻像素值，无权重计算。
        # PixelShuffle	有	    快	    高	超分辨率、嵌入式设备
        # 高精度需求：转置卷积或PixelShuffle。
        # 实时性优先：双线性插值或PixelShuffle。
        # 边缘锐度敏感：最近邻插值。
        # 减少伪影：PixelShuffle + 卷积组合

        self.upconv4 = torch.nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decode_layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*16, out_channels=features*8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features*8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.ReLU(),
        )
        self.upconv3 = torch.nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decode_layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            # 激活函数	    问题	为何ReLU更适配U-Net
            # Sigmoid	    梯度消失严重，输出非零中心，计算量大	不适用于深层网络
            # Tanh	        梯度消失（虽比Sigmoid轻），计算量稍大	效率低于ReLU
            # LeakyReLU	    负区间引入小梯度，缓解死亡神经元但增加参数（如斜率需手动设定）	ReLU+BN已足够，无需复杂设计
            # ELU	        计算复杂，负区间指数运算增加耗时	效率优先
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU()
        )
        self.upconv2 = torch.nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decode_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*2, out_channels=features*2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU()
        )
        self.upconv1 = torch.nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decode_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU()
        )
        self.out_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features, out_channels=64, kernel_size=1, padding=0, stride=1),
        )

        self.seg_head = EnhancedSegHead(in_ch=64, seg_ch=3)
        self.recon_head = EnhancedReconHead(in_ch=64, recon_ch=3)
    def forward(self, x):
        enc1 = self.encode_layer1(x)
        enc2 = self.encode_layer2(self.pool1(enc1))
        enc3 = self.encode_layer3(self.pool2(enc2))
        enc4 = self.encode_layer4(self.pool3(enc3))

        bottleneck = self.encode_decode_layer(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decode_layer4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decode_layer3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decode_layer2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decode_layer1(dec1)

        out = self.out_layer(dec1)
        seg = self.seg_head(out)
        recon = self.recon_head(out)
        return seg,recon

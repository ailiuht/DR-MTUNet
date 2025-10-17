import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ConvBlock, DynamicResidualGate, EllipseParamPredictor, EnhancedSegHead, EnhancedReconHead, \
    SCAttention


# ---------------------- 主干网络3 ----------------------
class DR_MTUNet(nn.Module):
    def __init__(self, in_ch=3, seg_ch=2, recon_ch=3, shape_dim=5):
        super().__init__()

        # 编码器
        self.enc1 = ConvBlock(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512)


        # 解码器
        self.dec4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # 辅助解码器
        self.decs4 = ConvBlock(1024, 512)
        self.ups3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decs3 = ConvBlock(512, 256)
        self.ups2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decs2 = ConvBlock(256, 128)
        self.ups1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decs1 = ConvBlock(128, 64)
        self.segs_head = EnhancedSegHead(in_ch=64, seg_ch=seg_ch)

        # 多任务头
        self.seg_head = EnhancedSegHead(in_ch=64, seg_ch=seg_ch)
        self.recon_head = EnhancedReconHead(in_ch=64, recon_ch=recon_ch)

        self._init_weights()
        self.bottleneck = nn.Sequential(
            ConvBlock(512, 1024),  # 扩展通道数
            nn.Conv2d(1024, 512, 1),  # 1x1卷积压缩通道
            nn.BatchNorm2d(512),
            nn.PReLU()
        )

        self.alpha = nn.Parameter(torch.zeros(1))
    def _init_weights(self):
        """综合初始化策略"""
        # 基础初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.25)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.25)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.25)
                if m.bias is not None: nn.init.zeros_(m.bias)

        # 特殊模块初始化
        for m in self.modules():
            # 注意力机制初始化
            if isinstance(m, (SCAttention, DynamicResidualGate)):
                for layer in m.modules():
                    if isinstance(layer, nn.Conv2d) and layer.out_channels == 1:
                        nn.init.zeros_(layer.weight)
                        if layer.bias is not None: nn.init.zeros_(layer.bias)

            # 分割头最后一层初始化
            if isinstance(m, EnhancedSegHead):
                nn.init.normal_(m.fusion[-1].weight, std=0.01)
                nn.init.zeros_(m.fusion[-1].bias)

            # 重建头最后一层初始化
            if isinstance(m, EnhancedReconHead):
                nn.init.normal_(m.final_conv[-2].weight, std=0.01)
                nn.init.zeros_(m.final_conv[-2].bias)

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)  # 64
        e2 = self.enc2(self.pool1(e1))  # 128
        e3 = self.enc3(self.pool2(e2))  # 256
        e4 = self.enc4(self.pool3(e3))  # 512

        bottleneck_out = self.bottleneck(e4)

        ds4 = torch.cat([e4, bottleneck_out], dim=1)
        ds4 = self.decs4(ds4)

        ds3 = self.ups3(ds4)
        ds3 = torch.cat([e3, ds3], dim=1)
        ds3 = self.decs3(ds3)

        ds2 = self.ups2(ds3)
        ds2 = torch.cat([e2, ds2], dim=1)
        ds2 = self.decs2(ds2)

        ds1 = self.ups1(ds2)
        ds1 = torch.cat([e1, ds1], dim=1)
        ds1 = self.decs1(ds1)

        segs = self.segs_head(ds1)

        # 动态门控跳跃连接
        d4 = torch.cat([e4, bottleneck_out], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        # 多任务输出
        seg = self.seg_head(d1)
        recon = self.recon_head(d1)

        alpha = torch.sigmoid(self.alpha)

        combined_seg = alpha * seg + (1 - alpha) * segs
        return combined_seg, recon, alpha

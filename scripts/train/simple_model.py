"""
Simple 3D CNN for Glacier Movement Prediction
Alternative to TimeSformer, better for small datasets
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3DBlock(nn.Module):
    """3D Convolutional block with batch norm and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResBlock3D(nn.Module):
    """3D Residual block"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv3DBlock(channels, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class SimpleGlacierNet(nn.Module):
    """
    Simple 3D CNN for glacier lake prediction
    Better suited for small datasets than TimeSformer
    """
    def __init__(self, in_channels=8, num_frames=6, img_size=128):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.img_size = img_size
        
        # Encoder
        self.encoder1 = Conv3DBlock(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.encoder2 = Conv3DBlock(32, 64, kernel_size=3, padding=1)
        self.res2 = ResBlock3D(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.encoder3 = Conv3DBlock(64, 128, kernel_size=3, padding=1)
        self.res3 = ResBlock3D(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.encoder4 = Conv3DBlock(128, 256, kernel_size=3, padding=1)
        self.res4 = ResBlock3D(256)
        
        # Bottleneck
        self.bottleneck = Conv3DBlock(256, 512, kernel_size=3, padding=1)
        
        # Decoder (2D after temporal pooling)
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        
        self.decoder4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.decoder3 = nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.decoder2 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.decoder1 = nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1)
        
        # Final output
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        # x: (B, T, C, H, W)
        # Rearrange to (B, C, T, H, W) for 3D convolution
        x = x.permute(0, 2, 1, 3, 4)
        
        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        
        e2 = self.encoder2(p1)
        e2 = self.res2(e2)
        p2 = self.pool2(e2)
        
        e3 = self.encoder3(p2)
        e3 = self.res3(e3)
        p3 = self.pool3(e3)
        
        e4 = self.encoder4(p3)
        e4 = self.res4(e4)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Temporal pooling
        b_2d = self.temporal_pool(b).squeeze(2)  # (B, 512, H, W)
        e3_2d = e3[:, :, -1, :, :]  # Use last temporal frame
        e2_2d = e2[:, :, -1, :, :]
        e1_2d = e1[:, :, -1, :, :]
        
        # Decoder
        d4 = F.relu(self.decoder4(b_2d))
        d4 = self.up4(d4)
        
        d3 = torch.cat([d4, e3_2d], dim=1)
        d3 = F.relu(self.decoder3(d3))
        d3 = self.up3(d3)
        
        d2 = torch.cat([d3, e2_2d], dim=1)
        d2 = F.relu(self.decoder2(d2))
        d2 = self.up2(d2)
        
        d1 = torch.cat([d2, e1_2d], dim=1)
        d1 = F.relu(self.decoder1(d1))
        
        # Final output
        output = self.final(d1)
        
        return output
    
    def get_num_params(self):
        """Count number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_simple_model(config):
    """Factory function to create simple CNN model"""
    model = SimpleGlacierNet(
        in_channels=8,
        num_frames=config.NUM_FRAMES,
        img_size=config.IMAGE_SIZE
    )
    
    print(f"Simple CNN model created with {model.get_num_params():,} parameters")
    return model

"""
Enhanced TimeSformer for Glacier Movement Prediction
Memory-optimized for 4GB GPU with divided space-time attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

class PatchEmbed(nn.Module):
    """Image to Patch Embedding with temporal dimension"""
    def __init__(self, img_size=128, patch_size=16, in_chans=8, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # Reshape to (B*T, C, H, W)
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        
        # Patch embedding
        x = self.proj(x)  # (B*T, embed_dim, H', W')
        x = rearrange(x, '(b t) c h w -> b t (h w) c', b=B, t=T)
        x = self.norm(x)
        
        return x

class TemporalAttention(nn.Module):
    """Temporal attention along time dimension"""
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        # x: (B, T, N, C) where N is num_patches
        B, T, N, C = x.shape
        
        # Rearrange to (B*N, T, C) for temporal attention
        x = rearrange(x, 'b t n c -> (b n) t c')
        
        qkv = self.qkv(x).reshape(B*N, T, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*N, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B*N, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Rearrange back
        x = rearrange(x, '(b n) t c -> b t n c', b=B, n=N)
        
        return x

class SpatialAttention(nn.Module):
    """Spatial attention along spatial dimension"""
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        # x: (B, T, N, C)
        B, T, N, C = x.shape
        
        # Rearrange to (B*T, N, C) for spatial attention
        x = rearrange(x, 'b t n c -> (b t) n c')
        
        qkv = self.qkv(x).reshape(B*T, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*T, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B*T, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Rearrange back
        x = rearrange(x, '(b t) n c -> b t n c', b=B, t=T)
        
        return x

class TransformerBlock(nn.Module):
    """Transformer block with divided space-time attention"""
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, qkv_bias=True, 
                 drop=0.1, attn_drop=0.1):
        super().__init__()
        
        # Temporal attention
        self.norm1 = nn.LayerNorm(dim)
        self.temporal_attn = TemporalAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        
        # Spatial attention
        self.norm2 = nn.LayerNorm(dim)
        self.spatial_attn = SpatialAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        
        # MLP
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x):
        # Temporal attention
        x = x + self.temporal_attn(self.norm1(x))
        
        # Spatial attention
        x = x + self.spatial_attn(self.norm2(x))
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        
        return x

class SegmentationHead(nn.Module):
    """Segmentation head to produce pixel-wise predictions"""
    def __init__(self, embed_dim=192, img_size=128, patch_size=16, out_channels=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_per_side = img_size // patch_size
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(16, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        # x: (B, T, N, C)
        B, T, N, C = x.shape
        
        # Use last temporal frame for prediction
        x = x[:, -1, :, :]  # (B, N, C)
        
        # Reshape to spatial dimensions
        H = W = self.num_patches_per_side
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Decode
        x = self.decoder(x)
        
        return x

class EnhancedTimeSformer(nn.Module):
    """
    Enhanced TimeSformer for Glacier Movement Prediction
    Optimized for 4GB GPU memory
    """
    def __init__(self, img_size=128, patch_size=16, in_chans=8, num_frames=6,
                 embed_dim=192, depth=4, num_heads=4, mlp_ratio=2.0,
                 qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Position embedding (spatial + temporal)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        # Segmentation head
        self.seg_head = SegmentationHead(embed_dim, img_size, patch_size, out_channels=1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_module_weights)
        
    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # x: (B, T, C, H, W)
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, T, N, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Segmentation head
        output = self.seg_head(x)  # (B, 1, H, W)
        
        return output
    
    def get_num_params(self):
        """Count number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_model(config):
    """Factory function to create model"""
    model = EnhancedTimeSformer(
        img_size=config.IMAGE_SIZE,
        patch_size=16,
        in_chans=8,  # 4 velocity channels + 4 static channels
        num_frames=config.NUM_FRAMES,
        embed_dim=192,  # Reduced from 768 for memory
        depth=4,  # Reduced from 12 for memory
        num_heads=4,  # Reduced from 12 for memory
        mlp_ratio=2.0,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1
    )
    
    print(f"Model created with {model.get_num_params():,} parameters")
    return model

"""
Weighted Feature Fusion Network
Based on Large Kernel Convolution and Transformer for Multi-Modal Remote Sensing Image Segmentation

This is the MAIN PROPOSED MODEL for your capstone project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class LargeKernelConv(nn.Module):
    """
    Large Kernel Convolution Block
    Uses large kernels (7x7, 11x11) to capture extensive spatial context
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, padding=3):
        super(LargeKernelConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Depthwise separable convolution for efficiency
        self.dw_conv = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                                 padding=padding, groups=out_channels, bias=False)
        self.pw_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        # Depthwise separable large kernel
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention for Transformer
    Captures global dependencies and long-range relationships
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block
    Combines Multi-Head Self-Attention with Feed-Forward Network
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                          attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class WeightedFeatureFusion(nn.Module):
    """
    Weighted Feature Fusion Module
    Adaptively fuses features from different modalities and branches
    """
    def __init__(self, in_channels, num_branches=2):
        super(WeightedFeatureFusion, self).__init__()
        self.num_branches = num_branches
        
        # Learnable fusion weights
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * num_branches, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels * num_branches, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        """
        features: list of feature maps from different branches
        """
        # Concatenate features
        concat_features = torch.cat(features, dim=1)
        
        # Calculate attention weights
        weights = self.attention(concat_features)
        
        # Split weights for each branch
        weights = torch.chunk(weights, self.num_branches, dim=1)
        
        # Weighted fusion
        fused = sum([f * w for f, w in zip(features, weights)])
        
        return fused


class MultiModalEncoder(nn.Module):
    """
    Multi-Modal Encoder for processing different data modalities
    Supports RGB, Infrared, and Multi-spectral inputs
    """
    def __init__(self, in_channels=3, base_channels=64):
        super(MultiModalEncoder, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder blocks with large kernels
        self.enc1 = LargeKernelConv(base_channels, base_channels * 2, kernel_size=7, padding=3)
        self.enc2 = LargeKernelConv(base_channels * 2, base_channels * 4, kernel_size=7, padding=3)
        self.enc3 = LargeKernelConv(base_channels * 4, base_channels * 8, kernel_size=11, padding=5)
        self.enc4 = LargeKernelConv(base_channels * 8, base_channels * 16, kernel_size=11, padding=5)
        
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x1 = self.conv1(x)
        
        x2 = self.pool(x1)
        x2 = self.enc1(x2)
        
        x3 = self.pool(x2)
        x3 = self.enc2(x3)
        
        x4 = self.pool(x3)
        x4 = self.enc3(x4)
        
        x5 = self.pool(x4)
        x5 = self.enc4(x5)
        
        return [x1, x2, x3, x4, x5]


class WeightedFeatureFusionNetwork(nn.Module):
    """
    Main Model: Weighted Feature Fusion Network
    Combines Large Kernel Convolutions with Transformer for Multi-Modal Segmentation
    """
    def __init__(self, in_channels=3, num_classes=7, base_channels=64, num_transformer_blocks=4):
        super(WeightedFeatureFusionNetwork, self).__init__()
        
        # Multi-modal encoder
        self.encoder = MultiModalEncoder(in_channels, base_channels)
        
        # Transformer layers for global context
        embed_dim = base_channels * 16
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=8, mlp_ratio=4.)
            for _ in range(num_transformer_blocks)
        ])
        
        # Position embedding for transformer
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, embed_dim))  # Assuming 14x14 feature map
        
        # Weighted fusion modules
        self.fusion1 = WeightedFeatureFusion(base_channels * 2, num_branches=2)
        self.fusion2 = WeightedFeatureFusion(base_channels * 4, num_branches=2)
        self.fusion3 = WeightedFeatureFusion(base_channels * 8, num_branches=2)
        
        # Decoder with skip connections
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final segmentation head
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        
        self._init_weights()
        
    def _init_weights(self):
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        x1, x2, x3, x4, x5 = features
        
        # Apply transformer on bottleneck features
        B, C, H, W = x5.shape
        x5_flat = x5.flatten(2).transpose(1, 2)  # B, HW, C
        
        # Add position embedding
        if x5_flat.shape[1] == self.pos_embed.shape[1]:
            x5_flat = x5_flat + self.pos_embed
        
        # Transformer blocks
        for blk in self.transformer_blocks:
            x5_flat = blk(x5_flat)
        
        # Reshape back
        x5 = x5_flat.transpose(1, 2).reshape(B, C, H, W)
        
        # Decoder with weighted fusion
        d4 = self.dec4(x5)
        d4 = self.fusion3([d4, x4])
        
        d3 = self.dec3(d4)
        d3 = self.fusion2([d3, x3])
        
        d2 = self.dec2(d3)
        d2 = self.fusion1([d2, x2])
        
        d1 = self.dec1(d2)
        
        # Final prediction
        out = self.final_conv(d1)
        
        return out


def get_model(num_classes=7, pretrained=False):
    """
    Factory function to create the model
    """
    model = WeightedFeatureFusionNetwork(
        in_channels=3,
        num_classes=num_classes,
        base_channels=64,
        num_transformer_blocks=4
    )
    
    return model


if __name__ == "__main__":
    # Test the model
    model = get_model(num_classes=7)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

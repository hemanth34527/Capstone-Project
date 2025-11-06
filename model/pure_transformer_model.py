"""
Pure Transformer Model for Remote Sensing Image Segmentation
Based on SETR (Segmentation Transformer) architecture
Uses only attention mechanisms without CNN components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them
    Similar to Vision Transformer (ViT)
    """
    def __init__(self, image_size=256, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x, H, W


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for transformer
    """
    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        return x + self.pos_embed


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism
    """
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention calculation
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation
    """
    def __init__(self, embed_dim=768, hidden_dim=3072, dropout=0.0):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with attention and feed-forward
    """
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout)
        
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of transformer blocks
    """
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super(TransformerEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


class DecoderHead(nn.Module):
    """
    Progressive upsampling decoder head
    Converts transformer features back to segmentation map
    """
    def __init__(self, embed_dim=768, num_classes=7, image_size=256, patch_size=16):
        super(DecoderHead, self).__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        
        # Progressive upsampling
        self.decode1 = nn.Sequential(
            nn.Conv2d(embed_dim, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.decode2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.decode3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.decode4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # Final segmentation head
        self.segmentation_head = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, x, H, W):
        # Reshape from (B, N, C) to (B, C, H, W)
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Progressive upsampling
        x = self.decode1(x)  # 2x
        x = self.decode2(x)  # 4x
        x = self.decode3(x)  # 8x
        x = self.decode4(x)  # 16x (original size)
        
        # Final segmentation
        x = self.segmentation_head(x)
        
        return x


class PureTransformerSegmentation(nn.Module):
    """
    Pure Transformer Model for Semantic Segmentation
    Based on SETR architecture - uses only attention mechanisms
    
    Architecture:
    1. Patch Embedding: Split image into patches
    2. Positional Encoding: Add position information
    3. Transformer Encoder: Stack of attention blocks
    4. Decoder Head: Progressive upsampling to segmentation map
    
    Key difference from hybrid models:
    - NO convolutional feature extraction
    - NO CNN backbone
    - ONLY attention mechanisms for feature learning
    """
    def __init__(
        self,
        image_size=256,
        patch_size=16,
        in_channels=3,
        num_classes=7,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super(PureTransformerSegmentation, self).__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Positional encoding
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = PositionalEncoding(num_patches, embed_dim)
        
        # Dropout after embedding
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Decoder head
        self.decoder = DecoderHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            image_size=image_size,
            patch_size=patch_size
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
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
        """
        Forward pass
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            out: Segmentation map (B, num_classes, H, W)
        """
        # Patch embedding
        x, H, W = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional encoding
        x = self.pos_embed(x)
        x = self.pos_drop(x)
        
        # Transformer encoding
        x = self.encoder(x)  # (B, num_patches, embed_dim)
        
        # Decode to segmentation map
        out = self.decoder(x, H, W)  # (B, num_classes, H, W)
        
        return out
    
    def get_num_params(self):
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Factory function for easy model creation
def get_pure_transformer(num_classes=7, image_size=256, model_size='base'):
    """
    Create Pure Transformer model with different sizes
    
    Args:
        num_classes: Number of segmentation classes
        image_size: Input image size
        model_size: 'tiny', 'small', 'base', or 'large'
    """
    configs = {
        'tiny': {
            'embed_dim': 384,
            'depth': 6,
            'num_heads': 6,
            'patch_size': 16
        },
        'small': {
            'embed_dim': 512,
            'depth': 8,
            'num_heads': 8,
            'patch_size': 16
        },
        'base': {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'patch_size': 16
        },
        'large': {
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'patch_size': 16
        }
    }
    
    config = configs.get(model_size, configs['base'])
    
    model = PureTransformerSegmentation(
        image_size=image_size,
        num_classes=num_classes,
        **config
    )
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing Pure Transformer Model...")
    
    # Create model
    model = get_pure_transformer(num_classes=7, image_size=256, model_size='base')
    print(f"Model created with {model.get_num_params() / 1e6:.2f}M parameters")
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    print(f"\nInput shape: {x.shape}")
    
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Verify output
    assert output.shape == (2, 7, 256, 256), "Output shape mismatch!"
    print("\n✓ Pure Transformer model test passed!")
    
    # Compare with different sizes
    print("\nModel size comparison:")
    for size in ['tiny', 'small', 'base', 'large']:
        m = get_pure_transformer(num_classes=7, model_size=size)
        params = m.get_num_params() / 1e6
        print(f"  {size.capitalize():6s}: {params:6.2f}M parameters")

"""
Generate Network Architecture Diagram for Weighted Fusion Model
Creates a professional visualization of the model architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

def create_architecture_diagram():
    """Create a comprehensive architecture diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define enhanced colors with gradients
    color_input = '#1e3a8a'  # Deep blue
    color_encoder = '#2563eb'  # Blue
    color_conv = '#3b82f6'  # Light blue
    color_transformer = '#f59e0b'  # Amber
    color_fusion = '#10b981'  # Emerald
    color_decoder = '#8b5cf6'  # Purple
    color_output = '#eab308'  # Yellow
    
    # Title with better styling
    title_box = FancyBboxPatch((0.5, 12.8), 19, 1, 
                               boxstyle="round,pad=0.2", 
                               facecolor='#1e293b', 
                               edgecolor='#3b82f6', linewidth=3)
    ax.add_patch(title_box)
    ax.text(10, 13.5, 'WEIGHTED FEATURE FUSION NETWORK', 
            fontsize=24, fontweight='bold', ha='center', va='center', color='white')
    ax.text(10, 13.1, 'Large Kernel Convolution + Transformer for Satellite Image Segmentation',
            fontsize=13, ha='center', va='center', style='italic', color='#94a3b8')
    
    # ===================== INPUT LAYER =====================
    input_box = FancyBboxPatch((7.5, 11.2), 5, 1.2, 
                               boxstyle="round,pad=0.15", 
                               facecolor=color_input, 
                               edgecolor='white', linewidth=3)
    ax.add_patch(input_box)
    ax.text(10, 11.95, 'INPUT IMAGE', fontsize=14, fontweight='bold', ha='center', va='center', color='white')
    ax.text(10, 11.55, 'Satellite Image (512×512×3)', fontsize=11, ha='center', va='center', style='italic', color='#bfdbfe')
    
    # Enhanced arrow down with shadow
    arrow1 = FancyArrowPatch((10, 11.2), (10, 10.5), 
                            arrowstyle='->', mutation_scale=30, 
                            linewidth=4, color='#1e293b',
                            connectionstyle="arc3,rad=0")
    ax.add_patch(arrow1)
    
    # ===================== ENCODER PATH =====================
    # Initial Conv with gradient effect
    enc1 = FancyBboxPatch((7.5, 9.6), 5, 0.9,
                          boxstyle="round,pad=0.1",
                          facecolor=color_encoder,
                          edgecolor='white', linewidth=2.5)
    ax.add_patch(enc1)
    ax.text(10, 10.15, 'INITIAL ENCODER', fontsize=12, fontweight='bold', ha='center', va='center', color='white')
    ax.text(10, 9.85, 'Conv3x3 + BatchNorm + ReLU', fontsize=9, ha='center', va='center', color='#bfdbfe')
    
    # Arrow down (split into two branches) - Enhanced
    ax.plot([10, 10], [9.6, 9.1], 'k-', linewidth=4)
    ax.plot([10, 5], [9.1, 9.1], 'k-', linewidth=4)
    ax.plot([10, 15], [9.1, 9.1], 'k-', linewidth=4)
    
    # Branch indicators
    ax.text(5, 9.3, '▼', fontsize=20, ha='center', va='center', color='#3b82f6')
    ax.text(15, 9.3, '▼', fontsize=20, ha='center', va='center', color='#f59e0b')
    
    # ===================== LEFT BRANCH: LARGE KERNEL CONVOLUTION =====================
    # Branch title box
    branch1_title = FancyBboxPatch((1, 8.5), 8, 0.6,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#1e40af',
                                   edgecolor='#60a5fa', linewidth=3)
    ax.add_patch(branch1_title)
    ax.text(5, 8.8, '🔷 LARGE KERNEL CONVOLUTION PATH', fontsize=12, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # Conv Block 1 - Enhanced
    conv1 = FancyBboxPatch((2, 7.2), 6, 1.1,
                           boxstyle="round,pad=0.1",
                           facecolor=color_conv,
                           edgecolor='white', linewidth=2.5)
    ax.add_patch(conv1)
    ax.text(5, 7.9, 'CONV BLOCK 1', fontsize=11, fontweight='bold', ha='center', va='center', color='white')
    ax.text(5, 7.6, 'Kernel: 7×7', fontsize=9, ha='center', va='center', color='white')
    ax.text(5, 7.35, 'Channels: 64 → 128', fontsize=9, ha='center', va='center', color='#dbeafe')
    
    # Arrow with label
    arrow_c1 = FancyArrowPatch((5, 7.2), (5, 6.6),
                              arrowstyle='->', mutation_scale=25,
                              linewidth=3.5, color='#1e40af')
    ax.add_patch(arrow_c1)
    
    # Conv Block 2 - Enhanced
    conv2 = FancyBboxPatch((2, 5.3), 6, 1.1,
                           boxstyle="round,pad=0.1",
                           facecolor=color_conv,
                           edgecolor='white', linewidth=2.5)
    ax.add_patch(conv2)
    ax.text(5, 6, 'CONV BLOCK 2', fontsize=11, fontweight='bold', ha='center', va='center', color='white')
    ax.text(5, 5.7, 'Kernel: 11×11 (Large)', fontsize=9, ha='center', va='center', color='white')
    ax.text(5, 5.45, 'Channels: 128 → 256', fontsize=9, ha='center', va='center', color='#dbeafe')
    
    # Arrow
    arrow_c2 = FancyArrowPatch((5, 5.3), (5, 4.7),
                              arrowstyle='->', mutation_scale=25,
                              linewidth=3.5, color='#1e40af')
    ax.add_patch(arrow_c2)
    
    # Conv Block 3 - Enhanced
    conv3 = FancyBboxPatch((2, 3.4), 6, 1.1,
                           boxstyle="round,pad=0.1",
                           facecolor=color_conv,
                           edgecolor='white', linewidth=2.5)
    ax.add_patch(conv3)
    ax.text(5, 4.1, 'CONV BLOCK 3', fontsize=11, fontweight='bold', ha='center', va='center', color='white')
    ax.text(5, 3.8, 'Kernel: 7×7 + ReLU', fontsize=9, ha='center', va='center', color='white')
    ax.text(5, 3.55, 'Channels: 256 → 512', fontsize=9, ha='center', va='center', color='#dbeafe')
    
    # ===================== RIGHT BRANCH: TRANSFORMER =====================
    # Branch title box
    branch2_title = FancyBboxPatch((11, 8.5), 8, 0.6,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#c2410c',
                                   edgecolor='#fb923c', linewidth=3)
    ax.add_patch(branch2_title)
    ax.text(15, 8.8, '🔶 TRANSFORMER ATTENTION PATH', fontsize=12, fontweight='bold',
            ha='center', va='center', color='white')
    
    # Patch Embedding - Enhanced
    trans1 = FancyBboxPatch((12, 7.2), 6, 1.1,
                            boxstyle="round,pad=0.1",
                            facecolor=color_transformer,
                            edgecolor='white', linewidth=2.5)
    ax.add_patch(trans1)
    ax.text(15, 7.9, 'PATCH EMBEDDING', fontsize=11, fontweight='bold', ha='center', va='center', color='white')
    ax.text(15, 7.6, 'Patch Size: 16×16', fontsize=9, ha='center', va='center', color='white')
    ax.text(15, 7.35, 'Embedding Dim: 512', fontsize=9, ha='center', va='center', color='#fef3c7')
    
    # Arrow
    arrow_t1 = FancyArrowPatch((15, 7.2), (15, 6.6),
                              arrowstyle='->', mutation_scale=25,
                              linewidth=3.5, color='#c2410c')
    ax.add_patch(arrow_t1)
    
    # Multi-Head Attention - Enhanced
    trans2 = FancyBboxPatch((12, 5.3), 6, 1.1,
                            boxstyle="round,pad=0.1",
                            facecolor=color_transformer,
                            edgecolor='white', linewidth=2.5)
    ax.add_patch(trans2)
    ax.text(15, 6, 'MULTI-HEAD ATTENTION', fontsize=11, fontweight='bold', ha='center', va='center', color='white')
    ax.text(15, 5.7, 'Heads: 8 | Dim: 512', fontsize=9, ha='center', va='center', color='white')
    ax.text(15, 5.45, 'Self-Attention Mechanism', fontsize=9, ha='center', va='center', color='#fef3c7')
    
    # Arrow
    arrow_t2 = FancyArrowPatch((15, 5.3), (15, 4.7),
                              arrowstyle='->', mutation_scale=25,
                              linewidth=3.5, color='#c2410c')
    ax.add_patch(arrow_t2)
    
    # Feed Forward Network - Enhanced
    trans3 = FancyBboxPatch((12, 3.4), 6, 1.1,
                            boxstyle="round,pad=0.1",
                            facecolor=color_transformer,
                            edgecolor='white', linewidth=2.5)
    ax.add_patch(trans3)
    ax.text(15, 4.1, 'FEED FORWARD NETWORK', fontsize=11, fontweight='bold', ha='center', va='center', color='white')
    ax.text(15, 3.8, 'FFN + LayerNorm', fontsize=9, ha='center', va='center', color='white')
    ax.text(15, 3.55, 'Hidden: 2048 → 512', fontsize=9, ha='center', va='center', color='#fef3c7')
    
    # ===================== FUSION LAYER =====================
    # Arrows to fusion - Enhanced with curves
    arrow_to_fus1 = FancyArrowPatch((5, 3.4), (10, 2.5),
                                    arrowstyle='->', mutation_scale=30,
                                    linewidth=4, color='#1e40af',
                                    connectionstyle="arc3,rad=.3")
    ax.add_patch(arrow_to_fus1)
    
    arrow_to_fus2 = FancyArrowPatch((15, 3.4), (10, 2.5),
                                    arrowstyle='->', mutation_scale=30,
                                    linewidth=4, color='#c2410c',
                                    connectionstyle="arc3,rad=-.3")
    ax.add_patch(arrow_to_fus2)
    
    # Fusion box - Highlighted
    fusion_outer = FancyBboxPatch((6.5, 1.5), 7, 1,
                                  boxstyle="round,pad=0.15",
                                  facecolor='#065f46',
                                  edgecolor='#34d399', linewidth=4)
    ax.add_patch(fusion_outer)
    
    fusion = FancyBboxPatch((6.7, 1.6), 6.6, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor=color_fusion,
                            edgecolor='white', linewidth=2)
    ax.add_patch(fusion)
    ax.text(10, 2.2, '⚡ WEIGHTED FEATURE FUSION', fontsize=13, fontweight='bold', ha='center', va='center', color='white')
    ax.text(10, 1.85, 'F_fused = α × F_conv + (1-α) × F_trans', fontsize=10, ha='center', va='center', 
            style='italic', color='#d1fae5', family='monospace')
    
    # Weight visualization - Enhanced
    weight_alpha = FancyBboxPatch((5.2, 2.3), 1, 0.5,
                                  boxstyle="round,pad=0.05",
                                  facecolor='#fbbf24',
                                  edgecolor='white', linewidth=2)
    ax.add_patch(weight_alpha)
    ax.text(5.7, 2.55, 'α', fontsize=14, fontweight='bold', ha='center', va='center', color='white')
    
    weight_beta = FancyBboxPatch((13.8, 2.3), 1, 0.5,
                                boxstyle="round,pad=0.05",
                                facecolor='#fbbf24',
                                edgecolor='white', linewidth=2)
    ax.add_patch(weight_beta)
    ax.text(14.3, 2.55, '1-α', fontsize=13, fontweight='bold', ha='center', va='center', color='white')
    
    # Arrow down from fusion
    arrow_fus = FancyArrowPatch((10, 1.5), (10, 0.9),
                                arrowstyle='->', mutation_scale=30,
                                linewidth=4, color='#065f46')
    ax.add_patch(arrow_fus)
    
    # ===================== DECODER PATH =====================
    # Decoder Block 1 - Enhanced
    dec1 = FancyBboxPatch((7, 0), 6, 0.8,
                          boxstyle="round,pad=0.1",
                          facecolor=color_decoder,
                          edgecolor='white', linewidth=2.5)
    ax.add_patch(dec1)
    ax.text(10, 0.5, 'DECODER + UPSAMPLING', fontsize=11, fontweight='bold', ha='center', va='center', color='white')
    ax.text(10, 0.2, 'Progressive Reconstruction + Skip Connections', fontsize=9, ha='center', va='center', color='#e9d5ff')
    
    # Arrow down
    arrow_dec1 = FancyArrowPatch((10, 0), (10, -0.6),
                                 arrowstyle='->', mutation_scale=30,
                                 linewidth=4, color='#6d28d9')
    ax.add_patch(arrow_dec1)
    
    # ===================== OUTPUT LAYER =====================
    # Output box - Enhanced
    output_box = FancyBboxPatch((6, -1.8), 8, 1.1,
                                boxstyle="round,pad=0.15",
                                facecolor=color_output,
                                edgecolor='white', linewidth=3)
    ax.add_patch(output_box)
    ax.text(10, -1.05, 'SEGMENTATION OUTPUT', fontsize=13, fontweight='bold', ha='center', va='center', color='#422006')
    ax.text(10, -1.35, 'Softmax → 7 Land Cover Classes', fontsize=10, ha='center', va='center', color='#78350f')
    ax.text(10, -1.6, '(Urban, Agriculture, Rangeland, Forest, Water, Barren, Unknown)', 
            fontsize=8, ha='center', va='center', style='italic', color='#92400e')
    
    # ===================== INFORMATION PANELS =====================
    # Left panel - Model Architecture
    info_left = FancyBboxPatch((0.3, -3.8), 6, 1.8,
                               boxstyle="round,pad=0.15",
                               facecolor='#f8fafc',
                               edgecolor='#3b82f6', linewidth=2)
    ax.add_patch(info_left)
    ax.text(3.3, -2.2, '🏗️ MODEL ARCHITECTURE', fontsize=11, fontweight='bold', ha='center', color='#1e40af')
    ax.text(0.8, -2.6, '• Total Parameters: 136M', fontsize=9, ha='left', color='#334155')
    ax.text(0.8, -2.9, '• Input Resolution: 512×512 RGB', fontsize=9, ha='left', color='#334155')
    ax.text(0.8, -3.2, '• Output Classes: 7 Categories', fontsize=9, ha='left', color='#334155')
    ax.text(0.8, -3.5, '• Architecture: Dual-Path Fusion', fontsize=9, ha='left', color='#334155')
    
    # Center panel - Training Configuration
    info_center = FancyBboxPatch((7, -3.8), 6, 1.8,
                                 boxstyle="round,pad=0.15",
                                 facecolor='#f8fafc',
                                 edgecolor='#10b981', linewidth=2)
    ax.add_patch(info_center)
    ax.text(10, -2.2, '⚙️ TRAINING CONFIG', fontsize=11, fontweight='bold', ha='center', color='#065f46')
    ax.text(7.5, -2.6, '• Dataset: DeepGlobe (803 images)', fontsize=9, ha='left', color='#334155')
    ax.text(7.5, -2.9, '• Loss: CE + Dice (50:50)', fontsize=9, ha='left', color='#334155')
    ax.text(7.5, -3.2, '• Optimizer: Adam (lr=1e-4)', fontsize=9, ha='left', color='#334155')
    ax.text(7.5, -3.5, '• Batch Size: 4 | Epochs: 5', fontsize=9, ha='left', color='#334155')
    
    # Right panel - Performance Metrics
    info_right = FancyBboxPatch((13.7, -3.8), 6, 1.8,
                                boxstyle="round,pad=0.15",
                                facecolor='#f8fafc',
                                edgecolor='#f59e0b', linewidth=2)
    ax.add_patch(info_right)
    ax.text(16.7, -2.2, '📊 PERFORMANCE', fontsize=11, fontweight='bold', ha='center', color='#92400e')
    ax.text(14.2, -2.6, '• Pixel Accuracy: 66.86%', fontsize=9, ha='left', color='#334155', weight='bold')
    ax.text(14.2, -2.9, '• Mean IoU: 19.50%', fontsize=9, ha='left', color='#334155', weight='bold')
    ax.text(14.2, -3.2, '• Best Class: Water (68.2%)', fontsize=9, ha='left', color='#334155')
    ax.text(14.2, -3.5, '• Speed: ~3 sec/batch', fontsize=9, ha='left', color='#334155')
    
    # Add decorative elements
    # Corner decorations
    corner1 = FancyBboxPatch((0.2, 12.7), 0.4, 0.4, boxstyle="round,pad=0.05",
                            facecolor='#3b82f6', edgecolor='none')
    ax.add_patch(corner1)
    corner2 = FancyBboxPatch((19.4, 12.7), 0.4, 0.4, boxstyle="round,pad=0.05",
                            facecolor='#3b82f6', edgecolor='none')
    ax.add_patch(corner2)
    
    plt.tight_layout()
    return fig

def main():
    """Generate and save the architecture diagram"""
    print("Generating Network Architecture Diagram...")
    
    fig = create_architecture_diagram()
    
    # Save with high resolution
    output_path = 'architecture_diagram.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    
    plt.close()
    print("\n============================================================")
    print("✅ Architecture diagram generated successfully!")
    print("============================================================")
    print(f"\nLocation: {output_path}")
    print("\n✨ Ready to add to your website!")

if __name__ == "__main__":
    main()

"""Training visualization tools.

Creates visualizations for:
- Model I/O structure diagrams
- Training loss curves
"""

import matplotlib.pyplot as plt
from typing import List, Optional


def plot_model_io_structure(save_path: str):
    """Plot model input/output structure diagram.
    
    Args:
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 9.5, 'LSTM Frequency Filter: Input/Output Structure', 
            ha='center', va='top', fontsize=18, fontweight='bold')
    
    input_box = plt.Rectangle((0.5, 6), 2, 2, facecolor='#E3F2FD', 
                              edgecolor='#1976D2', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 7.5, 'INPUT', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='#1976D2')
    ax.text(1.5, 7.0, '[S(t), C₁, C₂, C₃, C₄]', ha='center', va='center', fontsize=10)
    ax.text(1.5, 6.5, 'Dimension: 5', ha='center', va='center', fontsize=9, style='italic')
    
    lstm_box = plt.Rectangle((3.5, 6), 3, 2, facecolor='#FFF3E0', 
                             edgecolor='#F57C00', linewidth=2)
    ax.add_patch(lstm_box)
    ax.text(5, 7.5, 'LSTM NETWORK', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='#F57C00')
    ax.text(5, 7.0, 'Hidden State: (h_t, c_t)', ha='center', va='center', fontsize=10)
    ax.text(5, 6.5, 'State preserved between samples', ha='center', va='center', 
           fontsize=8, style='italic')
    
    output_box = plt.Rectangle((7.5, 6), 2, 2, facecolor='#E8F5E9', 
                               edgecolor='#388E3C', linewidth=2)
    ax.add_patch(output_box)
    ax.text(8.5, 7.5, 'OUTPUT', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='#388E3C')
    ax.text(8.5, 7.0, 'Pure Frequency', ha='center', va='center', fontsize=10)
    ax.text(8.5, 6.5, 'Dimension: 1', ha='center', va='center', fontsize=9, style='italic')
    
    ax.annotate('', xy=(3.5, 7), xytext=(2.5, 7), 
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(7.5, 7), xytext=(6.5, 7), 
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    detail1 = plt.Rectangle((0.5, 3.5), 4, 2, facecolor='#F5F5F5', 
                           edgecolor='gray', linewidth=1, linestyle='--')
    ax.add_patch(detail1)
    ax.text(2.5, 5.2, 'Input Components:', ha='center', va='top', 
           fontsize=11, fontweight='bold')
    ax.text(2.5, 4.8, '• S(t): Mixed noisy signal sample', ha='center', va='top', fontsize=9)
    ax.text(2.5, 4.5, '• C: One-hot selection vector', ha='center', va='top', fontsize=9)
    ax.text(2.5, 4.2, '  [1,0,0,0] → Extract f₁=1Hz', ha='center', va='top', fontsize=9)
    ax.text(2.5, 3.9, '  [0,1,0,0] → Extract f₂=3Hz', ha='center', va='top', fontsize=9)
    
    detail2 = plt.Rectangle((5.5, 3.5), 4, 2, facecolor='#F5F5F5', 
                           edgecolor='gray', linewidth=1, linestyle='--')
    ax.add_patch(detail2)
    ax.text(7.5, 5.2, 'Processing:', ha='center', va='top', 
           fontsize=11, fontweight='bold')
    ax.text(7.5, 4.8, '• Conditional regression', ha='center', va='top', fontsize=9)
    ax.text(7.5, 4.5, '• L=1: Sequence length of 1', ha='center', va='top', fontsize=9)
    ax.text(7.5, 4.2, '• State management critical', ha='center', va='top', fontsize=9)
    ax.text(7.5, 3.9, '• Learns frequency structure', ha='center', va='top', fontsize=9)
    
    example_box = plt.Rectangle((1, 0.5), 8, 2.5, facecolor='#FFF9C4', 
                               edgecolor='#F9A825', linewidth=2)
    ax.add_patch(example_box)
    ax.text(5, 2.7, 'Example Data Flow', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='#F9A825')
    ax.text(5, 2.3, 'Sample at t=0.000s, extracting f₂=3Hz:', ha='center', va='center', fontsize=10)
    ax.text(5, 1.9, '[0.8124, 0, 1, 0, 0] → LSTM → 0.0000', ha='center', va='center', 
           fontsize=10, family='monospace')
    ax.text(5, 1.5, 'Sample at t=0.001s, extracting f₂=3Hz:', ha='center', va='center', fontsize=10)
    ax.text(5, 1.1, '[0.7932, 0, 1, 0, 0] → LSTM → 0.0188', ha='center', va='center', 
           fontsize=10, family='monospace')
    ax.text(5, 0.7, '(Hidden state preserved between consecutive samples)', 
           ha='center', va='center', fontsize=8, style='italic', color='#F57C00')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_loss(
    train_losses: List[float],
    val_losses: Optional[List[float]],
    save_path: str
):
    """Plot training loss curve.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: Optional list of validation losses
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    
    if val_losses and len(val_losses) > 0:
        ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Training Progress: Loss vs Epoch', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    final_train_loss = train_losses[-1]
    ax.annotate(f'Final: {final_train_loss:.6f}', 
               xy=(len(train_losses), final_train_loss),
               xytext=(len(train_losses) * 0.7, final_train_loss * 1.2),
               arrowprops=dict(arrowstyle='->', color='blue'),
               fontsize=10, color='blue', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


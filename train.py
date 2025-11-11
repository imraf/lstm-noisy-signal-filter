"""Main training script for LSTM Frequency Filter.

Orchestrates the complete training and visualization pipeline.
"""

import torch
import json
from pathlib import Path
from datetime import datetime

from src.pipeline import execute_training_pipeline, generate_all_visualizations


def main():
    """Main execution function."""
    
    print("=" * 80)
    print("LSTM Frequency Filter - Training Pipeline")
    print("=" * 80)
    
    TRAIN_SEED = 11
    TEST_SEED = 42
    FREQUENCIES = [1.0, 3.0, 5.0, 7.0]
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Using device: {device}")
    
    results = execute_training_pipeline(
        train_seed=TRAIN_SEED,
        test_seed=TEST_SEED,
        frequencies=FREQUENCIES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        device=device,
        save_dir=output_dir,
        verbose=True
    )
    
    generate_all_visualizations(
        train_seed=TRAIN_SEED,
        test_seed=TEST_SEED,
        frequencies=FREQUENCIES,
        results=results,
        save_dir=output_dir,
        verbose=True
    )
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'train_seed': TRAIN_SEED,
            'test_seed': TEST_SEED,
            'frequencies': FREQUENCIES,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS
        },
        'metrics': {
            'train_mse': float(results['train_mse']),
            'test_mse': float(results['test_mse']),
            'generalization_gap': float(abs(results['test_mse'] - results['train_mse'])),
            'generalizes_well': abs(results['test_mse'] - results['train_mse']) < 0.01
        },
        'per_frequency': {
            'train': {f'f{i+1}_{freq}Hz': results['train_freq_metrics'][i] 
                     for i, freq in enumerate(FREQUENCIES)},
            'test': {f'f{i+1}_{freq}Hz': results['test_freq_metrics'][i] 
                    for i, freq in enumerate(FREQUENCIES)}
        }
    }
    
    results_file = output_dir / "results_summary.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n[SUMMARY]")
    print(f"  - Trained for {NUM_EPOCHS} epochs")
    print(f"  - Final Train MSE: {results['train_mse']:.8f}")
    print(f"  - Final Test MSE: {results['test_mse']:.8f}")
    print(f"  - Generated 14 visualization plots")
    print(f"  - Model saved to: {output_dir}/models")
    print(f"  - Visualizations saved to: {output_dir}/visualizations")
    print(f"  - Results saved to: {results_file}")
    print(f"\n[NEXT STEPS]")
    print(f"  1. Review visualizations in {output_dir}/visualizations")
    print(f"  2. Check results_summary.json for detailed metrics")
    print(f"  3. Run tests: pytest tests/ --cov=src")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

"""Main training script for LSTM Frequency Filter.

Orchestrates the complete training and visualization pipeline.
"""

import argparse
import torch
import json
import yaml
from pathlib import Path
from datetime import datetime

from src.pipeline import execute_training_pipeline, generate_all_visualizations


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train LSTM Frequency Filter',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--train-seed', type=int, default=11,
                        help='Random seed for training data')
    parser.add_argument('--test-seed', type=int, default=42,
                        help='Random seed for test data')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory for outputs')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use for training')
    parser.add_argument('--no-visualization', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--frequencies', type=float, nargs='+', default=[1.0, 3.0, 5.0, 7.0],
                        help='List of frequencies to extract')
    
    return parser.parse_args()


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_config_with_args(config: dict, args: argparse.Namespace):
    """Merge CLI arguments with config file, prioritizing CLI args."""
    if args.epochs is not None:
        config.setdefault('training', {})['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config.setdefault('training', {})['learning_rate'] = args.learning_rate
    if args.hidden_size is not None:
        config.setdefault('model', {})['hidden_size'] = args.hidden_size
    if args.num_layers is not None:
        config.setdefault('model', {})['num_layers'] = args.num_layers
    if args.train_seed is not None:
        config.setdefault('data', {})['train_seed'] = args.train_seed
    if args.test_seed is not None:
        config.setdefault('data', {})['test_seed'] = args.test_seed
    if args.frequencies is not None:
        config.setdefault('data', {})['frequencies'] = args.frequencies
    
    return config


def get_device(device_arg: str):
    """Get device based on argument and availability."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_arg)


def main():
    """Main execution function."""
    args = parse_args()
    
    print("=" * 80)
    print("LSTM Frequency Filter - Training Pipeline")
    print("=" * 80)
    
    if args.config:
        print(f"\n[INFO] Loading configuration from: {args.config}")
        config = load_config(args.config)
        config = merge_config_with_args(config, args)
        
        TRAIN_SEED = config.get('data', {}).get('train_seed', 11)
        TEST_SEED = config.get('data', {}).get('test_seed', 42)
        FREQUENCIES = config.get('data', {}).get('frequencies', [1.0, 3.0, 5.0, 7.0])
        HIDDEN_SIZE = config.get('model', {}).get('hidden_size', 64)
        NUM_LAYERS = config.get('model', {}).get('num_layers', 2)
        LEARNING_RATE = config.get('training', {}).get('learning_rate', 0.001)
        BATCH_SIZE = config.get('training', {}).get('batch_size', 32)
        NUM_EPOCHS = config.get('training', {}).get('num_epochs', 100)
        output_dir = Path(config.get('output', {}).get('base_dir', args.output_dir))
    else:
        print("\n[INFO] Using command-line arguments (no config file)")
        TRAIN_SEED = args.train_seed
        TEST_SEED = args.test_seed
        FREQUENCIES = args.frequencies
        HIDDEN_SIZE = args.hidden_size
        NUM_LAYERS = args.num_layers
        LEARNING_RATE = args.learning_rate
        BATCH_SIZE = args.batch_size
        NUM_EPOCHS = args.epochs
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device(args.device)
    print(f"\n[INFO] Using device: {device}")
    
    print(f"[INFO] Configuration:")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Hidden size: {HIDDEN_SIZE}")
    print(f"  - Number of layers: {NUM_LAYERS}")
    print(f"  - Frequencies: {FREQUENCIES}")
    print(f"  - Train seed: {TRAIN_SEED}")
    print(f"  - Test seed: {TEST_SEED}")
    
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
    
    if not args.no_visualization:
        generate_all_visualizations(
            train_seed=TRAIN_SEED,
            test_seed=TEST_SEED,
            frequencies=FREQUENCIES,
            results=results,
            save_dir=output_dir,
            verbose=True
        )
    else:
        print("\n[INFO] Skipping visualization generation (--no-visualization flag)")
        print(f"  - To generate visualizations later, run without --no-visualization")
    
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
    if not args.no_visualization:
        print(f"  - Generated 14 visualization plots")
    print(f"  - Model saved to: {output_dir}/models")
    if not args.no_visualization:
        print(f"  - Visualizations saved to: {output_dir}/visualizations")
    print(f"  - Results saved to: {results_file}")
    print(f"\n[NEXT STEPS]")
    if not args.no_visualization:
        print(f"  1. Review visualizations in {output_dir}/visualizations")
        print(f"  2. Check results_summary.json for detailed metrics")
        print(f"  3. Run tests: pytest tests/ --cov=src")
    else:
        print(f"  1. Check results_summary.json for detailed metrics")
        print(f"  2. Run tests: pytest tests/ --cov=src")
        print(f"  3. Generate visualizations: python train.py --config {args.config or 'config/default.yaml'}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

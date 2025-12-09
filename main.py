#!/usr/bin/env python3
"""
Main entry point for sentiment classification experiments.

Usage:
    python main.py --model [model_type] --num_iters [num_iters] --lr [lr] \
                   --num_repeat [num_repeat] --shuffle [bool]

Supported model types:
    - bow: Bag of Words
    - cbow: Continuous Bag of Words
    - deepcbow: Deep Continuous Bag of Words
    - lstm: LSTM classifier (single example)
    - lstm_batched: LSTM classifier (batched, recommended)
    - treelstm: Tree-LSTM classifier
    - treelstm_each_node: Tree-LSTM with node-level supervision
"""

import argparse
import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from exp_program.data import (
        load_data, build_vocabulary, load_pretrained_embeddings,
        NUM_CLASSES, download_and_extract_sst, download_embeddings
    )
    from exp_program.train import (
        ExperimentConfig, run_experiment, run_multiple_experiments,
        set_seed, get_device
    )
    from exp_program.visualize import (
        plot_training_curves, plot_comparison, plot_results_summary,
        save_results_table
    )
except ImportError:
    from data import (
        load_data, build_vocabulary, load_pretrained_embeddings,
        NUM_CLASSES, download_and_extract_sst, download_embeddings
    )
    from train import (
        ExperimentConfig, run_experiment, run_multiple_experiments,
        set_seed, get_device
    )
    from visualize import (
        plot_training_curves, plot_comparison, plot_results_summary,
        save_results_table
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sentiment classification experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--model", type=str, default="lstm_batched",
        choices=["bow", "cbow", "deepcbow", "lstm", "lstm_batched", 
                 "treelstm", "treelstm_each_node"],
        help="Model type to train"
    )
    
    # Training configuration
    parser.add_argument(
        "--num_iters", type=int, default=30000,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=25,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_repeat", type=int, default=1,
        help="Number of repeated experiments with different seeds"
    )
    
    # Data configuration
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Directory containing the SST data (auto-downloaded if not specified)"
    )
    parser.add_argument(
        "--embedding_file", type=str, default=None,
        help="Path to pre-trained word embeddings (auto-downloaded if not specified)"
    )
    
    # Model hyperparameters
    parser.add_argument(
        "--embedding_dim", type=int, default=300,
        help="Embedding dimension"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=168,
        help="Hidden layer dimension"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5,
        help="Dropout probability"
    )
    
    # Experiment options
    parser.add_argument(
        "--shuffle", action="store_true", default=False,
        help="Shuffle words in each example (to test word order importance)"
    )
    parser.add_argument(
        "--no_pretrained", action="store_true", default=False,
        help="Do not use pre-trained embeddings"
    )
    parser.add_argument(
        "--finetune", action="store_true", default=False,
        help="Fine-tune pre-trained embeddings"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--no_plot", action="store_true", default=False,
        help="Do not generate plots"
    )
    
    # Logging configuration
    parser.add_argument(
        "--use_tensorboard", action="store_true", default=False,
        help="Enable TensorBoard logging"
    )
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="Name for this run (used in tensorboard)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Sentiment Classification Experiment")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Iterations: {args.num_iters}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Repeats: {args.num_repeat}")
    print(f"Shuffle words: {args.shuffle}")
    print(f"Use pretrained: {not args.no_pretrained}")
    print(f"Finetune embeddings: {args.finetune}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Auto-download data if not specified
    data_dir = args.data_dir
    if data_dir is None:
        print("\nNo data directory specified. Downloading SST data...")
        data_dir = download_and_extract_sst("data")
    
    # Auto-download embeddings if not specified (and model uses them)
    embedding_file = args.embedding_file
    if embedding_file is None and args.model not in ["bow"]:
        print("\nNo embedding file specified. Downloading GloVe embeddings...")
        embedding_file = download_embeddings("glove", "data")
    
    # Load data
    print("\nLoading data...")
    train_data, dev_data, test_data = load_data(data_dir)
    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")
    
    # Load embeddings or build vocabulary
    vectors = None
    if not args.no_pretrained and embedding_file and os.path.exists(embedding_file):
        print(f"\nLoading pre-trained embeddings from {embedding_file}...")
        vocab, vectors = load_pretrained_embeddings(embedding_file, args.embedding_dim)
        print(f"Vocabulary size: {len(vocab)}")
    else:
        print("\nBuilding vocabulary from training data...")
        vocab = build_vocabulary(train_data)
        print(f"Vocabulary size: {len(vocab)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize TensorBoard if requested
    tb_writer = None
    if args.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            run_name = args.run_name or f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tb_log_dir = os.path.join(args.output_dir, "tensorboard", run_name)
            tb_writer = SummaryWriter(log_dir=tb_log_dir)
            print(f"\nTensorBoard logging enabled: {tb_log_dir}")
        except ImportError:
            print("\nWarning: tensorboard not installed. Skipping TensorBoard logging.")
            print("Install with: pip install tensorboard")
    
    # Create experiment config
    config = ExperimentConfig(
        model_type=args.model,
        num_iters=args.num_iters,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        shuffle_words=args.shuffle,
        use_pretrained=not args.no_pretrained,
        finetune_embeddings=args.finetune,
        seed=args.seed
    )
    
    # Run experiments
    if args.num_repeat > 1:
        results = run_multiple_experiments(
            config=config,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            vocab=vocab,
            vectors=vectors,
            num_repeats=args.num_repeat,
            output_dir=args.output_dir,
            tb_writer=tb_writer
        )
        
        # Save summary
        summary = {
            "model_type": args.model,
            "num_repeats": args.num_repeat,
            "test_accuracies": [r.metrics.test_accuracy for r in results],
            "mean_test_accuracy": np.mean([r.metrics.test_accuracy for r in results]),
            "std_test_accuracy": np.std([r.metrics.test_accuracy for r in results]),
            "config": config.to_dict()
        }
        
        summary_path = os.path.join(args.output_dir, f"{args.model}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_path}")
        
    else:
        result = run_experiment(
            config=config,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            vocab=vocab,
            vectors=vectors,
            output_dir=args.output_dir,
            tb_writer=tb_writer
        )
        results = [result]
    
    # Close TensorBoard writer
    if tb_writer is not None:
        tb_writer.close()
    
    # Generate plots
    if not args.no_plot:
        print("\nGenerating plots...")
        for i, result in enumerate(results):
            plot_path = os.path.join(args.output_dir, f"{args.model}_run{i+1}_curves.png")
            plot_training_curves(
                losses=result.metrics.train_losses,
                accuracies=result.metrics.dev_accuracies,
                title=f"{args.model} (Run {i+1})",
                save_path=plot_path
            )
            print(f"Training curves saved to {plot_path}")
    
    print("\n" + "=" * 60)
    print("Experiment completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

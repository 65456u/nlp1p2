"""
Visualization utilities for experiment results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from pathlib import Path


def plot_training_curves(
    losses: List[float],
    accuracies: List[float],
    title: str = "Training Curves",
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot training loss and validation accuracy curves.
    
    Args:
        losses: list of training losses
        accuracies: list of validation accuracies
        title: plot title
        save_path: path to save the figure (optional)
        show: whether to display the plot
    
    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(losses, 'b-', linewidth=1.5)
    ax1.set_xlabel('Print Step')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(accuracies, 'g-', linewidth=1.5)
    ax2.set_xlabel('Evaluation Step')
    ax2.set_ylabel('Dev Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_comparison(
    results: Dict[str, Dict],
    metric: str = 'test_accuracy',
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot comparison of multiple models.
    
    Args:
        results: dict mapping model name to results dict with mean and std
        metric: metric to compare
        title: plot title
        save_path: path to save the figure
        show: whether to display the plot
    
    Returns:
        matplotlib Figure
    """
    models = list(results.keys())
    means = [results[m].get('mean', results[m].get(metric, 0)) for m in models]
    stds = [results[m].get('std', 0) for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', edgecolor='black')
    
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.3f}±{std:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_accuracy_by_length(
    results: Dict[str, List[Tuple[int, float]]],
    title: str = "Accuracy by Sentence Length",
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot accuracy as a function of sentence length for different models.
    
    Args:
        results: dict mapping model name to list of (length, accuracy) tuples
        title: plot title
        save_path: path to save the figure
        show: whether to display the plot
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (model, data), color in zip(results.items(), colors):
        lengths, accs = zip(*sorted(data))
        ax.plot(lengths, accs, 'o-', label=model, color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Sentence Length')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_multiple_runs(
    runs: List[Dict],
    model_name: str,
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot training curves for multiple runs of the same model.
    
    Args:
        runs: list of result dicts, each with 'train_losses' and 'dev_accuracies'
        model_name: name of the model
        save_path: path to save the figure
        show: whether to display the plot
    
    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    
    for i, (run, color) in enumerate(zip(runs, colors)):
        losses = run.get('train_losses', [])
        accs = run.get('dev_accuracies', [])
        
        ax1.plot(losses, color=color, alpha=0.7, label=f'Run {i+1}')
        ax2.plot(accs, color=color, alpha=0.7, label=f'Run {i+1}')
    
    ax1.set_xlabel('Print Step')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Evaluation Step')
    ax2.set_ylabel('Dev Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.suptitle(f'{model_name} - Multiple Runs', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_results_summary(
    results_dir: str,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Load all results from a directory and create a summary plot.
    
    Args:
        results_dir: directory containing result JSON files
        save_path: path to save the figure
        show: whether to display the plot
    
    Returns:
        matplotlib Figure
    """
    # Find all summary files
    summary_files = list(Path(results_dir).glob("*_summary.json"))
    
    if not summary_files:
        # Try to find individual result files
        result_files = list(Path(results_dir).glob("*_result.json"))
        if not result_files:
            print(f"No result files found in {results_dir}")
            return None
        
        # Aggregate results by model type
        model_results = {}
        for f in result_files:
            with open(f, 'r') as fp:
                data = json.load(fp)
            model = data['config']['model_type']
            if model not in model_results:
                model_results[model] = []
            model_results[model].append(data['metrics']['test_accuracy'])
        
        results = {}
        for model, accs in model_results.items():
            results[model] = {
                'mean': np.mean(accs),
                'std': np.std(accs)
            }
    else:
        results = {}
        for f in summary_files:
            with open(f, 'r') as fp:
                data = json.load(fp)
            model = data['model_type']
            results[model] = {
                'mean': data['mean_test_accuracy'],
                'std': data.get('std_test_accuracy', 0)
            }
    
    if not results:
        print("No results to plot")
        return None
    
    return plot_comparison(
        results,
        metric='test_accuracy',
        title='Model Comparison - Test Accuracy',
        save_path=save_path,
        show=show
    )


def save_results_table(
    results_dir: str,
    output_path: str
):
    """
    Create a LaTeX table from experiment results.
    
    Args:
        results_dir: directory containing result files
        output_path: path to save the LaTeX table
    """
    # Find all summary files
    summary_files = list(Path(results_dir).glob("*_summary.json"))
    
    results = {}
    if summary_files:
        for f in summary_files:
            with open(f, 'r') as fp:
                data = json.load(fp)
            model = data['model_type']
            results[model] = {
                'mean': data['mean_test_accuracy'],
                'std': data.get('std_test_accuracy', 0),
                'n': data.get('num_repeats', 1)
            }
    else:
        # Try individual result files
        result_files = list(Path(results_dir).glob("*_result.json"))
        model_results = {}
        for f in result_files:
            with open(f, 'r') as fp:
                data = json.load(fp)
            model = data['config']['model_type']
            if model not in model_results:
                model_results[model] = []
            model_results[model].append(data['metrics']['test_accuracy'])
        
        for model, accs in model_results.items():
            results[model] = {
                'mean': np.mean(accs),
                'std': np.std(accs),
                'n': len(accs)
            }
    
    # Create LaTeX table
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Model & Test Accuracy & Runs \\\\",
        "\\midrule"
    ]
    
    for model in sorted(results.keys()):
        r = results[model]
        lines.append(f"{model} & {r['mean']:.3f} $\\pm$ {r['std']:.3f} & {r['n']} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Test accuracy for different models (mean $\\pm$ std over multiple runs)}",
        "\\label{tab:results}",
        "\\end{table}"
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"LaTeX table saved to {output_path}")


def plot_word_order_comparison(
    normal_results: Dict,
    shuffled_results: Dict,
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot comparison between normal and shuffled word order.
    
    Args:
        normal_results: results with normal word order
        shuffled_results: results with shuffled word order
        save_path: path to save the figure
        show: whether to display the plot
    
    Returns:
        matplotlib Figure
    """
    models = list(normal_results.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    normal_means = [normal_results[m].get('mean', normal_results[m].get('test_accuracy', 0)) for m in models]
    shuffled_means = [shuffled_results[m].get('mean', shuffled_results[m].get('test_accuracy', 0)) for m in models]
    
    bars1 = ax.bar(x - width/2, normal_means, width, label='Normal', color='steelblue')
    bars2 = ax.bar(x + width/2, shuffled_means, width, label='Shuffled', color='coral')
    
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Effect of Word Order on Model Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory containing result files")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for the plot")
    parser.add_argument("--table", type=str, default=None,
                       help="Output path for LaTeX table")
    
    args = parser.parse_args()
    
    if args.table:
        save_results_table(args.results_dir, args.table)
    
    plot_results_summary(args.results_dir, save_path=args.output, show=True)

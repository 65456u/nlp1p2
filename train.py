"""Training and evaluation functions for sentiment classification.
Based on NLP1 2025 Practical 2.
"""

import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Optional logging backends
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    from .data import (
        Example, Vocabulary, pad, get_examples, get_minibatch,
        NUM_CLASSES, build_augmented_dataset
    )
except ImportError:
    from data import (
        Example, Vocabulary, pad, get_examples, get_minibatch,
        NUM_CLASSES, build_augmented_dataset
    )


@dataclass
class ExperimentMetrics:
    """Container for experiment metrics."""
    train_losses: List[float] = field(default_factory=list)
    dev_accuracies: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    test_accuracy: float = 0.0
    best_dev_accuracy: float = 0.0
    best_iter: int = 0
    total_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    model_type: str
    num_iters: int
    learning_rate: float
    batch_size: int
    embedding_dim: int = 300
    hidden_dim: int = 100
    dropout: float = 0.5
    shuffle_words: bool = False
    use_pretrained: bool = True
    finetune_embeddings: bool = False
    seed: int = 42
    
    def to_dict(self) -> Dict:
        return asdict(self)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def prepare_example(example: Example, vocab: Vocabulary, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare a single example for input."""
    x = [vocab.w2i.get(t, 0) for t in example.tokens]
    x = torch.LongTensor([x]).to(device)
    y = torch.LongTensor([example.label]).to(device)
    return x, y


def prepare_minibatch(mb: List[Example], vocab: Vocabulary, device: torch.device,
                      shuffle_words: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare a minibatch for input."""
    maxlen = max([len(ex.tokens) for ex in mb])
    
    x = []
    for ex in mb:
        tokens = [vocab.w2i.get(t, 0) for t in ex.tokens]
        if shuffle_words:
            random.shuffle(tokens)
        tokens = pad(tokens, maxlen)
        x.append(tokens)
    
    x = torch.LongTensor(x).to(device)
    y = torch.LongTensor([ex.label for ex in mb]).to(device)
    
    return x, y


def prepare_treelstm_minibatch(mb: List[Example], vocab: Vocabulary, device: torch.device,
                               shuffle_words: bool = False) -> Tuple[Tuple[torch.Tensor, np.ndarray], torch.Tensor]:
    """Prepare a minibatch for TreeLSTM (with reversed sentences and transitions)."""
    maxlen = max([len(ex.tokens) for ex in mb])
    
    # Note: sentences are reversed for TreeLSTM
    x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen)[::-1] for ex in mb]
    x = torch.LongTensor(x).to(device)
    
    y = torch.LongTensor([ex.label for ex in mb]).to(device)
    
    maxlen_t = max([len(ex.transitions) for ex in mb])
    transitions = [pad(ex.transitions, maxlen_t, pad_value=2) for ex in mb]
    transitions = np.array(transitions).T  # time-major
    
    return (x, transitions), y


def evaluate(model: nn.Module, data: List[Example], vocab: Vocabulary, 
             device: torch.device, batch_size: int = 16,
             is_treelstm: bool = False, shuffle_words: bool = False) -> Tuple[int, int, float]:
    """Evaluate model accuracy on a dataset."""
    correct = 0
    total = 0
    model.eval()
    
    for mb in get_minibatch(data, batch_size=batch_size, shuffle=False):
        if is_treelstm:
            x, targets = prepare_treelstm_minibatch(mb, vocab, device)
        else:
            x, targets = prepare_minibatch(mb, vocab, device, shuffle_words=shuffle_words)
        
        with torch.no_grad():
            logits = model(x)
        
        predictions = logits.argmax(dim=-1).view(-1)
        correct += (predictions == targets.view(-1)).sum().item()
        total += targets.size(0)
    
    return correct, total, correct / float(total) if total > 0 else 0.0


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_data: List[Example],
    dev_data: List[Example],
    test_data: List[Example],
    vocab: Vocabulary,
    device: torch.device,
    num_iterations: int = 10000,
    print_every: int = 1000,
    eval_every: int = 1000,
    batch_size: int = 25,
    is_treelstm: bool = False,
    shuffle_words: bool = False,
    save_path: Optional[str] = None,
    tb_writer: Optional[Any] = None
) -> ExperimentMetrics:
    """
    Train a model with the given configuration.
    
    Args:
        tb_writer: TensorBoard SummaryWriter (optional)
    
    Returns:
        ExperimentMetrics with training history and final results
    """
    metrics = ExperimentMetrics()
    
    iter_i = 0
    train_loss = 0.
    print_num = 0
    start = time.time()
    criterion = nn.CrossEntropyLoss()
    best_eval = 0.
    best_iter = 0
    
    # Select preparation function based on model type
    if is_treelstm:
        prep_fn = lambda mb, v, d: prepare_treelstm_minibatch(mb, v, d, shuffle_words=shuffle_words)
    else:
        prep_fn = lambda mb, v, d: prepare_minibatch(mb, v, d, shuffle_words=shuffle_words)
    
    while True:
        for batch in get_minibatch(train_data, batch_size=batch_size):
            model.train()
            x, targets = prep_fn(batch, vocab, device)
            logits = model(x)
            
            B = targets.size(0)
            loss = criterion(logits.view([B, -1]), targets.view(-1))
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print_num += 1
            iter_i += 1
            
            # Print info
            if iter_i % print_every == 0:
                avg_loss = train_loss / print_num
                print(f"Iter {iter_i}: loss={avg_loss:.4f}, time={time.time()-start:.2f}s")
                metrics.train_losses.append(avg_loss)
                
                # Log to tensorboard
                if tb_writer is not None:
                    tb_writer.add_scalar('train/loss', avg_loss, iter_i)
                
                print_num = 0
                train_loss = 0.
            
            # Evaluate
            if iter_i % eval_every == 0:
                _, _, accuracy = evaluate(model, dev_data, vocab, device, 
                                         batch_size=batch_size, is_treelstm=is_treelstm,
                                         shuffle_words=shuffle_words)
                metrics.dev_accuracies.append(accuracy)
                print(f"Iter {iter_i}: dev acc={accuracy:.4f}")
                
                # Log to tensorboard
                if tb_writer is not None:
                    tb_writer.add_scalar('eval/dev_accuracy', accuracy, iter_i)
                
                # Save best model
                if accuracy > best_eval:
                    print("New highscore!")
                    best_eval = accuracy
                    best_iter = iter_i
                    
                    if save_path:
                        ckpt = {
                            "state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_eval": best_eval,
                            "best_iter": best_iter
                        }
                        torch.save(ckpt, save_path)
            
            # Done training
            if iter_i == num_iterations:
                print("Done training")
                
                # Load best model if saved
                if save_path and os.path.exists(save_path):
                    print("Loading best model")
                    ckpt = torch.load(save_path)
                    model.load_state_dict(ckpt["state_dict"])
                
                # Final evaluation
                _, _, train_acc = evaluate(model, train_data, vocab, device,
                                          batch_size=batch_size, is_treelstm=is_treelstm,
                                          shuffle_words=shuffle_words)
                _, _, dev_acc = evaluate(model, dev_data, vocab, device,
                                        batch_size=batch_size, is_treelstm=is_treelstm,
                                        shuffle_words=shuffle_words)
                _, _, test_acc = evaluate(model, test_data, vocab, device,
                                         batch_size=batch_size, is_treelstm=is_treelstm,
                                         shuffle_words=shuffle_words)
                
                print(f"Best model iter {best_iter}: "
                      f"train acc={train_acc:.4f}, dev acc={dev_acc:.4f}, test acc={test_acc:.4f}")
                
                metrics.train_accuracies.append(train_acc)
                metrics.test_accuracy = test_acc
                metrics.best_dev_accuracy = best_eval
                metrics.best_iter = best_iter
                metrics.total_time = time.time() - start
                
                return metrics


def train_with_node_supervision(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_data: List[Example],
    dev_data: List[Example],
    test_data: List[Example],
    vocab: Vocabulary,
    device: torch.device,
    num_iterations: int = 10000,
    print_every: int = 1000,
    eval_every: int = 1000,
    batch_size: int = 25,
    save_path: Optional[str] = None,
    filter_neutral: bool = True,
    tb_writer: Optional[Any] = None
) -> ExperimentMetrics:
    """
    Train TreeLSTM with node-level supervision.
    This uses augmented data where each node is a training example.
    
    Args:
        filter_neutral: If True, filter out neutral labels (label=2) from training.
                       This is important because ~69% of phrase-level labels are neutral,
                       which would cause the model to be heavily biased.
        tb_writer: TensorBoard SummaryWriter (optional)
    """
    # Build augmented dataset
    print("Building augmented dataset for node-level supervision...")
    train_data_augmented = build_augmented_dataset(train_data)
    print(f"Augmented training data: {len(train_data_augmented)} examples "
          f"(from {len(train_data)} sentences)")
    
    # Filter out neutral labels if requested
    if filter_neutral:
        train_data_augmented = [ex for ex in train_data_augmented if ex.label != 2]
        print(f"After filtering neutral: {len(train_data_augmented)} examples")
    
    # Use standard TreeLSTM training with augmented data
    return train_model(
        model=model,
        optimizer=optimizer,
        train_data=train_data_augmented,
        dev_data=dev_data,  # Evaluate on original dev set
        test_data=test_data,  # Test on original test set
        vocab=vocab,
        device=device,
        num_iterations=num_iterations,
        print_every=print_every,
        eval_every=eval_every,
        batch_size=batch_size,
        is_treelstm=True,
        save_path=save_path,
        tb_writer=tb_writer
    )


@dataclass
class ExperimentResult:
    """Full experiment result including config and metrics."""
    config: ExperimentConfig
    metrics: ExperimentMetrics
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "timestamp": self.timestamp
        }
    
    def save(self, path: str):
        """Save result to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentResult':
        """Load result from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        config = ExperimentConfig(**data["config"])
        metrics = ExperimentMetrics(**data["metrics"])
        return cls(config=config, metrics=metrics, timestamp=data.get("timestamp", ""))


def run_experiment(
    config: ExperimentConfig,
    train_data: List[Example],
    dev_data: List[Example],
    test_data: List[Example],
    vocab: Vocabulary,
    vectors: Optional[np.ndarray] = None,
    output_dir: str = "results",
    tb_writer: Optional[Any] = None
) -> ExperimentResult:
    """
    Run a single experiment with the given configuration.
    
    Args:
        config: experiment configuration
        train_data: training data
        dev_data: development data
        test_data: test data
        vocab: vocabulary
        vectors: pre-trained word vectors (optional)
        output_dir: directory to save results
        tb_writer: TensorBoard SummaryWriter (optional)
    
    Returns:
        ExperimentResult with config and metrics
    """
    try:
        from .models import get_model, print_parameters
    except ImportError:
        from models import get_model, print_parameters
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create model
    model = get_model(
        model_type=config.model_type,
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        output_dim=NUM_CLASSES,
        vocab=vocab
    )
    
    # Load pre-trained embeddings if available
    # BOW model uses embedding_dim = output_dim, so skip pretrained for BOW
    if vectors is not None and config.use_pretrained and config.model_type != 'bow':
        print("Loading pre-trained embeddings...")
        with torch.no_grad():
            model.embed.weight.data.copy_(torch.from_numpy(vectors))
            model.embed.weight.requires_grad = config.finetune_embeddings
    
    model = model.to(device)
    print_parameters(model)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Model save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"{config.model_type}_{timestamp}.pt")
    
    # Determine if this is a TreeLSTM model
    is_treelstm = config.model_type in ['treelstm', 'treelstm_each_node']
    
    # Train
    if config.model_type == 'treelstm_each_node':
        metrics = train_with_node_supervision(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            vocab=vocab,
            device=device,
            num_iterations=config.num_iters,
            batch_size=config.batch_size,
            save_path=save_path,
            tb_writer=tb_writer
        )
    else:
        metrics = train_model(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            vocab=vocab,
            device=device,
            num_iterations=config.num_iters,
            batch_size=config.batch_size,
            is_treelstm=is_treelstm,
            shuffle_words=config.shuffle_words,
            save_path=save_path,
            tb_writer=tb_writer
        )
    
    # Create result
    result = ExperimentResult(config=config, metrics=metrics)
    
    # Save result
    result_path = os.path.join(output_dir, f"{config.model_type}_{timestamp}_result.json")
    result.save(result_path)
    print(f"Results saved to {result_path}")
    
    return result


def run_multiple_experiments(
    config: ExperimentConfig,
    train_data: List[Example],
    dev_data: List[Example],
    test_data: List[Example],
    vocab: Vocabulary,
    vectors: Optional[np.ndarray] = None,
    num_repeats: int = 3,
    output_dir: str = "results",
    tb_writer: Optional[Any] = None
) -> List[ExperimentResult]:
    """
    Run multiple experiments with different seeds.
    
    Args:
        config: base experiment configuration (seed will be modified)
        train_data: training data
        dev_data: development data
        test_data: test data
        vocab: vocabulary
        vectors: pre-trained word vectors (optional)
        num_repeats: number of experiments to run
        output_dir: directory to save results
        tb_writer: TensorBoard SummaryWriter (optional)
    
    Returns:
        List of ExperimentResults
    """
    results = []
    base_seed = config.seed
    
    for i in range(num_repeats):
        print(f"\n{'='*60}")
        print(f"Running experiment {i+1}/{num_repeats}")
        print(f"{'='*60}\n")
        
        # Create config with modified seed
        exp_config = ExperimentConfig(
            model_type=config.model_type,
            num_iters=config.num_iters,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            shuffle_words=config.shuffle_words,
            use_pretrained=config.use_pretrained,
            finetune_embeddings=config.finetune_embeddings,
            seed=base_seed + i
        )
        
        result = run_experiment(
            config=exp_config,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            vocab=vocab,
            vectors=vectors,
            output_dir=output_dir,
            tb_writer=tb_writer
        )
        results.append(result)
    
    # Print summary statistics
    test_accs = [r.metrics.test_accuracy for r in results]
    mean_acc = np.mean(test_accs)
    std_acc = np.std(test_accs)
    
    print(f"\n{'='*60}")
    print(f"Summary for {config.model_type}")
    print(f"{'='*60}")
    print(f"Test accuracies: {test_accs}")
    print(f"Mean: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"{'='*60}\n")
    
    return results

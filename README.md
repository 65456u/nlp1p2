# Sentiment Analysis Experiment Framework

This is an experiment framework for sentiment classification on the Stanford Sentiment Treebank (SST) dataset.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running a single experiment

```bash
python main.py --model [model_type] --num_iters [num_iters] --lr [lr] --num_repeat [num_repeat] --shuffle [bool]
```

### Auto-Download Feature

**If you don't specify `--data_dir` or `--embedding_file`, the program will automatically download the required files:**

- **SST Dataset**: Downloaded from Stanford NLP and extracted to `data/trees/`
- **GloVe Embeddings**: Downloaded from gist to `data/glove.840B.300d.sst.txt`

This means you can run experiments without any prior setup:

```bash
# Just run - data will be downloaded automatically!
python main.py --model lstm_batched --num_iters 1000
```

### Arguments

- `--model`: Model type. Choices: `bow`, `cbow`, `deepcbow`, `lstm`, `lstm_batched`, `treelstm`, `treelstm_each_node`
- `--num_iters`: Number of training iterations (default: 30000)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 25)
- `--num_repeat`: Number of repeated experiments with different seeds (default: 1)
- `--shuffle`: If set, shuffle words in each example (tests word order importance)
- `--finetune`: If set, fine-tune pre-trained embeddings
- `--no_pretrained`: If set, do not use pre-trained embeddings
- `--data_dir`: Path to data directory (default: auto-download to `data/trees/`)
- `--embedding_file`: Path to embedding file (default: auto-download to `data/glove.840B.300d.sst.txt`)
- `--output_dir`: Directory to save results (default: results)
- `--seed`: Random seed (default: 42)

### Examples

```bash
# Train LSTM with 3 repetitions
python main.py --model lstm_batched --num_iters 30000 --lr 0.001 --num_repeat 3

# Train TreeLSTM with node-level supervision
python main.py --model treelstm_each_node --num_iters 30000 --lr 0.001

# Test word order importance with shuffled words
python main.py --model lstm_batched --num_iters 30000 --shuffle

# Train with fine-tuned embeddings
python main.py --model lstm_batched --num_iters 30000 --finetune
```

### Running all experiments

```bash
bash run_all_experiments.sh
```

This will run all model types with multiple seeds and generate comparison plots.

## Project Structure

```
exp_program/
├── __init__.py          # Package initialization
├── data.py              # Data loading and preprocessing
├── models.py            # Model architectures (BOW, CBOW, LSTM, TreeLSTM, etc.)
├── train.py             # Training and evaluation functions
├── main.py              # CLI entry point
├── visualize.py         # Visualization utilities
├── run_all_experiments.sh  # Script to run all experiments
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── tests/
    ├── __init__.py
    └── test_all.py      # Unit tests
```

## Supported Models

1. **BOW (Bag of Words)**: Simple bag-of-words with learned word embeddings directly mapped to sentiment classes.

2. **CBOW (Continuous Bag of Words)**: Bag-of-words with dense embeddings projected through a linear layer.

3. **DeepCBOW**: CBOW with multiple hidden layers and tanh activations.

4. **LSTM**: LSTM-based classifier that processes words sequentially.

5. **TreeLSTM**: Tree-structured LSTM that uses the parse tree structure of sentences.

6. **TreeLSTM with Node Supervision**: TreeLSTM trained on all nodes in the tree (not just root).

## Output

Results are saved in the output directory with the following structure:

```
results/
├── <model_type>/
│   ├── <model>_<timestamp>.pt          # Model checkpoint
│   ├── <model>_<timestamp>_result.json # Detailed results
│   ├── <model>_summary.json            # Summary statistics
│   └── summary.png                     # Training curves plot
└── results_table.tex                   # LaTeX table for report
```

## Running Tests

```bash
cd exp_program
python -m pytest tests/
```

Or run specific tests:

```bash
python -m pytest tests/test_all.py -v
```

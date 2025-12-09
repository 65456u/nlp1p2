#!/bin/bash
# Script to run all experiments for NLP1 Practical 2
# This will run all model types with multiple seeds and generate comparison plots

set -e

# Configuration
NUM_ITERS=30000
NUM_REPEAT=3
LR=0.001
BATCH_SIZE=25
OUTPUT_DIR="results"

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "=================================================="
echo "NLP1 Practical 2 - Full Experiment Suite"
echo "=================================================="
echo "Number of iterations: ${NUM_ITERS}"
echo "Number of repeats per model: ${NUM_REPEAT}"
echo "Learning rate: ${LR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=================================================="

# Function to run experiments for a model
run_model() {
    local model=$1
    local extra_args=${2:-""}
    
    echo ""
    echo "=================================================="
    echo "Running experiments for: ${model}"
    echo "=================================================="
    
    python main.py \
        --model ${model} \
        --num_iters ${NUM_ITERS} \
        --lr ${LR} \
        --batch_size ${BATCH_SIZE} \
        --num_repeat ${NUM_REPEAT} \
        --output_dir ${OUTPUT_DIR}/${model} \
        ${extra_args}
}

# 1. Bag of Words
run_model "bow"

# 2. Continuous Bag of Words
run_model "cbow"

# 3. Deep CBOW
run_model "deepcbow"

# 4. LSTM (batched)
run_model "lstm_batched"

# 5. LSTM with shuffled words (to test word order importance)
echo ""
echo "=================================================="
echo "Running LSTM with shuffled words (word order test)"
echo "=================================================="
python main.py \
    --model lstm_batched \
    --num_iters ${NUM_ITERS} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --num_repeat ${NUM_REPEAT} \
    --shuffle \
    --output_dir ${OUTPUT_DIR}/lstm_shuffled

# 6. TreeLSTM
run_model "treelstm"

# 7. TreeLSTM with node supervision
run_model "treelstm_each_node"

# 8. LSTM with fine-tuned embeddings
echo ""
echo "=================================================="
echo "Running LSTM with fine-tuned embeddings"
echo "=================================================="
python main.py \
    --model lstm_batched \
    --num_iters ${NUM_ITERS} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --num_repeat ${NUM_REPEAT} \
    --finetune \
    --output_dir ${OUTPUT_DIR}/lstm_finetuned

# Generate summary plots and tables
echo ""
echo "=================================================="
echo "Generating summary visualizations"
echo "=================================================="

python -c "
from exp_program.visualize import plot_results_summary, save_results_table
import os

# Plot comparison for all models
for subdir in os.listdir('${OUTPUT_DIR}'):
    subdir_path = os.path.join('${OUTPUT_DIR}', subdir)
    if os.path.isdir(subdir_path):
        plot_results_summary(subdir_path, 
                           save_path=os.path.join(subdir_path, 'summary.png'),
                           show=False)

# Create LaTeX table
save_results_table('${OUTPUT_DIR}', '${OUTPUT_DIR}/results_table.tex')
"

echo ""
echo "=================================================="
echo "All experiments completed!"
echo "=================================================="
echo "Results are saved in: ${OUTPUT_DIR}"
echo ""
echo "Summary:"
echo "- Individual model results: ${OUTPUT_DIR}/<model_name>/"
echo "- Summary plots: ${OUTPUT_DIR}/<model_name>/summary.png"
echo "- LaTeX results table: ${OUTPUT_DIR}/results_table.tex"
echo "=================================================="

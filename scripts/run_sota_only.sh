#!/bin/bash

set -e

echo "=========================================="
echo "SOTA COMPARISON (DeepSpeed + Megatron)"
echo "=========================================="

# Check if DeepSpeed is installed
if ! python -c "import deepspeed" 2>/dev/null; then
    echo ""
    echo "Installing DeepSpeed..."
    pip install deepspeed
    echo ""
fi

# Run SOTA comparison on 8 GPUs
echo "Running SOTA comparison on 8 GPUs..."
echo ""

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=29501 \
    experiments/sota_comparison.py

# Merge with existing results
echo ""
echo "Merging with baseline/ultimate results..."
python analysis/merge_with_baseline.py

# Generate SOTA analysis
echo ""
echo "Generating SOTA analysis..."
python analysis/sota_analysis.py

# Generate plots
echo ""
echo "Generating SOTA comparison plots..."
python utils/sota_plots.py

echo ""
echo "=========================================="
echo "✅ SOTA COMPARISON COMPLETE!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Raw SOTA data: results/sota_comparison/"
echo "  - Merged data: results/merged/"
echo "  - Tables: results/tables/sota_comparison.tex"
echo "  - Figures: results/figures/sota_*.pdf"
echo ""

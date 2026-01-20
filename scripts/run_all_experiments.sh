#!/bin/bash

# Complete experimental suite for ICML paper
# Run this to generate all results

set -e

echo "=========================================="
echo "ULTIMATE MOE - ICML EXPERIMENTAL SUITE"
echo "=========================================="

# Configuration
NNODES=1
NPROC_PER_NODE=8  # Adjust based on your GPU count
MASTER_ADDR=localhost
MASTER_PORT=29500

# Create results directory
mkdir -p results/{comprehensive,ablation,scalability,deepspeed}

echo ""
echo "Step 1: Running comprehensive baseline vs UltimateMoE comparison..."
echo ""

for world_size in 1 2 4 8; do
    echo "Testing with $world_size GPUs..."
    torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$world_size \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        experiments/benchmark_suite.py
    
    sleep 5  # Cool down between runs
done

echo ""
echo "Step 2: Running ablation study..."
echo ""

for world_size in 1 2 4 8; do
    echo "Ablation with $world_size GPUs..."
    torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$world_size \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        experiments/ablation_study.py
    
    sleep 5
done

echo ""
echo "Step 3: Running scalability tests..."
echo ""

for world_size in 1 2 4 8; do
    echo "Scalability test with $world_size GPUs..."
    torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$world_size \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        experiments/scalability_test.py
    
    sleep 5
done

echo ""
echo "Step 4: Aggregating results..."
echo ""

python analysis/aggregate_results.py

echo ""
echo "Step 5: Generating paper figures..."
echo ""

python analysis/paper_tables.py
python utils/visualization.py

echo ""
echo "=========================================="
echo "✅ ALL EXPERIMENTS COMPLETE!"
echo "=========================================="
echo ""
echo "Results location:"
echo "  - Raw data: results/"
echo "  - Tables: results/tables/"
echo "  - Figures: results/figures/"
echo ""

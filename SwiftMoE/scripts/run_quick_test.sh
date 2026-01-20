#!/bin/bash

# Quick test on 8 GPUs for debugging

echo "Quick test: 8 GPUs, BS=2048, 128 experts"

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=29500 \
    experiments/benchmark_suite.py

echo "✅ Quick test complete!"

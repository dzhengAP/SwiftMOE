"""
Main benchmark orchestration for ICML paper.
Runs comprehensive experiments across all configurations.
"""

import os
import sys
import json
import time
import torch
import torch.distributed as dist
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.baseline_moe import StandardMoE
from models.ultimate_moe import UltimateMoE


class ComprehensiveBenchmark:
    """
    Comprehensive benchmark suite for ICML paper.
    Tests multiple configurations systematically.
    """
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = f"cuda:{self.rank}"
        torch.cuda.set_device(self.rank)
        
        self.results = []
    
    def warmup(self, model: torch.nn.Module, x: torch.Tensor, iterations: int = 10):
        """Warmup run to stabilize timings"""
        for _ in range(iterations):
            _ = model(x)
        torch.cuda.synchronize()
    
    def measure_latency(
        self, 
        model: torch.nn.Module, 
        x: torch.Tensor, 
        iterations: int = 100
    ) -> Tuple[float, float, List[float]]:
        """
        Measure latency with statistical rigor.
        Returns: (mean_ms, std_ms, all_times_ms)
        """
        times = []
        
        for _ in range(iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start_event.record()
            
            _ = model(x)
            
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed = start_event.elapsed_time(end_event)
            times.append(elapsed)
        
        return np.mean(times), np.std(times), times
    
    def run_single_config(
        self,
        model_type: str,
        num_experts: int,
        d_model: int,
        batch_size: int,
        use_triton: bool = True,
        use_nccl_opt: bool = True,
        enable_profiling: bool = False
    ) -> Dict:
        """Run benchmark for a single configuration"""
        
        # Create model
        if model_type == "baseline":
            model = StandardMoE(
                num_experts=num_experts,
                d_model=d_model,
                use_fp16=True
            ).to(self.device)
        elif model_type == "ultimate":
            model = UltimateMoE(
                num_experts=num_experts,
                d_model=d_model,
                use_fp16=True,
                use_triton=use_triton,
                use_nccl_opt=use_nccl_opt,
                enable_profiling=enable_profiling
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create input
        x = torch.randn(batch_size, d_model, device=self.device, dtype=torch.float16)
        
        # Warmup
        self.warmup(model, x, iterations=10)
        
        # Reset memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure latency
        mean_lat, std_lat, all_times = self.measure_latency(model, x, iterations=100)
        
        # Get memory usage
        max_mem_allocated = torch.cuda.max_memory_allocated(self.device) / 1e9  # GB
        
        # Get profiling breakdown if enabled
        breakdown = {}
        if enable_profiling and hasattr(model, 'get_profile_summary'):
            breakdown = model.get_profile_summary()
        
        # Compute throughput (tokens/s across all GPUs)
        throughput = (batch_size * self.world_size) / (mean_lat / 1000)
        
        result = {
            'model_type': model_type,
            'world_size': self.world_size,
            'batch_size': batch_size,
            'num_experts': num_experts,
            'd_model': d_model,
            'use_triton': use_triton,
            'use_nccl_opt': use_nccl_opt,
            'latency_mean_ms': mean_lat,
            'latency_std_ms': std_lat,
            'latency_min_ms': np.min(all_times),
            'latency_max_ms': np.max(all_times),
            'latency_p50_ms': np.percentile(all_times, 50),
            'latency_p95_ms': np.percentile(all_times, 95),
            'latency_p99_ms': np.percentile(all_times, 99),
            'throughput_tokens_per_sec': throughput,
            'max_memory_gb': max_mem_allocated,
            **breakdown
        }
        
        if self.rank == 0:
            print(f"\n{'='*80}")
            print(f"Config: {model_type} | BS={batch_size} | Experts={num_experts} | WS={self.world_size}")
            print(f"Latency: {mean_lat:.2f} ± {std_lat:.2f} ms")
            print(f"Throughput: {throughput:.0f} tokens/s")
            print(f"Memory: {max_mem_allocated:.2f} GB")
            if breakdown:
                print(f"Breakdown: {breakdown}")
            print(f"{'='*80}\n")
        
        return result
    
    def run_baseline_vs_ultimate(self):
        """
        Main comparison: Baseline vs UltimateMoE
        """
        batch_sizes = [512, 1024, 2048, 4096, 8192]
        expert_counts = [64, 128, 256]
        d_model = 1024
        
        if self.rank == 0:
            print("\n" + "="*80)
            print("EXPERIMENT 1: Baseline vs UltimateMoE Comparison")
            print("="*80 + "\n")
        
        for num_experts in expert_counts:
            for batch_size in batch_sizes:
                # Baseline
                result_baseline = self.run_single_config(
                    model_type="baseline",
                    num_experts=num_experts,
                    d_model=d_model,
                    batch_size=batch_size,
                    enable_profiling=False
                )
                self.results.append(result_baseline)
                
                # UltimateMoE with profiling
                result_ultimate = self.run_single_config(
                    model_type="ultimate",
                    num_experts=num_experts,
                    d_model=d_model,
                    batch_size=batch_size,
                    use_triton=True,
                    use_nccl_opt=True,
                    enable_profiling=True
                )
                self.results.append(result_ultimate)
                
                # Compute speedup
                speedup = result_baseline['latency_mean_ms'] / result_ultimate['latency_mean_ms']
                
                if self.rank == 0:
                    print(f"Speedup: {speedup:.2f}×")
                
                # Synchronize between configs
                dist.barrier()
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save results to JSON"""
        if self.rank == 0:
            output_file = self.output_dir / f"{filename.replace('.json', '')}_ws{self.world_size}.json"
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            print(f"\n✅ Results saved to {output_file}\n")
    
    def run_all(self):
        """Run all experiments"""
        self.run_baseline_vs_ultimate()
        self.save_results("comprehensive_benchmark.json")


def main():
    benchmark = ComprehensiveBenchmark(output_dir="results/comprehensive")
    benchmark.run_all()
    
    if benchmark.rank == 0:
        print("\n" + "="*80)
        print("✅ COMPREHENSIVE BENCHMARK COMPLETE")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()

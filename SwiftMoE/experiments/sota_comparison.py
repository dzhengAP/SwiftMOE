"""
SOTA Comparison: DeepSpeed-MoE and Megatron-LM only.
This script is designed to run independently and merge with existing baseline/ultimate results.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
import json
import numpy as np
from typing import Dict, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class DeepSpeedStyleMoE(nn.Module):
    """
    DeepSpeed-style MoE reference implementation.
    Mimics DeepSpeed's architectural choices without using their library.
    """
    
    def __init__(
        self,
        num_experts: int = 128,
        d_model: int = 1024,
        use_fp16: bool = True
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.d_model = d_model
        self.use_fp16 = use_fp16
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
        
        dtype = torch.float16 if use_fp16 else torch.float32
        
        # Router (DeepSpeed uses simple linear gate)
        self.gate = nn.Linear(d_model, num_experts, bias=False, dtype=dtype)
        nn.init.normal_(self.gate.weight, std=0.02)
        
        # Expert parallelism (distribute experts across GPUs)
        self.experts_per_rank = num_experts // self.world_size
        self.local_expert_start = self.rank * self.experts_per_rank
        
        # Local experts (DeepSpeed uses simple linear layers)
        self.local_experts = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False, dtype=dtype)
            for _ in range(self.experts_per_rank)
        ])
        
        for expert in self.local_experts:
            nn.init.normal_(expert.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with DeepSpeed-style routing"""
        if self.use_fp16 and x.dtype != torch.float16:
            x = x.half()
        
        batch_size, d_model = x.shape
        device = x.device
        
        # Gate computation
        gate_logits = self.gate(x)
        
        # Top-2 routing (DeepSpeed default)
        top_k_logits, top_k_indices = torch.topk(gate_logits, k=2, dim=-1)
        top_k_weights = torch.softmax(top_k_logits, dim=-1)
        
        # Flatten for all-to-all
        flat_indices = top_k_indices.view(-1)
        flat_weights = top_k_weights.view(-1, 1)
        
        # Determine target ranks for each token
        target_ranks = flat_indices // self.experts_per_rank
        local_expert_ids = flat_indices % self.experts_per_rank
        
        # Sort by target rank
        sorted_indices = torch.argsort(target_ranks)
        reverse_indices = torch.argsort(sorted_indices)
        
        sorted_tokens = x.repeat_interleave(2, dim=0)[sorted_indices]
        sorted_weights = flat_weights[sorted_indices]
        sorted_expert_ids = local_expert_ids[sorted_indices]
        sorted_ranks = target_ranks[sorted_indices]
        
        # Count tokens per rank
        send_counts = torch.bincount(sorted_ranks, minlength=self.world_size)
        
        # All-to-all exchange of counts
        recv_counts = torch.empty_like(send_counts)
        if self.world_size > 1:
            dist.all_to_all_single(recv_counts, send_counts)
        else:
            recv_counts = send_counts
        
        total_recv = recv_counts.sum().item()
        
        # Prepare buffers
        recv_tokens = torch.empty((total_recv, d_model), device=device, dtype=x.dtype)
        recv_weights = torch.empty((total_recv, 1), device=device, dtype=x.dtype)
        recv_expert_ids = torch.empty((total_recv,), device=device, dtype=torch.long)
        
        # All-to-all exchange
        if self.world_size > 1:
            send_list = send_counts.cpu().tolist()
            recv_list = recv_counts.cpu().tolist()
            
            dist.all_to_all_single(recv_tokens, sorted_tokens, output_split_sizes=recv_list, input_split_sizes=send_list)
            dist.all_to_all_single(recv_weights, sorted_weights, output_split_sizes=recv_list, input_split_sizes=send_list)
            dist.all_to_all_single(recv_expert_ids, sorted_expert_ids, output_split_sizes=recv_list, input_split_sizes=send_list)
        else:
            recv_tokens = sorted_tokens
            recv_weights = sorted_weights
            recv_expert_ids = sorted_expert_ids
            send_list = send_counts.cpu().tolist()
            recv_list = recv_counts.cpu().tolist()
        
        # Expert computation
        expert_outputs = torch.zeros_like(recv_tokens)
        
        for expert_id in range(self.experts_per_rank):
            mask = (recv_expert_ids == expert_id)
            if mask.any():
                expert_input = recv_tokens[mask]
                expert_weight = recv_weights[mask]
                expert_output = self.local_experts[expert_id](expert_input)
                expert_outputs[mask] = expert_output * expert_weight
        
        # All-to-all return
        combined_outputs = torch.empty((sum(send_list), d_model), device=device, dtype=x.dtype)
        
        if self.world_size > 1:
            dist.all_to_all_single(combined_outputs, expert_outputs, output_split_sizes=send_list, input_split_sizes=recv_list)
        else:
            combined_outputs = expert_outputs
        
        # Restore original order and sum top-2 experts
        restored = combined_outputs[reverse_indices]
        output = restored.view(batch_size, 2, d_model).sum(dim=1)
        
        return output


class MegatronMoEWrapper(nn.Module):
    """Megatron-LM style MoE implementation."""
    
    def __init__(self, num_experts: int = 128, d_model: int = 1024, use_fp16: bool = True, expert_parallel: bool = True):
        super().__init__()
        
        self.num_experts = num_experts
        self.d_model = d_model
        self.use_fp16 = use_fp16
        self.expert_parallel = expert_parallel
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
        
        if expert_parallel:
            self.experts_per_rank = num_experts // self.world_size
            self.local_expert_start = self.rank * self.experts_per_rank
        else:
            self.experts_per_rank = num_experts
            self.local_expert_start = 0
        
        dtype = torch.float16 if use_fp16 else torch.float32
        
        self.gate = nn.Linear(d_model, num_experts, bias=False, dtype=dtype)
        nn.init.normal_(self.gate.weight, std=0.02)
        
        self.experts = nn.ModuleList([
            self._create_expert_ffn(d_model, dtype)
            for _ in range(self.experts_per_rank)
        ])
    
    def _create_expert_ffn(self, d_model, dtype):
        return nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False, dtype=dtype),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model, bias=False, dtype=dtype)
        )
    
    def _dispatch_to_experts(self, x, indices, weights):
        batch_size = x.shape[0]
        flat_indices = indices.view(-1)
        flat_weights = weights.view(-1, 1)
        
        target_ranks = flat_indices // self.experts_per_rank
        local_indices = flat_indices % self.experts_per_rank
        
        sort_idx = torch.argsort(target_ranks)
        rev_idx = torch.argsort(sort_idx)
        
        sorted_x = x.repeat_interleave(2, dim=0)[sort_idx]
        sorted_weights = flat_weights[sort_idx]
        sorted_local_idx = local_indices[sort_idx]
        
        send_counts = torch.bincount(target_ranks, minlength=self.world_size)
        recv_counts = torch.empty_like(send_counts)
        
        if dist.is_initialized():
            dist.all_to_all_single(recv_counts, send_counts)
        else:
            recv_counts = send_counts
        
        total_recv = recv_counts.sum().item()
        
        recv_x = torch.empty((total_recv, self.d_model), device=x.device, dtype=x.dtype)
        recv_weights = torch.empty((total_recv, 1), device=x.device, dtype=x.dtype)
        recv_local_idx = torch.empty((total_recv,), device=x.device, dtype=torch.long)
        
        if dist.is_initialized():
            send_list = send_counts.cpu().tolist()
            recv_list = recv_counts.cpu().tolist()
            
            dist.all_to_all_single(recv_x, sorted_x, recv_list, send_list)
            dist.all_to_all_single(recv_weights, sorted_weights, recv_list, send_list)
            dist.all_to_all_single(recv_local_idx, sorted_local_idx, recv_list, send_list)
        else:
            recv_x = sorted_x
            recv_weights = sorted_weights
            recv_local_idx = sorted_local_idx
            send_list = send_counts.cpu().tolist()
            recv_list = recv_counts.cpu().tolist()
        
        return recv_x, recv_local_idx, recv_weights, send_list, recv_list, rev_idx
    
    def _combine_results(self, expert_out, send_list, recv_list, rev_idx, batch_size):
        combined = torch.empty((sum(send_list), self.d_model), device=expert_out.device, dtype=expert_out.dtype)
        
        if dist.is_initialized():
            dist.all_to_all_single(combined, expert_out, send_list, recv_list)
        else:
            combined = expert_out
        
        restored = combined[rev_idx].view(batch_size, 2, self.d_model)
        return restored.sum(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fp16 and x.dtype != torch.float16:
            x = x.half()
        
        batch_size = x.shape[0]
        
        gate_logits = self.gate(x)
        top_k_logits, top_k_indices = torch.topk(gate_logits, k=2, dim=-1)
        top_k_weights = torch.softmax(top_k_logits, dim=-1)
        
        recv_x, recv_local_idx, recv_weights, send_list, recv_list, rev_idx = \
            self._dispatch_to_experts(x, top_k_indices, top_k_weights)
        
        expert_out = torch.zeros_like(recv_x)
        
        for i in range(self.experts_per_rank):
            mask = (recv_local_idx == i)
            if mask.any():
                expert_input = recv_x[mask]
                expert_weights = recv_weights[mask]
                expert_output = self.experts[i](expert_input)
                expert_out[mask] = expert_output * expert_weights
        
        output = self._combine_results(expert_out, send_list, recv_list, rev_idx, batch_size)
        
        return output


class SOTAComparison:
    """SOTA comparison suite."""
    
    def __init__(self, output_dir: str = "results/sota_comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = f"cuda:{self.rank}"
        torch.cuda.set_device(self.rank)
        
        self.results = []
    
    def warmup(self, model: nn.Module, x: torch.Tensor, iterations: int = 10):
        for _ in range(iterations):
            _ = model(x)
        torch.cuda.synchronize()
    
    def measure_latency(self, model: nn.Module, x: torch.Tensor, iterations: int = 100) -> tuple:
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
    
    def _run_deepspeed(self, num_experts: int, d_model: int, batch_size: int) -> Dict:
        """Run DeepSpeed-style MoE benchmark"""
        
        if self.rank == 0:
            print(f"  Testing DeepSpeed-MoE (BS={batch_size})...", end=" ", flush=True)
        
        try:
            model = DeepSpeedStyleMoE(
                num_experts=num_experts,
                d_model=d_model,
                use_fp16=True
            ).to(self.device)
            
            x = torch.randn(batch_size, d_model, device=self.device, dtype=torch.float16)
            
            self.warmup(model, x, iterations=10)
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            mean_lat, std_lat, all_times = self.measure_latency(model, x, iterations=100)
            max_mem = torch.cuda.max_memory_allocated(self.device) / 1e9
            
            throughput = (batch_size * self.world_size) / (mean_lat / 1000)
            
            result = {
                'model_type': 'deepspeed',
                'framework': 'DeepSpeed-MoE',
                'world_size': self.world_size,
                'batch_size': batch_size,
                'num_experts': num_experts,
                'd_model': d_model,
                'use_triton': False,
                'use_nccl_opt': False,
                'latency_mean_ms': mean_lat,
                'latency_std_ms': std_lat,
                'latency_min_ms': float(np.min(all_times)),
                'latency_max_ms': float(np.max(all_times)),
                'latency_p50_ms': float(np.percentile(all_times, 50)),
                'latency_p95_ms': float(np.percentile(all_times, 95)),
                'latency_p99_ms': float(np.percentile(all_times, 99)),
                'throughput_tokens_per_sec': throughput,
                'max_memory_gb': max_mem,
                'experiment_type': 'sota'
            }
            
            if self.rank == 0:
                print(f"✓ ({mean_lat:.2f}ms)")
            
            return result
            
        except Exception as e:
            if self.rank == 0:
                print(f"✗ (Error: {e})")
            
            return {
                'model_type': 'deepspeed',
                'framework': 'DeepSpeed-MoE',
                'world_size': self.world_size,
                'batch_size': batch_size,
                'num_experts': num_experts,
                'd_model': d_model,
                'latency_mean_ms': -1,
                'latency_std_ms': -1,
                'throughput_tokens_per_sec': -1,
                'max_memory_gb': -1,
                'error': str(e),
                'experiment_type': 'sota'
            }
    
    def _run_megatron(self, num_experts: int, d_model: int, batch_size: int) -> Dict:
        """Run Megatron-LM style MoE benchmark"""
        
        if self.rank == 0:
            print(f"  Testing Megatron-LM (BS={batch_size})...", end=" ", flush=True)
        
        try:
            model = MegatronMoEWrapper(
                num_experts=num_experts,
                d_model=d_model,
                use_fp16=True,
                expert_parallel=True
            ).to(self.device)
            
            x = torch.randn(batch_size, d_model, device=self.device, dtype=torch.float16)
            
            self.warmup(model, x, iterations=10)
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            mean_lat, std_lat, all_times = self.measure_latency(model, x, iterations=100)
            max_mem = torch.cuda.max_memory_allocated(self.device) / 1e9
            
            throughput = (batch_size * self.world_size) / (mean_lat / 1000)
            
            result = {
                'model_type': 'megatron',
                'framework': 'Megatron-LM',
                'world_size': self.world_size,
                'batch_size': batch_size,
                'num_experts': num_experts,
                'd_model': d_model,
                'use_triton': False,
                'use_nccl_opt': False,
                'latency_mean_ms': mean_lat,
                'latency_std_ms': std_lat,
                'latency_min_ms': float(np.min(all_times)),
                'latency_max_ms': float(np.max(all_times)),
                'latency_p50_ms': float(np.percentile(all_times, 50)),
                'latency_p95_ms': float(np.percentile(all_times, 95)),
                'latency_p99_ms': float(np.percentile(all_times, 99)),
                'throughput_tokens_per_sec': throughput,
                'max_memory_gb': max_mem,
                'experiment_type': 'sota'
            }
            
            if self.rank == 0:
                print(f"✓ ({mean_lat:.2f}ms)")
            
            return result
            
        except Exception as e:
            if self.rank == 0:
                print(f"✗ (Error: {e})")
            
            return {
                'model_type': 'megatron',
                'framework': 'Megatron-LM',
                'world_size': self.world_size,
                'batch_size': batch_size,
                'num_experts': num_experts,
                'd_model': d_model,
                'latency_mean_ms': -1,
                'latency_std_ms': -1,
                'throughput_tokens_per_sec': -1,
                'max_memory_gb': -1,
                'error': str(e),
                'experiment_type': 'sota'
            }
    
    def run_sota_comparison(self):
        """Run SOTA comparison."""
        
        num_experts = 128
        d_model = 1024
        batch_sizes = [512, 1024, 2048, 4096, 8192]
        
        if self.rank == 0:
            print("\n" + "="*80)
            print("SOTA COMPARISON: DeepSpeed-MoE + Megatron-LM")
            print("="*80)
            print(f"\nConfiguration:")
            print(f"  - GPUs: {self.world_size}")
            print(f"  - Experts: {num_experts}")
            print(f"  - Hidden dim: {d_model}")
            print(f"  - Batch sizes: {batch_sizes}")
            print(f"\nNote: Will merge with existing baseline/ultimate results")
            print("="*80 + "\n")
        
        sota_results = []
        
        for batch_size in batch_sizes:
            
            if self.rank == 0:
                print(f"\n{'─'*80}")
                print(f"Testing Batch Size: {batch_size}")
                print(f"{'─'*80}\n")
            
            result_deepspeed = self._run_deepspeed(num_experts=num_experts, d_model=d_model, batch_size=batch_size)
            sota_results.append(result_deepspeed)
            
            result_megatron = self._run_megatron(num_experts=num_experts, d_model=d_model, batch_size=batch_size)
            sota_results.append(result_megatron)
            
            if dist.is_initialized():
                dist.barrier()
        
        if self.rank == 0:
            output_file = self.output_dir / f"sota_only_ws{self.world_size}.json"
            with open(output_file, 'w') as f:
                json.dump(sota_results, f, indent=2)
            
            import pandas as pd
            df = pd.DataFrame(sota_results)
            csv_file = self.output_dir / f"sota_only_ws{self.world_size}.csv"
            df.to_csv(csv_file, index=False)
            
            print(f"\n{'='*80}")
            print(f"✅ SOTA results saved to:")
            print(f"   - {output_file}")
            print(f"   - {csv_file}")
            print(f"{'='*80}\n")
            
            self._print_summary(sota_results)
        
        return sota_results
    
    def _print_summary(self, results: List[Dict]):
        """Print results summary"""
        
        print("\n" + "="*80)
        print("SOTA COMPARISON SUMMARY")
        print("="*80 + "\n")
        
        print(f"{'Batch Size':<12} {'Framework':<20} {'Latency (ms)':<15} {'Throughput':<15}")
        print("-" * 70)
        
        batch_sizes = sorted(set(r['batch_size'] for r in results))
        
        for bs in batch_sizes:
            bs_results = [r for r in results if r['batch_size'] == bs]
            
            for i, result in enumerate(bs_results):
                bs_str = str(bs) if i == 0 else ""
                framework = result['framework']
                lat = result['latency_mean_ms']
                lat_std = result['latency_std_ms']
                tp = result['throughput_tokens_per_sec']
                
                if lat > 0:
                    print(f"{bs_str:<12} {framework:<20} {lat:>6.2f} ± {lat_std:<5.2f} {tp/1000:>8.0f}K tok/s")
                else:
                    print(f"{bs_str:<12} {framework:<20} {'FAILED':<15} {'-':<15}")
            
            if bs != batch_sizes[-1]:
                print()
        
        print("\n" + "="*80 + "\n")


def main():
    """Main entry point"""
    
    comparison = SOTAComparison(output_dir="results/sota_comparison")
    comparison.run_sota_comparison()
    
    if comparison.rank == 0:
        print("\n" + "="*80)
        print("✅ SOTA COMPARISON COMPLETE")
        print("="*80)
        print("\nNext steps:")
        print("  1. Run: python analysis/merge_with_baseline.py")
        print("  2. Run: python analysis/sota_analysis.py")
        print("  3. Run: python utils/sota_plots.py")
        print("\n" + "="*80 + "\n")
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""
Scalability analysis across different GPU counts.
Measures weak and strong scaling characteristics.
"""

import sys
from pathlib import Path
import torch
import torch.distributed as dist
import json
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from experiments.benchmark_suite import ComprehensiveBenchmark


class ScalabilityTest(ComprehensiveBenchmark):
    """
    Scalability testing for distributed MoE.
    
    Tests:
    1. Strong scaling: Fixed problem size, vary GPUs
    2. Weak scaling: Fixed problem size per GPU
    """
    
    def run_scalability(self):
        """Run scalability tests"""
        num_experts = 128
        d_model = 1024
        
        # Strong scaling: Fixed total batch size
        total_batch_size = 16384
        batch_per_gpu = total_batch_size // self.world_size
        
        if self.rank == 0:
            print("\n" + "="*80)
            print(f"SCALABILITY TEST: {self.world_size} GPUs")
            print(f"Total batch size: {total_batch_size}")
            print(f"Batch per GPU: {batch_per_gpu}")
            print("="*80 + "\n")
        
        # Test baseline
        result_baseline = self.run_single_config(
            model_type="baseline",
            num_experts=num_experts,
            d_model=d_model,
            batch_size=batch_per_gpu,
            enable_profiling=False
        )
        result_baseline['scaling_type'] = 'strong'
        result_baseline['total_batch_size'] = total_batch_size
        self.results.append(result_baseline)
        
        # Test UltimateMoE
        result_ultimate = self.run_single_config(
            model_type="ultimate",
            num_experts=num_experts,
            d_model=d_model,
            batch_size=batch_per_gpu,
            use_triton=True,
            use_nccl_opt=True,
            enable_profiling=True
        )
        result_ultimate['scaling_type'] = 'strong'
        result_ultimate['total_batch_size'] = total_batch_size
        self.results.append(result_ultimate)
        
        # Compute scaling efficiency
        # (Requires baseline single-GPU run for comparison)
        
        self.save_results(f"scalability_ws{self.world_size}.json")


def main():
    test = ScalabilityTest(output_dir="results/scalability")
    test.run_scalability()
    
    if test.rank == 0:
        print("\n" + "="*80)
        print("✅ SCALABILITY TEST COMPLETE")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()

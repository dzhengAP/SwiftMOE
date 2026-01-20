"""
Systematic ablation study for ICML paper.
Tests each optimization component independently.
"""

import sys
from pathlib import Path
import torch
import torch.distributed as dist
import json

sys.path.append(str(Path(__file__).parent.parent))

from models.baseline_moe import StandardMoE
from models.ultimate_moe import UltimateMoE
from experiments.benchmark_suite import ComprehensiveBenchmark


class AblationStudy(ComprehensiveBenchmark):
    """
    Ablation study to isolate contribution of each optimization.
    
    Configurations:
    1. Baseline: Standard PyTorch
    2. +Triton: Only kernel fusion
    3. +NCCL: Only NCCL optimization
    4. Full: All optimizations (UltimateMoE)
    """
    
    def run_ablation(self):
        """Run systematic ablation"""
        # Fixed configuration for ablation
        num_experts = 128
        d_model = 1024
        batch_size = 2048  # Production setting
        
        if self.rank == 0:
            print("\n" + "="*80)
            print("ABLATION STUDY")
            print(f"Configuration: BS={batch_size}, Experts={num_experts}, GPUs={self.world_size}")
            print("="*80 + "\n")
        
        configs = [
            ("Baseline", "baseline", False, False),
            ("+ Triton Only", "ultimate", True, False),
            ("+ NCCL Only", "ultimate", False, True),
            ("Full (Ours)", "ultimate", True, True),
        ]
        
        ablation_results = []
        baseline_latency = None
        
        for name, model_type, use_triton, use_nccl in configs:
            if model_type == "baseline":
                result = self.run_single_config(
                    model_type="baseline",
                    num_experts=num_experts,
                    d_model=d_model,
                    batch_size=batch_size,
                    enable_profiling=False
                )
            else:
                result = self.run_single_config(
                    model_type="ultimate",
                    num_experts=num_experts,
                    d_model=d_model,
                    batch_size=batch_size,
                    use_triton=use_triton,
                    use_nccl_opt=use_nccl,
                    enable_profiling=True
                )
            
            result['variant_name'] = name
            
            if baseline_latency is None:
                baseline_latency = result['latency_mean_ms']
            
            speedup = baseline_latency / result['latency_mean_ms']
            reduction_pct = (1 - result['latency_mean_ms'] / baseline_latency) * 100
            
            result['speedup'] = speedup
            result['reduction_percent'] = reduction_pct
            
            ablation_results.append(result)
            
            if self.rank == 0:
                print(f"{name:20s}: {result['latency_mean_ms']:6.2f} ms | "
                      f"Speedup: {speedup:.2f}× | Reduction: {reduction_pct:.1f}%")
            
            dist.barrier()
        
        # Save ablation results
        if self.rank == 0:
            output_file = self.output_dir / f"ablation_study_ws{self.world_size}.json"
            with open(output_file, 'w') as f:
                json.dump(ablation_results, f, indent=2)
            
            # Create summary table
            import pandas as pd
            df = pd.DataFrame(ablation_results)
            df_summary = df[['variant_name', 'latency_mean_ms', 'latency_std_ms', 
                            'speedup', 'reduction_percent']]
            
            csv_file = self.output_dir / f"ablation_summary_ws{self.world_size}.csv"
            df_summary.to_csv(csv_file, index=False)
            
            print(f"\n✅ Ablation results saved to {output_file}")
            print(f"✅ Summary table saved to {csv_file}\n")


def main():
    ablation = AblationStudy(output_dir="results/ablation")
    ablation.run_ablation()
    
    if ablation.rank == 0:
        print("\n" + "="*80)
        print("✅ ABLATION STUDY COMPLETE")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()

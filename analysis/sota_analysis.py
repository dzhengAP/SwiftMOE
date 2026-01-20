"""
Analyze merged SOTA comparison results.
Generate LaTeX tables and statistics for paper.
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np


def load_merged_results():
    """Load merged results"""
    merged_file = Path("results/merged/all_frameworks_merged.csv")
    
    if not merged_file.exists():
        print("❌ Merged results not found!")
        print("Please run: python analysis/merge_with_baseline.py")
        return None
    
    return pd.read_csv(merged_file)


def generate_sota_latex_table(df: pd.DataFrame) -> str:
    """Generate SOTA comparison LaTeX table"""
    
    # Filter for 8 GPUs, 128 experts
    df_filtered = df[
        (df['world_size'] == 8) &
        (df['num_experts'] == 128) &
        (df['latency_mean_ms'] > 0)  # Exclude failed runs
    ].copy()
    
    frameworks = ['PyTorch (Baseline)', 'DeepSpeed-MoE', 'Megatron-LM', 'UltimateMoE (Ours)']
    batch_sizes = sorted(df_filtered['batch_size'].unique())
    
    latex_lines = []
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{SOTA comparison on 8× V100 GPUs (128 experts, $d_{\text{model}}=1024$). All values in milliseconds (mean $\pm$ std). Speedup computed relative to PyTorch baseline.}")
    latex_lines.append(r"\label{tab:sota_comparison}")
    latex_lines.append(r"\begin{tabular}{l" + "c" * len(batch_sizes) + "c}")
    latex_lines.append(r"\toprule")
    
    # Header
    bs_headers = " & ".join([f"\\textbf{{{bs}}}" for bs in batch_sizes])
    latex_lines.append(f"\\textbf{{Framework}} & {bs_headers} & \\textbf{{Mean Speedup}} \\\\")
    latex_lines.append(r"\midrule")
    
    # Get baseline latencies
    baseline_lats = {}
    df_baseline = df_filtered[df_filtered['framework'] == 'PyTorch (Baseline)']
    for _, row in df_baseline.iterrows():
        baseline_lats[row['batch_size']] = row['latency_mean_ms']
    
    # Generate rows
    for framework in frameworks:
        df_fw = df_filtered[df_filtered['framework'] == framework]
        
        if df_fw.empty:
            continue
        
        row_values = []
        speedups = []
        
        for bs in batch_sizes:
            df_bs = df_fw[df_fw['batch_size'] == bs]
            
            if not df_bs.empty:
                lat_mean = df_bs['latency_mean_ms'].values[0]
                lat_std = df_bs['latency_std_ms'].values[0]
                
                row_values.append(f"{lat_mean:.2f} $\\pm$ {lat_std:.2f}")
                
                if bs in baseline_lats and baseline_lats[bs] > 0:
                    speedup = baseline_lats[bs] / lat_mean
                    speedups.append(speedup)
            else:
                row_values.append("-")
        
        mean_speedup = np.mean(speedups) if speedups else 1.0
        
        # Format framework name
        if "Ours" in framework:
            fw_text = r"\textbf{" + framework.replace("(Ours)", "") + "}"
            speedup_text = f"\\textbf{{{mean_speedup:.2f}$\\times$}}"
        else:
            fw_text = framework
            speedup_text = f"{mean_speedup:.2f}$\\times$"
        
        row = f"{fw_text} & " + " & ".join(row_values) + f" & {speedup_text} \\\\"
        latex_lines.append(row)
    
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table*}")
    
    return "\n".join(latex_lines)


def print_paper_summary(df: pd.DataFrame):
    """Print summary statistics for paper writing"""
    
    print("\n" + "="*80)
    print("PAPER SUMMARY STATISTICS")
    print("="*80 + "\n")
    
    # Filter for 8 GPUs, 128 experts
    df_filtered = df[
        (df['world_size'] == 8) &
        (df['num_experts'] == 128) &
        (df['latency_mean_ms'] > 0)
    ]
    
    # Get baseline
    df_baseline = df_filtered[df_filtered['framework'] == 'PyTorch (Baseline)']
    
    frameworks_to_compare = ['DeepSpeed-MoE', 'Megatron-LM', 'UltimateMoE (Ours)']
    
    print("📊 SPEEDUP vs BASELINE (8 GPUs, 128 experts):\n")
    
    for framework in frameworks_to_compare:
        df_fw = df_filtered[df_filtered['framework'] == framework]
        
        if df_fw.empty:
            print(f"  {framework}: NO DATA")
            continue
        
        speedups = []
        
        for _, fw_row in df_fw.iterrows():
            bs = fw_row['batch_size']
            base_row = df_baseline[df_baseline['batch_size'] == bs]
            
            if not base_row.empty:
                base_lat = base_row['latency_mean_ms'].values[0]
                fw_lat = fw_row['latency_mean_ms']
                speedup = base_lat / fw_lat
                speedups.append(speedup)
        
        if speedups:
            print(f"  {framework}:")
            print(f"    Mean speedup: {np.mean(speedups):.2f}×")
            print(f"    Max speedup: {np.max(speedups):.2f}×")
            print(f"    Min speedup: {np.min(speedups):.2f}×")
            print(f"    Median speedup: {np.median(speedups):.2f}×")
            print()
    
    # Production setting (BS=2048)
    print("📊 PRODUCTION SETTING (BS=2048, 128 experts, 8 GPUs):\n")
    
    df_prod = df_filtered[df_filtered['batch_size'] == 2048]
    
    if not df_prod.empty:
        baseline_row = df_prod[df_prod['framework'] == 'PyTorch (Baseline)']
        
        if not baseline_row.empty:
            baseline_lat = baseline_row['latency_mean_ms'].values[0]
            
            print(f"  {'Framework':<30} {'Latency':<15} {'Speedup':<10}")
            print(f"  {'-'*55}")
            print(f"  {'PyTorch (Baseline)':<30} {baseline_lat:>6.2f}ms {1.00:>8.2f}×")
            
            for framework in frameworks_to_compare:
                df_fw = df_prod[df_prod['framework'] == framework]
                
                if not df_fw.empty:
                    lat = df_fw['latency_mean_ms'].values[0]
                    speedup = baseline_lat / lat
                    
                    marker = "★" if "Ours" in framework else " "
                    print(f"{marker} {framework:<29} {lat:>6.2f}ms {speedup:>8.2f}×")
    
    print("\n" + "="*80 + "\n")


def main():
    """Generate SOTA analysis"""
    
    df = load_merged_results()
    
    if df is None:
        return
    
    # Generate LaTeX table
    latex_table = generate_sota_latex_table(df)
    
    output_dir = Path("results/tables")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with open(output_dir / "sota_comparison.tex", 'w') as f:
        f.write("% SOTA Comparison Table for ICML Paper\n")
        f.write("% Auto-generated from benchmark results\n\n")
        f.write(latex_table)
    
    print(f"✅ LaTeX table saved to {output_dir}/sota_comparison.tex\n")
    
    # Print summary for paper writing
    print_paper_summary(df)
    
    # Save summary JSON
    summary = {
        'configurations': {
            'gpus': 8,
            'experts': 128,
            'd_model': 1024,
            'batch_sizes': [512, 1024, 2048, 4096, 8192]
        },
        'frameworks_tested': df['framework'].unique().tolist(),
        'total_experiments': len(df)
    }
    
    with open(output_dir.parent / "sota_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("="*80)
    print("✅ SOTA ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

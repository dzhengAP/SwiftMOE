"""
Generate SOTA comparison visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


# ICML style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 300,
})


def load_merged_data():
    """Load merged SOTA data"""
    merged_file = Path("results/merged/all_frameworks_merged.csv")
    
    if not merged_file.exists():
        print("❌ Please run merge first: python analysis/merge_with_baseline.py")
        return None
    
    return pd.read_csv(merged_file)


def plot_sota_bar_comparison():
    """Generate bar chart comparing all frameworks"""
    
    df = load_merged_data()
    if df is None:
        return
    
    # Filter for 8 GPUs, 128 experts
    df_filtered = df[
        (df['world_size'] == 8) &
        (df['num_experts'] == 128) &
        (df['latency_mean_ms'] > 0)
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Latency bars
    batch_sizes = sorted(df_filtered['batch_size'].unique())
    frameworks = ['PyTorch (Baseline)', 'DeepSpeed-MoE', 'Megatron-LM', 'UltimateMoE (Ours)']
    
    x = np.arange(len(batch_sizes))
    width = 0.2
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    
    for i, framework in enumerate(frameworks):
        df_fw = df_filtered[df_filtered['framework'] == framework]
        
        if df_fw.empty:
            continue
        
        lats = []
        for bs in batch_sizes:
            df_bs = df_fw[df_fw['batch_size'] == bs]
            if not df_bs.empty:
                lats.append(df_bs['latency_mean_ms'].values[0])
            else:
                lats.append(0)
        
        offset = (i - len(frameworks)/2 + 0.5) * width
        bars = ax1.bar(x + offset, lats, width, label=framework, 
                      color=colors[i], alpha=0.85)
        
        # Highlight UltimateMoE
        if 'Ours' in framework:
            for bar in bars:
                bar.set_edgecolor('black')
                bar.set_linewidth(2.5)
    
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('End-to-End Latency (ms)')
    ax1.set_title('(a) Latency Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes)
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Speedup over baseline
    for framework in ['DeepSpeed-MoE', 'Megatron-LM', 'UltimateMoE (Ours)']:
        df_fw = df_filtered[df_filtered['framework'] == framework]
        
        if df_fw.empty:
            continue
        
        speedups = []
        valid_bs = []
        
        for bs in batch_sizes:
            base_row = df_filtered[(df_filtered['framework'] == 'PyTorch (Baseline)') & 
                                  (df_filtered['batch_size'] == bs)]
            fw_row = df_fw[df_fw['batch_size'] == bs]
            
            if not base_row.empty and not fw_row.empty:
                base_lat = base_row['latency_mean_ms'].values[0]
                fw_lat = fw_row['latency_mean_ms'].values[0]
                speedup = base_lat / fw_lat
                
                speedups.append(speedup)
                valid_bs.append(bs)
        
        marker = 's' if 'Ours' in framework else 'o'
        linewidth = 3 if 'Ours' in framework else 2
        markersize = 10 if 'Ours' in framework else 8
        
        ax2.plot(valid_bs, speedups, marker=marker, label=framework,
                linewidth=linewidth, markersize=markersize, alpha=0.85)
    
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, 
               linewidth=1.5, label='Baseline (1×)')
    
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Speedup vs PyTorch Baseline')
    ax2.set_title('(b) Relative Speedup')
    ax2.set_xscale('log')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path("results/figures")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    plt.savefig(output_dir / 'sota_comparison.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'sota_comparison.png', bbox_inches='tight')
    plt.close()
    
    print("✅ Generated SOTA comparison plot")


def plot_memory_comparison():
    """Compare memory footprint across frameworks"""
    
    df = load_merged_data()
    if df is None:
        return
    
    df_filtered = df[
        (df['world_size'] == 8) &
        (df['num_experts'] == 128) &
        (df['max_memory_gb'] > 0)
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    frameworks = ['PyTorch (Baseline)', 'DeepSpeed-MoE', 'Megatron-LM', 'UltimateMoE (Ours)']
    markers = ['o', 's', '^', 'D']
    
    for framework, marker in zip(frameworks, markers):
        df_fw = df_filtered[df_filtered['framework'] == framework]
        
        if df_fw.empty:
            continue
        
        df_fw = df_fw.sort_values('batch_size')
        
        linewidth = 3 if 'Ours' in framework else 2
        markersize = 10 if 'Ours' in framework else 8
        
        ax.plot(
            df_fw['batch_size'],
            df_fw['max_memory_gb'],
            marker=marker,
            label=framework,
            linewidth=linewidth,
            markersize=markersize,
            alpha=0.85
        )
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Peak Memory (GB)')
    ax.set_title('Memory Footprint Comparison (8 GPUs, 128 Experts)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path("results/figures")
    plt.savefig(output_dir / 'sota_memory_comparison.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'sota_memory_comparison.png', bbox_inches='tight')
    plt.close()
    
    print("✅ Generated memory comparison plot")


def main():
    """Generate all SOTA visualizations"""
    
    print("\n" + "="*80)
    print("GENERATING SOTA COMPARISON PLOTS")
    print("="*80 + "\n")
    
    try:
        plot_sota_bar_comparison()
        plot_memory_comparison()
        
        print("\n" + "="*80)
        print("✅ SOTA PLOTS COMPLETE")
        print("="*80)
        print("\nGenerated files:")
        print("  - results/figures/sota_comparison.pdf")
        print("  - results/figures/sota_memory_comparison.pdf")
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have run:")
        print("  1. bash scripts/run_sota_only.sh")
        print("  2. python analysis/merge_with_baseline.py")


if __name__ == "__main__":
    main()

"""
Generate ICML-quality figures.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# ICML style configuration
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 2,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def load_data():
    """Load aggregated results"""
    df_stats = pd.read_csv("results/aggregated/summary_statistics.csv")
    df_speedup = pd.read_csv("results/aggregated/speedup_analysis.csv")
    return df_stats, df_speedup


def plot_latency_breakdown(df: pd.DataFrame, output_dir: Path):
    """Plot latency breakdown (stacked bar chart)"""
    # Filter for 8 GPUs, 128 experts, UltimateMoE with profiling
    df_filtered = df[
        (df['world_size'] == 8) & 
        (df['num_experts'] == 128) &
        (df['model_type'] == 'ultimate')
    ].copy()
    
    if df_filtered.empty or 'Router' not in df_filtered.columns:
        print("⚠️  No profiling data available for latency breakdown")
        return
    
    # Create stacked bar chart
    stages = ['Router', 'Dispatch_Comm', 'Expert_Compute', 'Combine_Comm']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    batch_sizes = sorted(df_filtered['batch_size'].unique())
    x = np.arange(len(batch_sizes))
    width = 0.6
    
    bottom = np.zeros(len(batch_sizes))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
    
    for i, stage in enumerate(stages):
        values = [df_filtered[df_filtered['batch_size'] == bs][stage].values[0] 
                 if not df_filtered[df_filtered['batch_size'] == bs].empty else 0
                 for bs in batch_sizes]
        
        ax.bar(x, values, width, label=stage, bottom=bottom, color=colors[i], alpha=0.8)
        bottom += values
    
    ax.set_xlabel('Batch Size (tokens)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('(a) Latency Breakdown (WS=8, Experts=128)')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(axis='y', alpha=0.3)
    
    plt.savefig(output_dir / 'latency_breakdown.pdf')
    plt.savefig(output_dir / 'latency_breakdown.png')
    plt.close()
    
    print(f"✅ Generated latency breakdown plot")


def plot_throughput_latency(df: pd.DataFrame, output_dir: Path):
    """Plot throughput vs latency Pareto frontier"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot baseline
    df_baseline = df[df['model_type'] == 'baseline']
    for num_exp in df_baseline['num_experts'].unique():
        df_exp = df_baseline[df_baseline['num_experts'] == num_exp]
        ax.scatter(
            df_exp['latency_mean_ms'],
            df_exp['throughput_tokens_per_sec'],
            label=f'Baseline ({num_exp} experts)',
            alpha=0.6,
            s=100,
            marker='o'
        )
    
    # Plot ultimate
    df_ultimate = df[df['model_type'] == 'ultimate']
    for num_exp in df_ultimate['num_experts'].unique():
        df_exp = df_ultimate[df_exp['num_experts'] == num_exp]
        ax.scatter(
            df_exp['latency_mean_ms'],
            df_exp['throughput_tokens_per_sec'],
            label=f'UltimateMoE ({num_exp} experts)',
            alpha=0.6,
            s=100,
            marker='s'
        )
    
    ax.set_xlabel('Latency (ms, log scale)')
    ax.set_ylabel('Throughput (tokens/s)')
    ax.set_title('(b) Throughput vs. Latency Pareto Frontier')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / 'throughput_latency.pdf')
    plt.savefig(output_dir / 'throughput_latency.png')
    plt.close()
    
    print(f"✅ Generated throughput-latency plot")


def plot_scalability(df_speedup: pd.DataFrame, output_dir: Path):
    """Plot scaling efficiency"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter for fixed batch size
    batch_size = 2048
    df_filtered = df_speedup[df_speedup['batch_size'] == batch_size]
    
    for num_exp in df_filtered['num_experts'].unique():
        df_exp = df_filtered[df_filtered['num_experts'] == num_exp]
        
        # Compute speedup relative to single GPU
        if 1 in df_exp['world_size'].values:
            base_throughput = df_exp[df_exp['world_size'] == 1]['throughput_tokens_per_sec_ultimate'].values[0]
            
            world_sizes = []
            speedups = []
            
            for ws in sorted(df_exp['world_size'].unique()):
                tp = df_exp[df_exp['world_size'] == ws]['throughput_tokens_per_sec_ultimate'].values[0]
                world_sizes.append(ws)
                speedups.append(tp / base_throughput)
            
            ax.plot(world_sizes, speedups, marker='o', label=f'{num_exp} experts', linewidth=2)
    
    # Plot ideal linear scaling
    max_ws = df_filtered['world_size'].max()
    ax.plot([1, max_ws], [1, max_ws], 'k--', alpha=0.5, label='Ideal Linear', linewidth=1.5)
    
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Speedup Ratio')
    ax.set_title(f'(d) Scalability: Speedup (BS={batch_size})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / 'scalability.pdf')
    plt.savefig(output_dir / 'scalability.png')
    plt.close()
    
    print(f"✅ Generated scalability plot")


def plot_memory_footprint(df: pd.DataFrame, output_dir: Path):
    """Plot memory consumption"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter for 8 GPUs
    df_filtered = df[(df['world_size'] == 8) & (df['model_type'] == 'ultimate')]
    
    for num_exp in df_filtered['num_experts'].unique():
        df_exp = df_filtered[df_filtered['num_experts'] == num_exp]
        df_exp = df_exp.sort_values('batch_size')
        
        ax.plot(
            df_exp['batch_size'],
            df_exp['max_memory_gb'],
            marker='s',
            label=f'{num_exp} experts',
            linewidth=2
        )
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Memory Usage (GB)')
    ax.set_title('(c) Peak HBM Consumption (WS=8)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / 'memory_footprint.pdf')
    plt.savefig(output_dir / 'memory_footprint.png')
    plt.close()
    
    print(f"✅ Generated memory footprint plot")


def plot_ablation_bars(output_dir: Path):
    """Plot ablation study results"""
    # Load ablation data
    ablation_files = list(Path("results/ablation").glob("ablation_summary_*.csv"))
    
    if not ablation_files:
        print("⚠️  No ablation results found")
        return
    
    # Use 8-GPU results
    df_ablation = pd.read_csv("results/ablation/ablation_summary_ws8.csv")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    variants = df_ablation['variant_name']
    latencies = df_ablation['latency_mean_ms']
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    bars = ax.bar(range(len(variants)), latencies, color=colors, alpha=0.8)
    
    # Add speedup labels
    baseline_lat = latencies.iloc[0]
    for i, (bar, lat) in enumerate(zip(bars, latencies)):
        speedup = baseline_lat / lat
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f'{speedup:.2f}×',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=15, ha='right')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Ablation Study (BS=2048, 128 Experts, 8 GPUs)')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_bars.pdf')
    plt.savefig(output_dir / 'ablation_bars.png')
    plt.close()
    
    print(f"✅ Generated ablation bar chart")


def generate_all_figures():
    """Generate all paper figures"""
    print("\n" + "="*80)
    print("GENERATING ICML PAPER FIGURES")
    print("="*80 + "\n")
    
    # Create output directory
    output_dir = Path("results/figures")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    try:
        df_stats, df_speedup = load_data()
        
        # Load raw data for profiling info
        import glob
        import json
        
        all_raw_data = []
        for json_file in glob.glob("results/**/*.json", recursive=True):
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_raw_data.extend(data)
                else:
                    all_raw_data.append(data)
        
        df_raw = pd.DataFrame(all_raw_data)
        
        # Generate plots
        plot_latency_breakdown(df_raw, output_dir)
        plot_throughput_latency(df_raw, output_dir)
        plot_scalability(df_speedup, output_dir)
        plot_memory_footprint(df_raw, output_dir)
        plot_ablation_bars(output_dir)
        
        print("\n" + "="*80)
        print(f"✅ ALL FIGURES GENERATED IN {output_dir}")
        print("="*80 + "\n")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run aggregation first: python analysis/aggregate_results.py")


if __name__ == "__main__":
    generate_all_figures()

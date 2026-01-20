"""
统一聚合所有实验结果（comprehensive + sota_comparison）
"""

import json
import glob
from pathlib import Path
import pandas as pd
import numpy as np


def load_all_results(results_base_dir: str = "results"):
    """从所有子目录加载结果"""
    all_data = []
    
    # 加载 comprehensive benchmarks
    comprehensive_files = glob.glob(f"{results_base_dir}/comprehensive/*.json")
    for file_path in comprehensive_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    item['experiment_type'] = 'comprehensive'
                    all_data.append(item)
            else:
                data['experiment_type'] = 'comprehensive'
                all_data.append(data)
    
    # 加载 SOTA comparison
    sota_files = glob.glob(f"{results_base_dir}/sota_comparison/*.json")
    for file_path in sota_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    item['experiment_type'] = 'sota'
                    all_data.append(item)
            else:
                data['experiment_type'] = 'sota'
                all_data.append(data)
    
    # 加载 ablation
    ablation_files = glob.glob(f"{results_base_dir}/ablation/*.json")
    for file_path in ablation_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    item['experiment_type'] = 'ablation'
                    all_data.append(item)
            else:
                data['experiment_type'] = 'ablation'
                all_data.append(data)
    
    return pd.DataFrame(all_data)


def compute_unified_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """计算统一的统计数据"""
    
    # 分组统计
    grouped = df.groupby([
        'experiment_type', 'model_type', 'world_size', 
        'batch_size', 'num_experts'
    ], dropna=False)
    
    stats = grouped.agg({
        'latency_mean_ms': ['mean', 'std', 'min', 'max'],
        'throughput_tokens_per_sec': ['mean', 'std'],
        'max_memory_gb': ['mean', 'max']
    }).reset_index()
    
    return stats


def main():
    print("\n" + "="*80)
    print("AGGREGATING ALL EXPERIMENTAL RESULTS")
    print("="*80 + "\n")
    
    df = load_all_results("results")
    
    print(f"✅ Loaded {len(df)} result records")
    print(f"   - Comprehensive: {len(df[df['experiment_type'] == 'comprehensive'])}")
    print(f"   - SOTA: {len(df[df['experiment_type'] == 'sota'])}")
    print(f"   - Ablation: {len(df[df['experiment_type'] == 'ablation'])}")
    print()
    
    # Compute statistics
    stats = compute_unified_statistics(df)
    
    # Save
    output_dir = Path("results/aggregated")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    df.to_csv(output_dir / "all_results_raw.csv", index=False)
    stats.to_csv(output_dir / "all_results_statistics.csv", index=False)
    
    print(f"✅ Saved to {output_dir}/")
    print("   - all_results_raw.csv")
    print("   - all_results_statistics.csv")
    print()
    
    # Compute speedups (如果有 baseline 和其他模型)
    if 'baseline' in df['model_type'].values:
        speedup_df = compute_speedup_table(df)
        speedup_df.to_csv(output_dir / "speedup_analysis.csv", index=False)
        print(f"✅ Saved speedup analysis")
    
    print("\n" + "="*80)
    print("✅ AGGREGATION COMPLETE")
    print("="*80 + "\n")


def compute_speedup_table(df: pd.DataFrame) -> pd.DataFrame:
    """计算 speedup（兼容 comprehensive 和 sota 数据）"""
    
    speedups = []
    
    # 对于每个配置，计算相对于 baseline 的加速
    for exp_type in df['experiment_type'].unique():
        df_exp = df[df['experiment_type'] == exp_type]
        
        # 获取 baseline
        df_baseline = df_exp[df_exp['model_type'] == 'baseline']
        
        for _, base_row in df_baseline.iterrows():
            ws = base_row['world_size']
            bs = base_row['batch_size']
            ne = base_row['num_experts']
            base_lat = base_row['latency_mean_ms']
            
            # 查找其他模型
            df_others = df_exp[
                (df_exp['world_size'] == ws) &
                (df_exp['batch_size'] == bs) &
                (df_exp['num_experts'] == ne) &
                (df_exp['model_type'] != 'baseline')
            ]
            
            for _, other_row in df_others.iterrows():
                speedup_record = {
                    'experiment_type': exp_type,
                    'world_size': ws,
                    'batch_size': bs,
                    'num_experts': ne,
                    'model_type': other_row['model_type'],
                    'framework': other_row.get('framework', other_row['model_type']),
                    'baseline_latency_ms': base_lat,
                    'optimized_latency_ms': other_row['latency_mean_ms'],
                    'speedup': base_lat / other_row['latency_mean_ms'],
                    'reduction_percent': (1 - other_row['latency_mean_ms'] / base_lat) * 100
                }
                speedups.append(speedup_record)
    
    return pd.DataFrame(speedups)


if __name__ == "__main__":
    main()

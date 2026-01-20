"""
Merge SOTA results with existing baseline/ultimate results.
Creates unified dataset for paper tables.
"""

import json
import pandas as pd
from pathlib import Path


def load_comprehensive_results():
    """Load baseline and ultimate results from comprehensive benchmark"""
    comprehensive_dir = Path("results/comprehensive")
    
    all_data = []
    for json_file in comprehensive_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
    
    return pd.DataFrame(all_data)


def load_sota_results():
    """Load DeepSpeed and Megatron results"""
    sota_dir = Path("results/sota_comparison")
    
    all_data = []
    for json_file in sota_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
    
    return pd.DataFrame(all_data)


def merge_results():
    """Merge all results into unified dataset"""
    
    print("\n" + "="*80)
    print("MERGING COMPREHENSIVE + SOTA RESULTS")
    print("="*80 + "\n")
    
    # Load both datasets
    df_comprehensive = load_comprehensive_results()
    df_sota = load_sota_results()
    
    print(f"Comprehensive results: {len(df_comprehensive)} records")
    print(f"SOTA results: {len(df_sota)} records")
    
    # Add framework labels if missing
    if 'framework' not in df_comprehensive.columns:
        df_comprehensive['framework'] = df_comprehensive['model_type'].map({
            'baseline': 'PyTorch (Baseline)',
            'ultimate': 'UltimateMoE (Ours)'
        })
    
    # Combine
    df_merged = pd.concat([df_comprehensive, df_sota], ignore_index=True)
    
    print(f"Merged total: {len(df_merged)} records\n")
    
    # Filter for 8 GPUs, 128 experts (SOTA comparison config)
    df_sota_config = df_merged[
        (df_merged['world_size'] == 8) &
        (df_merged['num_experts'] == 128)
    ].copy()
    
    # Create comparison table
    frameworks = ['PyTorch (Baseline)', 'DeepSpeed-MoE', 'Megatron-LM', 'UltimateMoE (Ours)']
    
    print("="*80)
    print("SOTA COMPARISON (8 GPUs, 128 Experts)")
    print("="*80 + "\n")
    
    print(f"{'Framework':<25} {'BS=512':<12} {'BS=1024':<12} {'BS=2048':<12} {'BS=4096':<12} {'BS=8192':<12}")
    print("-" * 85)
    
    baseline_lats = {}
    
    for framework in frameworks:
        df_fw = df_sota_config[df_sota_config['framework'] == framework]
        
        if df_fw.empty:
            print(f"{framework:<25} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            continue
        
        row_str = f"{framework:<25}"
        
        for bs in [512, 1024, 2048, 4096, 8192]:
            df_bs = df_fw[df_fw['batch_size'] == bs]
            
            if not df_bs.empty:
                lat = df_bs['latency_mean_ms'].values[0]
                
                # Store baseline for speedup calculation
                if 'Baseline' in framework:
                    baseline_lats[bs] = lat
                
                # Calculate speedup if we have baseline
                if bs in baseline_lats and baseline_lats[bs] > 0:
                    speedup = baseline_lats[bs] / lat
                    row_str += f"{lat:>6.2f}ms ({speedup:.2f}×) "
                else:
                    row_str += f"{lat:>6.2f}ms      "
            else:
                row_str += f"{'N/A':<12}"
        
        print(row_str)
    
    print("\n" + "="*80 + "\n")
    
    # Save merged dataset
    output_dir = Path("results/merged")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    df_merged.to_csv(output_dir / "all_frameworks_merged.csv", index=False)
    df_sota_config.to_csv(output_dir / "sota_comparison_8gpu_128exp.csv", index=False)
    
    with open(output_dir / "all_frameworks_merged.json", 'w') as f:
        json.dump(df_merged.to_dict('records'), f, indent=2)
    
    print(f"✅ Merged results saved to {output_dir}/")
    print(f"   - all_frameworks_merged.csv")
    print(f"   - all_frameworks_merged.json")
    print(f"   - sota_comparison_8gpu_128exp.csv")
    print("\n" + "="*80 + "\n")
    
    return df_merged


def main():
    df_merged = merge_results()
    
    # Compute summary statistics
    print("Computing summary statistics...")
    
    df_sota = df_merged[
        (df_merged['world_size'] == 8) &
        (df_merged['num_experts'] == 128)
    ]
    
    frameworks = df_sota['framework'].unique()
    
    for framework in frameworks:
        df_fw = df_sota[df_sota['framework'] == framework]
        
        if len(df_fw) > 0:
            mean_lat = df_fw['latency_mean_ms'].mean()
            mean_tp = df_fw['throughput_tokens_per_sec'].mean()
            mean_mem = df_fw['max_memory_gb'].mean()
            
            print(f"\n{framework}:")
            print(f"  Avg Latency: {mean_lat:.2f}ms")
            print(f"  Avg Throughput: {mean_tp/1000:.0f}K tokens/s")
            print(f"  Avg Memory: {mean_mem:.2f}GB")
    
    print("\n" + "="*80)
    print("✅ MERGE COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

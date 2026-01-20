"""
Comprehensive test script to verify all components are present and functional.
Run this before executing any experiments.
"""

import sys
import os
from pathlib import Path
import importlib.util
from typing import Dict, List, Tuple

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}\n")


def print_result(status: bool, message: str):
    """Print test result"""
    if status:
        print(f"{Colors.GREEN}✅{Colors.RESET} {message}")
    else:
        print(f"{Colors.RED}❌{Colors.RESET} {message}")
    return status


def check_file_exists(filepath: str) -> bool:
    """Check if file exists"""
    return Path(filepath).exists()


def check_module_imports(module_path: str, required_classes: List[str]) -> Tuple[bool, List[str]]:
    """Check if module can be imported and contains required classes"""
    try:
        spec = importlib.util.spec_from_file_location("test_module", module_path)
        if spec is None or spec.loader is None:
            return False, [f"Cannot load module: {module_path}"]
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        missing = []
        for cls in required_classes:
            if not hasattr(module, cls):
                missing.append(cls)
        
        return len(missing) == 0, missing
    
    except Exception as e:
        return False, [str(e)]


def test_directory_structure() -> Dict[str, bool]:
    """Test if all required directories exist"""
    print_header("TEST 1: Directory Structure")
    
    required_dirs = [
        "models",
        "experiments",
        "utils",
        "analysis",
        "scripts",
        "results",
    ]
    
    results = {}
    
    for dir_name in required_dirs:
        exists = Path(dir_name).is_dir()
        results[dir_name] = print_result(exists, f"Directory: {dir_name}/")
        
        if not exists:
            print(f"  {Colors.YELLOW}→ Create with: mkdir -p {dir_name}{Colors.RESET}")
    
    return results


def test_model_files() -> Dict[str, bool]:
    """Test model implementation files"""
    print_header("TEST 2: Model Implementation Files")
    
    model_tests = {
        'models/__init__.py': [],
        'models/baseline_moe.py': ['StandardMoE'],
        'models/ultimate_moe.py': [
            'UltimateMoE', 
            'MoEAsyncCommHandler',
            'setup_nccl_optimizations',
            'fused_router_dispatch_kernel'
        ],
        'models/deepspeed_wrapper.py': [
            'DeepSpeedMoEWrapper',
            'MegatronMoEWrapper'
        ]
    }
    
    results = {}
    
    for filepath, required_classes in model_tests.items():
        file_exists = check_file_exists(filepath)
        
        if not file_exists:
            results[filepath] = print_result(False, f"File: {filepath}")
            print(f"  {Colors.YELLOW}→ This file needs to be created{Colors.RESET}")
            continue
        
        if not required_classes:
            results[filepath] = print_result(True, f"File: {filepath}")
            continue
        
        can_import, missing = check_module_imports(filepath, required_classes)
        
        if can_import:
            results[filepath] = print_result(True, f"File: {filepath}")
            print(f"  {Colors.GREEN}  Classes: {', '.join(required_classes)}{Colors.RESET}")
        else:
            results[filepath] = print_result(False, f"File: {filepath}")
            print(f"  {Colors.RED}  Missing: {', '.join(missing)}{Colors.RESET}")
    
    return results


def test_experiment_files() -> Dict[str, bool]:
    """Test experiment script files"""
    print_header("TEST 3: Experiment Scripts")
    
    experiment_tests = {
        'experiments/__init__.py': [],
        'experiments/benchmark_suite.py': ['ComprehensiveBenchmark'],
        'experiments/sota_comparison.py': ['SOTAComparison', 'DeepSpeedMoEWrapper', 'MegatronMoEWrapper'],
        'experiments/ablation_study.py': ['AblationStudy'],
        'experiments/scalability_test.py': ['ScalabilityTest']
    }
    
    results = {}
    
    for filepath, required_classes in experiment_tests.items():
        file_exists = check_file_exists(filepath)
        
        if not file_exists:
            results[filepath] = print_result(False, f"File: {filepath}")
            print(f"  {Colors.YELLOW}→ This file needs to be created{Colors.RESET}")
            continue
        
        if not required_classes:
            results[filepath] = print_result(True, f"File: {filepath}")
            continue
        
        can_import, missing = check_module_imports(filepath, required_classes)
        
        if can_import:
            results[filepath] = print_result(True, f"File: {filepath}")
        else:
            results[filepath] = print_result(False, f"File: {filepath}")
            print(f"  {Colors.RED}  Missing: {', '.join(missing)}{Colors.RESET}")
    
    return results


def test_analysis_files() -> Dict[str, bool]:
    """Test analysis script files"""
    print_header("TEST 4: Analysis Scripts")
    
    analysis_files = [
        'analysis/__init__.py',
        'analysis/aggregate_results.py',
        'analysis/merge_with_baseline.py',
        'analysis/sota_analysis.py',
        'analysis/paper_tables.py',
        'analysis/statistical_analysis.py'
    ]
    
    results = {}
    
    for filepath in analysis_files:
        exists = check_file_exists(filepath)
        results[filepath] = print_result(exists, f"File: {filepath}")
        
        if not exists:
            print(f"  {Colors.YELLOW}→ This file needs to be created{Colors.RESET}")
    
    return results


def test_utility_files() -> Dict[str, bool]:
    """Test utility script files"""
    print_header("TEST 5: Utility Scripts")
    
    utility_files = [
        'utils/__init__.py',
        'utils/visualization.py',
        'utils/advanced_plots.py',
        'utils/sota_plots.py'
    ]
    
    results = {}
    
    for filepath in utility_files:
        exists = check_file_exists(filepath)
        results[filepath] = print_result(exists, f"File: {filepath}")
        
        if not exists:
            print(f"  {Colors.YELLOW}→ This file needs to be created{Colors.RESET}")
    
    return results


def test_bash_scripts() -> Dict[str, bool]:
    """Test bash execution scripts"""
    print_header("TEST 6: Bash Scripts")
    
    bash_scripts = [
        'scripts/run_quick_test.sh',
        'scripts/run_all_experiments.sh',
        'scripts/run_sota_only.sh',
        'scripts/analyze_current_results.sh'
    ]
    
    results = {}
    
    for filepath in bash_scripts:
        exists = check_file_exists(filepath)
        
        if exists:
            # Check if executable
            is_executable = os.access(filepath, os.X_OK)
            if is_executable:
                results[filepath] = print_result(True, f"Script: {filepath} (executable)")
            else:
                results[filepath] = print_result(True, f"Script: {filepath}")
                print(f"  {Colors.YELLOW}→ Make executable: chmod +x {filepath}{Colors.RESET}")
        else:
            results[filepath] = print_result(False, f"Script: {filepath}")
            print(f"  {Colors.YELLOW}→ This file needs to be created{Colors.RESET}")
    
    return results


def test_dependencies() -> Dict[str, bool]:
    """Test if required Python packages are installed"""
    print_header("TEST 7: Python Dependencies")
    
    required_packages = {
        'torch': 'PyTorch',
        'triton': 'Triton',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'scipy': 'SciPy (for statistical tests)',
        'deepspeed': 'DeepSpeed (optional for SOTA)'
    }
    
    results = {}
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            results[package] = print_result(True, f"{description}")
        except ImportError:
            is_optional = 'optional' in description.lower()
            
            if is_optional:
                print_result(True, f"{description} {Colors.YELLOW}(not installed, optional){Colors.RESET}")
                results[package] = True
            else:
                results[package] = print_result(False, f"{description}")
                print(f"  {Colors.YELLOW}→ Install with: pip install {package}{Colors.RESET}")
    
    return results


def test_torch_distributed() -> bool:
    """Test if torch.distributed is available"""
    print_header("TEST 8: Distributed Training Support")
    
    try:
        import torch
        import torch.distributed as dist
        
        has_nccl = torch.cuda.is_available() and torch.cuda.nccl.is_available()
        has_gloo = dist.is_gloo_available()
        
        print_result(True, f"torch.distributed available")
        print_result(has_nccl, f"NCCL backend (GPU): {torch.cuda.nccl.version() if has_nccl else 'N/A'}")
        print_result(has_gloo, f"Gloo backend (CPU)")
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print_result(num_gpus > 0, f"Available GPUs: {num_gpus}")
            
            if num_gpus > 0:
                for i in range(num_gpus):
                    props = torch.cuda.get_device_properties(i)
                    print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
        else:
            print_result(False, "CUDA not available")
        
        return has_nccl
    
    except Exception as e:
        print_result(False, f"Error checking distributed: {e}")
        return False


def test_triton_kernels() -> bool:
    """Test if Triton kernels can compile"""
    print_header("TEST 9: Triton Kernel Compilation")
    
    try:
        import torch
        import triton
        import triton.language as tl
        
        # Test simple kernel
        @triton.jit
        def test_kernel(x_ptr, output_ptr, N: tl.constexpr):
            pid = tl.program_id(0)
            idx = pid * 64 + tl.arange(0, 64)
            mask = idx < N
            x = tl.load(x_ptr + idx, mask=mask, other=0.0)
            tl.store(output_ptr + idx, x * 2.0, mask=mask)
        
        if torch.cuda.is_available():
            x = torch.randn(128, device='cuda')
            output = torch.empty_like(x)
            
            grid = (triton.cdiv(128, 64),)
            test_kernel[grid](x, output, 128)
            
            torch.cuda.synchronize()
            
            print_result(True, "Triton kernel compilation works")
            print_result(True, f"Triton version: {triton.__version__}")
            return True
        else:
            print_result(False, "CUDA not available for Triton test")
            return False
    
    except Exception as e:
        print_result(False, f"Triton kernel test failed: {e}")
        return False


def test_existing_results() -> bool:
    """Check if there are existing benchmark results"""
    print_header("TEST 10: Existing Results")
    
    results_found = False
    
    # Check comprehensive results
    comp_dir = Path("results/comprehensive")
    if comp_dir.exists():
        json_files = list(comp_dir.glob("*.json"))
        if json_files:
            print_result(True, f"Found comprehensive results: {len(json_files)} files")
            results_found = True
        else:
            print_result(False, "No comprehensive results found")
    else:
        print_result(False, "results/comprehensive/ directory not found")
    
    # Check SOTA results
    sota_dir = Path("results/sota_comparison")
    if sota_dir.exists():
        json_files = list(sota_dir.glob("*.json"))
        if json_files:
            print_result(True, f"Found SOTA results: {len(json_files)} files")
            results_found = True
        else:
            print_result(False, "No SOTA results yet (will be generated)")
    else:
        print_result(False, "results/sota_comparison/ not found (will be created)")
    
    # Check merged results
    merged_dir = Path("results/merged")
    if merged_dir.exists():
        if (merged_dir / "all_frameworks_merged.csv").exists():
            print_result(True, "Found merged results")
            results_found = True
        else:
            print_result(False, "No merged results yet")
    else:
        print_result(False, "results/merged/ not found (will be created)")
    
    return results_found


def generate_missing_init_files():
    """Generate missing __init__.py files"""
    print_header("GENERATING MISSING __init__.py FILES")
    
    init_files = [
        'models/__init__.py',
        'experiments/__init__.py',
        'utils/__init__.py',
        'analysis/__init__.py'
    ]
    
    for init_file in init_files:
        if not Path(init_file).exists():
            Path(init_file).parent.mkdir(exist_ok=True, parents=True)
            with open(init_file, 'w') as f:
                f.write('"""Package initialization"""\n')
            print_result(True, f"Created: {init_file}")
        else:
            print_result(True, f"Exists: {init_file}")


def test_function_existence() -> Dict[str, bool]:
    """Test if key functions exist in files"""
    print_header("TEST 11: Function Existence Check")
    
    function_tests = {
        'models/baseline_moe.py': {
            'classes': ['StandardMoE'],
            'methods': ['forward', '_route_tokens', '_dispatch_tokens', '_expert_forward', '_combine_tokens']
        },
        'models/ultimate_moe.py': {
            'classes': ['UltimateMoE', 'MoEAsyncCommHandler'],
            'methods': ['forward', '_expert_compute', 'get_profile_summary']
        },
        'experiments/benchmark_suite.py': {
            'classes': ['ComprehensiveBenchmark'],
            'methods': ['run_single_config', 'measure_latency', 'warmup']
        }
    }
    
    results = {}
    
    for filepath, checks in function_tests.items():
        if not Path(filepath).exists():
            results[filepath] = print_result(False, f"File missing: {filepath}")
            continue
        
        try:
            spec = importlib.util.spec_from_file_location("test_module", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            all_good = True
            missing_items = []
            
            # Check classes
            for cls_name in checks.get('classes', []):
                if not hasattr(module, cls_name):
                    all_good = False
                    missing_items.append(f"class {cls_name}")
                else:
                    cls = getattr(module, cls_name)
                    # Check methods
                    for method_name in checks.get('methods', []):
                        if not hasattr(cls, method_name):
                            all_good = False
                            missing_items.append(f"{cls_name}.{method_name}")
            
            if all_good:
                results[filepath] = print_result(True, f"File: {filepath}")
                print(f"  {Colors.GREEN}  All required components present{Colors.RESET}")
            else:
                results[filepath] = print_result(False, f"File: {filepath}")
                print(f"  {Colors.RED}  Missing: {', '.join(missing_items)}{Colors.RESET}")
        
        except Exception as e:
            results[filepath] = print_result(False, f"File: {filepath}")
            print(f"  {Colors.RED}  Error: {e}{Colors.RESET}")
    
    return results


def test_gpu_availability() -> bool:
    """Test GPU availability and configuration"""
    print_header("TEST 12: GPU Configuration")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print_result(False, "CUDA not available")
            return False
        
        num_gpus = torch.cuda.device_count()
        print_result(num_gpus >= 1, f"Available GPUs: {num_gpus}")
        
        if num_gpus < 8:
            print(f"  {Colors.YELLOW}⚠️ Recommended: 8 GPUs for full experiments{Colors.RESET}")
            print(f"  {Colors.YELLOW}  You can still run with {num_gpus} GPU(s){Colors.RESET}")
        
        # Test each GPU
        for i in range(min(num_gpus, 8)):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {memory_gb:.1f} GB")
            print(f"    Compute: {props.major}.{props.minor}")
            
            # Test basic operation
            try:
                x = torch.randn(100, 100, device=f'cuda:{i}')
                y = torch.mm(x, x)
                torch.cuda.synchronize()
                print(f"    {Colors.GREEN}✓ Basic operations work{Colors.RESET}")
            except Exception as e:
                print(f"    {Colors.RED}✗ Error: {e}{Colors.RESET}")
        
        return True
    
    except Exception as e:
        print_result(False, f"GPU test failed: {e}")
        return False


def create_test_data_sample():
    """Create a tiny test run to verify everything works"""
    print_header("TEST 13: Quick Functional Test")
    
    try:
        import torch
        import torch.distributed as dist
        
        if not torch.cuda.is_available():
            print_result(False, "CUDA not available, skipping functional test")
            return False
        
        print("Testing model instantiation (single GPU)...")
        
        # Test baseline model
        sys.path.insert(0, str(Path.cwd()))
        
        try:
            from models.baseline_moe import StandardMoE
            
            model = StandardMoE(num_experts=8, d_model=128, use_fp16=True)
            model = model.cuda()
            
            x = torch.randn(32, 128, device='cuda', dtype=torch.float16)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
            
            print_result(True, "StandardMoE (Baseline) works")
        
        except Exception as e:
            print_result(False, f"StandardMoE failed: {e}")
            return False
        
        # Test ultimate model
        try:
            from models.ultimate_moe import UltimateMoE
            
            model = UltimateMoE(num_experts=8, d_model=128, use_fp16=True, use_triton=False)
            model = model.cuda()
            
            x = torch.randn(32, 128, device='cuda', dtype=torch.float16)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (32, 128), f"Output shape mismatch: {output.shape}"
            
            print_result(True, "UltimateMoE works (Triton disabled)")
        
        except Exception as e:
            print_result(False, f"UltimateMoE failed: {e}")
            return False
        
        # Test Triton kernel
        try:
            from models.ultimate_moe import UltimateMoE
            
            model = UltimateMoE(num_experts=8, d_model=128, use_fp16=True, use_triton=True)
            model = model.cuda()
            
            x = torch.randn(32, 128, device='cuda', dtype=torch.float16)
            
            with torch.no_grad():
                output = model(x)
            
            print_result(True, "UltimateMoE works (Triton enabled)")
        
        except Exception as e:
            print_result(False, f"Triton kernel failed: {e}")
            print(f"  {Colors.YELLOW}→ This might be OK if Triton compilation needs NCCL init{Colors.RESET}")
        
        return True
    
    except Exception as e:
        print_result(False, f"Functional test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_summary_report(all_results: Dict[str, Dict[str, bool]]):
    """Generate final summary report"""
    print_header("FINAL SUMMARY")
    
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        category_passed = sum(results.values())
        category_total = len(results)
        
        total_tests += category_total
        passed_tests += category_passed
        
        status = "✅" if category_passed == category_total else "⚠️"
        print(f"{status} {category}: {category_passed}/{category_total} passed")
    
    print(f"\n{Colors.BOLD}Overall: {passed_tests}/{total_tests} tests passed{Colors.RESET}")
    
    if passed_tests == total_tests:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! Ready to run experiments.{Colors.RESET}\n")
        print("Next steps:")
        print(f"  {Colors.BLUE}1. bash scripts/run_quick_test.sh{Colors.RESET}")
        print(f"  {Colors.BLUE}2. bash scripts/run_sota_only.sh{Colors.RESET}")
        print(f"  {Colors.BLUE}3. python analysis/merge_with_baseline.py{Colors.RESET}")
        return True
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️ Some tests failed. Please fix issues above.{Colors.RESET}\n")
        return False


def main():
    """Run all tests"""
    
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("="*80)
    print("ULTIMATE MOE - COMPREHENSIVE SETUP TEST SUITE")
    print("="*80)
    print(f"{Colors.RESET}\n")
    
    # Generate missing __init__ files first
    generate_missing_init_files()
    
    # Run all tests
    all_results = {}
    
    all_results['Directory Structure'] = test_directory_structure()
    all_results['Model Files'] = test_model_files()
    all_results['Experiment Files'] = test_experiment_files()
    all_results['Analysis Files'] = test_analysis_files()
    all_results['Utility Files'] = test_utility_files()
    all_results['Bash Scripts'] = test_bash_scripts()
    all_results['Dependencies'] = test_dependencies()
    
    # Special tests (not counted in pass/fail)
    test_torch_distributed()
    test_triton_kernels()
    test_existing_results()
    
    # Only run functional test if basic structure is OK
    basic_structure_ok = all(
        all(v for v in all_results['Model Files'].values()),
    )
    
    if basic_structure_ok:
        create_test_data_sample()
    
    # Generate summary
    success = generate_summary_report(all_results)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

"""
Script to run Experiment 2: Column ordering effects on vanilla TabPFN.

Usage:
    python run_experiment_2.py                    # Run full experiment
    python run_experiment_2.py --no-resume       # Start fresh
"""

import argparse
from experiment_2 import run_experiment_2
from utils.dag_utils import print_dag_info
from utils.scm_data import get_dag_and_config


def main():
    parser = argparse.ArgumentParser(description='Run Experiment 2: Column ordering effects')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh (ignore checkpoint)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Show DAG info
    dag, col_names, _ = get_dag_and_config(False)
    print("=" * 60)
    print("EXPERIMENT 2: Column Ordering Effects on Vanilla TabPFN")
    print("=" * 60)
    print("\nResearch Question:")
    print("Does column ordering affect synthetic data quality when TabPFN")
    print("uses its implicit autoregressive mechanism (no DAG provided)?")
    print("\nCurrent SCM structure:")
    print_dag_info(dag, col_names)
    print()
    
    # Configuration (only full version)
    print("Running FULL experiment...")
    config = {
        'train_sizes': [20, 50, 100, 200, 500],
        'ordering_strategies': ['original', 'topological', 'worst', 'random', 'reverse'],
        'n_repetitions': 10,
        'test_size': 2000,
        'n_permutations': 3,
        'metrics': ['max_corr_diff', 'propensity_mse', 'kmarginal'],
        'include_categorical': False,
        'n_estimators': 3,
        'random_seed_base': 42
    }
    output_dir = args.output or "experiment_2_results"
    
    # Calculate total configurations
    total_configs = (len(config['train_sizes']) * 
                    len(config['ordering_strategies']) * 
                    config['n_repetitions'])
    
    print(f"\nExperiment Configuration:")
    print(f"  Training sizes: {config['train_sizes']}")
    print(f"  Ordering strategies: {config['ordering_strategies']}")
    print(f"  Repetitions: {config['n_repetitions']}")
    print(f"  Total configurations: {total_configs}")
    print(f"  Resume: {not args.no_resume}")
    print(f"  Output: {output_dir}")
    
    # Run experiment
    results = run_experiment_2(
        config=config,
        output_dir=output_dir,
        resume=not args.no_resume
    )
    
    # Print summary
    if results is not None and len(results) > 0:
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        # Best and worst orderings per metric
        for metric in config['metrics']:
            print(f"\n{metric.upper()}:")
            mean_by_order = results.groupby('order_strategy')[metric].mean()
            best_order = mean_by_order.idxmin()
            worst_order = mean_by_order.idxmax()
            
            print(f"  Best ordering: {best_order} ({mean_by_order[best_order]:.4f})")
            print(f"  Worst ordering: {worst_order} ({mean_by_order[worst_order]:.4f})")
            print(f"  Difference: {mean_by_order[worst_order] - mean_by_order[best_order]:.4f}")


if __name__ == "__main__":
    main()

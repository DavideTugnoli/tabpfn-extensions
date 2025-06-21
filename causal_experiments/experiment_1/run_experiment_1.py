"""
Simple script to run Experiment 1.

Usage:
    python run_experiment_1.py                    # Fair comparison (topological order)
    python run_experiment_1.py --order original  # Original order (neutral)
    python run_experiment_1.py --order worst     # Worst case for vanilla
"""

import argparse
from experiment_1 import run_experiment_1
from utils.dag_utils import get_ordering_strategies, print_dag_info
from utils.scm_data import get_dag_and_config

def main():
    parser = argparse.ArgumentParser(description='Run Experiment 1')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh (ignore checkpoint)')
    parser.add_argument('--order', type=str, default='topological',
                       choices=['original', 'topological', 'worst', 'random', 'reverse'],
                       help='Column ordering strategy')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Show DAG info first
    dag, col_names, _ = get_dag_and_config(False)
    print("Current SCM structure:")
    print_dag_info(dag, col_names)
    print()
    
    # Full experiment config
    config = {
        'train_sizes': [20, 50, 100, 200, 500],
        'n_repetitions': 10,
        'test_size': 2000,
        'n_permutations': 3,
        'metrics': ['max_corr_diff', 'propensity_mse', 'kmarginal'],
        'include_categorical': False,
        'n_estimators': 3,
        'random_seed_base': 42,
        'column_order_strategy': args.order
    }
    output_dir = args.output or f"experiment_1_{args.order}"
    
    print("Starting Experiment 1...")
    print(f"Column order strategy: {args.order}")
    print(f"Resume: {not args.no_resume}")
    print(f"Output: {output_dir}")
    print(f"Total iterations: {len(config['train_sizes']) * config['n_repetitions']}")
    
    results = run_experiment_1(
        config=config,
        output_dir=output_dir,
        resume=not args.no_resume
    )
    
    print(f"\nExperiment completed! Results shape: {results.shape}")


if __name__ == "__main__":
    main()
"""
Experiment 1: Effect of DAG and training set size.
Clean, generic, works with any SCM/DAG.

Usage:
    python experiment_1.py                    # Fair comparison (topological order)
    python experiment_1.py --order original  # Original order (neutral)
    python experiment_1.py --order worst     # Worst case for vanilla
"""
import sys
import os
import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from io import StringIO
import warnings
import argparse
warnings.filterwarnings('ignore')

# Add the causal_experiments directory to the path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# TabPFN imports
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised

# Local imports
from utils.scm_data import generate_scm_data, get_dag_and_config
from utils.metrics import FaithfulDataEvaluator
from utils.dag_utils import get_ordering_strategies, reorder_data_and_dag, print_dag_info
from utils.checkpoint_utils import save_checkpoint, get_checkpoint_info, cleanup_checkpoint
from utils.experiment_utils import generate_synthetic_data_quiet, reorder_data_and_columns

# Centralized default config
DEFAULT_CONFIG = {
    'train_sizes': [20, 50, 100, 200, 500],
    'n_repetitions': 10,
    'test_size': 2000,
    'n_permutations': 3,
    'metrics': ['mean_corr_difference', 'max_corr_difference', 'propensity_metrics', 'k_marginal_tvd'],
    'include_categorical': False,
    'n_estimators': 3,
    'random_seed_base': 42,
    'column_order_strategy': 'original',
}

# Utility: Evaluate metrics

def evaluate_metrics(X_test, X_synth, col_names, categorical_cols, k_for_kmarginal=2):
    evaluator = FaithfulDataEvaluator()
    cat_col_names = [col_names[i] for i in categorical_cols] if categorical_cols else []
    return evaluator.evaluate(
        pd.DataFrame(X_test, columns=col_names),
        pd.DataFrame(X_synth, columns=col_names),
        categorical_columns=cat_col_names if cat_col_names else None,
        k_for_kmarginal=k_for_kmarginal
    )

# Pipeline: With DAG (no reordering)

def run_with_dag(X_train, X_test, dag, col_names, categorical_cols, config, seed, train_size, repetition):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = TabPFNClassifier(n_estimators=config['n_estimators'], device=device)
    reg = TabPFNRegressor(n_estimators=config['n_estimators'], device=device)
    model = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
    if categorical_cols:
        model.set_categorical_features(categorical_cols)
    model.fit(torch.from_numpy(X_train).float())
    X_synth = generate_synthetic_data_quiet(
        model, config['test_size'], dag, config['n_permutations']
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    metrics = evaluate_metrics(X_test, X_synth, col_names, categorical_cols)
    base_info = {
        'train_size': train_size,
        'repetition': repetition,
        'seed': seed,
        'categorical': config['include_categorical'],
        'column_order_strategy': 'none',
        'column_order': 'none',
    }
    def flatten_metrics():
        flat = {}
        for metric in config['metrics']:
            value = metrics.get(metric)
            if isinstance(value, dict):
                for submetric, subvalue in value.items():
                    flat[f'{metric}_{submetric}'] = subvalue
            else:
                flat[metric] = value
        return flat
    return {**base_info, 'dag_used': str(dag), **flatten_metrics()}

# Pipeline: No DAG (with reordering)

def run_no_dag(X_train, X_test, col_names, categorical_cols, column_order, column_order_name, config, seed, train_size, repetition):
    X_train_reordered, col_names_reordered, categorical_cols_reordered = reorder_data_and_columns(
        X_train, col_names, categorical_cols, column_order
    )
    X_test_reordered, _, _ = reorder_data_and_columns(
        X_test, col_names, categorical_cols, column_order
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = TabPFNClassifier(n_estimators=config['n_estimators'], device=device)
    reg = TabPFNRegressor(n_estimators=config['n_estimators'], device=device)
    model = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
    if categorical_cols_reordered:
        model.set_categorical_features(categorical_cols_reordered)
    model.fit(torch.from_numpy(X_train_reordered).float())
    X_synth = generate_synthetic_data_quiet(
        model, config['test_size'], None, config['n_permutations']
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    metrics = evaluate_metrics(X_test_reordered, X_synth, col_names_reordered, categorical_cols_reordered)
    base_info = {
        'train_size': train_size,
        'repetition': repetition,
        'seed': seed,
        'categorical': config['include_categorical'],
        'column_order_strategy': '',
        'column_order': '',
    }
    def flatten_metrics():
        flat = {}
        for metric in config['metrics']:
            value = metrics.get(metric)
            if isinstance(value, dict):
                for submetric, subvalue in value.items():
                    flat[f'{metric}_{submetric}'] = subvalue
            else:
                flat[metric] = value
        return flat
    return {**base_info, 'dag_used': '', **flatten_metrics()}

# Main iteration orchestrator

def run_single_iteration(train_size, repetition, config, X_test, correct_dag, col_names, categorical_cols, no_dag_column_order, column_order_name):
    print(f"  Running train_size={train_size}, rep={repetition+1}/{config['n_repetitions']}")
    seed = config['random_seed_base'] + repetition
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    X_train_original = generate_scm_data(train_size, seed, config['include_categorical'])
    print(f"    Using pre-calculated column order: {column_order_name} = {no_dag_column_order}")
    # With DAG: no reordering
    row_with_dag = run_with_dag(
        X_train_original, X_test, correct_dag, col_names, categorical_cols, config, seed, train_size, repetition
    )
    # Without DAG: with reordering (using pre-calculated order)
    row_without_dag = run_no_dag(
        X_train_original, X_test, col_names, categorical_cols, no_dag_column_order, column_order_name, config, seed, train_size, repetition
    )
    return [row_with_dag, row_without_dag]

def run_experiment_1(config=None, output_dir="experiment_1_results", resume=True):
    """
    Main experiment function with column ordering control.
    """
    # Use centralized config and update with any overrides
    base_config = DEFAULT_CONFIG.copy()
    if config is not None:
        base_config.update(config)
    config = base_config
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(script_dir) / 'results'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Experiment 1 - Output dir: {output_dir}")
    print(f"Config: {config}")
    
    # Setup
    correct_dag, col_names, categorical_cols = get_dag_and_config(config['include_categorical'])
    X_test_original = generate_scm_data(config['test_size'], 123, config['include_categorical'])
    
    # Pre-calculate column order for no_dag case (ONCE!)
    available_orderings = get_ordering_strategies(correct_dag)
    column_order_name = config.get('column_order_strategy')
    if column_order_name not in available_orderings:
        raise ValueError(f"Unknown ordering strategy: {column_order_name}. "
                        f"Available: {list(available_orderings.keys())}")
    no_dag_column_order = available_orderings[column_order_name]
    print(f"Pre-calculated column order for no_dag case: {column_order_name} = {no_dag_column_order}")
    
    # Determine file suffix based on column_order_strategy
    strategy = config.get('column_order_strategy')
    suffix = strategy
    
    raw_results_file = output_dir / f"raw_results_{suffix}.csv"
    raw_results_final_file = output_dir / f"raw_results_final_{suffix}.csv"
    
    # Check for checkpoint
    if resume:
        results_so_far, start_train_idx, start_rep = get_checkpoint_info(output_dir)
    else:
        results_so_far, start_train_idx, start_rep = [], 0, 0
    
    # Run experiment
    total_iterations = len(config['train_sizes']) * config['n_repetitions']
    completed = len(results_so_far)
    
    print(f"Total iterations: {total_iterations}, Already completed: {completed}")
    
    try:
        for train_idx, train_size in enumerate(config['train_sizes'][start_train_idx:], start_train_idx):
            rep_start = start_rep if train_idx == start_train_idx else 0
            for rep in range(rep_start, config['n_repetitions']):
                results = run_single_iteration(
                    train_size, rep, config, X_test_original, 
                    correct_dag, col_names, categorical_cols, no_dag_column_order, column_order_name
                )
                results_so_far.extend(results)
                # Save to CSV incrementally
                df_current = pd.DataFrame(results_so_far)
                df_current.to_csv(raw_results_file, index=False)
                # Save checkpoint
                save_checkpoint(results_so_far, train_idx, rep + 1, output_dir)
                # Progress
                completed += 2
                print(f"    Progress: {completed}/{total_iterations*2} ({100*completed/(total_iterations*2):.1f}%)")
                print(f"    Results saved to: {raw_results_file}")
            start_rep = 0
    except KeyboardInterrupt:
        print("\nExperiment interrupted. Progress saved!")
        return pd.DataFrame(results_so_far)
    # Experiment completed
    print("\nExperiment completed!")
    # Clean up checkpoint
    cleanup_checkpoint(output_dir)
    # Final results
    df_results = pd.DataFrame(results_so_far)
    df_results.to_csv(raw_results_final_file, index=False)
    print(f"Results saved to: {raw_results_final_file}")
    print(f"Total results: {len(df_results)}")
    return df_results

def main():
    """Main CLI interface for Experiment 1."""
    parser = argparse.ArgumentParser(description='Run Experiment 1')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh (ignore checkpoint)')
    parser.add_argument('--order', type=str, default=None,
                       choices=['original', 'topological', 'worst', 'random', 'reverse', 'both'],
                       help='Column ordering strategy')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (auto-generated if not specified)')
    args = parser.parse_args()

    # Show DAG info first
    dag, col_names, _ = get_dag_and_config(False)
    print("Current SCM structure:")
    print_dag_info(dag, col_names)
    print()

    # Determine which orderings to run
    if args.order is None or args.order == 'both':
        orderings_to_run = ['topological', 'original']
    else:
        orderings_to_run = [args.order]

    all_results = []
    for ordering in orderings_to_run:
        config = DEFAULT_CONFIG.copy()
        config['column_order_strategy'] = ordering
        output_dir = args.output or f"experiment_1_{ordering}"
        print(f"\n{'='*50}")
        print(f"Starting Experiment 1 with ordering: {ordering}")
        print(f"Column order strategy: {ordering}")
        print(f"Resume: {not args.no_resume}")
        print(f"Output: {output_dir}")
        print(f"Total iterations: {len(config['train_sizes']) * config['n_repetitions']}")
        print(f"{'='*50}")
        results = run_experiment_1(
            config=config,
            output_dir=output_dir,
            resume=not args.no_resume
        )
        all_results.append(results)
    # Combine all results if multiple orderings were run
    if len(all_results) > 1:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_output_dir = args.output or "experiment_1_combined"
        combined_output_dir = Path(combined_output_dir)
        combined_output_dir.mkdir(exist_ok=True)
        combined_results.to_csv(combined_output_dir / "combined_results.csv", index=False)
        print(f"\nCombined results saved to: {combined_output_dir}")
        print(f"Combined results shape: {combined_results.shape}")
    print(f"\nAll experiments completed!")


if __name__ == "__main__":
    main()
"""
Experiment 2: Column ordering effects on vanilla TabPFN.

This experiment tests whether column ordering affects synthetic data quality
when TabPFN uses its implicit autoregressive mechanism (no DAG provided).

Usage:
    python experiment_2.py                    # Run full experiment
    python experiment_2.py --no-resume       # Start fresh
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

# Add the causal_experiments directory to the path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# TabPFN imports - use local imports to avoid HPO dependency issues
from tabpfn_extensions.utils import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised.unsupervised import TabPFNUnsupervisedModel

# Create a namespace for the unsupervised module
class UnsupervisedNamespace:
    TabPFNUnsupervisedModel = TabPFNUnsupervisedModel

unsupervised = UnsupervisedNamespace()

warnings.filterwarnings('ignore')

# Local imports
from utils.scm_data import generate_scm_data, get_dag_and_config
from utils.metrics import FaithfulDataEvaluator
from utils.dag_utils import get_ordering_strategies, print_dag_info
from utils.checkpoint_utils import save_checkpoint, get_checkpoint_info, cleanup_checkpoint
from utils.experiment_utils import generate_synthetic_data_quiet, reorder_data_and_columns

# Centralized default config
DEFAULT_CONFIG = {
    'train_sizes': [20, 50, 100, 200, 500],
    'ordering_strategies': ['original', 'topological', 'worst', 'random'],
    'n_repetitions': 10,
    'test_size': 2000,
    'n_permutations': 3,
    'metrics': ['mean_corr_difference', 'max_corr_difference', 'propensity_metrics', 'k_marginal_tvd'],
    'include_categorical': False,
    'n_estimators': 3,
    'random_seed_base': 42
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

# Pipeline: Vanilla TabPFN with column reordering

def run_vanilla_tabpfn(X_train, X_test, col_names, categorical_cols, column_order, order_strategy, config, seed, train_size, repetition):
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
        'column_order_strategy': order_strategy,
        'column_order': str(column_order),
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
    return {**base_info, **flatten_metrics()}

# Main configuration orchestrator

def run_single_configuration(train_size, order_strategy, repetition, config, 
                           X_test, correct_dag, col_names, categorical_cols, pre_calculated_column_order):
    print(f"    Order: {order_strategy}, Rep: {repetition+1}/{config['n_repetitions']}")
    seed = config['random_seed_base'] + repetition
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    X_train_original = generate_scm_data(train_size, seed, config['include_categorical'])
    print(f"    Using pre-calculated column order: {order_strategy} = {pre_calculated_column_order}")
    return run_vanilla_tabpfn(X_train_original, X_test, col_names, categorical_cols, pre_calculated_column_order, order_strategy, config, seed, train_size, repetition)

def run_experiment_2(config=None, output_dir="experiment_2_results", resume=True):
    """
    Main experiment function for testing column ordering effects.
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
    
    print(f"Experiment 2 - Output dir: {output_dir}")
    print(f"Config: {config}")
    
    # Setup
    correct_dag, col_names, categorical_cols = get_dag_and_config(config['include_categorical'])
    X_test_original = generate_scm_data(config['test_size'], 123, config['include_categorical'])
    
    # Pre-calculate all column orders for each strategy (ONCE!)
    available_orderings = get_ordering_strategies(correct_dag)
    pre_calculated_orders = {}
    for order_strategy in config['ordering_strategies']:
        if order_strategy not in available_orderings:
            raise ValueError(f"Unknown ordering strategy: {order_strategy}. "
                            f"Available: {list(available_orderings.keys())}")
        pre_calculated_orders[order_strategy] = available_orderings[order_strategy]
        print(f"Pre-calculated column order for {order_strategy}: {pre_calculated_orders[order_strategy]}")
    
    # Check for checkpoint
    if resume:
        results_so_far, start_train_idx, start_rep = get_checkpoint_info(output_dir)
    else:
        results_so_far, start_train_idx, start_rep = [], 0, 0
    
    # Run experiment
    total_iterations = (len(config['train_sizes']) * 
                       len(config['ordering_strategies']) * 
                       config['n_repetitions'])
    completed = len(results_so_far)
    
    print(f"Total iterations: {total_iterations}, Already completed: {completed}")
    
    try:
        for train_idx, train_size in enumerate(config['train_sizes'][start_train_idx:], start_train_idx):
            
            for order_strategy in config['ordering_strategies']:
                
                rep_start = start_rep if train_idx == start_train_idx else 0
                
                for rep in range(rep_start, config['n_repetitions']):
                    
                    result = run_single_configuration(
                        train_size, order_strategy, rep, config, X_test_original,
                        correct_dag, col_names, categorical_cols, pre_calculated_orders[order_strategy]
                    )
                    
                    results_so_far.append(result)
                    
                    # Save to CSV incrementally
                    df_current = pd.DataFrame(results_so_far)
                    df_current.to_csv(output_dir / "raw_results.csv", index=False)
                    
                    # Save checkpoint
                    save_checkpoint(results_so_far, train_idx, rep + 1, output_dir)
                    
                    # Progress
                    completed += 1
                    print(f"    Progress: {completed}/{total_iterations} ({100*completed/total_iterations:.1f}%)")
                    print(f"    Results saved to: {output_dir}/raw_results.csv")
                
                start_rep = 0  # Reset start_rep after first strategy
    
    except KeyboardInterrupt:
        print("\nExperiment interrupted. Progress saved!")
        return pd.DataFrame(results_so_far)
    
    # Experiment completed
    print("\nExperiment completed!")
    
    # Clean up checkpoint
    cleanup_checkpoint(output_dir)
    
    # Final results
    df_results = pd.DataFrame(results_so_far)
    df_results.to_csv(output_dir / "raw_results_final.csv", index=False)
    
    print(f"Results saved to: {output_dir}")
    print(f"Total results: {len(df_results)}")
    
    return df_results


def main():
    """Main CLI interface for Experiment 2."""
    parser = argparse.ArgumentParser(description='Run Experiment 2: Column ordering effects')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh (ignore checkpoint)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Show SCM info (only for reference, not used in experiment)
    dag, col_names, _ = get_dag_and_config(False)
    print("=" * 60)
    print("EXPERIMENT 2: Column Ordering Effects on Vanilla TabPFN")
    print("=" * 60)
    print("\nResearch Question:")
    print("Does column ordering affect synthetic data quality when TabPFN")
    print("uses its implicit autoregressive mechanism (no DAG provided)?")
    print("\nSCM structure (for ordering strategies reference):")
    print_dag_info(dag, col_names)
    print()
    
    # Use centralized config
    print("Running FULL experiment...")
    config = DEFAULT_CONFIG.copy()
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
        
        # Get actual metric columns from results
        metric_columns = [col for col in results.columns if col not in ['train_size', 'repetition', 'categorical', 'seed', 'column_order_strategy', 'column_order']]
        
        # Best and worst orderings per metric
        for metric in metric_columns:
            print(f"\n{metric.upper()}:")
            mean_by_order = results.groupby('order_strategy')[metric].mean()
            best_order = mean_by_order.idxmin()
            worst_order = mean_by_order.idxmax()
            
            print(f"  Best ordering: {best_order} ({mean_by_order[best_order]:.4f})")
            print(f"  Worst ordering: {worst_order} ({mean_by_order[worst_order]:.4f})")
            print(f"  Difference: {mean_by_order[worst_order] - mean_by_order[best_order]:.4f}")


if __name__ == "__main__":
    main()
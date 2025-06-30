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
from utils.dag_utils import get_ordering_strategies, reorder_data_and_dag, print_dag_info
from utils.checkpoint_utils import save_checkpoint, get_checkpoint_info, cleanup_checkpoint


def generate_synthetic_data_quiet(model, n_samples, dag=None, n_permutations=3):
    """Generate synthetic data with TabPFN, suppressing output."""
    plt.ioff()
    plt.close('all')
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    try:
        X_synthetic = model.generate_synthetic_data(
            n_samples=n_samples,
            t=1.0,
            n_permutations=n_permutations,
            dag=dag
        ).cpu().numpy()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        plt.close('all')
    
    return X_synthetic


def run_single_configuration(train_size, order_strategy, repetition, config, 
                           X_test, correct_dag, col_names, categorical_cols):
    """
    Run one configuration: train_size + order_strategy + repetition.
    
    NOTE: For Experiment 2, we only test vanilla TabPFN (no DAG provided).
    """
    print(f"    Order: {order_strategy}, Rep: {repetition+1}/{config['n_repetitions']}")
    
    # Set seeds
    seed = config['random_seed_base'] + repetition
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Generate training data (always in "original" order first)
    X_train_original = generate_scm_data(train_size, seed, config['include_categorical'])
    
    # Get ordering
    available_orderings = get_ordering_strategies(correct_dag)
    
    if order_strategy not in available_orderings:
        raise ValueError(f"Unknown ordering strategy: {order_strategy}. "
                        f"Available: {list(available_orderings.keys())}")
    
    column_order = available_orderings[order_strategy]
    
    # Reorder training data (NO DAG reordering needed for vanilla)
    X_train, _ = reorder_data_and_dag(X_train_original, correct_dag, column_order)
    X_train_tensor = torch.from_numpy(X_train).float()
    
    # Also reorder test data for consistent evaluation
    X_test_reordered, _ = reorder_data_and_dag(X_test, correct_dag, column_order)
    
    # Reorder column names and categorical indices
    col_names_reordered = [col_names[i] for i in column_order]
    categorical_cols_reordered = None
    if categorical_cols:
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(column_order)}
        categorical_cols_reordered = [old_to_new[col] for col in categorical_cols if col in old_to_new]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # === VANILLA TabPFN (NO DAG) ===
    clf = TabPFNClassifier(n_estimators=config['n_estimators'], device=device)
    reg = TabPFNRegressor(n_estimators=config['n_estimators'], device=device)
    model = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
    
    if categorical_cols_reordered:
        model.set_categorical_features(categorical_cols_reordered)
    
    model.fit(X_train_tensor)
    
    # Generate synthetic data WITHOUT DAG
    X_synth = generate_synthetic_data_quiet(
        model, config['test_size'], None, config['n_permutations']
    )
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # === EVALUATE ===
    evaluator = FaithfulDataEvaluator()
    
    # Ensure DataFrame inputs for evaluator
    if not isinstance(X_test_reordered, pd.DataFrame):
        X_test_reordered = pd.DataFrame(X_test_reordered, columns=col_names_reordered)
    if not isinstance(X_synth, pd.DataFrame):
        X_synth = pd.DataFrame(X_synth, columns=col_names_reordered)
    metrics = evaluator.evaluate(X_test_reordered, X_synth, col_names_reordered, categorical_cols_reordered)
    
    # Build result
    result = {
        'train_size': train_size,
        'order_strategy': order_strategy,
        'column_order': str(column_order),
        'dag_used': str(column_order),  # Add DAG information (same as column_order for vanilla)
        'repetition': repetition,
        'categorical': config['include_categorical']
    }
    
    # Add all metrics
    for metric, value in metrics.items():
        if isinstance(value, dict):
            for submetric, subvalue in value.items():
                result[f'{metric}_{submetric}'] = subvalue
        else:
            result[metric] = value
    
    return result


def run_experiment_2(config=None, output_dir="experiment_2_results", resume=True):
    """
    Main experiment function for testing column ordering effects.
    """
    # Default config
    if config is None:
        config = {
            'train_sizes': [20, 50, 100, 200, 500],
            'ordering_strategies': ['original', 'topological', 'worst', 'random'],
            'n_repetitions': 10,
            'test_size': 2000,
            'n_permutations': 3,
            'metrics': ['mean_corr_difference', 'max_corr_difference', 'propensity_mse', 'k_marginal_tvd'],
            'include_categorical': False,
            'n_estimators': 3,
            'random_seed_base': 42
        }
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Experiment 2 - Output dir: {output_dir}")
    print(f"Config: {config}")
    
    # Setup
    correct_dag, col_names, categorical_cols = get_dag_and_config(config['include_categorical'])
    X_test_original = generate_scm_data(config['test_size'], 123, config['include_categorical'])
    
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
            
            rep_start = start_rep if train_idx == start_train_idx else 0
            
            for rep in range(rep_start, config['n_repetitions']):
                
                for order_strategy in config['ordering_strategies']:
                    
                    result = run_single_configuration(
                        train_size, order_strategy, rep, config, X_test_original,
                        correct_dag, col_names, categorical_cols
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
        'ordering_strategies': ['original', 'topological', 'worst', 'random'],
        'n_repetitions': 10,
        'test_size': 2000,
        'n_permutations': 3,
        'metrics': ['mean_corr_difference', 'max_corr_difference', 'propensity_mse', 'k_marginal_tvd'],
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
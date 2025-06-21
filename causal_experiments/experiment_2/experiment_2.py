"""
Experiment 2: Effect of column ordering on vanilla TabPFN.

This experiment tests whether column order affects vanilla TabPFN quality (without DAG).
We compare different orderings to understand the implicit autoregressive mechanism.
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
warnings.filterwarnings('ignore')

# TabPFN imports
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised

# Local imports
from ..utils.scm_data import generate_scm_data, get_dag_and_config
from ..utils.metrics import SyntheticDataEvaluator
from ..utils.dag_utils import get_ordering_strategies, reorder_data_and_dag, print_dag_info


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
    evaluator = SyntheticDataEvaluator(config['metrics'])
    
    metrics = evaluator.evaluate(
        X_test_reordered, X_synth, 
        col_names_reordered, categorical_cols_reordered
    )
    
    # Build result
    result = {
        'train_size': train_size,
        'order_strategy': order_strategy,
        'column_order': str(column_order),
        'repetition': repetition,
        'categorical': config['include_categorical']
    }
    
    # Add all metrics
    for metric, value in metrics.items():
        result[metric] = value
    
    return result


def save_checkpoint(results_so_far, current_config_idx, output_dir):
    """Save checkpoint for resuming."""
    checkpoint = {
        'results': results_so_far,
        'current_config_idx': current_config_idx
    }
    
    checkpoint_file = Path(output_dir) / "checkpoint.pkl"
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(output_dir):
    """Load checkpoint if exists."""
    checkpoint_file = Path(output_dir) / "checkpoint.pkl"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return None


def run_experiment_2(config=None, output_dir="experiment_2_results", resume=True):
    """
    Main experiment function for testing column ordering effects.
    
    This experiment tests multiple ordering strategies on vanilla TabPFN (no DAG).
    """
    # Default config
    if config is None:
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
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Experiment 2 - Output dir: {output_dir}")
    print(f"Config: {config}")
    
    # Setup
    correct_dag, col_names, categorical_cols = get_dag_and_config(config['include_categorical'])
    X_test_original = generate_scm_data(config['test_size'], 123, config['include_categorical'])
    
    # Show DAG structure
    print("\nDAG Structure for reference:")
    print_dag_info(correct_dag, col_names)
    
    # Generate all configurations
    configurations = []
    for train_size in config['train_sizes']:
        for order_strategy in config['ordering_strategies']:
            for rep in range(config['n_repetitions']):
                configurations.append({
                    'train_size': train_size,
                    'order_strategy': order_strategy,
                    'repetition': rep
                })
    
    total_configs = len(configurations)
    print(f"\nTotal configurations: {total_configs}")
    
    # Check for checkpoint
    results_so_far = []
    start_idx = 0
    
    if resume:
        checkpoint = load_checkpoint(output_dir)
        if checkpoint:
            print("Resuming from checkpoint!")
            results_so_far = checkpoint['results']
            start_idx = checkpoint['current_config_idx']
            print(f"  Resuming from configuration {start_idx}/{total_configs}")
    
    # Run experiment
    print(f"\nStarting experiment...")
    completed = len(results_so_far)
    
    try:
        for idx in range(start_idx, total_configs):
            cfg = configurations[idx]
            
            print(f"\n[{idx+1}/{total_configs}] Train size: {cfg['train_size']}, "
                  f"Order: {cfg['order_strategy']}, Rep: {cfg['repetition']+1}")
            
            result = run_single_configuration(
                cfg['train_size'], 
                cfg['order_strategy'], 
                cfg['repetition'],
                config, 
                X_test_original, 
                correct_dag, 
                col_names, 
                categorical_cols
            )
            
            results_so_far.append(result)
            
            # Save to CSV incrementally
            df_current = pd.DataFrame(results_so_far)
            df_current.to_csv(output_dir / "raw_results.csv", index=False)
            
            # Save checkpoint
            save_checkpoint(results_so_far, idx + 1, output_dir)
            
            # Progress
            completed += 1
            print(f"    Progress: {completed}/{total_configs} ({100*completed/total_configs:.1f}%)")
    
    except KeyboardInterrupt:
        print("\nExperiment interrupted. Progress saved!")
        return pd.DataFrame(results_so_far)
    
    # Experiment completed
    print("\nExperiment completed!")
    
    # Clean up checkpoint
    checkpoint_file = output_dir / "checkpoint.pkl"
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    # Final results
    df_results = pd.DataFrame(results_so_far)
    df_results.to_csv(output_dir / "raw_results_final.csv", index=False)
    
    # Basic summary statistics
    print("\nBasic results summary:")
    print("=" * 60)
    
    # Group by order strategy and compute mean metrics
    for metric in config['metrics']:
        print(f"\n{metric} (mean ± std) by ordering strategy:")
        summary = df_results.groupby('order_strategy')[metric].agg(['mean', 'std'])
        print(summary.round(4))
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Total results: {len(df_results)}")
    
    return df_results


if __name__ == "__main__":
    # Quick test with smaller config
    test_config = {
        'train_sizes': [50, 100],
        'ordering_strategies': ['original', 'topological', 'worst'],
        'n_repetitions': 2,
        'test_size': 500,
        'n_permutations': 2,
        'metrics': ['max_corr_diff', 'propensity_mse'],
        'include_categorical': False,
        'n_estimators': 3,
        'random_seed_base': 42
    }
    
    results = run_experiment_2(test_config, "test_exp2_quick")
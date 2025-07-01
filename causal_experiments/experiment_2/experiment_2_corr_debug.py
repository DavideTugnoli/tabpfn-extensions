"""
DEBUG VERSION of Experiment 2: Column ordering effects on vanilla TabPFN.

This debug version tests only train_size=20 with 'original' strategy to understand
why Max Correlation Difference values are so similar.
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
import hashlib

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


def data_hash(data):
    """Calculate hash of data for debugging."""
    return hashlib.md5(str(data).encode()).hexdigest()[:8]


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
    print(f"\n{'='*60}")
    print(f"DEBUGGING CONFIG: Train={train_size}, Order={order_strategy}, Rep={repetition}")
    print(f"{'='*60}")
    
    # Set seeds
    seed = config['random_seed_base'] + repetition
    print(f"[DEBUG] Setting seed: {seed} (base={config['random_seed_base']} + rep={repetition})")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Generate training data (always in "original" order first)
    print(f"[DEBUG] Generating training data with seed {seed}...")
    X_train_original = generate_scm_data(train_size, seed, config['include_categorical'])
    print(f"[DEBUG] X_train_original shape: {X_train_original.shape}")
    print(f"[DEBUG] X_train_original hash: {data_hash(X_train_original)}")
    print(f"[DEBUG] X_train_original stats: mean={np.mean(X_train_original):.6f}, std={np.std(X_train_original):.6f}")
    
    # Get ordering
    available_orderings = get_ordering_strategies(correct_dag)
    
    if order_strategy not in available_orderings:
        raise ValueError(f"Unknown ordering strategy: {order_strategy}. "
                        f"Available: {list(available_orderings.keys())}")
    
    column_order = available_orderings[order_strategy]
    print(f"[DEBUG] Column order for {order_strategy}: {column_order}")
    
    # Reorder training data (NO DAG reordering needed for vanilla)
    X_train, _ = reorder_data_and_dag(X_train_original, correct_dag, column_order)
    X_train_tensor = torch.from_numpy(X_train).float()
    print(f"[DEBUG] X_train after reordering hash: {data_hash(X_train)}")
    
    # Also reorder test data for consistent evaluation
    X_test_reordered, _ = reorder_data_and_dag(X_test, correct_dag, column_order)
    print(f"[DEBUG] X_test_reordered hash: {data_hash(X_test_reordered)}")
    
    # Reorder column names and categorical indices
    col_names_reordered = [col_names[i] for i in column_order]
    categorical_cols_reordered = None
    if categorical_cols:
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(column_order)}
        categorical_cols_reordered = [old_to_new[col] for col in categorical_cols if col in old_to_new]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEBUG] Using device: {device}")
    
    # === VANILLA TabPFN (NO DAG) ===
    print(f"[DEBUG] Creating TabPFN model...")
    clf = TabPFNClassifier(n_estimators=config['n_estimators'], device=device)
    reg = TabPFNRegressor(n_estimators=config['n_estimators'], device=device)
    model = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
    
    if categorical_cols_reordered:
        model.set_categorical_features(categorical_cols_reordered)
    
    print(f"[DEBUG] Fitting model on training data...")
    model.fit(X_train_tensor)
    
    # Generate synthetic data WITHOUT DAG
    print(f"[DEBUG] Generating synthetic data (n_samples={config['test_size']}, dag=None)...")
    X_synth = generate_synthetic_data_quiet(
        model, config['test_size'], None, config['n_permutations']
    )
    print(f"[DEBUG] X_synth shape: {X_synth.shape}")
    print(f"[DEBUG] X_synth hash: {data_hash(X_synth)}")
    print(f"[DEBUG] X_synth stats: mean={np.mean(X_synth):.6f}, std={np.std(X_synth):.6f}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # === EVALUATE ===
    evaluator = FaithfulDataEvaluator()
    
    # Ensure DataFrame inputs for evaluator
    if not isinstance(X_test_reordered, pd.DataFrame):
        X_test_reordered = pd.DataFrame(X_test_reordered, columns=col_names_reordered)
    if not isinstance(X_synth, pd.DataFrame):
        X_synth = pd.DataFrame(X_synth, columns=col_names_reordered)
    
    # === DEBUG: Detailed correlation analysis ===
    print(f"\n[DEBUG] CORRELATION ANALYSIS:")
    real_corr = X_test_reordered.corr()
    synth_corr = X_synth.corr()
    diff_corr = (real_corr - synth_corr).abs()
    
    print(f"[DEBUG] Real correlation matrix:")
    print(real_corr.round(6))
    print(f"[DEBUG] Synthetic correlation matrix:")
    print(synth_corr.round(6))
    print(f"[DEBUG] Absolute difference matrix:")
    print(diff_corr.round(10))
    print(f"[DEBUG] Max correlation difference: {diff_corr.values.max():.2e}")
    print(f"[DEBUG] Mean correlation difference: {np.mean(np.abs(np.triu(diff_corr.values, k=1))):.2e}")
    
    # Save correlation matrices for this repetition
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(script_dir) / 'debug_results'
    output_dir.mkdir(exist_ok=True)
    
    corr_file = output_dir / f"corr_matrices_rep{repetition}.csv"
    debug_df = pd.DataFrame({
        'real_corr': real_corr.values.flatten(),
        'synth_corr': synth_corr.values.flatten(),
        'abs_diff': diff_corr.values.flatten()
    })
    debug_df.to_csv(corr_file, index=False)
    print(f"[DEBUG] Correlation matrices saved to: {corr_file}")
    
    metrics = evaluator.evaluate(
        X_test_reordered,
        X_synth,
        categorical_columns=col_names_reordered,
        k_for_kmarginal=2
    )
    
    print(f"[DEBUG] Metrics computed:")
    for k, v in metrics.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                print(f"  {k}_{sk}: {sv}")
        else:
            print(f"  {k}: {v}")
    
    # Build result
    result = {
        'train_size': train_size,
        'order_strategy': order_strategy,
        'column_order': str(column_order),
        'repetition': repetition,
        'categorical': config['include_categorical'],
        'seed_used': seed,
        'train_data_hash': data_hash(X_train),
        'synth_data_hash': data_hash(X_synth)
    }
    
    # Add all metrics
    for metric, value in metrics.items():
        if isinstance(value, dict):
            for submetric, subvalue in value.items():
                result[f'{metric}_{submetric}'] = subvalue
        else:
            result[metric] = value
    
    return result


def run_experiment_2_debug(config=None, output_dir="experiment_2_debug_results", resume=False):
    """
    DEBUG version of experiment function - limited scope for investigation.
    """
    # Debug config - only train_size=20, order_strategy='original'
    if config is None:
        config = {
            'train_sizes': [20],  # Only one train size
            'ordering_strategies': ['original'],  # Only one strategy
            'n_repetitions': 10,
            'test_size': 2000,
            'n_permutations': 3,
            'metrics': ['mean_corr_difference', 'max_corr_difference', 'propensity_mse', 'k_marginal_tvd'],
            'include_categorical': False,
            'n_estimators': 3,
            'random_seed_base': 42
        }
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(script_dir) / 'debug_results'
    output_dir.mkdir(exist_ok=True)
    
    print(f"DEBUG Experiment 2 - Output dir: {output_dir}")
    print(f"Config: {config}")
    
    # Setup
    correct_dag, col_names, categorical_cols = get_dag_and_config(config['include_categorical'])
    print(f"[DEBUG] Generating test data with fixed seed 123...")
    X_test_original = generate_scm_data(config['test_size'], 123, config['include_categorical'])
    print(f"[DEBUG] X_test_original hash: {data_hash(X_test_original)}")
    print(f"[DEBUG] X_test_original stats: mean={np.mean(X_test_original):.6f}, std={np.std(X_test_original):.6f}")
    
    results_so_far = []
    
    # Run experiment
    total_iterations = (len(config['train_sizes']) * 
                       len(config['ordering_strategies']) * 
                       config['n_repetitions'])
    
    print(f"Total iterations: {total_iterations}")
    
    try:
        for train_size in config['train_sizes']:
            for order_strategy in config['ordering_strategies']:
                for rep in range(config['n_repetitions']):
                    
                    result = run_single_configuration(
                        train_size, order_strategy, rep, config, X_test_original,
                        correct_dag, col_names, categorical_cols
                    )
                    
                    results_so_far.append(result)
                    
                    # Save to CSV incrementally with DEBUG prefix
                    df_current = pd.DataFrame(results_so_far)
                    df_current.to_csv(output_dir / "DEBUG_raw_results.csv", index=False)
                    
                    print(f"\n[DEBUG] Progress: {len(results_so_far)}/{total_iterations}")
                    print(f"[DEBUG] Results saved to: {output_dir}/DEBUG_raw_results.csv")
    
    except KeyboardInterrupt:
        print("\nDebug experiment interrupted. Progress saved!")
        return pd.DataFrame(results_so_far)
    
    # Experiment completed
    print("\nDebug experiment completed!")
    
    # Final results
    df_results = pd.DataFrame(results_so_far)
    df_results.to_csv(output_dir / "DEBUG_raw_results_final.csv", index=False)
    
    print(f"\n{'='*60}")
    print(f"DEBUG SUMMARY")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"Total results: {len(df_results)}")
    
    # Check for identical values
    if 'max_corr_difference' in df_results.columns:
        max_corr_values = df_results['max_corr_difference'].values
        unique_values = np.unique(max_corr_values)
        print(f"\nMax Corr Diff analysis:")
        print(f"  Unique values: {len(unique_values)}")
        print(f"  Values: {unique_values}")
        if len(unique_values) == 1:
            print(f"  *** ALL VALUES ARE IDENTICAL! ***")
        else:
            print(f"  Range: {max_corr_values.min():.2e} to {max_corr_values.max():.2e}")
    
    return df_results


def main():
    """Main debug interface."""
    print("=" * 60)
    print("DEBUG EXPERIMENT 2: Max Correlation Difference Investigation")
    print("=" * 60)
    print("\nLimited scope:")
    print("- Train size: 20 only")
    print("- Order strategy: 'original' only") 
    print("- Repetitions: 10")
    print("- Focus: Understanding why Max Corr Diff values are identical")
    
    # Run debug experiment
    results = run_experiment_2_debug()
    
    if results is not None and len(results) > 0:
        print(f"\n[SUCCESS] Debug experiment completed!")
        print(f"Check debug_results/ folder for detailed analysis.")


if __name__ == "__main__":
    main() 
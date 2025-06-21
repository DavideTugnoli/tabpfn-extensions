"""
Experiment 1: Effect of DAG and training set size.
Clean, generic, works with any SCM/DAG.
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
from utils.scm_data import generate_scm_data, get_dag_and_config
from utils.metrics import SyntheticDataEvaluator
from utils.dag_utils import get_ordering_strategies, reorder_data_and_dag

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

def run_single_iteration(train_size, repetition, config, X_test, correct_dag, col_names, categorical_cols):
    """
    Run one iteration: train_size + repetition.
    Now GENERIC - works with any DAG!
    """
    print(f"  Running train_size={train_size}, rep={repetition+1}/{config['n_repetitions']}")
    
    # Set seeds
    seed = config['random_seed_base'] + repetition
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Generate training data (always in "original" order first)
    X_train_original = generate_scm_data(train_size, seed, config['include_categorical'])
    
    # Get ordering strategy (GENERIC for any DAG!)
    column_order_name = config.get('column_order_strategy', 'original')
    available_orderings = get_ordering_strategies(correct_dag)  # ← GENERIC!
    
    if column_order_name not in available_orderings:
        raise ValueError(f"Unknown ordering strategy: {column_order_name}. "
                        f"Available: {list(available_orderings.keys())}")
    
    column_order = available_orderings[column_order_name]
    print(f"    Using column order: {column_order_name} = {column_order}")
    
    # Reorder training data and DAG (GENERIC function!)
    X_train, dag_reordered = reorder_data_and_dag(X_train_original, correct_dag, column_order)
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
    
    # === WITH DAG ===
    clf_dag = TabPFNClassifier(n_estimators=config['n_estimators'], device=device)
    reg_dag = TabPFNRegressor(n_estimators=config['n_estimators'], device=device)
    model_dag = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf_dag, tabpfn_reg=reg_dag)
    
    if categorical_cols_reordered:
        model_dag.set_categorical_features(categorical_cols_reordered)
    
    model_dag.fit(X_train_tensor)
    X_synth_dag = generate_synthetic_data_quiet(
        model_dag, config['test_size'], dag_reordered, config['n_permutations']
    )
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # === WITHOUT DAG ===
    clf_no_dag = TabPFNClassifier(n_estimators=config['n_estimators'], device=device)
    reg_no_dag = TabPFNRegressor(n_estimators=config['n_estimators'], device=device)
    model_no_dag = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf_no_dag, tabpfn_reg=reg_no_dag)
    
    if categorical_cols_reordered:
        model_no_dag.set_categorical_features(categorical_cols_reordered)
    
    # Fit and generate WITHOUT DAG (same reordered data!)
    model_no_dag.fit(X_train_tensor)
    X_synth_no_dag = generate_synthetic_data_quiet(
        model_no_dag, config['test_size'], None, config['n_permutations']
    )
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # === EVALUATE ===
    evaluator = SyntheticDataEvaluator(config['metrics'])
    
    metrics_dag = evaluator.evaluate(
        X_test_reordered, X_synth_dag, 
        col_names_reordered, categorical_cols_reordered
    )
    metrics_no_dag = evaluator.evaluate(
        X_test_reordered, X_synth_no_dag, 
        col_names_reordered, categorical_cols_reordered
    )
    
    # Build result
    result = {
        'train_size': train_size,
        'repetition': repetition,
        'categorical': config['include_categorical'],
        'column_order_strategy': column_order_name,
        'column_order': str(column_order)  # For debugging
    }
    
    for metric in config['metrics']:
        result[f'{metric}_with_dag'] = metrics_dag[metric]
        result[f'{metric}_without_dag'] = metrics_no_dag[metric]
    
    return result

# Rest of the functions remain the same...
def save_checkpoint(results_so_far, current_train_idx, current_rep, output_dir):
    """Save simple checkpoint."""
    checkpoint = {
        'results': results_so_far,
        'current_train_idx': current_train_idx,
        'current_rep': current_rep
    }
    
    checkpoint_file = Path(output_dir) / "checkpoint.pkl"
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint, f)

def load_checkpoint(output_dir):
    """Load simple checkpoint."""
    checkpoint_file = Path(output_dir) / "checkpoint.pkl"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return None

def run_experiment_1(config=None, output_dir="experiment_1_results", resume=True):
    """
    Main experiment function with column ordering control.
    """
    # Default config
    if config is None:
        config = {
            'train_sizes': [20, 50, 100, 200, 500],
            'n_repetitions': 10,
            'test_size': 2000,
            'n_permutations': 3,
            'metrics': ['max_corr_diff', 'propensity_mse', 'kmarginal'],
            'include_categorical': False,
            'n_estimators': 3,
            'random_seed_base': 42,
            'column_order_strategy': 'original'  # NEW: Control column ordering!
        }
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Experiment 1 - Output dir: {output_dir}")
    print(f"Config: {config}")
    
    # Setup
    correct_dag, col_names, categorical_cols = get_dag_and_config(config['include_categorical'])
    X_test_original = generate_scm_data(config['test_size'], 123, config['include_categorical'])
    
    # Check for checkpoint
    results_so_far = []
    start_train_idx = 0
    start_rep = 0
    
    if resume:
        checkpoint = load_checkpoint(output_dir)
        if checkpoint:
            print("Resuming from checkpoint!")
            results_so_far = checkpoint['results']
            start_train_idx = checkpoint['current_train_idx']
            start_rep = checkpoint['current_rep']
            print(f"  Resuming from train_size_idx={start_train_idx}, rep={start_rep}")
    
    # Run experiment
    total_iterations = len(config['train_sizes']) * config['n_repetitions']
    completed = len(results_so_far)
    
    print(f"Total iterations: {total_iterations}, Already completed: {completed}")
    
    try:
        for train_idx, train_size in enumerate(config['train_sizes'][start_train_idx:], start_train_idx):
            
            rep_start = start_rep if train_idx == start_train_idx else 0
            
            for rep in range(rep_start, config['n_repetitions']):
                
                result = run_single_iteration(
                    train_size, rep, config, X_test_original, 
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
            
            start_rep = 0
    
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
    
    print(f"Results saved to: {output_dir}")
    print(f"Total results: {len(df_results)}")
    
    return df_results

if __name__ == "__main__":
    # Test different orderings
    test_config = {
        'train_sizes': [20, 50],
        'n_repetitions': 2,
        'test_size': 1000,
        'n_permutations': 3,
        'metrics': ['max_corr_diff', 'propensity_mse'],
        'include_categorical': False,
        'n_estimators': 3,
        'random_seed_base': 42,
        'column_order_strategy': 'topological'  # TEST: Use best order
    }
    
    results = run_experiment_1(test_config, "test_topological_order")
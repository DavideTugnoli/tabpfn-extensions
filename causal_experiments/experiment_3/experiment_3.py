"""
Experiment 3: Robustness to incorrect DAGs.

This experiment tests whether providing an incorrect DAG is better or worse
than providing no DAG at all. We compare multiple DAG conditions:
- correct: The true DAG
- no_dag: No DAG provided (vanilla TabPFN)
- wrong_parents: DAG with incorrect parent relationships
- missing_edges: DAG missing some true edges
- extra_edges: DAG with spurious edges added

Usage:
    python experiment_3.py                    # Run full experiment
    python experiment_3.py --no-resume       # Start fresh
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

# TabPFN imports - use local imports to avoid HPO dependency issues
from tabpfn_extensions.utils import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised.unsupervised import TabPFNUnsupervisedModel

# Create a namespace for the unsupervised module
class UnsupervisedNamespace:
    TabPFNUnsupervisedModel = TabPFNUnsupervisedModel

unsupervised = UnsupervisedNamespace()

# Local imports
from utils.scm_data import generate_scm_data, get_dag_and_config
from utils.metrics import FaithfulDataEvaluator
from utils.dag_utils import get_ordering_strategies, reorder_data_and_dag, print_dag_info, create_dag_variations
from utils.checkpoint_utils import save_checkpoint, get_checkpoint_info, cleanup_checkpoint

# Centralized default config
DEFAULT_CONFIG = {
    'train_sizes': [20, 50, 100, 200, 500],
    'dag_types': ['correct', 'no_dag', 'wrong_parents', 'missing_edges', 'extra_edges'],
    'n_repetitions': 10,
    'test_size': 2000,
    'n_permutations': 3,
    'metrics': ['mean_corr_difference', 'max_corr_difference', 'propensity_metrics', 'k_marginal_tvd'],
    'include_categorical': False,
    'n_estimators': 3,
    'random_seed_base': 42
}

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


def run_single_configuration(train_size, dag_type, repetition, config, 
                           X_test, correct_dag, col_names, categorical_cols,
                           dag_variations):
    """
    Run one configuration: train_size + dag_type + repetition.
    """
    print(f"    DAG type: {dag_type}, Rep: {repetition+1}/{config['n_repetitions']}")
    
    # Set seeds
    seed = config['random_seed_base'] + repetition
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Generate training data
    X_train = generate_scm_data(train_size, seed, config['include_categorical'])
    X_train_tensor = torch.from_numpy(X_train).float()
    
    # Get the DAG to use
    dag_to_use = dag_variations[dag_type]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and train model
    clf = TabPFNClassifier(n_estimators=config['n_estimators'], device=device)
    reg = TabPFNRegressor(n_estimators=config['n_estimators'], device=device)
    model = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
    
    if categorical_cols:
        model.set_categorical_features(categorical_cols)
    
    model.fit(X_train_tensor)
    
    # Generate synthetic data with specified DAG
    X_synth = generate_synthetic_data_quiet(
        model, config['test_size'], dag_to_use, config['n_permutations']
    )
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Ensure DataFrame inputs for evaluator
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=col_names)
    if not isinstance(X_synth, pd.DataFrame):
        X_synth = pd.DataFrame(X_synth, columns=col_names)
    
    # === EVALUATE ===
    evaluator = FaithfulDataEvaluator()
    
    # BUGFIX: Get the correct list of categorical column NAMES for the evaluator
    cat_col_names = []
    if categorical_cols:
        cat_col_names = [col_names[i] for i in categorical_cols]

    metrics = evaluator.evaluate(
        X_test,
        X_synth,
        categorical_columns=cat_col_names if cat_col_names else None,
        k_for_kmarginal=2
    )
    
    # Build result
    result = {
        'train_size': train_size,
        'dag_type': dag_type,
        'dag_used': str(dag_to_use) if dag_to_use is not None else 'None',
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
    
    # Add some DAG structure info for debugging
    if dag_to_use is not None:
        result['dag_edges'] = sum(len(parents) for parents in dag_to_use.values())
    else:
        result['dag_edges'] = 0
    
    return result


def run_experiment_3(config=None, output_dir="experiment_3_results", resume=True):
    """
    Main experiment function for testing DAG robustness.
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
    
    print(f"Experiment 3 - Output dir: {output_dir}")
    print(f"Config: {config}")
    
    # Setup
    correct_dag, col_names, categorical_cols = get_dag_and_config(config['include_categorical'])
    X_test_original = generate_scm_data(config['test_size'], 123, config['include_categorical'])
    
    # Create DAG variations
    dag_variations = create_dag_variations(correct_dag)
    
    # Check for checkpoint
    if resume:
        results_so_far, start_train_idx, start_rep = get_checkpoint_info(output_dir)
    else:
        results_so_far, start_train_idx, start_rep = [], 0, 0
    
    # Run experiment
    total_iterations = len(config['train_sizes']) * len(config['dag_types']) * config['n_repetitions']
    completed = len(results_so_far)
    
    print(f"Total iterations: {total_iterations}, Already completed: {completed}")
    
    try:
        for train_idx, train_size in enumerate(config['train_sizes'][start_train_idx:], start_train_idx):
            
            rep_start = start_rep if train_idx == start_train_idx else 0
            
            for rep in range(rep_start, config['n_repetitions']):
                
                for dag_type in config['dag_types']:
                    
                    result = run_single_configuration(
                        train_size, dag_type, rep, config, X_test_original,
                        correct_dag, col_names, categorical_cols, dag_variations
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
    """Main CLI interface for Experiment 3."""
    parser = argparse.ArgumentParser(description='Run Experiment 3: Robustness to incorrect DAGs')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh (ignore checkpoint)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Show experiment info
    print("=" * 60)
    print("EXPERIMENT 3: Robustness to Incorrect DAGs")
    print("=" * 60)
    print("\nResearch Question:")
    print("Is providing an incorrect DAG better or worse than providing")
    print("no DAG at all? How robust is TabPFN to DAG misspecification?")
    
    # Show correct DAG
    dag, col_names, _ = get_dag_and_config(False)
    print("\nCorrect SCM structure:")
    print_dag_info(dag, col_names)
    
    # Show DAG variations
    dag_variations = create_dag_variations(dag)
    print("\n\nDAG variations to test:")
    print("-" * 40)
    print("1. correct: The true DAG")
    print("2. no_dag: No DAG provided (vanilla TabPFN)")
    print("3. wrong_parents: Completely wrong parent relationships")
    print("4. missing_edges: Some true edges removed")
    print("5. extra_edges: Spurious edges added")
    
    # Use centralized config
    print("\n\nRunning FULL experiment...")
    config = DEFAULT_CONFIG.copy()
    output_dir = args.output or "experiment_3_results"
    
    # Calculate total configurations
    total_configs = (len(config['train_sizes']) * 
                    len(config['dag_types']) * 
                    config['n_repetitions'])
    
    print(f"\nExperiment Configuration:")
    print(f"  Training sizes: {config['train_sizes']}")
    print(f"  DAG types: {config['dag_types']}")
    print(f"  Repetitions: {config['n_repetitions']}")
    print(f"  Total configurations: {total_configs}")
    print(f"  Resume: {not args.no_resume}")
    print(f"  Output: {output_dir}")
    
    # Run experiment
    results = run_experiment_3(
        config=config,
        output_dir=output_dir,
        resume=not args.no_resume
    )
    
    # Print detailed summary
    if results is not None and len(results) > 0:
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        # Overall comparison
        # Get actual metric columns from results
        metric_columns = [col for col in results.columns if col not in ['train_size', 'dag_type', 'dag_used', 'repetition', 'categorical', 'dag_edges']]
        
        for metric in metric_columns:
            print(f"\n{metric.upper()} Results:")
            print("-" * 40)
            
            # Mean by DAG type
            mean_by_dag = results.groupby('dag_type')[metric].mean()
            
            # Sort by performance (lower is better)
            sorted_dags = mean_by_dag.sort_values()
            
            print("Performance ranking (best to worst):")
            for i, (dag_type, value) in enumerate(sorted_dags.items(), 1):
                print(f"  {i}. {dag_type}: {value:.4f}")
            
            # Compare to correct DAG
            if 'correct' in mean_by_dag.index:
                correct_value = mean_by_dag['correct']
                print(f"\nComparison to correct DAG ({correct_value:.4f}):")
                
                for dag_type in config['dag_types']:
                    if dag_type != 'correct' and dag_type in mean_by_dag.index:
                        diff = mean_by_dag[dag_type] - correct_value
                        pct_worse = (diff / correct_value) * 100
                        print(f"  {dag_type}: {diff:+.4f} ({pct_worse:+.1f}%)")


if __name__ == "__main__":
    main()
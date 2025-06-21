# src/metrics.py
"""
Evaluation metrics for synthetic data quality assessment.

This module provides metrics to compare real vs synthetic tabular data:
- Max Correlation Difference
- Propensity MSE  
- K-marginal Distribution Distance
"""

import numpy as np
import pandas as pd
import sys
from io import StringIO
from itertools import combinations
from typing import Dict, List, Optional, Union
from syntheval import SynthEval


class SyntheticDataEvaluator:
    """
    Evaluates the quality of synthetic data compared to real data.
    
    Provides multiple complementary metrics that focus on multivariate 
    relationships rather than univariate distributions.
    """
    
    def __init__(self, metrics: List[str] = None):
        """
        Initialize evaluator with specified metrics.
        
        Args:
            metrics: List of metrics to compute. 
                    Default: ['max_corr_diff', 'propensity_mse', 'kmarginal']
        """
        self.available_metrics = ['max_corr_diff', 'propensity_mse', 'kmarginal']
        self.metrics = metrics or self.available_metrics.copy()
        
        # Validate metrics
        invalid_metrics = set(self.metrics) - set(self.available_metrics)
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}. "
                           f"Available: {self.available_metrics}")
    
    def evaluate(self, X_real: Union[np.ndarray, pd.DataFrame], 
                 X_synthetic: Union[np.ndarray, pd.DataFrame],
                 column_names: Optional[List[str]] = None,
                 categorical_columns: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Evaluate synthetic data quality using multiple metrics.
        
        Args:
            X_real: Real data
            X_synthetic: Synthetic data  
            column_names: Column names for the data
            categorical_columns: Indices of categorical columns
            
        Returns:
            Dictionary with metric names as keys and scores as values
        """
        # Convert to DataFrame if needed
        X_real_df = self._ensure_dataframe(X_real, column_names)
        X_synthetic_df = self._ensure_dataframe(X_synthetic, column_names)
        
        # Handle categorical columns
        if categorical_columns:
            X_real_df, X_synthetic_df = self._process_categorical_columns(
                X_real_df, X_synthetic_df, categorical_columns
            )
        
        results = {}
        
        # Compute SynthEval metrics (max_corr_diff, propensity_mse)
        if any(metric in ['max_corr_diff', 'propensity_mse'] for metric in self.metrics):
            syntheval_results = self._compute_syntheval_metrics(X_real_df, X_synthetic_df)
            results.update(syntheval_results)
        
        # Compute K-marginal distance
        if 'kmarginal' in self.metrics:
            results['kmarginal'] = self._compute_kmarginal_distance(
                X_real_df.values, X_synthetic_df.values, 
                categorical_columns, X_real_df.columns.tolist()
            )
        
        # Filter to requested metrics only
        return {metric: results[metric] for metric in self.metrics if metric in results}
    
    def _ensure_dataframe(self, data: Union[np.ndarray, pd.DataFrame], 
                         column_names: Optional[List[str]]) -> pd.DataFrame:
        """Convert data to DataFrame if it's a numpy array."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        else:
            if column_names is None:
                column_names = [f'col_{i}' for i in range(data.shape[1])]
            return pd.DataFrame(data, columns=column_names)
    
    def _process_categorical_columns(self, X_real: pd.DataFrame, X_synthetic: pd.DataFrame,
                                   categorical_columns: List[int]) -> tuple:
        """Convert specified columns to categorical type."""
        X_real = X_real.copy()
        X_synthetic = X_synthetic.copy()
        
        for col_idx in categorical_columns:
            col_name = X_real.columns[col_idx]
            X_real[col_name] = X_real[col_name].astype('category')
            X_synthetic[col_name] = X_synthetic[col_name].astype('category')
        
        return X_real, X_synthetic
    
    def _compute_syntheval_metrics(self, X_real: pd.DataFrame, 
                                  X_synthetic: pd.DataFrame) -> Dict[str, float]:
        """Compute metrics using SynthEval library."""
        evaluator = SynthEval(X_real)
        
        # Suppress output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        try:
            evaluator.evaluate(
                X_synthetic, 
                None,
                **{
                    "corr_diff": {"return_mats": True},
                    "p_mse": {"k_folds": 5, "max_iter": 100, "solver": "liblinear"},
                }
            )
            
            results = {}
            
            if 'max_corr_diff' in self.metrics:
                results['max_corr_diff'] = evaluator._raw_results["corr_diff"]["diff_cor_mat"].abs().values.max()
            
            if 'propensity_mse' in self.metrics:
                results['propensity_mse'] = evaluator._raw_results["p_mse"]["avg pMSE"]
            
            return results
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def _compute_kmarginal_distance(self, real_data: np.ndarray, synthetic_data: np.ndarray,
                                   categorical_indices: Optional[List[int]] = None,
                                   column_names: Optional[List[str]] = None,
                                   k: int = 2, n_bins: int = 10) -> float:
        """
        Compute k-marginal distribution distance.
        
        This metric discretizes continuous variables and computes the average
        Total Variation Distance between k-dimensional marginal distributions.
        
        Args:
            real_data: Real data array
            synthetic_data: Synthetic data array
            categorical_indices: Indices of categorical columns
            column_names: Column names
            k: Dimension of marginals to compare
            n_bins: Number of bins for discretization
            
        Returns:
            K-marginal distance (lower is better, range [0,1])
        """
        # Create DataFrames
        if column_names is None:
            column_names = [f'col_{i}' for i in range(real_data.shape[1])]
        
        df_real = pd.DataFrame(real_data, columns=column_names)
        df_synth = pd.DataFrame(synthetic_data, columns=column_names)
        
        # Identify categorical vs numeric columns
        categorical_set = set(categorical_indices) if categorical_indices else set()
        numeric_cols = [i for i in range(len(df_real.columns)) if i not in categorical_set]
        
        # Discretize numeric columns using percentiles from real data
        for col_idx in numeric_cols:
            col = df_real.columns[col_idx]
            
            # Create bins based on real data percentiles
            percentiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.percentile(df_real[col].dropna(), percentiles)
            bin_edges = np.unique(bin_edges)  # Remove duplicates
            
            # Apply binning to both datasets
            df_real[col] = pd.cut(df_real[col], bins=bin_edges, labels=False, include_lowest=True)
            df_synth[col] = pd.cut(df_synth[col], bins=bin_edges, labels=False, include_lowest=True)
            
            # Handle out-of-range values in synthetic data
            df_synth[col] = df_synth[col].fillna(-1)  # Mark as outliers
        
        # Generate k-dimensional feature combinations
        feature_combinations = list(combinations(range(len(df_real.columns)), k))
        
        # Limit combinations for computational efficiency
        if len(feature_combinations) > 1000:
            np.random.seed(42)
            indices = np.random.choice(len(feature_combinations), 1000, replace=False)
            feature_combinations = [feature_combinations[i] for i in indices]
        
        # Compute Total Variation Distance for each k-marginal
        total_tvd = 0.0
        
        for feat_indices in feature_combinations:
            cols = [df_real.columns[i] for i in feat_indices]
            
            # Get marginal distributions (normalized frequencies)
            real_counts = df_real.groupby(cols).size()
            real_density = real_counts / len(df_real)
            
            synth_counts = df_synth.groupby(cols).size()
            synth_density = synth_counts / len(df_synth)
            
            # Compute Total Variation Distance
            all_keys = set(real_density.index) | set(synth_density.index)
            tvd = 0.0
            
            for key in all_keys:
                real_val = real_density.get(key, 0.0)
                synth_val = synth_density.get(key, 0.0)
                tvd += abs(real_val - synth_val)
            
            # TVD is sum of absolute differences / 2
            total_tvd += tvd / 2.0
        
        # Average TVD across all k-marginals
        mean_tvd = total_tvd / len(feature_combinations)
        
        return float(mean_tvd)


def get_metric_descriptions() -> Dict[str, str]:
    """
    Get descriptions of available metrics.
    
    Returns:
        Dictionary mapping metric names to their descriptions
    """
    return {
        'max_corr_diff': 'Maximum absolute difference between correlation matrices (lower is better)',
        'propensity_mse': 'Mean squared error of propensity scores from real/synthetic classifier (lower is better)', 
        'kmarginal': 'Average Total Variation Distance between k-dimensional marginal distributions (lower is better)'
    }
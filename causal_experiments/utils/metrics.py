"""
Evaluation metrics for synthetic data quality assessment.

This module provides metrics to compare real vs synthetic tabular data:
- Max Correlation Difference (max_corr_diff)
- Mean Correlation Difference (mean_corr_diff)
- Propensity MSE (propensity_mse)
- K-marginal Distribution Distance (kmarginal)
"""

import numpy as np
import pandas as pd
import sys
from io import StringIO
from typing import Dict, List, Optional, Union
from syntheval import SynthEval

from metrics.k_marginal import KMarginal, bin_data_for_k_marginal
from metrics.syntheval_facade import evaluate_correlation_difference, evaluate_pmse

class SyntheticDataEvaluator:
    """
    Evaluates the quality of synthetic data compared to real data.
    Provides multiple complementary metrics.
    """

    def __init__(self, metrics: List[str] = None):
        """
        Initialize evaluator with specified metrics.
        Default: ['mean_corr_diff', 'max_corr_diff', 'propensity_mse', 'kmarginal']
        """
        self.available_metrics = ['mean_corr_diff', 'max_corr_diff', 'propensity_mse', 'kmarginal']
        self.metrics = metrics or self.available_metrics.copy()
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
        """
        X_real_df = self._ensure_dataframe(X_real, column_names)
        X_synthetic_df = self._ensure_dataframe(X_synthetic, column_names)

        results = {}

        # Correlation metrics (mean/max) and propensity_mse
        if any(metric in ['mean_corr_diff', 'max_corr_diff', 'propensity_mse'] for metric in self.metrics):
            syntheval_results = self._compute_syntheval_metrics(
                X_real_df, X_synthetic_df, categorical_columns
            )
            results.update(syntheval_results)

        # K-marginal
        if 'kmarginal' in self.metrics:
            results['kmarginal'] = self._compute_kmarginal_distance(
                X_real_df, X_synthetic_df, categorical_columns
            )

        # Filter to requested metrics only
        return {metric: results[metric] for metric in self.metrics if metric in results}

    def _ensure_dataframe(self, data: Union[np.ndarray, pd.DataFrame],
                         column_names: Optional[List[str]]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        else:
            if column_names is None:
                column_names = [f'col_{i}' for i in range(data.shape[1])]
            return pd.DataFrame(data, columns=column_names)

    def _compute_syntheval_metrics(self, X_real: pd.DataFrame,
                                   X_synthetic: pd.DataFrame,
                                   categorical_columns: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Compute SynthEval-based metrics:
        - mean_corr_diff (mean of upper triangle, excluding diagonal)
        - max_corr_diff (max absolute difference)
        - propensity_mse (average pMSE)
        """
        X_real_for_eval = X_real.copy()
        X_synthetic_for_eval = X_synthetic.copy()

        cat_col_names = []
        if categorical_columns:
            for col_idx in categorical_columns:
                col_name = X_real.columns[col_idx]
                cat_col_names.append(col_name)
                X_real_for_eval[col_name] = X_real_for_eval[col_name].astype(int)
                X_synthetic_for_eval[col_name] = X_synthetic_for_eval[col_name].astype(int)

        evaluator = SynthEval(X_real_for_eval)

        # Suppress printout
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            evaluator.evaluate(
                X_synthetic_for_eval,
                cat_cols=cat_col_names if cat_col_names else None,
                **{
                    "corr_diff": {"return_mats": True},
                    "p_mse": {"k_folds": 5, "max_iter": 100, "solver": "liblinear"},
                }
            )

            results = {}
            # Correlation matrix difference
            if 'mean_corr_diff' in self.metrics or 'max_corr_diff' in self.metrics:
                # Usa la funzione di facade per uniformità (già fedele ai file)
                corr_diff_mat = evaluate_correlation_difference(evaluator, X_synthetic_for_eval)
                abs_diff = np.abs(corr_diff_mat.values)
                tri_upper = abs_diff[np.triu_indices(abs_diff.shape[0], k=1)]
                if 'mean_corr_diff' in self.metrics:
                    results['mean_corr_diff'] = tri_upper.mean()
                if 'max_corr_diff' in self.metrics:
                    results['max_corr_diff'] = abs_diff.max()
            # Propensity
            if 'propensity_mse' in self.metrics:
                pmse_res = evaluate_pmse(evaluator, X_synthetic_for_eval)
                results['propensity_mse'] = pmse_res.get('avg pMSE', -1.0)

            return results

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _compute_kmarginal_distance(self, real_data: pd.DataFrame,
                                    synthetic_data: pd.DataFrame,
                                    categorical_indices: Optional[List[int]] = None,
                                    k: int = 2) -> float:
        """
        Faithful calculation using bin_data_for_k_marginal and KMarginal.
        """
        # Defensive: synth columns must match real columns (same names)
        if not all(real_data.columns == synthetic_data.columns):
            synthetic_data = synthetic_data.copy()
            synthetic_data.columns = real_data.columns

        real_binned, _, syn_binned = bin_data_for_k_marginal(
            real_data, real_data, synthetic_data, categorical_features=[]
        )
        try:
            km = KMarginal(real_binned, syn_binned, k=k)
            tdds = 0
            for marg in km.marginal_pairs():
                _, _, abs_den_diff = km.marginal_densities(km.td, marg)
                tdds += abs_den_diff.sum()
            mean_tdds = tdds / len(km.marginals)
            return mean_tdds / 2
        except Exception:
            return 1.0  # Fail-safe

def get_metric_descriptions() -> Dict[str, str]:
    return {
        'mean_corr_diff': 'Mean absolute difference (off-diagonal) between correlation matrices (lower is better)',
        'max_corr_diff': 'Maximum absolute difference between correlation matrices (lower is better)',
        'propensity_mse': 'Mean squared error of propensity scores from real/synthetic classifier (lower is better)',
        'kmarginal': 'Average Total Variation Distance between k-dimensional marginal distributions (lower is better)'
    }

"""
This module provides the FaithfulDataEvaluator class to assess the quality of
synthetic data.

The calculation logic is a faithful re-implementation of the methodology
found in the reference files ('k_marginal.py' and 'syntheval_facade.py')
to ensure 100% consistent results.

This module is self-contained and does not require other local files.
"""

import pandas as pd
import numpy as np
import itertools
import random
import warnings
from typing import Dict, List, Tuple

# The only required external dependency is SynthEval
from syntheval import SynthEval


class FaithfulDataEvaluator:
    """
    Calculates data quality metrics for synthetic data, faithfully adhering
    to a reference methodology.
    """

    def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        k_for_kmarginal: int = 2
    ) -> Dict[str, float]:
        """
        Runs the complete evaluation and returns a dictionary of metrics.

        Args:
            real_data (pd.DataFrame): The DataFrame containing the real data.
            synthetic_data (pd.DataFrame): The DataFrame containing the synthetic data.
            k_for_kmarginal (int, optional): The order of marginals to compute. Defaults to 2.

        Returns:
            Dict[str, float]: A dictionary containing the names and values of the metrics.
        """
        results = {}

        # --- Metrics based on the SynthEval library ---
        syntheval_metrics = self._compute_syntheval_metrics(real_data, synthetic_data)
        results.update(syntheval_metrics)

        # --- K-Marginal Metric ---
        kmarginal_tvd = self._compute_kmarginal_metric(real_data, synthetic_data, k=k_for_kmarginal)
        results['k_marginal_tvd'] = kmarginal_tvd

        return results

    def _compute_syntheval_metrics(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculates metrics that rely on the SynthEval library, using the
        exact parameters from the reference implementation.
        """
        # Initialize the evaluator with the real data
        evaluator = SynthEval(real_data, cat_cols=[])  # Assuming numeric data as per examples

        # Run the evaluation in a single pass for efficiency
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            evaluator.evaluate(
                synthetic_data,
                **{
                    "corr_diff": {"return_mats": True},
                    "p_mse": {
                        "k_folds": 5,
                        "max_iter": 100,
                        "solver": "liblinear",
                    }
                }
            )

        eval_results = evaluator.get_results()
        metrics = {}

        # 1. Extract correlation difference and compute max and mean
        if 'corr_diff' in eval_results:
            diff_matrix = eval_results['corr_diff']['diff_cor_mat']
            abs_diff_values = np.abs(diff_matrix.values)

            # Max corr distance
            metrics['max_corr_distance'] = np.max(abs_diff_values)

            # Mean corr distance (on the upper triangle to avoid duplicates)
            upper_triangle_indices = np.triu_indices(abs_diff_values.shape[0], k=1)
            metrics['mean_corr_distance'] = np.mean(abs_diff_values[upper_triangle_indices])
        else:
            metrics['max_corr_distance'] = -1.0
            metrics['mean_corr_distance'] = -1.0

        # 2. Extract Propensity MSE
        if 'p_mse' in eval_results:
            metrics['propensity_mse'] = eval_results['p_mse'].get('avg pMSE', -1.0)
        else:
            metrics['propensity_mse'] = -1.0

        return metrics

    def _compute_kmarginal_metric(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        k: int
    ) -> float:
        """
        Calculates the K-Marginal TVD metric by following the reference
        file's discretization and calculation logic.
        """
        # Step 1: Discretize (bin) the data using the reference methodology
        real_binned, syn_binned = self._discretize_for_kmarginal(real_data, synthetic_data)

        # Step 2: Calculate the mean TVD on the marginals
        features = real_binned.columns.tolist()
        if len(features) < k:
            return 1.0  # Cannot compute marginals

        # Generate marginal combinations
        marginals = list(itertools.combinations(sorted(features), k))

        # Subsample if there are too many marginals (as per original file)
        subsample_threshold = 1000
        if len(marginals) > subsample_threshold:
            marginals = random.sample(marginals, subsample_threshold)

        if not marginals:
            return 1.0

        total_density_diff_sum = 0
        for marg in marginals:
            marg = list(marg)
            # Calculate densities (probability distributions)
            t_den = real_binned.groupby(marg).size() / len(real_binned)
            s_den = syn_binned.groupby(marg).size() / len(syn_binned)

            # Sum of absolute differences for the current marginal
            abs_den_diff = t_den.subtract(s_den, fill_value=0).abs()
            total_density_diff_sum += abs_den_diff.sum()

        # Calculate the mean (this is 'mean_tdds' from the original file)
        mean_total_density_diff = total_density_diff_sum / len(marginals)

        # The TVD is half of this sum
        mean_tvd = mean_total_density_diff / 2.0

        return mean_tvd

    def _discretize_for_kmarginal(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Replicates the percentile-rank-based binning logic from the reference file.
        """
        # Identify numeric columns with enough unique values to be binned
        numeric_features = [
            col for col in real_data.select_dtypes(include=np.number).columns
            if real_data[col].nunique() >= 20
        ]

        # --- Bin the real (training) data ---
        real_binned = real_data.copy()
        for col in numeric_features:
            # Handle non-NA values
            not_na_mask = real_binned[col].notna()
            # Calculate percentile ranks
            ranked_pct = real_binned.loc[not_na_mask, col].rank(pct=True)
            # Assign to one of 20 bins (0-19)
            real_binned.loc[not_na_mask, col] = ranked_pct.apply(lambda x: int(20 * x) if x < 1 else 19)
            # Assign a special bin for NA values
            real_binned.loc[real_data[col].isna(), col] = -1

        # --- Bin the synthetic data based on the real data's bin structure ---
        syn_binned = synthetic_data.copy()
        for col in numeric_features:
            # Handle non-NA values
            syn_not_na_mask = syn_binned[col].notna()
            syn_numeric_values = pd.to_numeric(syn_binned.loc[syn_not_na_mask, col])
            
            binned_syn_values = syn_numeric_values.copy()
            max_value_of_previous_bin = -np.inf
            
            # Sort unique bins from the real data (excluding the NA bin)
            unique_bins = sorted([b for b in real_binned[col].unique() if b != -1])
            
            for i, bin_val in enumerate(unique_bins):
                # Find the original maximum value in the real data for this bin
                indices_for_bin = real_binned[real_binned[col] == bin_val].index
                max_value_in_bin = real_data.loc[indices_for_bin, col].max()

                if i < len(unique_bins) - 1: # Not the last bin
                    min_value_of_current_bin = max_value_of_previous_bin
                    binned_syn_values.loc[
                        (syn_numeric_values > min_value_of_current_bin) &
                        (syn_numeric_values <= max_value_in_bin)
                    ] = bin_val
                else: # This is the last bin
                    min_value_of_current_bin = max_value_of_previous_bin
                    binned_syn_values.loc[syn_numeric_values > min_value_of_current_bin] = bin_val
                
                max_value_of_previous_bin = max_value_in_bin

            syn_binned.loc[syn_not_na_mask, col] = binned_syn_values
            # Assign a special bin for NA values
            syn_binned.loc[synthetic_data[col].isna(), col] = -1

        # Return only the relevant binned columns as integers
        return real_binned[numeric_features].astype(int), syn_binned[numeric_features].astype(int)


# --- Example of How to Use This File ---
if __name__ == '__main__':
    print("Example usage of the FaithfulDataEvaluator class")

    # 1. Create sample data
    np.random.seed(42)
    sample_real_data = pd.DataFrame({
        'feature_A': np.random.randn(500),
        'feature_B': np.random.rand(500) * 100,
        'feature_C': np.random.poisson(10, 500)
    })
    # Add some missing values for testing the binning logic
    sample_real_data.loc[sample_real_data.sample(frac=0.05).index, 'feature_A'] = np.nan

    # Create synthetic data with slight differences
    sample_synthetic_data = pd.DataFrame({
        'feature_A': np.random.randn(500) * 1.2 + 0.2,
        'feature_B': np.random.rand(500) * 95,
        'feature_C': np.random.poisson(10.5, 500)
    })

    # 2. Initialize and use the evaluator
    evaluator = FaithfulDataEvaluator()

    print("\nRunning evaluation...")
    all_metrics = evaluator.evaluate(sample_real_data, sample_synthetic_data, k_for_kmarginal=2)

    # 3. Print the results
    print("\n--- Evaluation Results ---")
    for metric_name, value in all_metrics.items():
        print(f"{metric_name:<25}: {value:.6f}")
    print("--------------------------")
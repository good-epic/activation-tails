from typing import Dict, List, Tuple, Optional, Union, Sequence, Any
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import gc
from transformer_lens import HookedTransformer
from activation_distribution import LayerwiseDistributions
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

class CategorizedDistributions:
    """
    Wrapper class to manage LayerwiseDistributions with multiple category levels.
    Supports flexible category-based access and querying.
    Category structure is built dynamically as distributions are added.
    """
    
    def __init__(self):
        """Initialize empty categorized distributions."""
        self.distributions = {}
        self.indices = defaultdict(lambda: defaultdict(set))
        self.category_sets = []  # List of sets of categories at each level
    
    def add_distribution(self, categories: Tuple[str, ...], distribution: LayerwiseDistributions):
        """
        Add a distribution with its categories.
        
        Args:
            categories: Tuple of category values
            distribution: LayerwiseDistributions object
        """
        # Update our knowledge of category structure
        while len(self.category_sets) < len(categories):
            self.category_sets.append(set())
        
        for i, cat in enumerate(categories):
            self.category_sets[i].add(cat)
        
        # Store the distribution
        self.distributions[categories] = distribution
        
        # Update all relevant indices
        for n in range(1, len(categories) + 1):
            for positions in combinations(range(len(categories)), n):
                # Create key for these positions
                key = tuple(categories[p] for p in positions)
                self.indices[positions][key].add(categories)
    
    def get_distribution(self, *args, positions: Optional[Tuple[int, ...]] = None) -> Union[LayerwiseDistributions, List[LayerwiseDistributions]]:
        """
        Get distribution(s) matching the provided categories.
        
        Can be called in two ways:
        1. cd.get_distribution('science', 'formal') - matches categories in order from start
        2. cd.get_distribution('formal', positions=(1,)) - matches specific positions
        
        Args:
            *args: Category values
            positions: Optional tuple specifying which positions to match.
                      If None, matches from start.
        
        Returns:
            Single LayerwiseDistributions if exact match, or list of matches if partial
        """
        categories = args
        
        if len(categories) > len(self.category_sets):
            raise ValueError(f"Too many categories provided. Maximum is {len(self.category_sets)}")
            
        if positions is None:
            # Default behavior - match from start
            positions = tuple(range(len(categories)))
        else:
            # Position-specific matching
            if len(positions) != len(categories):
                raise ValueError("Number of positions must match number of categories")
            if not all(0 <= p < len(self.category_sets) for p in positions):
                raise ValueError(f"Positions must be between 0 and {len(self.category_sets)-1}")
            
        # Look up in index
        matching_keys = self.indices[positions][categories]
        
        if not matching_keys:
            pos_str = f" at positions {positions}" if positions else ""
            raise KeyError(f"No distributions found matching categories {categories}{pos_str}")
        
        if len(matching_keys) == 1:
            (key,) = matching_keys
            return self.distributions[key]
        else:
            return [self.distributions[key] for key in matching_keys]




    def _create_label_mappings(self, unique_tuples: List[Tuple[str, ...]]) -> Tuple[Dict[Tuple, str], Dict[str, Tuple]]:
        """
        Create bidirectional mappings between category tuples and string labels.
        
        Args:
            unique_tuples: List of unique category tuples
            
        Returns:
            tuple_to_label: Dictionary mapping tuples to string labels
            label_to_tuple: Dictionary mapping string labels back to tuples
        """
        tuple_to_label = {tup: f"category_{i}" for i, tup in enumerate(unique_tuples)}
        label_to_tuple = {v: k for k, v in tuple_to_label.items()}
        return tuple_to_label, label_to_tuple

    def _compute_confusion_matrices(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute confusion matrices and related metrics for multi-category classification results.
        
        Args:
            results_df: DataFrame with test results including true_category_N columns
                    and top_N_categories predictions
            
        Returns:
            Dictionary containing:
                - full_tuple: Statistics for complete category tuples
                - dimensions: Per-dimension statistics
        """
        # Get number of category levels
        n_categories = len([col for col in results_df.columns if col.startswith('true_category_')])
        
        # Extract true and predicted categories
        true_tuples = [
            tuple(row[f'true_category_{i+1}'] for i in range(n_categories))
            for _, row in results_df.iterrows()
        ]
        
        # Parse top predictions (taking top 1 for confusion matrix)
        pred_tuples = []
        for _, row in results_df.iterrows():
            categories = row['top_1_categories'].split('_')
            if len(categories) != n_categories:
                # If prediction doesn't have the right number of categories,
                # pad with None values
                categories.extend([None] * (n_categories - len(categories)))
            pred_tuples.append(tuple(categories))
        
        # Get unique tuples and create label mappings
        unique_tuples = sorted(list(set(true_tuples + [t for t in pred_tuples if all(x is not None for x in t)])))
        tuple_to_label = {tup: f"category_{i}" for i, tup in enumerate(unique_tuples)}
        label_to_tuple = {v: k for k, v in tuple_to_label.items()}
        
        # Convert tuples to labels, handling None values
        true_labels = [tuple_to_label[tup] for tup in true_tuples]
        pred_labels = []
        for tup in pred_tuples:
            if all(x is not None for x in tup) and tup in tuple_to_label:
                pred_labels.append(tuple_to_label[tup])
            else:
                # For incomplete/invalid predictions, use a special label
                pred_labels.append("invalid_prediction")
        
        # Add "invalid_prediction" to label mappings if needed
        if "invalid_prediction" in pred_labels:
            label_to_tuple["invalid_prediction"] = ("INVALID",) * n_categories
        
        # Compute full tuple confusion matrix
        all_labels = list(label_to_tuple.keys())
        conf_mat = confusion_matrix(true_labels, pred_labels, labels=all_labels)
        
        # Calculate precision, recall, and F1 matrices
        n_classes = len(all_labels)
        precision_mat = np.zeros((n_classes, n_classes))
        recall_mat = np.zeros((n_classes, n_classes))
        f1_mat = np.zeros((n_classes, n_classes))
        
        for i in range(n_classes):
            for j in range(n_classes):
                # Precision
                col_sum = conf_mat[:, j].sum()
                precision_mat[i, j] = conf_mat[i, j] / col_sum if col_sum > 0 else 0
                
                # Recall
                row_sum = conf_mat[i, :].sum()
                recall_mat[i, j] = conf_mat[i, j] / row_sum if row_sum > 0 else 0
                
                # F1
                if precision_mat[i, j] + recall_mat[i, j] > 0:
                    f1_mat[i, j] = 2 * (precision_mat[i, j] * recall_mat[i, j]) / (precision_mat[i, j] + recall_mat[i, j])
        
        # Calculate per-dimension metrics
        dimension_stats = {}
        
        for dim in range(n_categories):
            # Get values for this dimension
            true_dim = [t[dim] for t in true_tuples]
            pred_dim = []
            
            for p in pred_tuples:
                if dim < len(p) and p[dim] is not None:
                    pred_dim.append(p[dim])
                else:
                    pred_dim.append("invalid_prediction")
            
            # Get unique values for this dimension
            unique_dim_values = sorted(list(set(true_dim + [p for p in pred_dim if p != "invalid_prediction"])))
            if "invalid_prediction" in pred_dim:
                unique_dim_values.append("invalid_prediction")
            
            # Compute confusion matrix for this dimension
            dim_conf_mat = confusion_matrix(true_dim, pred_dim, labels=unique_dim_values)
            
            # Calculate precision, recall, and F1 for this dimension
            n_dim_classes = len(unique_dim_values)
            dim_precision = np.zeros((n_dim_classes, n_dim_classes))
            dim_recall = np.zeros((n_dim_classes, n_dim_classes))
            dim_f1 = np.zeros((n_dim_classes, n_dim_classes))
            
            for i in range(n_dim_classes):
                for j in range(n_dim_classes):
                    # Precision
                    col_sum = dim_conf_mat[:, j].sum()
                    dim_precision[i, j] = dim_conf_mat[i, j] / col_sum if col_sum > 0 else 0
                    
                    # Recall
                    row_sum = dim_conf_mat[i, :].sum()
                    dim_recall[i, j] = dim_conf_mat[i, j] / row_sum if row_sum > 0 else 0
                    
                    # F1
                    if dim_precision[i, j] + dim_recall[i, j] > 0:
                        dim_f1[i, j] = 2 * (dim_precision[i, j] * dim_recall[i, j]) / (dim_precision[i, j] + dim_recall[i, j])
            
            dimension_stats[f"dimension_{dim}"] = {
                "confusion_matrix": {
                    "matrix": dim_conf_mat,
                    "labels": unique_dim_values
                },
                "precision_matrix": {
                    "matrix": dim_precision,
                    "labels": unique_dim_values
                },
                "recall_matrix": {
                    "matrix": dim_recall,
                    "labels": unique_dim_values
                },
                "f1_matrix": {
                    "matrix": dim_f1,
                    "labels": unique_dim_values
                }
            }
        
        # Return all statistics
        return {
            "full_tuple": {
                "confusion_matrix": {
                    "matrix": conf_mat,
                    "labels": [label_to_tuple[label] for label in all_labels]
                },
                "precision_matrix": {
                    "matrix": precision_mat,
                    "labels": [label_to_tuple[label] for label in all_labels]
                },
                "recall_matrix": {
                    "matrix": recall_mat,
                    "labels": [label_to_tuple[label] for label in all_labels]
                },
                "f1_matrix": {
                    "matrix": f1_mat,
                    "labels": [label_to_tuple[label] for label in all_labels]
                }
            },
            "dimensions": dimension_stats
        }
    

    def _calculate_summary_statistics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate summary statistics from test results.
        
        Args:
            results_df: DataFrame with test results including true_category_N columns
                    and top_N_categories predictions
            
        Returns:
            Dictionary containing confusion matrices and other metrics
        """
        # Compute confusion matrices and related metrics
        stats = self._compute_confusion_matrices(results_df)
        
        # Get number of category levels
        n_categories = len([col for col in results_df.columns if col.startswith('true_category_')])
        
        # Extract true categories
        true_tuples = [
            tuple(row[f'true_category_{i+1}'] for i in range(n_categories))
            for _, row in results_df.iterrows()
        ]
        
        # Add top-k accuracy for full tuples
        stats['full_tuple']['top_k_accuracy'] = {}
        for k in [1, 2, 3]:
            correct = 0
            total = len(true_tuples)
            
            for idx, true_tuple in enumerate(true_tuples):
                pred_cats = results_df.iloc[idx][f'top_{k}_categories'].split('_')
                if len(pred_cats) == len(true_tuple) and all(p == t for p, t in zip(pred_cats, true_tuple)):
                    correct += 1
                    
            stats['full_tuple']['top_k_accuracy'][k] = correct / total if total > 0 else 0.0
        
        # Add per-dimension top-k accuracy
        for dim in range(n_categories):
            true_dim = [t[dim] for t in true_tuples]
            
            stats['dimensions'][f'dimension_{dim}']['top_k_accuracy'] = {}
            for k in [1, 2, 3]:
                correct = 0
                total = len(true_dim)
                
                for idx, true_val in enumerate(true_dim):
                    pred_cats = results_df.iloc[idx][f'top_{k}_categories'].split('_')
                    if dim < len(pred_cats) and pred_cats[dim] == true_val:
                        correct += 1
                        
                stats['dimensions'][f'dimension_{dim}']['top_k_accuracy'][k] = correct / total if total > 0 else 0.0
        
        return stats
    
    
    def set_tail_cutoffs(self) -> None:
        """
        Set tail cutoffs for all activation distributions in all layerwise distributions.
        """
        # Loop through all category combinations
        for category_tuple in self.keys():
            # Get layerwise distributions for this category
            layerwise_dist = self[category_tuple]
            
            # Loop through each layer's activation distribution
            for layer_idx in range(len(layerwise_dist)):
                activation_dist = layerwise_dist[layer_idx]
                activation_dist.set_tail_cutoffs()

    
    def compute_test_likelihoods(
            self,
            model: HookedTransformer,
            test_sentences: List[str],
            true_categories: List[Tuple[str, ...]],
            hook_names: List[str],
            batch_size: int = 20,
            device: str = "cuda",
            drop_last_word: bool = True,
            std_threshold: Optional[float] = None,
            detect_tails: bool = False
        ) -> Tuple[pd.DataFrame, Dict]:
            """
            Compute log likelihoods for test sentences.
            
            Added argument:
                std_threshold: If set, only consider points beyond this many standard 
                            deviations from the mean
            """
            if len(test_sentences) != len(true_categories):
                raise ValueError("Number of sentences must match number of category tuples")
                
            results = []
            
            # Process test sentences in batches
            for i in tqdm(range(0, len(test_sentences), batch_size), desc="Computing likelihoods"):
                batch_sentences = test_sentences[i:i + batch_size]
                if drop_last_word:
                    batch_sentences = [s.rsplit(' ', 1)[0] for s in batch_sentences]
                
                # Get sequence lengths for each sentence in batch
                seq_lengths = [len(model.to_str_tokens(s)) for s in batch_sentences]
                    
                with torch.inference_mode():
                    _, batch_cache = model.run_with_cache(batch_sentences, names_filter=hook_names)
                    
                    # Process each sentence in batch
                    for b, (sentence, seq_length) in enumerate(zip(batch_sentences, seq_lengths)):
                        sentence_activations = [
                            batch_cache[hook_name][b, seq_length-1, :].cpu().numpy()
                            for hook_name in hook_names
                        ]
                        
                        row_data = {
                            'sentence': test_sentences[i + b],
                            'processed_sentence': sentence,
                            'sequence_length': seq_length
                        }
                        
                        # Add true categories
                        true_cats = true_categories[i + b]
                        for idx, cat in enumerate(true_cats, 1):
                            row_data[f'true_category_{idx}'] = cat
                        
                        # Calculate log-likelihood for each category combination
                        ll_values = []
                        for cat_tuple in self.keys():
                            layer_lls = []
                            for layer_idx, layer_activation in enumerate(sentence_activations):
                                ll = self[cat_tuple][layer_idx].log_likelihood(
                                    layer_activation,
                                    std_threshold=std_threshold,
                                    detect_tails=detect_tails
                                )
                                layer_lls.append(np.sum(ll))
                            
                            row_data[f'll_layers_{"_".join(cat_tuple)}'] = layer_lls
                            total_ll = sum(layer_lls)
                            ll_values.append((cat_tuple, total_ll))
                            row_data[f'll_{"_".join(cat_tuple)}'] = total_ll
                        
                        # Sort and store top predictions
                        ll_values.sort(key=lambda x: x[1], reverse=True)
                        for k in range(min(3, len(ll_values))):
                            cats, ll = ll_values[k]
                            row_data[f'top_{k+1}_categories'] = '_'.join(cats)
                            row_data[f'top_{k+1}_ll'] = ll
                        
                        results.append(row_data)
                
                del batch_cache
                gc.collect()
                torch.cuda.empty_cache()
            
            results_df = pd.DataFrame(results)
            stats = self._calculate_summary_statistics(results_df)
            
            return results_df, stats

    
    def get_categories(self, level: Optional[int] = None) -> Union[List[str], Dict[int, List[str]]]:
        """Get unique categories at specified level(s)."""
        if level is not None:
            if not 0 <= level < len(self.category_sets):
                raise ValueError(f"Level must be between 0 and {len(self.category_sets)-1}")
            return sorted(self.category_sets[level])
        return {i: sorted(cats) for i, cats in enumerate(self.category_sets)}
    
    def n_category_levels(self) -> int:
        """Get number of category levels."""
        return len(self.category_sets)
    
    def keys(self) -> List[Tuple[str, ...]]:
        """Get all category combinations as tuples."""
        return list(self.distributions.keys())
    
    def values(self) -> List[LayerwiseDistributions]:
        """Get all LayerwiseDistributions objects."""
        return list(self.distributions.values())
    
    def items(self) -> List[Tuple[Tuple[str, ...], LayerwiseDistributions]]:
        """Get all (categories, distribution) pairs."""
        return list(self.distributions.items())
    
    def __len__(self) -> int:
        """Return number of stored distributions."""
        return len(self.distributions)
    
    def __contains__(self, categories: Tuple[str, ...]) -> bool:
        """Check if exact category combination exists."""
        return categories in self.distributions
    
    def __iter__(self):
        """Iterator over category tuples."""
        return iter(self.distributions)
    
    def __getitem__(self, categories: Tuple[str, ...]) -> LayerwiseDistributions:
        """Dictionary-style access to distributions."""
        try:
            return self.distributions[categories]
        except KeyError:
            raise KeyError(f"No distribution found for categories: {categories}")

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional
from tqdm.auto import tqdm
import gc
import scipy.stats as st
from transformer_lens import HookedTransformer
from activation_distribution import LayerwiseDistributions, ActivationDistribution, InterpolatedKDE
from categorized_distributions import CategorizedDistributions
import ast
import logging
import json
import random

def setup_logging():
    """Configure logging to work with tqdm"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[TqdmLoggingHandler()]
    )
    return logging.getLogger()

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def load_from_text(
    model: HookedTransformer,
    hook_names: List[str],
    sentences: Dict = None,
    sentences_file: str = None,
    batch_size: int = 20,
    device: str = "cuda",
    drop_last_word: bool = True,
    kernel_width_scalar: float = 1.0) -> CategorizedDistributions:
    """
    Load activation distributions from raw text data into CategorizedDistributions.
    Category structure is inferred from the input data.
    
    Args:
        model: HookedTransformer model
        hook_names: List of hook names to collect activations from
        sentences: Nested dictionary with categories as keys
        sentences_file: Alternative to sentences, path to JSON file
        batch_size: Batch size for processing
        device: Device to use for computation
        drop_last_word: Whether to drop the last word of each sentence
    
    Returns:
        CategorizedDistributions object
    """
    if sentences is not None and sentences_file is not None:
        raise ValueError("Can only load sentences from file or take in sentences dict, not both")
    
    logger = setup_logging()
    n_layers = len(hook_names)
    
    # Validate hook names match model architecture
    if n_layers != model.cfg.n_layers:
        raise ValueError(f"Number of hooks ({n_layers}) doesn't match model layers ({model.cfg.n_layers})")
    
    # Load sentences if needed
    if sentences_file is not None:
        with open(sentences_file, 'r') as sf:
            sentences = json.load(sf)

        # Create CategorizedDistributions with discovered category names
    cat_dist = CategorizedDistributions()
    
    # Process sentences for each category combination
    def process_nested_dict(d, current_categories=()):
        if isinstance(d, list):
            # We've reached the sentences
            if drop_last_word:
                sentences_to_process = [s.rsplit(' ', 1)[0] for s in d]
            else:
                sentences_to_process = d
                
            # Initialize storage for each hook
            hook_activations = {hook_name: [] for hook_name in hook_names}
            
            # Process sentences in batches
            for i in tqdm(range(0, len(sentences_to_process), batch_size),
                         desc=f"Processing {current_categories}", leave=False):
                batch_sentences = sentences_to_process[i:i + batch_size]
                seq_lengths = [len(model.to_str_tokens(s)) for s in batch_sentences]
                
                with torch.inference_mode():
                    _, batch_cache = model.run_with_cache(
                        batch_sentences,
                        names_filter=hook_names
                    )
                    
                    # Collect activations for each hook
                    for hook_name in hook_names:
                        batch_activations = torch.stack([
                            batch_cache[hook_name][b, length-1, :]
                            for b, length in enumerate(seq_lengths)
                        ])
                        hook_activations[hook_name].append(batch_activations)
                
                del batch_cache
                gc.collect()
                torch.cuda.empty_cache()
            
            # Create LayerwiseDistributions for this category combination
            layer_distributions = []
            for hook_name in hook_names:
                combined_activations = torch.cat(hook_activations[hook_name], dim=0)
                layer_distributions.append(
                    ActivationDistribution(combined_activations.cpu().numpy())
                )
                del combined_activations
            
            # Add to CategorizedDistributions
            cat_dist.add_distribution(
                current_categories,
                LayerwiseDistributions(layer_distributions, kernel_width_scalar=kernel_width_scalar)
            )
            
            del hook_activations
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info(f"Completed processing categories: {current_categories}")
        else:
            # Continue recursing through the dictionary, adding each key to categories
            for key, value in d.items():
                process_nested_dict(value, current_categories + (key,))
    
    # Start processing from root
    process_nested_dict(sentences)
    
    return cat_dist


def load_from_csv(
    filepath: str,
    category_names: List[str] = None) -> CategorizedDistributions:
    """
    Load activation distributions from CSV file into CategorizedDistributions.
    
    Args:
        filepath: Path to CSV file
        category_names: Optional list of category names. If None, will detect from column names
                       using the format category_1, category_2, etc.
        
    Returns:
        CategorizedDistributions object
    """
    # Read CSV file
    df = pd.read_csv(filepath)
    
    # If category_names not provided, detect from columns
    if category_names is None:
        category_cols = [col for col in df.columns if col.startswith('category_')]
        if not category_cols:
            raise ValueError("No category columns found in CSV file")
        # Sort by number to ensure correct order
        category_names = sorted(category_cols, key=lambda x: int(x.split('_')[1]))
    
    # Validate required columns - now including mean and std
    required_columns = category_names + ['layer_num', 'x_kde', 'density', 'mean', 'std']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        if missing_columns == {'mean', 'std'}:
            # For backward compatibility: if only mean and std are missing, 
            # we can continue but won't support tail analysis
            print("Warning: mean and std columns not found. Tail analysis will not be available.")
            df['mean'] = None
            df['std'] = None
        else:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Initialize CategorizedDistributions
    cat_dist = CategorizedDistributions()
    
    # Convert string representations to arrays if needed
    if isinstance(df['x_kde'].iloc[0], str):
        df['x_kde'] = df['x_kde'].apply(ast.literal_eval)
    if isinstance(df['density'].iloc[0], str):
        df['density'] = df['density'].apply(ast.literal_eval)
    
    # Get unique category combinations
    categories_df = df[category_names].drop_duplicates()
    
    for _, category_row in tqdm(categories_df.iterrows(), 
                               desc="Processing categories",
                               total=len(categories_df)):
        # Get data for this category combination
        category_tuple = tuple(category_row[name] for name in category_names)
        category_mask = pd.Series(True, index=df.index)
        for name, value in zip(category_names, category_tuple):
            category_mask &= df[name] == value
        
        category_data = df[category_mask].sort_values('layer_num')
        
        # Verify all layers are present
        n_layers = len(category_data)
        layer_nums = set(category_data['layer_num'])
        expected_layers = set(range(n_layers))
        if layer_nums != expected_layers:
            raise ValueError(f"Missing layers for categories {category_tuple}. "
                           f"Expected {expected_layers}, got {layer_nums}")
        
        # Create distributions for each layer
        layer_distributions = []
        for _, row in category_data.iterrows():
            kde = InterpolatedKDE(None, precomputed_points=(row['x_kde'], row['density']))
            layer_distributions.append(
                ActivationDistribution(
                    kde=kde,
                    mean=row['mean'] if pd.notnull(row['mean']) else None,
                    std=row['std'] if pd.notnull(row['std']) else None
                )
            )
        
        cat_dist.add_distribution(category_tuple, LayerwiseDistributions(layer_distributions))
    
    return cat_dist


def split_json_data(
    data: Dict,
    test_split: float,
    seed: int
) -> Tuple[Dict, List[str], List[Tuple[str, ...]]]:
    """
    Split JSON data into train and test sets.
    Only splits sentences within each category combination,
    maintaining same categories in train and test.
    
    Args:
        data: Nested dictionary of categories and sentences
        test_split: Proportion of sentences to use for testing
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of:
        - Training data dictionary
        - List of test sentences
        - List of category tuples for test sentences
    """
    random.seed(seed)
    
    def get_category_tuples(d: Dict, current_tuple: Tuple = ()) -> List[Tuple[Tuple[str, ...], List[str]]]:
        """Recursively collect all category tuples and their sentences."""
        if isinstance(d, list):
            return [(current_tuple, d)]
        
        result = []
        for key, value in d.items():
            result.extend(get_category_tuples(value, current_tuple + (key,)))
        return result
    
    # Get all category tuples and their sentences
    category_data = get_category_tuples(data)
    
    # Split sentences for each category tuple
    train_data = {}
    test_pairs = []
    
    for category_tuple, sentences in category_data:
        # Shuffle sentences for this category combination
        shuffled_sentences = sentences.copy()
        random.shuffle(shuffled_sentences)
        
        # Split into train and test
        split_idx = int(len(shuffled_sentences) * (1 - test_split))
        train_sentences = shuffled_sentences[:split_idx]
        test_sentences = shuffled_sentences[split_idx:]
        
        # Add test sentences to test pairs
        test_pairs.extend([(category_tuple, sent) for sent in test_sentences])
        
        # Add train sentences to nested dictionary
        current_dict = train_data
        for cat in category_tuple[:-1]:
            if cat not in current_dict:
                current_dict[cat] = {}
            current_dict = current_dict[cat]
        
        if category_tuple[-1] not in current_dict:
            current_dict[category_tuple[-1]] = []
        current_dict[category_tuple[-1]].extend(train_sentences)
    
    # Extract test sentences and categories
    test_categories, test_sentences = zip(*test_pairs) if test_pairs else ([], [])
    
    return train_data, list(test_sentences), list(test_categories)

#!/usr/bin/env python

import argparse
import json
import os
from pathlib import Path
import torch
import gc
from transformer_lens import HookedTransformer
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional

from load_data import load_from_text, split_json_data, load_from_csv
from categorized_distributions import CategorizedDistributions

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate activation distribution classifiers')
    
    # Input/Output
    parser.add_argument('--data-files', nargs='+', required=False,
                       help='One or more JSON files containing categorized sentences')
    parser.add_argument('--kde-points-dir', type=str,
                       help='Directory containing saved KDE points (alternative to data-files)')
    parser.add_argument('--test-data-file', type=str,
                       help='JSON file containing test sentences (optional)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results')
    
    # Model parameters
    parser.add_argument('--models', nargs='+', 
                       default=['pythia-410m', 'gpt2-medium'],
                       help='List of models to evaluate')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='Batch size for processing')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    parser.add_argument('--drop-last-word', type=bool, default=True,
                       help='Whether to drop the last word of each sentence')
    parser.add_argument('--kernel-width-scalar', type=float, default=1.0,
                       help='Scalar for KDE bandwidth')
    parser.add_argument('--std-threshold', type=float,
                       help='Only consider points beyond this many standard deviations from mean')
    parser.add_argument('--detect-tails', action='store_true',
                       help='Whether to detect distribution tails')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for train/test split')
    
    args = parser.parse_args()
    
    if not args.data_files and not args.kde_points_dir:
        parser.error("Either --data-files or --kde-points-dir must be provided")
    if args.data_files and args.kde_points_dir:
        parser.error("Cannot provide both --data-files and --kde-points-dir")
        
    return args

def plot_matrix(matrix: np.ndarray, labels: list, title: str, output_path: str):
    """Helper function to plot matrices with proper formatting."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_kde_points(cat_dist: CategorizedDistributions, model_dir: Path):
    """Save KDE interpolation points and statistics for each category and layer."""
    kde_data = []
    
    for categories in cat_dist.keys():
        distributions = cat_dist[categories]
        
        for layer_idx in range(len(distributions)):
            dist = distributions[layer_idx]
            
            row_data = {
                'layer_num': layer_idx,
                'x_kde': dist.kde.x_points.tolist(),
                'density': dist.kde.y_points.tolist(),
                'mean': dist.mean,
                'std': dist.std
            }
            
            for i, cat in enumerate(categories, 1):
                row_data[f'category_{i}'] = cat
                
            kde_data.append(row_data)
    
    df = pd.DataFrame(kde_data)
    df.to_csv(model_dir / "kde_points.csv", index=False)


def load_test_data_from_split(test_data_file: str) -> tuple[list[str], list[tuple[str, ...]]]:
    """
    Load test sentences and their categories from a JSON file.
    Expected format:
    {
        "test_sentences": ["sentence 1", ..., "sentence N"]
        "test_categories": [["cat_11", "cat_12", ...], ...,  ["cat_N1", "cat_N2", ...]]
    }

    Can only have one category per test sentence, as long as that matches the training data
    """
    with open(test_data_file, 'r') as f:
        data = json.load(f)
    
    test_sentences = []
    test_categories = []
    
    test_sentences = data['test_sentences']
    test_categories = data['test_categories']
    
    return test_sentences, test_categories


def save_analysis_results(results_df: pd.DataFrame, stats: dict, model_dir: Path):
    """Save analysis results to files in easily loadable formats."""
    
    # Save raw test results DataFrame (now includes per-layer likelihoods)
    results_df.to_csv(model_dir / "test_results.csv", index=False)
    
    # Extract and save per-layer results
    layer_cols = [col for col in results_df.columns if col.startswith('ll_layers_')]
    if layer_cols:
        layer_results = results_df[['sentence'] + layer_cols].copy()
        layer_results.to_csv(model_dir / "per_layer_likelihoods.csv", index=False)
    
    # Save full tuple statistics
    full_tuple_stats = stats['full_tuple']
    
    # Save confusion matrix data
    conf_matrix_df = pd.DataFrame(
        full_tuple_stats['confusion_matrix']['matrix'],
        index=full_tuple_stats['confusion_matrix']['labels'],
        columns=full_tuple_stats['confusion_matrix']['labels']
    )
    conf_matrix_df.to_csv(model_dir / "full_tuple_confusion_matrix.csv")
    
    # Save precision matrix
    precision_df = pd.DataFrame(
        full_tuple_stats['precision_matrix']['matrix'],
        index=full_tuple_stats['precision_matrix']['labels'],
        columns=full_tuple_stats['precision_matrix']['labels']
    )
    precision_df.to_csv(model_dir / "full_tuple_precision_matrix.csv")
    
    # Save recall matrix
    recall_df = pd.DataFrame(
        full_tuple_stats['recall_matrix']['matrix'],
        index=full_tuple_stats['recall_matrix']['labels'],
        columns=full_tuple_stats['recall_matrix']['labels']
    )
    recall_df.to_csv(model_dir / "full_tuple_recall_matrix.csv")
    
    # Save F1 matrix
    f1_df = pd.DataFrame(
        full_tuple_stats['f1_matrix']['matrix'],
        index=full_tuple_stats['f1_matrix']['labels'],
        columns=full_tuple_stats['f1_matrix']['labels']
    )
    f1_df.to_csv(model_dir / "full_tuple_f1_matrix.csv")
    
    # Save top-k accuracies
    top_k_df = pd.DataFrame.from_dict(
        full_tuple_stats['top_k_accuracy'], 
        orient='index',
        columns=['accuracy']
    )
    top_k_df.to_csv(model_dir / "full_tuple_top_k_accuracy.csv")
    
    # Save per-dimension statistics
    for dim, dim_stats in stats['dimensions'].items():
        dim_dir = model_dir / dim
        dim_dir.mkdir(exist_ok=True)
        
        # Save confusion matrix
        dim_conf_df = pd.DataFrame(
            dim_stats['confusion_matrix']['matrix'],
            index=dim_stats['confusion_matrix']['labels'],
            columns=dim_stats['confusion_matrix']['labels']
        )
        dim_conf_df.to_csv(dim_dir / "confusion_matrix.csv")
        
        # Save precision matrix
        dim_prec_df = pd.DataFrame(
            dim_stats['precision_matrix']['matrix'],
            index=dim_stats['precision_matrix']['labels'],
            columns=dim_stats['precision_matrix']['labels']
        )
        dim_prec_df.to_csv(dim_dir / "precision_matrix.csv")
        
        # Save recall matrix
        dim_recall_df = pd.DataFrame(
            dim_stats['recall_matrix']['matrix'],
            index=dim_stats['recall_matrix']['labels'],
            columns=dim_stats['recall_matrix']['labels']
        )
        dim_recall_df.to_csv(dim_dir / "recall_matrix.csv")
        
        # Save F1 matrix
        dim_f1_df = pd.DataFrame(
            dim_stats['f1_matrix']['matrix'],
            index=dim_stats['f1_matrix']['labels'],
            columns=dim_stats['f1_matrix']['labels']
        )
        dim_f1_df.to_csv(dim_dir / "f1_matrix.csv")
        
        # Save top-k accuracies
        dim_topk_df = pd.DataFrame.from_dict(
            dim_stats['top_k_accuracy'],
            orient='index',
            columns=['accuracy']
        )
        dim_topk_df.to_csv(dim_dir / "top_k_accuracy.csv")
    
    # Save all statistics as a single JSON for convenience
    # First convert numpy arrays and other non-serializable types to JSON-safe formats
    def convert_to_json_safe(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_safe(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)  # Convert tuples to lists
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

    # Convert stats dictionary to JSON-safe format
    json_safe_stats = convert_to_json_safe(stats)
    
    # Save to JSON file
    with open(model_dir / "all_statistics.json", 'w') as f:
        json.dump(json_safe_stats, f, indent=2)



def process_single_model(
    model_name: str,
    train_data: dict,
    test_sentences: list,
    test_categories: list,
    output_dir: Path,
    batch_size: int = 20,
    drop_last_word: bool = True,
    kernel_width_scalar: float = 1.0,
    std_threshold: Optional[float] = None,
    detect_tails: bool = False):
    """Process a single model through the full pipeline."""
    
    # Create model-specific output directory
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nProcessing model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device="cuda")
    hook_names = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
    
    try:
        # Create distributions
        print("Creating activation distributions...")
        cat_dist = load_from_text(
            model=model,
            hook_names=hook_names,
            sentences=train_data,
            batch_size=batch_size,
            drop_last_word=drop_last_word,
            kernel_width_scalar=kernel_width_scalar
        )
        
        save_kde_points(cat_dist, model_dir)
        if detect_tails:
            cat_dist.set_tail_cutoffs()

        # Save basic information
        with open(model_dir / "info.json", "w") as f:
            info = {
                "model_name": model_name,
                "n_layers": len(hook_names),
                "n_train_categories": len(cat_dist.keys()),
                "n_test_sentences": len(test_sentences),
                "batch_size": batch_size,
                "kernel_width_scalar": kernel_width_scalar,
                "drop_last_word": drop_last_word,
                "timestamp": datetime.now().isoformat()
            }
            json.dump(info, f, indent=4)
        
        # Run analysis using built-in methods
        print("Running analysis...")
        results_df, stats = cat_dist.compute_test_likelihoods(
                model=model,
                test_sentences=test_sentences,
                true_categories=test_categories,
                hook_names=hook_names,
                batch_size=batch_size,
                drop_last_word=drop_last_word,
                std_threshold=std_threshold,
                detect_tails=detect_tails)
        
        # Save results and create visualizations
        save_analysis_results(results_df, stats, model_dir)
        
    finally:
        # Clean up GPU memory
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()



def process_single_model_from_kde(
    model_name: str,
    kde_points_dir: Path,
    test_sentences: list,
    test_categories: list,
    output_dir: Path,
    batch_size: int = 20,
    drop_last_word: bool = True,
    std_threshold: Optional[float] = None,
    detect_tails: bool = False):
    """Process a single model using saved KDE points."""
    
    # Create model-specific output directory
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nProcessing model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device="cuda")
    hook_names = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
    
    try:
        # Load distributions from saved KDE points
        kde_points_file = Path(kde_points_dir) / model_name / "kde_points.csv"
        if not kde_points_file.exists():
            raise FileNotFoundError(f"No KDE points file found for model {model_name}")
            
        print("Loading KDE points...")
        cat_dist = load_from_csv(kde_points_file)
        if detect_tails:
            cat_dist.set_tail_cutoffs()
        
        # Run analysis using built-in methods
        print("Running analysis...")
        results_df, stats = cat_dist.compute_test_likelihoods(
            model=model,
            test_sentences=test_sentences,
            true_categories=test_categories,
            hook_names=hook_names,
            batch_size=batch_size,
            drop_last_word=drop_last_word,
            std_threshold=std_threshold,
            detect_tails=detect_tails)
        
        # Save results
        save_analysis_results(results_df, stats, model_dir)
        
    finally:
        # Clean up GPU memory
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        

def get_next_index(base_dir: Path, date_prefix: str) -> int:
    """
    Find the next available index for today's date.
    
    Args:
        base_dir: Base directory path
        date_prefix: Today's date in YYYYMMDD format
        
    Returns:
        Next available index number
    """
    # Find all directories matching today's date prefix
    existing_dirs = [d for d in base_dir.glob(f"{date_prefix}_*") if d.is_dir()]
    
    if not existing_dirs:
        return 1
        
    # Extract indices from directory names
    indices = []
    for dir_path in existing_dirs:
        try:
            index = int(dir_path.name.split('_')[1])
            indices.append(index)
        except (IndexError, ValueError):
            continue
            
    return max(indices, default=0) + 1

def main():
    args = parse_args()
    
    # Create base output directory if it doesn't exist
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get today's date prefix and next index
    date_prefix = datetime.now().strftime("%Y%m%d")
    next_index = get_next_index(base_output_dir, date_prefix)
    
    # Create output directory with date and index
    output_dir = base_output_dir / f"{date_prefix}_{next_index}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save command line arguments
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Load test data if provided
    if args.test_data_file:
        print("\nLoading test data...")
        test_sentences, test_categories = load_test_data_from_split(args.test_data_file)
    
    if args.kde_points_dir:
        # If no test data file provided, load from original split
        if not args.test_data_file:
            split_file = Path(args.kde_points_dir) / "data_split.json"
            if not split_file.exists():
                raise FileNotFoundError(f"No data split file found in {args.kde_points_dir}")
                
            with open(split_file, 'r') as f:
                split_data = json.load(f)
                test_sentences = split_data['test_sentences']
                test_categories = split_data['test_categories']
            
        # Process each model using saved KDE points
        for model_name in args.models:
            process_single_model_from_kde(
                model_name=model_name,
                kde_points_dir=args.kde_points_dir,
                test_sentences=test_sentences,
                test_categories=test_categories,
                output_dir=output_dir,
                batch_size=args.batch_size,
                drop_last_word=args.drop_last_word,
                std_threshold=args.std_threshold,
                detect_tails=args.detect_tails
            )
    else:
        # Load and combine all data files
        all_data = {}
        for file_path in args.data_files:
            with open(file_path, 'r') as f:
                file_data = json.load(f)
                # Merge dictionaries, assuming compatible structures
                for key1 in file_data:
                    if key1 not in all_data:
                        all_data[key1] = {}
                    for key2 in file_data[key1]:
                        if key2 not in all_data[key1]:
                            all_data[key1][key2] = []
                        all_data[key1][key2].extend(file_data[key1][key2])
        
        if args.test_data_file:
            # Use all data for training if separate test set provided
            train_data = all_data
        else:
            # Split into train/test
            print("\nSplitting data...")
            train_data, test_sentences, test_categories = split_json_data(
                all_data,
                args.test_split,
                args.seed
            )
        
        # Save train/test split
        with open(output_dir / "data_split.json", "w") as f:
            json.dump({
                "train_data": train_data,
                "test_sentences": test_sentences,
                "test_categories": test_categories
            }, f, indent=4)
        
        # Process each model from raw data
        for model_name in args.models:
            process_single_model(
                model_name=model_name,
                train_data=train_data,
                test_sentences=test_sentences,
                test_categories=test_categories,
                output_dir=output_dir,
                batch_size=args.batch_size,
                drop_last_word=args.drop_last_word,
                kernel_width_scalar=args.kernel_width_scalar,
                std_threshold=args.std_threshold,
                detect_tails=args.detect_tails
            )
    
    print(f"\nAll results saved to {output_dir}")

if __name__ == "__main__":
    main()

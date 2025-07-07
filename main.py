#!/usr/bin/env python3
"""
Smart Contract Vulnerability Detection using Wide + TabTransformer Neural Network

This script implements a hybrid neural network architecture combining:
- Wide component: Shallow processing for linear patterns
- TabTransformer component: Deep transformer for complex patterns

Usage:
    python main.py contracts_re.txt -vt re --lr 0.00015 --epochs 60 --batch_size 4
"""

import os
import sys
import time
import warnings
import pandas as pd
import numpy as np

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Local imports - FIXED
from config.enhanced_fragment_vectorizer import EnhancedFragmentVectorizer
from config.models.wide_tabtransformer import WideTabTransformer
from config.arg_parser import parameter_parser

# Configuration
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)

def print_header():
    """Print application header"""
    print("=" * 80)
    print("SMART CONTRACT VULNERABILITY DETECTION")
    print("Wide + TabTransformer Neural Network")
    print("Enhanced Vectorization System")
    print("=" * 80)

def print_parameters(args):
    """Print all parameters"""
    print("\nParameters:")
    print("-" * 40)
    for arg in vars(args):
        print(f"{arg:20}: {getattr(args, arg)}")
    print("-" * 40)

def parse_smart_contracts(filename):
    """
    Parse smart contract file and extract code fragments with labels
    
    Args:
        filename: Path to smart contract file
        
    Yields:
        tuple: (fragment_code, vulnerability_label)
    """
    print(f'Parsing smart contracts from: {filename}')
    
    with open(filename, "r", encoding="utf8") as file:
        fragment = []
        fragment_label = 0
        
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
                
            # Fragment separator
            if "-" * 40 in line and fragment:
                yield fragment, fragment_label
                fragment = []
            # Label line
            elif stripped.split()[0].isdigit():
                if fragment:
                    if stripped.isdigit():
                        fragment_label = int(stripped)
                    else:
                        fragment.append(stripped)
                else:
                    fragment.append(stripped)
            # Code line
            else:
                fragment.append(stripped)

def create_dataset(filename, args):
    """
    Create enhanced vectorized dataset from smart contract file
    
    Args:
        filename: Path to smart contract file
        args: Parsed arguments with vectorization parameters
        
    Returns:
        pd.DataFrame: Dataset with enhanced vectors and labels
    """
    print("\n" + "=" * 60)
    print("ENHANCED DATASET CREATION")
    print("=" * 60)
    
    # Collect fragments
    fragments = []
    vectorizer = EnhancedFragmentVectorizer(args.vec_length)
    
    print("Collecting code fragments with enhanced analysis...")
    start_time = time.time()
    
    for count, (fragment, label) in enumerate(parse_smart_contracts(filename), 1):
        print(f"Processing fragment {count:4d}", end="\r")
        vectorizer.add_fragment(fragment)
        fragments.append({"fragment": fragment, "label": label})
    
    collection_time = time.time() - start_time
    print(f"\nCollected {count} fragments in {collection_time:.2f}s")
    print(f"Forward slices: {vectorizer.forward_slices}")
    print(f"Backward slices: {vectorizer.backward_slices}")
    
    # Train enhanced Word2Vec model
    print("\nTraining enhanced Word2Vec model...")
    start_time = time.time()
    vectorizer.train_model()
    training_time = time.time() - start_time
    print(f"Enhanced Word2Vec training completed in {training_time:.2f}s")
    
    # Print vocabulary statistics if requested
    if args.vocab_stats:
        stats = vectorizer.get_vocabulary_stats()
        print(f"\nVocabulary Statistics:")
        print(f"- Total unique tokens: {stats['total_tokens']}")
        print(f"- Final vocabulary size: {stats['final_vocab_size']}")
        print(f"- Average fragment length: {stats['avg_fragment_length']:.2f}")
        print(f"- Average complexity score: {stats['avg_complexity']:.2f}")
        print(f"- Most common tokens: {[token for token, count in stats['most_common_tokens'][:10]]}")
    
    # Vectorize fragments with enhanced method
    print("\nVectorizing fragments with enhanced method...")
    start_time = time.time()
    
    dataset = []
    for i, fragment_data in enumerate(fragments):
        print(f"Vectorizing {i+1:4d}/{len(fragments)}", end="\r")
        vector = vectorizer.vectorize(fragment_data["fragment"])
        dataset.append({"vector": vector, "label": fragment_data["label"]})
    
    vectorization_time = time.time() - start_time
    print(f"\nEnhanced vectorization completed in {vectorization_time:.2f}s")
    
    # Create DataFrame
    df = pd.DataFrame(dataset)
    
    # Print enhanced statistics
    print("\n" + "=" * 60)
    print("ENHANCED DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Vulnerable samples (1): {sum(df['label'] == 1)}")
    print(f"Safe samples (0): {sum(df['label'] == 0)}")
    print(f"Vector shape: {df.iloc[0]['vector'].shape}")
    print(f"Vulnerability ratio: {sum(df['label'] == 1)/len(df)*100:.2f}%")
    print(f"Vector non-zero ratio: {np.count_nonzero(df.iloc[0]['vector']) / df.iloc[0]['vector'].size * 100:.2f}%")
    
    return df

def main():
    """Main execution function"""
    # Parse arguments
    args = parameter_parser()
    
    # Print header and parameters
    print_header()
    print_parameters(args)
    
    # Prepare dataset path
    base_name = os.path.splitext(os.path.basename(args.filename))[0]
    dataset_path = f"config/train_data/{base_name}_enhanced_vectors.pkl"
    
    # Create data directory
    os.makedirs("config/train_data", exist_ok=True)
    
    print(f"\nDataset path: {dataset_path}")
    
    # Load or create dataset
    if os.path.exists(dataset_path):
        print("Loading existing enhanced dataset...")
        dataset = pd.read_pickle(dataset_path)
        print("Enhanced dataset loaded successfully!")
    else:
        print("Creating new enhanced dataset...")
        dataset = create_dataset(args.filename, args)
        print(f"Saving enhanced dataset to {dataset_path}...")
        dataset.to_pickle(dataset_path)
        print("Enhanced dataset saved successfully!")
    
    # Model training and evaluation
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize model
    print("Initializing Wide + TabTransformer model...")
    model = WideTabTransformer(dataset, args)
    model.get_model_summary()
    
    # Train model
    history = model.train()
    
    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time:.2f}s")
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    results = model.evaluate()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Architecture: Wide + TabTransformer")
    print(f"Vectorization: Enhanced with vulnerability patterns")
    print(f"Dataset: {args.filename}")
    print(f"Vulnerability Type: {args.vt}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Final Accuracy: {results['accuracy']:.4f}")
    print(f"Final F1-Score: {results['f1_score']:.4f}")
    print("=" * 60)

if __name__ == '__main__':
    # Ensure UTF-8 encoding
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
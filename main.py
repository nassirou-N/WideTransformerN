#!/usr/bin/env python3
"""
Smart Contract Vulnerability Detection using Wide + TabTransformer Neural Network

This script implements a hybrid neural network architecture combining:
- Wide component: Shallow processing for linear patterns
- TabTransformer component: Deep transformer for complex patterns

Usage:
    python main.py contracts_re.txt -vt re --lr 0.00012 --epochs 80 --batch_size 6 --num_transformer_layers 4 --num_heads 12 --embedding_dim 96 --dropout 0.25
"""

import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from IPython.display import display, Image

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

# Configure matplotlib for Colab
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
plt.ioff()

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
        print(f"{arg:25}: {getattr(args, arg)}")
    print("-" * 40)

def parse_smart_contracts(filename):
    """
    Parse smart contract file and extract code fragments with labels
    Version corrigée qui ignore les identifiants de fichiers
    
    Args:
        filename: Path to smart contract file
        
    Yields:
        tuple: (fragment_code, vulnerability_label)
    """
    print(f'Parsing smart contracts from: {filename}')
    
    # Pattern pour détecter les identifiants de fichiers (ex: "00 50020.sol")
    file_pattern = re.compile(r'^\d+\s+\d+\.sol$')
    
    with open(filename, "r", encoding="utf8") as file:
        fragment = []
        fragment_label = 0
        
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            
            # Ignorer les identifiants de fichiers
            if file_pattern.match(stripped):
                continue
                
            # Fragment separator
            if "-" * 40 in line and fragment:
                yield fragment, fragment_label
                fragment = []
            # Label line (plus strict : seulement '0' ou '1')
            elif stripped in ['0', '1']:
                fragment_label = int(stripped)
            # Code line
            else:
                fragment.append(stripped)

def debug_parse_smart_contracts(filename, max_fragments=3):
    """
    Version debug pour vérifier le parsing
    
    Args:
        filename: Path to smart contract file
        max_fragments: Nombre maximum de fragments à afficher
    """
    print(f"\n{'='*60}")
    print("DEBUG: ANALYSE DU PARSING")
    print(f"{'='*60}")
    
    fragments_analyzed = 0
    total_fragments = 0
    vulnerable_count = 0
    safe_count = 0
    
    for i, (fragment, label) in enumerate(parse_smart_contracts(filename)):
        total_fragments += 1
        if label == 1:
            vulnerable_count += 1
        else:
            safe_count += 1
            
        if fragments_analyzed < max_fragments:
            print(f"\n--- Fragment {i+1} (Label: {label}) ---")
            print(f"Nombre de lignes: {len(fragment)}")
            
            # Afficher les premières lignes
            for j, line in enumerate(fragment[:6]):
                print(f"{j+1:2d}: {line}")
            
            if len(fragment) > 6:
                print(f"    ... ({len(fragment) - 6} lignes supplémentaires)")
            
            # Vérifier s'il y a encore des identifiants de fichiers
            polluted_lines = [line for line in fragment if line.endswith('.sol')]
            if polluted_lines:
                print(f"⚠️  ATTENTION: Lignes polluées détectées: {polluted_lines}")
            else:
                print("✅ Fragment propre (pas d'identifiants de fichiers)")
                
            fragments_analyzed += 1
    
    print(f"\n{'='*60}")
    print(f"STATISTIQUES DU DATASET:")
    print(f"- Total des fragments: {total_fragments}")
    print(f"- Contrats vulnérables (1): {vulnerable_count}")
    print(f"- Contrats sûrs (0): {safe_count}")
    print(f"- Ratio de vulnérabilité: {vulnerable_count/total_fragments*100:.2f}%")
    print(f"{'='*60}")

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


def plot_training_curves(history, save_path="training_curves.png", show_in_colab=True):
    """Plot training curves for accuracy and loss - Colab compatible"""
    plt.figure(figsize=(15, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, marker='o', markersize=4)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
    plt.title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, marker='s', markersize=4)
    plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Wide + TabTransformer Training Progress', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to: {save_path}")
    
    # Display in Colab
    if show_in_colab and 'google.colab' in sys.modules:
        plt.show()
    else:
        # For non-Colab environments or if show_in_colab is False
        try:
            display(Image(filename=save_path))
        except:
            print(f"Plot saved but cannot display. Check: {save_path}")
    
    plt.close()
    
    # Print final values
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\nFinal Training Metrics:")
    print(f"- Training Accuracy: {final_train_acc:.4f}")
    print(f"- Validation Accuracy: {final_val_acc:.4f}")
    print(f"- Training Loss: {final_train_loss:.4f}")
    print(f"- Validation Loss: {final_val_loss:.4f}")
    
    # Check for overfitting
    acc_gap = final_train_acc - final_val_acc
    if acc_gap > 0.1:
        print(f"\n⚠️  Warning: Potential overfitting detected (accuracy gap: {acc_gap:.4f})")
    else:
        print(f"\n✅ Model generalization looks good (accuracy gap: {acc_gap:.4f})")

def plot_metrics_comparison(results, save_path="metrics_comparison.png", show_in_colab=True):
    """Plot a comparison of different evaluation metrics - Colab compatible"""
    plt.figure(figsize=(10, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        results['accuracy'],
        results['precision'],
        results['recall'],
        results['f1_score']
    ]
    
    # Create bar plot with custom colors
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = plt.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylim(0, 1.1)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal lines for reference
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect score')
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics comparison saved to: {save_path}")
    
    # Display in Colab
    if show_in_colab and 'google.colab' in sys.modules:
        plt.show()
    else:
        try:
            display(Image(filename=save_path))
        except:
            print(f"Plot saved but cannot display. Check: {save_path}")
    
    plt.close()

def plot_confusion_matrix_style(results, save_path="confusion_matrix_analysis.png", show_in_colab=True):
    """Create a visual representation of the confusion matrix results"""
    plt.figure(figsize=(8, 6))
    
    # Create data for the plot
    fp_rate = results['fp_rate']
    fn_rate = results['fn_rate']
    tp_rate = results['recall']  # True Positive Rate = Recall
    tn_rate = 1 - fp_rate  # True Negative Rate = 1 - False Positive Rate
    
    categories = ['True Positive\nRate', 'True Negative\nRate', 'False Positive\nRate', 'False Negative\nRate']
    rates = [tp_rate, tn_rate, fp_rate, fn_rate]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    bars = plt.bar(categories, rates, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylim(0, 1.1)
    plt.ylabel('Rate', fontsize=12)
    plt.title('Classification Rates Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix analysis saved to: {save_path}")
    
    if show_in_colab and 'google.colab' in sys.modules:
        plt.show()
    else:
        try:
            display(Image(filename=save_path))
        except:
            print(f"Plot saved but cannot display. Check: {save_path}")
    
    plt.close()

def main():
    """Main execution function"""
    IN_COLAB = 'google.colab' in sys.modules
    
    if IN_COLAB:
        print("🔵 Running in Google Colab environment")
        # Ensure matplotlib works properly in Colab
        import matplotlib
        matplotlib.use('module://ipykernel.pylab.backend_inline')
    
    # Parse arguments
    args = parameter_parser()
    
    # Print header and parameters
    print_header()
    print_parameters(args)
    
    # 🆕 DEBUG DU PARSING
    print(f"\n{'='*60}")
    print("VERIFICATION DU PARSING (MODE DEBUG)")
    print(f"{'='*60}")
    
    # Test du parsing avec debug
    debug_parse_smart_contracts(args.filename, max_fragments=3)
    
    # Demander confirmation avant de continuer (en mode interactif seulement)
    if sys.stdin.isatty():  # Vérifie si on est en mode interactif
        user_input = input("\nLe parsing semble-t-il correct ? (y/n) [y]: ").lower().strip()
        if user_input and user_input != 'y':
            print("Parsing interrompu. Vérifiez les données d'entrée.")
            return
    else:
        print("\nMode non-interactif détecté, continuation automatique...")
    
    # Prepare dataset path
    base_name = os.path.splitext(os.path.basename(args.filename))[0]
    dataset_path = f"config/train_data/{base_name}_enhanced_vectors.pkl"
    
    # Create data directory
    os.makedirs("config/train_data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
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
    
    # Plot training curves
    print("\n" + "=" * 60)
    print("GENERATING TRAINING CURVES")
    print("=" * 60)
    
    plot_training_curves(
        history, 
        save_path=f"plots/{base_name}_training_curves.png",
        show_in_colab=IN_COLAB
    )

    # Evaluate model
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    results = model.evaluate()

    # Plot metrics comparison
    print("\n" + "=" * 60)
    print("GENERATING METRICS COMPARISON")
    print("=" * 60)
    
    plot_metrics_comparison(
        results,
        save_path=f"plots/{base_name}_metrics_comparison.png",
        show_in_colab=IN_COLAB
    )
    
    # Plot confusion matrix analysis
    print("\n" + "=" * 60)
    print("GENERATING CONFUSION MATRIX ANALYSIS")
    print("=" * 60)
    
    plot_confusion_matrix_style(
        results,
        save_path=f"plots/{base_name}_confusion_matrix_analysis.png",
        show_in_colab=IN_COLAB
    )
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Architecture: Wide + TabTransformer")
    print(f"Vectorization: Enhanced with vulnerability patterns")
    print(f"Dataset: {args.filename}")
    print(f"Vulnerability Type: {args.vt}")
    print(f"Training Parameters:")
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Transformer Layers: {args.num_transformer_layers}")
    print(f"  - Attention Heads: {args.num_heads}")
    print(f"  - Embedding Dim: {args.embedding_dim}")
    print(f"  - Dropout: {args.dropout}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Final Accuracy: {results['accuracy']:.4f}")
    print(f"Final F1-Score: {results['f1_score']:.4f}")
    print(f"\nPlots saved in: plots/")
    
    if IN_COLAB:
        print("\n📊 All plots have been displayed inline in Colab")
    
    print("=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 60)

if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass  

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
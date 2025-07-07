import argparse

def parameter_parser():
    """
    Enhanced parameter parser for Smart Contract Vulnerability Detection 
    using Wide + TabTransformer with improved vectorization
    """
    parser = argparse.ArgumentParser(
        description='Smart Contract Vulnerability Detection Using Wide + TabTransformer Neural Network'
    )

    # Required arguments
    parser.add_argument('filename', type=str, 
                       help="Path to smart contract file to process")
    parser.add_argument('-vt', type=str, choices=['ts', 're', 'io'], 
                       help="Vulnerability type: ts(timestamp), re(reentrancy), io(integer overflow)")
    
    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--dropout', type=float, default=0.2, 
                       help='Dropout rate (default: 0.2)')
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=4, 
                       help='Batch size (default: 4)')
    
    # Enhanced vectorization parameters
    parser.add_argument('--vec_length', type=int, default=150, 
                       help='Word2Vec vector dimension (default: 150)')
    parser.add_argument('--w2v_window', type=int, default=8,
                       help='Word2Vec context window size (default: 8)')
    parser.add_argument('--w2v_min_count', type=int, default=2,
                       help='Word2Vec minimum token frequency (default: 2)')
    parser.add_argument('--w2v_epochs', type=int, default=20,
                       help='Word2Vec training epochs (default: 20)')
    parser.add_argument('--w2v_negative', type=int, default=10,
                       help='Word2Vec negative sampling (default: 10)')
    
    # Model architecture parameters
    parser.add_argument('--wide_features', type=int, default=50, 
                       help='Number of features for wide component (default: 50)')
    
    # TabTransformer architecture parameters
    parser.add_argument('--num_transformer_layers', type=int, default=3, 
                       help='Number of transformer layers (default: 3)')
    parser.add_argument('--num_heads', type=int, default=8, 
                       help='Number of attention heads (default: 8)')
    parser.add_argument('--embedding_dim', type=int, default=64, 
                       help='Transformer embedding dimension (default: 64)')
    parser.add_argument('--mlp_hidden_dim', type=int, default=128, 
                       help='MLP hidden dimension in transformer (default: 128)')
    
    # Advanced options
    parser.add_argument('--use_enhanced_vectorization', action='store_true', default=True,
                       help='Use enhanced vectorization (default: True)')
    parser.add_argument('--complexity_weighting', action='store_true', default=False,
                       help='Use complexity-based sample weighting (default: False)')
    parser.add_argument('--vocab_stats', action='store_true', default=False,
                       help='Print detailed vocabulary statistics (default: False)')

    return parser.parse_args()
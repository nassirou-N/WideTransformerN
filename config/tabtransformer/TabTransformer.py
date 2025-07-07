import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention

class TabTransformer(Layer):
    """
    TabTransformer layer for processing tabular data with transformer architecture.
    Designed specifically for smart contract vulnerability detection.
    """
    
    def __init__(self, 
                 num_transformer_layers=3,
                 num_heads=8,
                 embedding_dim=64,
                 mlp_hidden_dim=128,
                 dropout_rate=0.2,
                 **kwargs):
        super(TabTransformer, self).__init__(**kwargs)
        
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout_rate = dropout_rate
        
        # Input feature embedding
        self.feature_embedding = Dense(embedding_dim, activation='relu')
        
        # Initialize transformer components
        self._build_transformer_layers()
    
    def _build_transformer_layers(self):
        """Initialize transformer layers and components"""
        self.attention_layers = []
        self.layer_norms_1 = []
        self.layer_norms_2 = []
        self.mlp_layers = []
        self.dropout_layers = []
        
        for i in range(self.num_transformer_layers):
            # Multi-head attention
            self.attention_layers.append(
                MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.embedding_dim // self.num_heads,
                    dropout=self.dropout_rate
                )
            )
            
            # Layer normalizations
            self.layer_norms_1.append(LayerNormalization())
            self.layer_norms_2.append(LayerNormalization())
            
            # MLP (Feed Forward Network)
            mlp = tf.keras.Sequential([
                Dense(self.mlp_hidden_dim, activation='relu'),
                Dropout(self.dropout_rate),
                Dense(self.embedding_dim)
            ])
            self.mlp_layers.append(mlp)
            
            # Dropout layers
            self.dropout_layers.append(Dropout(self.dropout_rate))

    def call(self, inputs, training=None, **kwargs):
        """Forward pass through TabTransformer"""
        # Project input features to embedding dimension
        x = self.feature_embedding(inputs)
        
        # Apply transformer layers
        for i in range(self.num_transformer_layers):
            # Multi-head attention with residual connection
            attention_output = self.attention_layers[i](x, x, training=training)
            attention_output = self.dropout_layers[i](attention_output, training=training)
            x = self.layer_norms_1[i](x + attention_output)
            
            # MLP with residual connection
            mlp_output = self.mlp_layers[i](x, training=training)
            x = self.layer_norms_2[i](x + mlp_output)
        
        return x

    def get_config(self):
        config = super(TabTransformer, self).get_config()
        config.update({
            "num_transformer_layers": self.num_transformer_layers,
            "num_heads": self.num_heads,
            "embedding_dim": self.embedding_dim,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config
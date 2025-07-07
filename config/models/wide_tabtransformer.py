import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import warnings
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Normalization, Concatenate, Flatten, Dropout, Dense, Input, GlobalAveragePooling1D
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split

from config.tabtransformer.TabTransformer import TabTransformer

# Suppress warnings and set seeds
warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

class WideTabTransformer:
    """
    Wide + TabTransformer model for smart contract vulnerability detection.
    
    Architecture:
    - Wide component: Shallow processing of first N features
    - TabTransformer component: Deep transformer processing of remaining features
    - Fusion: Concatenation and final classification
    """
    
    def __init__(self, data, args):
        self.args = args
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        
        # Prepare data
        self._prepare_data(data)
        
        # Build model
        self.model = self._build_model()
        
        print(f"Model initialized with {len(self.x_train_wide)} training samples")
        print(f"Wide features: {self.x_train_wide.shape[1:]}")
        print(f"TabTransformer features: {self.x_train_transformer.shape[1:]}")
    
    def _prepare_data(self, data):
        """Prepare and split data for training"""
        self.vectors = np.stack(data.iloc[:, 0].values)
        self.labels = data.iloc[:, 1].values
        
        # Get balanced indices
        positive_idxs = np.where(self.labels == 1)[0]
        negative_idxs = np.where(self.labels == 0)[0]
        idxs = np.concatenate([positive_idxs, negative_idxs])
        
        # Train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            self.vectors[idxs], self.labels[idxs],
            test_size=0.2, stratify=self.labels[idxs], random_state=42
        )
        
        # Split features for Wide and TabTransformer components
        wide_features = self.args.wide_features
        self.x_train_wide = x_train[:, :wide_features]
        self.x_train_transformer = x_train[:, wide_features:]
        self.x_test_wide = x_test[:, :wide_features]
        self.x_test_transformer = x_test[:, wide_features:]
        
        # Convert labels to categorical
        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)
        
        # Calculate class weights for balanced training
        classes = np.array([0, 1])
        class_weights = compute_class_weight(
            class_weight='balanced', classes=classes, y=self.labels
        )
        self.class_weight = {index: weight for index, weight in enumerate(class_weights)}
        
        print(f"Class distribution - Safe: {sum(self.labels == 0)}, Vulnerable: {sum(self.labels == 1)}")
        print(f"Class weights: {self.class_weight}")
    
    def _build_model(self):
        """Build Wide + TabTransformer model"""
        # Define inputs
        input_wide = Input(shape=self.x_train_wide.shape[1:], name='wide_input')
        input_transformer = Input(shape=self.x_train_transformer.shape[1:], name='transformer_input')
        
        # Wide component (shallow processing)
        wide = Normalization(name='wide_normalization')(input_wide)
        wide_flattened = Flatten(name='wide_flatten')(wide)
        
        # TabTransformer component (deep processing)
        transformer_input = Normalization(name='transformer_normalization')(input_transformer)
        
        # Apply TabTransformer
        tabtransformer = TabTransformer(
            num_transformer_layers=self.args.num_transformer_layers,
            num_heads=self.args.num_heads,
            embedding_dim=self.args.embedding_dim,
            mlp_hidden_dim=self.args.mlp_hidden_dim,
            dropout_rate=self.args.dropout
        )(transformer_input)
        
        # Global average pooling to aggregate sequence
        transformer_pooled = GlobalAveragePooling1D(name='transformer_pooling')(tabtransformer)
        
        # Additional processing for transformer output
        transformer_dense = Dense(128, activation='relu', name='transformer_dense')(transformer_pooled)
        transformer_dense = Dropout(self.args.dropout, name='transformer_dropout')(transformer_dense)
        
        # Fusion: Concatenate wide and transformer components
        merged = Concatenate(axis=-1, name='fusion')([wide_flattened, transformer_dense])
        
        # Final classification layers
        dense_1 = Dense(256, activation='relu', name='dense_1')(merged)
        dense_1 = Dropout(self.args.dropout, name='dropout_1')(dense_1)
        
        dense_2 = Dense(128, activation='relu', name='dense_2')(dense_1)
        dense_2 = Dropout(self.args.dropout, name='dropout_2')(dense_2)
        
        # Output layer
        output = Dense(2, activation='softmax', name='output')(dense_2)
        
        # Create and compile model
        model = Model(inputs=[input_wide, input_transformer], outputs=output, name='WideTabTransformer')
        
        optimizer = Adam(learning_rate=self.lr)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self):
        """Train the model"""
        print("\nStarting training...")
        print(f"Epochs: {self.epochs}, Batch size: {self.batch_size}, Learning rate: {self.lr}")
        
        history = self.model.fit(
            [self.x_train_wide, self.x_train_transformer], 
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=self.class_weight,
            validation_data=([self.x_test_wide, self.x_test_transformer], self.y_test),
            verbose=1
        )
        
        print("Training completed!")
        return history
    
    def evaluate(self):
        """Evaluate model performance"""
        print("\nEvaluating model...")
        
        # Get predictions
        predictions = self.model.predict(
            [self.x_test_wide, self.x_test_transformer], 
            batch_size=self.batch_size, 
            verbose=0
        )
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(self.y_test, axis=1)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(true_classes, predicted_classes).ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Print results
        print(f"\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        print(f"False Positive Rate: {fp_rate:.4f}")
        print(f"False Negative Rate: {fn_rate:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"TN: {tn}, FP: {fp}")
        print(f"FN: {fn}, TP: {tp}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'fp_rate': fp_rate,
            'fn_rate': fn_rate
        }
    
    def get_model_summary(self):
        """Print model architecture summary"""
        print("\nModel Architecture:")
        self.model.summary()
        
        print(f"\nTabTransformer Configuration:")
        print(f"- Transformer layers: {self.args.num_transformer_layers}")
        print(f"- Attention heads: {self.args.num_heads}")
        print(f"- Embedding dimension: {self.args.embedding_dim}")
        print(f"- MLP hidden dimension: {self.args.mlp_hidden_dim}")
        print(f"- Wide features: {self.args.wide_features}")
        print(f"- Dropout rate: {self.args.dropout}")
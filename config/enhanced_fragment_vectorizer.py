import re
import warnings
import numpy as np
from gensim.models import Word2Vec
from collections import Counter, defaultdict
import hashlib

# Set print options to display the entire array
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")

# Enhanced operator sets with vulnerability-specific patterns
operators3 = {'<<=', '>>=', '**='}
operators2 = {
    '->', '++', '--', '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=', '-=', '*=', '/=', 
    '%=', '&=', '^=', '|=', '=>'
}
operators1 = {
    '(', ')', '[', ']', '.', '+', '-', '*', '&', '/',
    '%', '<', '>', '^', '|', '=', ',', '?', ':', ';',
    '{', '}', '!', '~'
}

# Vulnerability-specific keywords and patterns
VULNERABILITY_KEYWORDS = {
    'reentrancy': {
        'call', 'value', 'send', 'transfer', 'withdraw', 'balance',
        'msg.sender', 'this.balance', 'call.value', 'external'
    },
    'overflow': {
        'uint', 'int', 'SafeMath', 'add', 'sub', 'mul', 'div',
        'require', 'assert', 'overflow', 'underflow'
    },
    'timestamp': {
        'now', 'block.timestamp', 'block.number', 'block.difficulty',
        'block.coinbase', 'timestamp', 'time'
    }
}

CRITICAL_PATTERNS = {
    'dangerous_call': r'\.call\.value\(',
    'external_call': r'\.call\(',
    'balance_access': r'\.balance',
    'msg_sender': r'msg\.sender',
    'require_pattern': r'require\(',
    'if_pattern': r'if\s*\(',
    'function_pattern': r'function\s+\w+',
    'modifier_pattern': r'modifier\s+\w+',
    'mapping_pattern': r'mapping\s*\(',
    'payable_pattern': r'payable'
}

class EnhancedFragmentVectorizer:
    def __init__(self, vector_length):
        self.fragments = []
        self.vector_length = vector_length
        self.forward_slices = 0
        self.backward_slices = 0
        self.cnt = 1
        
        # Enhanced features
        self.token_frequency = Counter()
        self.pattern_frequency = Counter()
        self.vocabulary = set()
        self.fragment_metadata = []
        
        # Statistical features
        self.fragment_lengths = []
        self.complexity_scores = []
        
    @staticmethod
    def extract_critical_patterns(code_line):
        """Extract vulnerability-specific patterns from code"""
        patterns_found = []
        for pattern_name, pattern_regex in CRITICAL_PATTERNS.items():
            if re.search(pattern_regex, code_line, re.IGNORECASE):
                patterns_found.append(f"PATTERN_{pattern_name.upper()}")
        return patterns_found
    
    @staticmethod
    def calculate_complexity_score(tokens):
        """Calculate complexity score based on control structures and patterns"""
        complexity = 0
        control_keywords = {'if', 'else', 'for', 'while', 'require', 'assert', 'modifier'}
        
        for token in tokens:
            if token.lower() in control_keywords:
                complexity += 1
            elif token in operators2 or token in operators3:
                complexity += 0.5
                
        return complexity
    
    @staticmethod
    def enhanced_tokenize(line):
        """Enhanced tokenization with better handling of Solidity patterns"""
        # First, extract and preserve critical patterns
        critical_patterns = EnhancedFragmentVectorizer.extract_critical_patterns(line)
        
        tmp, w = [], []
        i = 0
        
        while i < len(line):
            # Skip whitespace but preserve structure
            if line[i] == ' ':
                if w:  # Only add non-empty words
                    tmp.append(''.join(w))
                    w = []
                # Skip multiple spaces
                while i < len(line) and line[i] == ' ':
                    i += 1
                continue
                
            # Handle 3-character operators
            elif i + 2 < len(line) and line[i:i + 3] in operators3:
                if w:
                    tmp.append(''.join(w))
                    w = []
                tmp.append(line[i:i + 3])
                i += 3
                
            # Handle 2-character operators
            elif i + 1 < len(line) and line[i:i + 2] in operators2:
                if w:
                    tmp.append(''.join(w))
                    w = []
                tmp.append(line[i:i + 2])
                i += 2
                
            # Handle 1-character operators
            elif line[i] in operators1:
                if w:
                    tmp.append(''.join(w))
                    w = []
                tmp.append(line[i])
                i += 1
                
            # Regular character
            else:
                w.append(line[i])
                i += 1
        
        # Add the last word if any
        if w:
            tmp.append(''.join(w))
        
        # Filter out empty tokens and add critical patterns
        tokens = [token for token in tmp if token.strip()]
        tokens.extend(critical_patterns)
        
        return tokens

    @staticmethod
    def tokenize_fragment(fragment):
        """Enhanced fragment tokenization with metadata extraction"""
        tokenized = []
        function_regex = re.compile(r'function(\d)+')
        backwards_slice = False
        
        # Fragment-level metadata
        total_lines = len(fragment)
        has_external_calls = False
        has_state_changes = False
        vulnerability_indicators = 0
        
        for line in fragment:
            tokens = EnhancedFragmentVectorizer.enhanced_tokenize(line)
            tokenized.extend(tokens)
            
            # Check for function pattern
            if len(list(filter(function_regex.match, tokens))) > 0:
                backwards_slice = True
            
            # Analyze line for vulnerability indicators
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in VULNERABILITY_KEYWORDS['reentrancy']):
                vulnerability_indicators += 1
                if 'call' in line_lower and 'value' in line_lower:
                    has_external_calls = True
            
            if 'balance' in line_lower and ('=' in line or '+=' in line or '-=' in line):
                has_state_changes = True
        
        # Add structural tokens for better representation
        structural_tokens = [
            f"LINES_{min(total_lines, 20)}",  # Capped at 20 for vocabulary management
            f"VULN_INDICATORS_{min(vulnerability_indicators, 10)}"
        ]
        
        if has_external_calls:
            structural_tokens.append("HAS_EXTERNAL_CALLS")
        if has_state_changes:
            structural_tokens.append("HAS_STATE_CHANGES")
            
        tokenized.extend(structural_tokens)
        
        return tokenized, backwards_slice

    def add_fragment(self, fragment):
        """Enhanced fragment addition with metadata collection"""
        tokenized_fragment, backwards_slice = self.tokenize_fragment(fragment)
        
        # Calculate fragment statistics
        complexity = self.calculate_complexity_score(tokenized_fragment)
        self.complexity_scores.append(complexity)
        self.fragment_lengths.append(len(tokenized_fragment))
        
        # Update vocabulary and frequency counters
        self.vocabulary.update(tokenized_fragment)
        self.token_frequency.update(tokenized_fragment)
        
        # Store fragment with enhanced information
        fragment_info = {
            'tokens': tokenized_fragment,
            'backwards_slice': backwards_slice,
            'complexity': complexity,
            'length': len(tokenized_fragment),
            'fragment_id': self.cnt
        }
        
        self.fragments.append(tokenized_fragment)
        self.fragment_metadata.append(fragment_info)
        
        if backwards_slice:
            self.backward_slices += 1
        else:
            self.forward_slices += 1
            
        self.cnt += 1

    def create_enhanced_word2vec_model(self):
        """Create enhanced Word2Vec model with optimized parameters"""
        # Filter out very rare tokens (appear less than 2 times)
        filtered_fragments = []
        for fragment in self.fragments:
            filtered_tokens = [token for token in fragment 
                             if self.token_frequency[token] >= 2]
            if filtered_tokens:  # Only add non-empty fragments
                filtered_fragments.append(filtered_tokens)
        
        # Enhanced Word2Vec parameters
        model = Word2Vec(
            sentences=filtered_fragments,
            vector_size=self.vector_length,
            window=8,  # Increased window for better context
            min_count=2,  # Minimum frequency threshold
            workers=4,  # Parallel processing
            sg=1,  # Skip-gram model (better for rare words)
            hs=0,  # Use negative sampling
            negative=10,  # Number of negative samples
            epochs=20,  # More training epochs
            alpha=0.025,  # Learning rate
            min_alpha=0.0001,  # Minimum learning rate
            seed=42  # Reproducibility
        )
        
        return model

    def vectorize(self, fragment):
        """Enhanced vectorization with improved sequence handling"""
        tokenized_fragment, backwards_slice = self.tokenize_fragment(fragment)
        
        # Initialize vector matrix
        vectors = np.zeros(shape=(100, self.vector_length), dtype=np.float32)
        
        # Get valid tokens (those in vocabulary)
        valid_tokens = [token for token in tokenized_fragment 
                       if token in self.embeddings.key_to_index]
        
        if not valid_tokens:
            # If no valid tokens, return zero vector
            return vectors
        
        # Enhanced positioning strategy
        if backwards_slice:
            # For backward slices, fill from the end
            start_idx = max(0, 100 - len(valid_tokens))
            for i, token in enumerate(valid_tokens[-100:]):
                if token in self.embeddings.key_to_index:
                    vectors[start_idx + i] = self.embeddings[token]
        else:
            # For forward slices, use strategic positioning
            if len(valid_tokens) <= 100:
                # If fragment fits, place sequentially
                for i, token in enumerate(valid_tokens):
                    if token in self.embeddings.key_to_index:
                        vectors[i] = self.embeddings[token]
            else:
                # If fragment is too long, use sampling strategy
                # Take first 50, last 30, and sample 20 from middle
                indices = (list(range(50)) + 
                          list(np.linspace(50, len(valid_tokens)-31, 20, dtype=int)) +
                          list(range(len(valid_tokens)-30, len(valid_tokens))))
                
                for i, token_idx in enumerate(indices[:100]):
                    token = valid_tokens[token_idx]
                    if token in self.embeddings.key_to_index:
                        vectors[i] = self.embeddings[token]
        
        return vectors

    def train_model(self):
        """Enhanced model training with vocabulary optimization"""
        print(f"Training enhanced Word2Vec model...")
        print(f"Total vocabulary size: {len(self.vocabulary)}")
        print(f"Average fragment length: {np.mean(self.fragment_lengths):.2f}")
        print(f"Average complexity score: {np.mean(self.complexity_scores):.2f}")
        
        # Create and train enhanced model
        model = self.create_enhanced_word2vec_model()
        self.embeddings = model.wv
        
        # Print vocabulary statistics
        print(f"Final vocabulary size: {len(self.embeddings.key_to_index)}")
        print(f"Most common tokens: {[token for token, count in self.token_frequency.most_common(10)]}")
        
        # Clean up memory
        del model
        del self.fragments  # Keep only metadata and embeddings
        
        # Verify critical tokens are in vocabulary
        critical_tokens = ['call', 'value', 'balance', 'msg.sender', 'require']
        missing_critical = [token for token in critical_tokens 
                          if token not in self.embeddings.key_to_index]
        if missing_critical:
            print(f"Warning: Critical tokens missing from vocabulary: {missing_critical}")

    def get_vocabulary_stats(self):
        """Get detailed vocabulary statistics"""
        return {
            'total_tokens': len(self.vocabulary),
            'final_vocab_size': len(self.embeddings.key_to_index) if hasattr(self, 'embeddings') else 0,
            'avg_fragment_length': np.mean(self.fragment_lengths),
            'avg_complexity': np.mean(self.complexity_scores),
            'most_common_tokens': self.token_frequency.most_common(20)
        }
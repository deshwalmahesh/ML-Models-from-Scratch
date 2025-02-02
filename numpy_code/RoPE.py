"""
Implementation of RoPE
"""
import numpy as np

class NumpyRoPE:
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        """
        Rotary Position Embeddings (RoPE) implementation in NumPy. RoPE works by rotating pairs of features in a complex plane, where the rotation angle is determined by the position of the token and a frequency scaling factor.
        Links: 
          0. My Annotated version of paper: https://github.com/deshwalmahesh/DataScience-StudyMaterial/blob/main/annotated_papers/RoPE%3A%20Rotary%20Position%20Embedding.pdf
          1. Code with Formula: https://nn.labml.ai/transformers/rope/index.html
          2. YT Video: https://youtu.be/Mn_9W1nCFLo?si=Y43txNlEoWT3TROR&t=1471
          3. YT Viddeo for the [faster...slower] analogy: https://youtu.be/GQPOtyITy54?si=FGgb0ZYJbMGuusFO&t=358
        Args:
            dim: The embedding dimension. Must be even as features are processed in pairs
            max_seq_len: Maximum sequence length for caching rotation matrices. These are the MAX tokens. You need to pre define it. Every model has different apetite
            base: Base for frequency scaling (10000 is standard I think it from the from the original transformer papers for the vanilla Cos-Sin embedding)
        """
        if dim % 2 != 0: raise ValueError(f"Dimension {dim} must be divisible by 2") 
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self._build_rotation_matrices() # pre compute as it's one time thing
    
    def _build_rotation_matrices(self):
        """
        Precomputes rotation matrices for each position up to max_seq_len. This is one time effort actually because it applies to every position in sequence
        Built for the logic:
          1. Create frequency bands (theta) that decrease geometrically. Each pair of features will rotate at a different frequency. (This is the crux of this algo. I'll explain 2 more times with different analogy)
          2. For each position, compute rotation angles by multiplying position with theta. So first Embedding "PAIR" gets rotated by 1.0, then the econd by 0.1 etc etc (hypothetically). Just like Second - Minute -> Hour hands in clock
          3. Create rotation matrices using cos and sin of these angles (there's this formula to multiply my )
        """
        # Earlier pairs rotate faster (capturing fine-grained position info) (like seconds, minutes hands) BUT later pairs rotate slower (preserving more of original embedding) (like hours, days and Weeks stlye)
        exponents = np.arange(0, self.dim, 2).astype(float)
        theta = 1.0 / (self.base ** (exponents / self.dim))
        
        pos_idx = np.arange(self.max_seq_len) # Create position indices. Which means position of token "i": Shape of [max_seq_len]
        angles = pos_idx[:, None] * theta[None, :]  # Compute rotation angles for each position and frequency. Shape: [max_seq_len, dim/2]
        
        # Pre Compute sin and cos for the rotation matrices. Shape is [max_seq_len, dim/2] because we spli the embedding in d/2 pairs. The roration goes something like: x_rotated = x * cos(mθ) - y * sin(mθ) || y_rotated = y * cos(mθ) + x * sin(mθ)
        self.cos_cached = np.cos(angles)
        self.sin_cached = np.sin(angles)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        How it works:
        1. Split features into first and second halves PAIRS (PAY  ATTENTION to Pairs) as [(i, i+d/2), i+1, i+1+d/2) ......]. So each pair becomes coordinate as [(x1, y1), (x2, y2), .... (x_d/2, y_d/2)]
            If for a RANDOM TOKEN embedding (with embed dimension d = 8) is [0.5, 0.3, 0.7, 0.2, 0.4, 0.6, 0.1, 0.8], the PAIRS become [(0.5, 0.4) , (0.3, 0.6),  (0.7, 0.1), (0.2, 0.8)]
        2. For each position, rotate EACH (xi,yi) PAIR by its corresponding angle: [x1, y1] -> [x1*cos(θ) - y1*sin(θ), y1*cos(θ) + x1*sin(θ)]
           The rotation angles (θ I think it's in radians) for EACH of the (xi,yi) pair according to formula (given d = 8) is: [1.0, 0.1, 0.001, 0.0001] you see the analogy now? [Second, Minute, Hour, Day]
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
               The last dimension (dim) contains the features to be rotated
        Returns:
            Tensor of same shape as input but with rotary position embeddings applied
        """
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        
        # Split the features into two halves that will form rotation pairs
        x1 = x[:, :, : self.dim//2] # first half of features act as the "x" coordinate of (xi,yi)
        x2 = x[:, :, self.dim//2 : self.dim] # second half of features act as the "y" coordinate of (xi,yi) 
        
        # Get rotation matrices for the current sequence length
        cos = self.cos_cached[:seq_len]  # [seq_len, dim/2]
        sin = self.sin_cached[:seq_len]  # [seq_len, dim/2]
        
        # Reshape for broadcasting
        cos = cos[None, :, :]  # [1, seq_len, dim/2]
        sin = sin[None, :, :]  # [1, seq_len, dim/2]
        
        # Apply rotation to EACH + PAIR of features:
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x2 * cos + x1 * sin
      
        return np.concatenate([x1_rotated, x2_rotated], axis=-1) # Concatenate the rotated pairs back together

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Just a fancy way of calling forward() to make the overall class callable. Like you do rope(inputs) OR rope.forward(inputs)"""
        return self.forward(x)


# Example usage
rope = NumpyRoPE(dim=8)

# Create sample input [batch=1, seq_len=3, dim=8] which means [1 sentence having, 3 tokens, and each token is converted to Embedidng Dimension] i.e  [["Mr Saleem Shady"]]
x = np.array([
    [0.5, 0.3, 0.7, 0.2, 0.4, 0.6, 0.1, 0.8],  # token 1
    [0.9, 0.4, 0.2, 0.5, 0.3, 0.7, 0.6, 0.1],  # token 2
    [0.1, 0.8, 0.3, 0.6, 0.4, 0.2, 0.9, 0.5]   # token 3
])[None, :, :]  # Add a batch dimension

x_rotated = rope(x) # you see, the fancy boi helps here
print("Input shape:", x.shape)
print("Output shape:", x_rotated.shape)
print("\nFirst token before rotation:", x[0, 0])
print("First token after rotation:", x_rotated[0, 0])

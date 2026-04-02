"""
nanoJEPA Configuration

This file contains all hyperparameters for the model, training, and data.
Kept simple and flat for clarity.
"""
from dataclasses import dataclass

@dataclass
class Config:
    # Model Architecture
    block_size: int = 512        # Maximum context length
    vocab_size: int = 50257      # GPT-2 vocab size
    n_layer: int = 6             # Number of transformer layers
    n_head: int = 8              # Number of attention heads
    n_embd: int = 512            # Embedding dimension
    dropout: float = 0.1         # Dropout rate
    bias: bool = True            # Use bias in Linears and LayerNorms

    # Training
    batch_size: int = 16         # Memory safety for RTX 3050 (8GB)
    learning_rate: float = 1e-4  # Learning rate (lowered for stability)
    num_epochs: int = 25         # Configured for competition run
    eval_interval: int = 50      # How often to evaluate
    eval_iters: int = 10         # How many batches to use for evaluation
    device: str = 'cpu'          # 'cuda', 'cpu', or 'mps'

    # JEPA Specifics
    jepa_weight: float = 1.0     # Lambda for JEPA loss (L_total = L_token + lambda * L_jepa)
    
    # Special Tokens (will be added to tokenizer)
    # [SEP] separates Question and Answer
    # [PRED] is the token used for latent prediction
    # We will likely map these to unused tokens in GPT-2 vocab or add them.
    # For simplicity in this educational repo, we might reuse rare tokens or extend vocab.
    # extended vocab approach:
    # 50257 = [SEP]
    # 50258 = [PRED]
    sep_token_id: int = 50257
    pred_token_id: int = 50258
    
    # We will resize model embeddings to accommodate these if necessary.
    final_vocab_size: int = 50259 

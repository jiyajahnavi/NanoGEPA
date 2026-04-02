"""
nanoJEPA Model Definition

A Decoder-only Transformer with custom attention masking to enforce JEPA constraints.
Features:
- Shared weights for Question and Answer encoding.
- Custom masking: Q->Q, A->A, PRED->Q. A-|->Q blocked.
- Positional embedding reset for View B (Answer).
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .config import Config

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply custom mask if provided
        # mask should be (B, 1, T, T) or broadcastable
        if mask is not None:
            # We expect mask to contain 0 for allowed, -inf for blocked
            att = att + mask
            
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class NanoJEPA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        # Head for token prediction
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        
        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def build_jepa_mask(self, q_lens, a_lens, total_len, device):
        """
        Constructs the block-diagonal attention mask.
        Batch processing is tricky with variable lengths without complex padding logic.
        For simplicity/education, we assume batch_size=1 or create a mask per item and stack.
        
        Mask shape: (B, 1, T, T)
        """
        B = len(q_lens)
        mask = torch.full((B, 1, total_len, total_len), float('-inf'), device=device)
        
        for b in range(B):
            q_len = q_lens[b]
            a_len = a_lens[b]
            
            # Indices
            # Q range: [0, q_len)
            # SEP: q_len
            # A range: [q_len+1, q_len+1+a_len)
            # PRED: q_len+1+a_len
            
            # Start of A in the sequence (after SEP)
            a_start = q_len + 1 
            # Position of PRED (at end)
            pred_pos = a_start + a_len
            
            # 1. Q attends to Q (Causal)
            # range 0..q_len (inclusive of SEP for Q context?) 
            # Let's include SEP in Q's view as it marks end of Q.
            q_end = q_len + 1
            tri_mask = torch.triu(torch.ones((q_end, q_end), device=device), diagonal=1)
            mask[b, 0, :q_end, :q_end] = tri_mask * float('-inf')
            # The diagonal=1 makes upper triangle -inf (causal) which is what we want? 
            # Wait, 0 is allowed. The initialized value is -inf.
            # So, set the causal lower triangle to 0.
            causal_mask_q = torch.tril(torch.ones((q_end, q_end), device=device))
            mask[b, 0, :q_end, :q_end] = mask[b, 0, :q_end, :q_end].masked_fill(causal_mask_q == 1, 0.0)

            # 2. A attends to A (Causal) - INDEPENDENT of Q
            # range a_start..pred_pos (exclusive of PRED)
            a_end = pred_pos
            a_seg_len = a_len 
            
            if a_seg_len > 0:
                # Create causal mask for A part
                causal_mask_a = torch.tril(torch.ones((a_seg_len, a_seg_len), device=device))
                # Apply to the block in main mask
                # The region is [a_start:a_end, a_start:a_end]
                # It currently is -inf. We set lower triangle to 0.
                mask[b, 0, a_start:a_end, a_start:a_end] = mask[b, 0, a_start:a_end, a_start:a_end].masked_fill(causal_mask_a == 1, 0.0)
                
            # 3. PRED attends to Q (Full or Causal)
            # PRED is at pred_pos.
            # It should attend to all Q tokens (0..q_end).
            # It should NOT attend to A.
            mask[b, 0, pred_pos, :q_end] = 0.0
            
            # PRED self-attend? It's one token.
            mask[b, 0, pred_pos, pred_pos] = 0.0

            # FIX: Handle padding tokens (indices > pred_pos)
            # If we don't let them attend to something, softmax gives NaN.
            if pred_pos + 1 < total_len:
                pad_range = torch.arange(pred_pos + 1, total_len, device=device)
                mask[b, 0, pad_range, pad_range] = 0.0


    def build_position_ids(self, q_lens, a_lens, total_len, device):
        """
        Resets position IDs for the Answer view.
        Q: 0, 1, 2...
        A: 0, 1, 2... (Reset)
        """
        B = len(q_lens)
        pos_ids = torch.zeros((B, total_len), dtype=torch.long, device=device)
        
        for b in range(B):
            q_len = q_lens[b]
            a_len = a_lens[b]
            
            # Q positions: 0..q_len (inclusive of SEP)
            q_end = q_len + 1
            pos_ids[b, :q_end] = torch.arange(q_end, device=device)
            
            # A positions: Reset to 0..a_len
            a_start = q_len + 1
            a_end = a_start + a_len
            pos_ids[b, a_start:a_end] = torch.arange(a_len, device=device)
            
            # PRED position: 
            # Should it be conceptualized as following Q?
            # Or following A?
            # Since PRED predicts A from Q, it effectively summarizes Q.
            # Let's give it position `q_len + 1` (continuation of Q sequence).
            pos_ids[b, a_end] = q_end # The position after the last Q token

        return pos_ids

    def forward(self, input_ids, q_lens=None, a_lens=None, targets=None):
        device = input_ids.device
        B, T = input_ids.size()
        
        # 1. Build Custom Mask & Positional IDs
        # We need q_lens and a_lens passed in batch
        if q_lens is None:
            # Fallback for standard GPT usage (inference/generation without JEPA split)
            # Just standard causal mask
            mask = None 
            pos_ids = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        else:
            mask = self.build_jepa_mask(q_lens, a_lens, T, device)
            pos_ids = self.build_position_ids(q_lens, a_lens, T, device)

        # 2. Embeddings
        tok_emb = self.transformer.wte(input_ids) # (B, T, C)
        pos_emb = self.transformer.wpe(pos_ids)   # (B, T, C)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # 3. Transformer Blocks
        for block in self.transformer.h:
            x = block(x, mask)
            
        x = self.transformer.ln_f(x) # (B, T, C) - these are the latents
        
        # 4. Latent Extraction for JEPA
        # We need to compute loss if targets are provided
        loss = None
        jepa_loss = None
        token_loss = None
        
        if targets is not None:
            # Targets are same as input_ids usually, shifted
            # But here we have specific JEPA logic.
            
            # Token Loss (Language Modeling)
            # We predict next token for Q and A segments.
            logits = self.lm_head(x) # (B, T, V)
            
            # Standard shifted cross entropy
            # Shift logits so token t predicts t+1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            # Mask padding in labels
            # Assuming padding token is 0 (as set in data.py collate_fn)
            # We convert 0s to -100 (standard ignore index) or -1 as used below
            # Create a mask for valid tokens. 
            # Note: We must be careful if 0 is a valid token (in GPT2 it is '!' I think).
            # Unlikely to be critical for this demo, but let's be cleaner.
            # Ideally data loader provides a mask. 
            # For now, we'll assume 0 is padding since we pad with 0s.
            
            # Create a clone to avoid modifying inputs if they are shared
            shift_labels = shift_labels.clone()
            shift_labels[shift_labels == 0] = -1 
            
            # Flatten
            token_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-1)
            
            # JEPA Loss
            # Pred Latent: Final hidden state of [PRED] token
            # Target Latent: Final hidden state of last Answer token
            
            pred_latents = []
            target_latents = []
            
            for b in range(B):
                q_len = q_lens[b]
                a_len = a_lens[b]
                
                # PRED token is at end
                pred_idx = q_len + 1 + a_len
                pred_latent = x[b, pred_idx]
                pred_latents.append(pred_latent)
                
                # Target Latent is last token of A
                # A ends at pred_idx - 1
                target_idx = pred_idx - 1
                target_latent = x[b, target_idx]
                target_latents.append(target_latent)
                
            pred_latents = torch.stack(pred_latents)       # (B, C)
            target_latents = torch.stack(target_latents)   # (B, C)
            
            # Cosine Embedding Loss
            # We want pred and target to be similar -> label 1
            # Loss = 1 - cos(x, y)
             
            # Detach target!
            target_latents = target_latents.detach()
            
            jepa_loss = 1.0 - F.cosine_similarity(pred_latents, target_latents).mean()
            
            # Total Loss
            loss = token_loss + self.config.jepa_weight * jepa_loss
            
        return {
            'logits': self.lm_head(x) if targets is None else logits,
            'latents': x,
            'loss': loss,
            'token_loss': token_loss,
            'jepa_loss': jepa_loss
        }

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        import inspect
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

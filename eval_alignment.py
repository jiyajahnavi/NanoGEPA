"""
Evaluation: JEPA Latent Alignment

Compares Cosine Similarity between [PRED] and Answer Latent
for:
1. Baseline (lambda=0.0)
2. JEPA (lambda=0.5)

Plots result to latent_alignment.png
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os

from nanoJEPA.model import NanoJEPA
from nanoJEPA.config import Config
from nanoJEPA.data import GSM8KDataset, collate_fn

def train_and_track(jepa_weight, steps=50, eval_interval=10):
    print(f"\n--- Training with lambda={jepa_weight} ---")
    
    # 1. Config & Model
    config = Config(jepa_weight=jepa_weight, max_iters=steps, learning_rate=1e-4)
    # Ensure vocab size matches data logic (manual fix from train.py)
    config.vocab_size = config.final_vocab_size 
    
    model = NanoJEPA(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=config.learning_rate, betas=(0.9, 0.95), device_type=device)
    
    # 2. Data
    # Use a small consistency subset for training to see effects quickly
    dataset = GSM8KDataset(split='train', config=config)
    # Hack: limit dataset size strictly for speed in this demo
    dataset.items = dataset.items[:200]
    
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    data_iter = iter(loader)
    
    # Tracking
    history = {'step': [], 'sim': []}
    
    for i in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
            
        input_ids = batch['input_ids'].to(device)
        q_lens = batch['q_lens'].to(device)
        a_lens = batch['a_lens'].to(device)
        
        # Forward
        outputs = model(input_ids, q_lens=q_lens, a_lens=a_lens, targets=input_ids)
        
        # Update
        loss = outputs['loss']
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track every N steps
        if i % eval_interval == 0:
            # Re-compute sim purely for logging (or take from outputs if available/representative)
            # The model outputs 'jepa_loss' which is 1 - AvgCosSim.
            # So AvgCosSim = 1 - jepa_loss
            
            # Note: outputs['jepa_loss'] is a detached scalar tensor
            jepa_loss = outputs['jepa_loss'].item()
            cos_sim = 1.0 - jepa_loss
            
            print(f"Step {i}: Sim={cos_sim:.4f}")
            history['step'].append(i)
            history['sim'].append(cos_sim)
            
    return history

def main():
    # settings
    steps = 100
    
    # 1. Baseline
    hist_baseline = train_and_track(jepa_weight=0.0, steps=steps)
    
    # 2. JEPA
    hist_jepa = train_and_track(jepa_weight=0.5, steps=steps)
    
    # 3. Plot
    plt.figure(figsize=(20, 10))
    plt.plot(hist_baseline['step'], hist_baseline['sim'], label='Baseline (λ=0.0)', marker='o')
    plt.plot(hist_jepa['step'], hist_jepa['sim'], label='JEPA (λ=0.5)', marker='x')
    
    plt.title('Cosine Similarity: [PRED] vs Answer Latent')
    plt.xlabel('Training Steps')
    plt.ylabel('Average Cosine Similarity')
    plt.legend()
    plt.grid(True)
    
    out_path = 'latent_alignment.png'
    plt.savefig(out_path)
    print(f"\nPlot saved to {out_path}")

if __name__ == '__main__':
    main()

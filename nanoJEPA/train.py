"""
Training Script for nanoJEPA

Train the model on GSM8K with:
1. Token Loss (Autoregressive)
2. JEPA Loss (Latent Prediction)
"""

import os
import time
import math
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .model import NanoJEPA
from .config import Config
from .data import GSM8KDataset, collate_fn

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not available. Using CPU.")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------
train_dataset = GSM8KDataset(split='train', config=config)
# Use a smaller subset for quick educational run if needed, but GSM8K is small enough (7.5k)
# train_dataset.items = train_dataset.items[:1000] 

train_loader = DataLoader(
    train_dataset, 
    batch_size=config.batch_size, 
    shuffle=True, 
    collate_fn=collate_fn,
    drop_last=True
)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# Update vocab size in config if dataset added new tokens (tiktoken is fixed though)
# We assumed manual token IDs for SEP and PRED in config.
# Ensure embedding layer is big enough.
config.vocab_size = config.final_vocab_size 

model = NanoJEPA(config)
model.to(device)

# -----------------------------------------------------------------------------
# Optimizer & Scheduler
# -----------------------------------------------------------------------------
optimizer = model.configure_optimizers(weight_decay=1e-1, learning_rate=config.learning_rate, betas=(0.9, 0.95), device_type=device)

dataset_size = len(train_dataset)
steps_per_epoch = math.ceil(dataset_size / config.batch_size)
max_iters = config.num_epochs * steps_per_epoch

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-5)

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
model.train()
t0 = time.time()

print(f"Starting training for {config.num_epochs} epochs. ({steps_per_epoch} steps/epoch, {max_iters} total iters)")

training_log = []
hist_token_loss = []
hist_jepa_loss = []

# AMP Scaler setup
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
first_batch_logged = False

for epoch in range(1, config.num_epochs + 1):
    epoch_token_loss = 0.0
    epoch_jepa_loss = 0.0
    epoch_total_loss = 0.0
    epoch_cos_sim = 0.0
    
    data_iter = iter(train_loader)
    
    for step in range(steps_per_epoch):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        # Move to device
        input_ids = batch['input_ids'].to(device)
        q_lens = batch['q_lens'].to(device)
        a_lens = batch['a_lens'].to(device)
        
        # Forward path with AMP
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(input_ids, q_lens=q_lens, a_lens=a_lens, targets=input_ids)
            
            loss = outputs['loss']
            token_loss = outputs['token_loss']
            jepa_loss = outputs['jepa_loss']
            
            # Calculate cosine similarity from JEPA loss
            cos_sim = 1.0 - jepa_loss.item()
        
        # Backward path with AMP
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        # Gradient clipping needs unscaling first
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if not first_batch_logged and torch.cuda.is_available():
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            first_batch_logged = True
            
        epoch_total_loss += loss.item()
        epoch_token_loss += token_loss.item()
        epoch_jepa_loss += jepa_loss.item()
        epoch_cos_sim += cos_sim
        
        # Sub-epoch Logging
        global_step = (epoch - 1) * steps_per_epoch + step
        if global_step % 50 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"Epoch {epoch} | Step {global_step}/{max_iters} | Loss {loss.item():.4f} | Token {token_loss.item():.4f} | JEPA {jepa_loss.item():.4f} | Sim {cos_sim:.2f} | {dt*1000:.2f}ms")
            
            if torch.isnan(loss):
                print("CRITICAL: Loss is NaN! Stopping training.")
                break
                
    if torch.isnan(loss):
        break

    # End of Epoch
    avg_total = epoch_total_loss / steps_per_epoch
    avg_token = epoch_token_loss / steps_per_epoch
    avg_jepa = epoch_jepa_loss / steps_per_epoch
    avg_sim = epoch_cos_sim / steps_per_epoch
    
    print(f"--- Epoch {epoch} Summary: Avg Loss: {avg_total:.4f}, Avg Token: {avg_token:.4f}, Avg JEPA: {avg_jepa:.4f}, Avg Sim: {avg_sim:.4f} ---")
    
    epoch_stats = {
        "epoch": epoch,
        "avg_total_loss": avg_total,
        "avg_token_loss": avg_token,
        "avg_jepa_loss": avg_jepa,
        "avg_cos_sim": avg_sim
    }
    training_log.append(epoch_stats)
    hist_token_loss.append(avg_token)
    hist_jepa_loss.append(avg_jepa)

print("Training finished.")

# Save logs & plots
os.makedirs("out", exist_ok=True)
with open("out/training_log.json", "w") as f:
    json.dump(training_log, f, indent=4)

plt.figure()
plt.plot(range(1, config.num_epochs + 1), hist_token_loss, label="Token Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Token Loss Curve")
plt.legend()
plt.savefig("out/token_loss_curve.png")
plt.close()

plt.figure()
plt.plot(range(1, config.num_epochs + 1), hist_jepa_loss, label="JEPA Loss", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("JEPA Loss Curve")
plt.legend()
plt.savefig("out/jepa_loss_curve.png")
plt.close()

# Save model
torch.save(model.state_dict(), "out/nanojepa.pt")
print("Model, logs, and plots saved to out/")

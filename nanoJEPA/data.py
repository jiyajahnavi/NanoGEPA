"""
Data Loading and Preprocessing for nanoJEPA

Dataset: GSM8K (Grade School Math 8K)
Target:
    View A: Question
    View B: Numerical Answer
    Format: [QUESTION] [SEP] [ANSWER] [PRED]

We use tiktoken (gpt2) for tokenization.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from datasets import load_dataset
import re
from .config import Config

class GSM8KDataset(Dataset):
    def __init__(self, split='train', config=None):
        self.config = config or Config()
        
        # Load GSM8K dataset
        # 'main' config is the default for gsm8k
        print(f"Loading GSM8K {split} split...")
        self.dataset = load_dataset("gsm8k", "main", split=split)
        
        # Tokenizer
        self.enc = tiktoken.get_encoding("gpt2")
        
        # Precompute items to ensure consistent length and valid data
        self.items = []
        for item in self.dataset:
            processed = self.process_item(item)
            if processed:
                self.items.append(processed)
                
        print(f"Loaded {len(self.items)} valid samples for {split}.")

    def process_item(self, item):
        question = item['question']
        answer_raw = item['answer']
        
        # Extract numerical answer (View B)
        # GSM8K answers are arguably chain of thought often ending in #### <number>
        # We want JUST the final number for View B to keep it distinct and "latent-focused".
        # Regex to find the number after ####
        match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_raw)
        if match:
            numeric_answer = match.group(1).replace(',', '') # remove commas
        else:
            return None # Skip if no clear numeric answer
            
        # Tokenize (View A)
        q_tokens = self.enc.encode(question)
        
        # Tokenize (View B)
        # We process the number as a string
        a_tokens = self.enc.encode(numeric_answer)
        
        # Construct Sequence: [Q] [SEP] [A] [PRED]
        # Truncate if necessary (keeping space for special tokens)
        # 2 special tokens: SEP, PRED
        max_len = self.config.block_size
        
        # We need to construct: [q_tokens] + [SEP] + [a_tokens] + [PRED]
        # AND we need labels for training.
        
        # The prompt implies we process Q and A. 
        # Token loss is on next-token prediction for Q and A.
        # JEPA loss is on latent of A vs latent of PRED.
        
        # Check lengths
        total_len = len(q_tokens) + 1 + len(a_tokens) + 1
        if total_len > max_len:
            # simple truncation of question (keep answer)
            q_len = max_len - (len(a_tokens) + 2)
            if q_len <= 0: return None
            q_tokens = q_tokens[:q_len]
            
        input_ids = q_tokens + [self.config.sep_token_id] + a_tokens + [self.config.pred_token_id]
        
        # Create attention mask helpers (conceptually).
        # We will handle the complex masking in the model forward pass or collate.
        # Here we just return the input_ids.
        
        # Padding
        inputs = torch.tensor(input_ids, dtype=torch.long)
        
        # Store lengths for masking logic later
        q_len = len(q_tokens)
        a_len = len(a_tokens)
        
        return {
            'input_ids': inputs,
            'q_len': q_len,
            'a_len': a_len
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def collate_fn(batch):
    # dynamic padding to max length in batch or fixed block_size
    # For simplicity, we pad to longest in batch
    max_len = max([x['input_ids'].size(0) for x in batch])
    
    input_ids_list = []
    q_lens = []
    a_lens = []
    
    for x in batch:
        ids = x['input_ids']
        pad_len = max_len - ids.size(0)
        # Use 0 or specialized pad token? GPT-2 often uses eot or 0.
        # We'll use 0 for padding (ignoring it in loss).
        padded = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)], dim=0)
        input_ids_list.append(padded)
        q_lens.append(x['q_len'])
        a_lens.append(x['a_len'])
        
    return {
        'input_ids': torch.stack(input_ids_list),
        'q_lens': torch.tensor(q_lens),
        'a_lens': torch.tensor(a_lens)
    }

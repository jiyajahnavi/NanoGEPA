import torch
import tiktoken
import os
import re

from nanoJEPA.model import NanoJEPA
from nanoJEPA.config import Config
from nanoJEPA.data import GSM8KDataset

def evaluate_accuracy():
    print("Loading Config and Model for Evaluation...")
    config = Config()
    config.vocab_size = config.final_vocab_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = NanoJEPA(config)
    checkpoint_path = "out/nanojepa.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded checkpoint successfully.")
    else:
        print("Model checkpoint not found. Evaluating with random weights.")
        
    model.to(device)
    model.eval()
    
    enc = tiktoken.get_encoding("gpt2")
    
    print("Loading test dataset...")
    # gsm8k uses 'train' and 'test' splits
    dataset = GSM8KDataset(split='test', config=config)
    
    # We only take the first 100 valid samples
    num_samples = min(100, len(dataset.items))
    print(f"Evaluating on {num_samples} samples...")
    
    correct_count = 0
    
    with torch.no_grad():
        for i in range(num_samples):
            item = dataset.items[i]
            input_tensor_full = item['input_ids']
            q_len = item['q_len']
            a_len = item['a_len']
            
            # Extract q_tokens and a_tokens
            # format: [q_tokens] [SEP] [a_tokens] [PRED]
            # SEP position is at q_len
            q_tokens = input_tensor_full[:q_len].tolist()
            a_tokens_gt = input_tensor_full[q_len+1 : q_len+1+a_len].tolist()
            
            ground_truth_answer = enc.decode(a_tokens_gt).strip()
            
            # Prepare generation input: [q_tokens] [SEP]
            input_ids = q_tokens + [config.sep_token_id]
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            
            generated_tokens = []
            max_new_tokens = 20
            
            for _ in range(max_new_tokens):
                outputs = model(input_tensor)
                logits = outputs['logits']
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()
                
                # Stop generation
                if next_token == config.sep_token_id or next_token == enc.eot_token or next_token == config.pred_token_id or next_token == 50256:
                    break
                    
                input_ids.append(next_token)
                generated_tokens.append(next_token)
                input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
                
            generated_answer = enc.decode(generated_tokens).strip()
            
            # Clean up generated answer (extract numbers only for robust comparison)
            match = re.search(r"-?[\d]+(?:\.\d+)?", generated_answer)
            gen_num = match.group(0) if match else generated_answer
            
            if gen_num == ground_truth_answer:
                correct_count += 1
                
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_samples} samples. Correct so far: {correct_count}")
                
    accuracy = (correct_count / num_samples) * 100.0
    print("-" * 50)
    print(f"Final Exact-Match Accuracy on {num_samples} GSM8K validation samples: {accuracy:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    evaluate_accuracy()

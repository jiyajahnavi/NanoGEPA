
import torch
import os

checkpoint_path = "out/nanojepa.pt"

if not os.path.exists(checkpoint_path):
    print(f"Error: {checkpoint_path} not found.")
else:
    try:
        # Load the checkpoint
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"Type: {type(state_dict)}")
        
        print("\n--- Keys and Shapes ---")
        total_params = 0
        has_nan = False
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                num_params = value.numel()
                total_params += num_params
                if torch.isnan(value).any():
                    print(f"[WARNING] {key} contains NaNs!")
                    has_nan = True
            
        print(f"\nTotal Parameters: {total_params}")
        print(f"Total Parameters (Millions): {total_params/1e6:.2f}M")
        
        if has_nan:
            print("\nCRITICAL: Model weights contain NaNs. Training likely diverged.")
        else:
            print("\nModel weights appear valid (no NaNs found).")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

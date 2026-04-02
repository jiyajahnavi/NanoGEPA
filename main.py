"""
nanoJEPA Gradio Interface (Competition Version)

This is a research demonstration interface for a JEPA-trained language model.
This interface is intentionally minimal to avoid conflating JEPA-based reasoning with chatbot-style generation.

It demonstrates:
1. Input projection (Question)
2. Latent reasoning (JEPA)
3. Output projection (Answer)
"""

import gradio as gr
import torch
import torch.nn.functional as F
import tiktoken
import os

from nanoJEPA.model import NanoJEPA
from nanoJEPA.config import Config

# -----------------------------------------------------------------------------
# 1. Setup & Model Loading
# -----------------------------------------------------------------------------

# Load Config
config = Config()
config.vocab_size = config.final_vocab_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model on {device}...")

model = NanoJEPA(config)
checkpoint_path = "out/nanojepa.pt"

if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")
    print(f"nanoJEPA trained for {config.num_epochs} epochs.")
else:
    print(f"WARNING: Checkpoint {checkpoint_path} not found. Using random weights.")

model.to(device)
model.eval()

# Tokenizer
enc = tiktoken.get_encoding("gpt2")

# -----------------------------------------------------------------------------
# 2. Inference Logic
# -----------------------------------------------------------------------------

def answer_question(question: str) -> str:
    """
    Processes the math question and generates a numeric answer.
    Strictly restricted to GSM8K-style inputs.
    """
    # --- Input Validation ---
    if not question or not question.strip():
        return "This demo is restricted to GSM8K-style math questions.\nPlease enter a numerical reasoning problem."
    
    # Simple heuristic to reject chit-chat
    conversational_triggers = ["hello", "hi", "how are you", "who are you", "chat", "write a poem"]
    q_lower = question.lower()
    if any(trigger in q_lower for trigger in conversational_triggers) and len(question.split()) < 5:
        return "This demo is restricted to GSM8K-style math questions.\nPlease enter a numerical reasoning problem."

    # --- Preprocessing ---
    # Construct Input: [QUESTION] [SEP]
    # Note: In training we used [Q] [SEP] [A] [PRED].
    # For inference, we generate A given Q.
    
    q_tokens = enc.encode(question)
    input_ids = q_tokens + [config.sep_token_id]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # --- Generation (Greedy Decoding) ---
    max_new_tokens = 20
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = model(input_tensor)
            logits = outputs['logits']
            
            # Next token prediction (greedy)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            
            # Stop conditions
            if next_token == config.sep_token_id or next_token == 50256: 
                 break
            
            # Append
            input_ids.append(next_token)
            generated_tokens.append(next_token)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            
    # --- Post-processing ---
    answer_text = enc.decode(generated_tokens)
    
    # Format Output as requested
    output_formatted = f"Answer:\n{answer_text.strip()}\n\nNote:\n“Reasoning performed in latent space using JEPA.\nThis demo is restricted to GSM8K-style math problems.”"
    
    return output_formatted

# -----------------------------------------------------------------------------
# 3. Gradio UI
# -----------------------------------------------------------------------------

css = """
footer {display: none !important;}
.gradio-container {min-height: 0px !important;}
"""

with gr.Blocks(css=css, title="nanoJEPA Demo") as demo:
    gr.Markdown("# nanoJEPA — JEPA-based Latent Reasoning Demo")
    gr.Markdown("### A minimal educational JEPA-for-language model (GSM8K)")
    
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(
                label="Input", 
                placeholder="Enter a math question (e.g. 'John has 3 apples...')", 
                lines=3
            )
            submit_btn = gr.Button("Compute Answer", variant="primary")
            
        with gr.Column():
            out = gr.Textbox(
                label="nanoJEPA Answer", 
                lines=5,
                interactive=False
            )
            
    gr.Markdown("""
    **Footer:**
    This system is not a chatbot.
    It demonstrates JEPA-based latent reasoning, not conversational AI.
    """)

    submit_btn.click(fn=answer_question, inputs=inp, outputs=out)
    inp.submit(fn=answer_question, inputs=inp, outputs=out)

# Launch
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

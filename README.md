# nanoJEPA  
## Joint Embedding Predictive Architecture for Language Reasoning (GSM8K)

nanoJEPA is a minimal, research-oriented implementation of a **Joint Embedding Predictive Architecture (JEPA)** applied to language reasoning tasks.

Unlike standard large language models that rely purely on next-token prediction, nanoJEPA is trained to **predict Answer representations from Question representations in latent space**, explicitly separating reasoning from text generation.

This repository demonstrates how latent alignment objectives affect reasoning structure in language models.

---

## Motivation

Modern LLMs optimize:

P(token_t | token_<t)

While effective for fluency, this objective does not explicitly enforce structured reasoning in representation space.

Inspired by:

- **JEPA / world-model ideas** proposed by Yann LeCun  
- **LLM-JEPA research** exploring latent prediction for language  
- **nanoGPT** for architectural clarity  

nanoJEPA investigates:

> Can reasoning be learned as latent state prediction instead of text reconstruction?

---

## Core Idea

Given two views of the same semantic state:

- **View A**: Question (math problem)
- **View B**: Final numeric Answer

nanoJEPA learns:

Question Latent → Answer Latent

instead of simply predicting answer tokens autoregressively.

Text generation is treated as a projection from latent space — not the reasoning process itself.

---

## Architecture

![nanoJEPA Architecture](nanoJEPA_architecture.png)

### Design Principles

- **Single decoder-only Transformer**
- No encoder–decoder split
- No external retrieval
- No pretrained backbone
- No chain-of-thought supervision

### View Isolation (JEPA Masking)

Custom attention masking enforces:

1. Question → Question only (causal)
2. Answer → Answer only (causal, independent)
3. `[PRED]` → Question only
4. `[PRED]` cannot attend to Answer

This forces the model to predict Answer representations without direct exposure.

---

## Model Configuration

| Component | Value |
|------------|--------|
| Layers | 6 |
| Attention Heads | 8 |
| Hidden Dimension | 512 |
| Parameters | ~45M |
| Dataset | GSM8K (~7.5k samples) |
| Training | 25 Full Epochs |
| Hardware | NVIDIA RTX 3050 |

---

## Training Objective

The total loss is:

L_total = L_token + λ * L_jepa

Where:

- **Token Loss**: Standard cross-entropy (stabilization)
- **JEPA Loss**: Cosine similarity between predicted and true Answer latent

L_jepa = 1 − cos(pred_latent, answer_latent)

---

## Experimental Results

### After 25 Full Epochs

- **Final Token Loss:** 0.1186  
- **Final JEPA Loss:** 0.0525  
- **Final Cosine Similarity:** 0.9475  

These results show stable latent alignment across training.

![TOken Loss](out/token_loss_curve.png)
![JEPA Loss](out/jepa_loss_curve.png)

### Key Observation

Ablation experiments (λ = 0 vs λ > 0) demonstrate:

- Without JEPA loss, latent alignment collapses.
- With JEPA loss, representation geometry remains stable.

This confirms:

> Next-token prediction alone does not preserve reasoning structure.

---

## Exact-Match Accuracy

Exact-match evaluation on 100 GSM8K validation samples yields:

- **0.00%**

This is expected.

nanoJEPA is trained from scratch on a small dataset without pretraining.  
The purpose of this system is to validate the **JEPA alignment mechanism**, not to achieve competitive GSM8K performance.

---

## Interactive Demo

A minimal Gradio interface is provided for demonstration:

```bash
python main.py
```

Design constraints:

- Single-turn input
- Deterministic greedy decoding
- No chat history
- No external LLM usage
- Max generation length enforced

The demo visualizes JEPA-based latent reasoning rather than conversational AI.

---

## Evaluation Tools

- `eval_alignment.py` — Latent cosine similarity evaluation
- `evaluate_accuracy.py` — Exact-match accuracy testing
- Automatic loss curve generation:
    - `token_loss_curve.png`
    - `jepa_loss_curve.png`

---

## File Structure

- `config.py`              # Hyperparameters
- `data.py`                # GSM8K loading & preprocessing
- `model.py`               # Transformer + JEPA masking
- `train.py`               # Training loop
- `evaluate_accuracy.py`   # Exact-match evaluation
- `eval_alignment.py`      # Latent alignment evaluation
- `main.py`                # Gradio demo

---

## Limitations

- Small model size (~45M)
- No large-scale pretraining
- Limited dataset (7.5k samples)
- Not optimized for fluent generation

nanoJEPA is a research prototype designed to investigate latent reasoning objectives.

---

## Conclusion

nanoJEPA demonstrates that:

- Reasoning can be framed as latent representation prediction.
- JEPA loss stabilizes semantic alignment.
- Text generation is not equivalent to reasoning.
- Latent geometry collapses under standard next-token training.

This project provides a transparent, reproducible baseline for JEPA-style language modeling experiments.

---

## Acknowledgements

Inspired by:

- Yann LeCun’s JEPA and world-model vision
- LLM-JEPA research
- nanoGPT implementation philosophy

Special thanks to:

- **Aditi Khatana**
- **Rishabh Yadav**

for discussions and contributions during development.
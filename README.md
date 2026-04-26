# GPT From Scratch: Understanding & Building Transformer Architecture

A comprehensive educational notebook that walks through the fundamentals of transformer architecture and implements a decoder-only GPT model from scratch using PyTorch.

## Overview

This project provides an in-depth exploration of transformer components, starting from basic character-level language modeling and progressively building up to a complete decoder-only transformer architecture. All code is implemented from first principles to maximize understanding of how these models work.

## What You'll Learn

### 1. **Data Preparation & Encoding**
- Downloading and exploring the TinyShakespeare dataset
- Character-level tokenization
- Creating train/validation splits
- Understanding context windows (block size) and batch sampling

### 2. **Bigram Language Model**
- Implementing a simple baseline model
- Token and positional embeddings
- Basic forward pass and loss computation
- Text generation using multinomial sampling
- Training with AdamW optimizer

### 3. **Self-Attention Mechanism**
- Mathematical foundation of attention
- Implementing single-head attention
- Masked (causal) attention for autoregressive models
- Scaled dot-product attention
- Key insights: attention as a communication mechanism

### 4. **Multi-Head Attention**
- Parallel attention heads
- Concatenation and projection
- Positional encoding strategies

### 5. **Transformer Building Blocks**
- **Feed-Forward Networks**: MLP layers with ReLU activation
- **Layer Normalization**: Pre-layer norm architecture with residual connections
- **Transformer Block**: Combining attention and feed-forward with skip connections
- **Decoder-Only Transformer**: Full model architecture

### 6. **Complete Decoder-Only Transformer**
- Multi-layer stacked blocks
- Dropout for regularization
- Efficient batch processing
- Text generation at scale

## Architecture Highlights

```
Input (character indices)
    ↓
Token Embedding (vocab_size → n_embd)
    ↓
Positional Embedding + Token Embedding
    ↓
[Transformer Block] × n_layer
    ├─ Multi-Head Attention (n_head heads)
    ├─ Residual Connection
    ├─ Layer Normalization
    ├─ Feed-Forward Network
    └─ Residual Connection
    ↓
Layer Normalization
    ↓
Linear Head (n_embd → vocab_size)
    ↓
Output (logits for next character prediction)
```

## Key Hyperparameters

- **`n_embd`**: Embedding dimension size (128)
- **`n_head`**: Number of attention heads (2)
- **`n_layer`**: Number of transformer blocks (2)
- **`block_size`**: Context window / sequence length (64)
- **`batch_size`**: Samples per batch (16)
- **`learning_rate`**: AdamW learning rate (3e-4)
- **`dropout`**: Regularization dropout rate (0.2)
- **`max_iters`**: Total training iterations (1500)

## Requirements

- Python 3.7+
- PyTorch (CPU or GPU)
- NumPy

## Usage

1. **Open the notebook**: `gpt_dev.ipynb`
2. **Run cells sequentially**: Each section builds on previous concepts
3. **Dataset**: Automatically downloads TinyShakespeare (4.5MB)
4. **Device**: Automatically selects GPU if available, otherwise CPU
5. **Output**: Generates 2000 character samples after training

### Quick Start
```python
# The final model generates Shakespeare-style text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = m.generate(context, max_new_tokens=2000)
print(decode(generated_text[0].tolist()))
```

## Model Performance

- **Training iterations**: 1500
- **Eval interval**: Every 500 steps
- **Final model size**: ~1.3M parameters
- **Expected final loss**: < 1.5 (on TinyShakespeare)

## Sections Breakdown

| Section | Key Concepts |
|---------|--------------|
| Imports & Dataset | Setup and data loading |
| Encoding/Decoding | Character-to-token mapping |
| Train/Val Split | 90/10 data split |
| Bigram Model | Simple baseline + generation |
| Self-Attention | Core mechanism of transformers |
| Multi-Head Attention | Parallel attention heads |
| Feed-Forward | Non-linear processing layer |
| Transformer Block | Complete transformer unit |
| Decoder-Only Transformer | Full model training & inference |

## Important Notes

- **Causal Masking**: Upper triangular masking prevents attending to future tokens (essential for autoregressive generation)
- **Positional Encoding**: Allows model to understand token order; simple learnable embeddings used here
- **Residual Connections**: Enables training of deep networks by allowing gradients to flow directly
- **Layer Normalization**: Stabilizes training and improves convergence
- **Scaled Attention**: Division by √(head_size) prevents attention weights from saturating

## Educational Value

This notebook is designed for:
- Understanding transformer internals
- Learning PyTorch deep learning workflows
- Seeing how complex models emerge from basic components
- Experimenting with hyperparameters and architectural changes

## Next Steps for Enhancement

- Experiment with different hyperparameter combinations
- Implement multi-query or grouped-query attention
- Add validation metrics (perplexity, etc.)
- Try different datasets
- Implement checkpoint saving/loading
- Add beam search for text generation

## References

- "Attention Is All You Need" (Vaswani et al., 2017)
- Inspired by Andrej Karpathy's makemore project
- TinyShakespeare dataset from karpathy/char-rnn

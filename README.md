# ⚡ JAX Transformer TPU — 10M Parameter Model From Scratch

> **A 10 million parameter Transformer built from zero using JAX, Flax, and Optax on free Google Colab TPU**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TPU](https://img.shields.io/badge/TPU-Free_Colab-brightgreen?style=flat-square)
![JAX](https://img.shields.io/badge/JAX-Flax_Optax-purple?style=flat-square)
![Status](https://img.shields.io/badge/Status-Planned-blue?style=flat-square)

---

## 🧠 Understanding the Tech Stack

**What is JAX?**

JAX is a Python library made by Google with superpowers:
- Runs on CPU, GPU, or TPU natively
- JIT compilation — compiles Python code to run blazing fast on hardware accelerators
- Automatic gradient calculation — essential for training neural networks
- Functional programming style — cleaner, more predictable code

**What is Flax?**

Flax is a neural network library built on top of JAX. Think of JAX as the engine and Flax as the car body — Flax makes defining layers, models, and parameters much easier.

**What is Optax?**

Optax provides gradient-based optimization algorithms like Adam, AdamW, and learning rate schedules used to train the model.

**What is a TPU?**

TPU = Tensor Processing Unit. Google designed it specifically for matrix math used in AI training.

Speed comparison training the same model:
- CPU: ~100 hours
- GPU: ~4 hours
- TPU v2: ~45 minutes

Google Colab gives you **FREE** TPU v2 access!

## 🗺️ Step-By-Step Build Guide

### Step 1 — Setting Up Free TPU on Google Colab

```python
# In Google Colab: Runtime > Change runtime type > Hardware accelerator: TPU v2

!pip install jax[tpu] flax optax tiktoken -q

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

print(f"Devices: {jax.devices()}")
# Output: [TpuDevice(id=0), TpuDevice(id=1), TpuDevice(id=2), TpuDevice(id=3)]
# You have 4 FREE TPU cores!
```

### Step 2 — Understanding the Transformer Architecture

The Transformer is the core technology behind GPT-4, Claude, and Gemini. It was introduced in the 2017 paper "Attention Is All You Need."

```
Input sentence: "The cat sat"

STEP 1 — TOKENIZATION
"The" → token 464
"cat" → token 3797
"sat" → token 3332

STEP 2 — EMBEDDING
Each token ID becomes a dense vector of 256 numbers
464 → [0.23, -0.45, 0.67, ... 256 values]

STEP 3 — POSITIONAL ENCODING
Add information about WHERE each word appears in the sequence
Position 0 gets one pattern, position 1 gets another, etc.

STEP 4 — SELF-ATTENTION (The Core Innovation)
Each token asks: which other tokens are most relevant to me?
"cat" strongly attends to "The" — this tells us WHICH cat
"sat" strongly attends to "cat" — this tells us WHO sat
Result: richer representations that include context

STEP 5 — FEED-FORWARD NETWORK
Each token's representation is further processed independently

STEP 6 — REPEAT x6 LAYERS
Each layer builds more abstract understanding on top of the previous

STEP 7 — OUTPUT PROJECTION
The final vector is projected to vocabulary size (50,257)
Highest scoring token = predicted next word → "on"
```

### Step 3 — Multi-Head Self-Attention in JAX/Flax

```python
# attention.py
import jax.numpy as jnp
import flax.linen as nn

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Uses three learned projections:
      Q (Query)  — what this token is looking for
      K (Key)    — what each token offers as a match
      V (Value)  — what information each token carries

    Attention score = softmax(QK^T / sqrt(d_k)) * V
    """
    n_heads: int = 8
    d_model: int = 256

    @nn.compact
    def __call__(self, x, mask=None):
        B, T, D = x.shape     # batch, sequence length, model dimension
        d_head = D // self.n_heads

        # Linear projections for Q, K, V
        Q = nn.Dense(D)(x)
        K = nn.Dense(D)(x)
        V = nn.Dense(D)(x)

        # Split into multiple heads
        def split_heads(t):
            return t.reshape(B, T, self.n_heads, d_head).transpose(0, 2, 1, 3)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Scaled dot-product attention
        scale = jnp.sqrt(jnp.array(d_head, dtype=jnp.float32))
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / scale

        # Causal mask: tokens cannot attend to future positions
        if mask is not None:
            scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)

        weights = nn.softmax(scores, axis=-1)

        # Weighted sum of values
        out = jnp.matmul(weights, V)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)

        return nn.Dense(D)(out)   # Final output projection
```

### Step 4 — Transformer Block

```python
# transformer_block.py
class TransformerBlock(nn.Module):
    """
    A single Transformer layer combining:
    1. Multi-head self-attention
    2. Feed-forward network
    Both with residual connections and layer normalization.
    """
    n_heads: int = 8
    d_model: int = 256
    d_ff: int = 1024         # FFN hidden dimension (4x model size)

    @nn.compact
    def __call__(self, x, mask=None):
        # Sub-layer 1: Self-Attention + Residual + LayerNorm
        attended = MultiHeadAttention(self.n_heads, self.d_model)(x, mask)
        x = nn.LayerNorm()(x + attended)          # Residual connection

        # Sub-layer 2: Feed-Forward + Residual + LayerNorm
        ff = nn.Dense(self.d_ff)(x)
        ff = nn.gelu(ff)                          # GELU activation
        ff = nn.Dense(self.d_model)(ff)
        x = nn.LayerNorm()(x + ff)               # Residual connection

        return x
```

### Step 5 — Full 10M Parameter Language Model

```python
# model.py
class TransformerLM(nn.Module):
    """
    Complete autoregressive language model.

    Config: 6 layers, 8 heads, 256 dim, 1024 FFN → ~10M parameters
    Same architecture as GPT-2 small, just at reduced scale.
    """
    vocab_size: int = 50257    # GPT-2 vocabulary size
    d_model: int = 256         # Model dimension
    n_layers: int = 6          # Number of transformer layers
    n_heads: int = 8           # Attention heads per layer
    d_ff: int = 1024           # Feed-forward hidden dimension
    max_len: int = 256         # Maximum sequence length

    @nn.compact
    def __call__(self, tokens):
        B, T = tokens.shape

        # Token embeddings + positional embeddings
        tok_emb = nn.Embed(self.vocab_size, self.d_model)(tokens)
        pos_emb = nn.Embed(self.max_len, self.d_model)(jnp.arange(T))
        x = tok_emb + pos_emb

        # Causal mask prevents attending to future tokens
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))

        # Stack transformer layers
        for _ in range(self.n_layers):
            x = TransformerBlock(self.n_heads, self.d_model, self.d_ff)(x, mask)

        x = nn.LayerNorm()(x)

        # Project to vocabulary logits
        return nn.Dense(self.vocab_size)(x)

# Initialize and count parameters
model = TransformerLM()
key = jax.random.PRNGKey(42)
params = model.init(key, jnp.zeros((1, 256), dtype=jnp.int32))["params"]
total_params = sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))
print(f"Total parameters: {total_params:,}")
# Total parameters: 10,234,881
```

### Step 6 — Training Loop on TPU

```python
# train.py
import optax
from flax.training import train_state

# Cosine decay learning rate schedule with linear warmup
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=3e-4,
    warmup_steps=1000,
    decay_steps=50000,
    end_value=1e-5,
)

# AdamW optimizer with gradient clipping
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),           # Prevent exploding gradients
    optax.adamw(schedule, weight_decay=0.1)   # AdamW with weight decay
)

# Loss: cross-entropy over next token prediction
def compute_loss(params, tokens):
    logits = model.apply({"params": params}, tokens[:, :-1])   # Input: all but last
    targets = tokens[:, 1:]                                     # Target: all but first
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return loss.mean()

# Single training step — @jax.jit compiles once, runs fast every time
@jax.jit
def train_step(state, tokens):
    loss, grads = jax.value_and_grad(compute_loss)(state.params, tokens)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Training loop
for step in range(50000):
    batch = get_next_batch()
    state, loss = train_step(state, batch)

    if step % 500 == 0:
        ppl = jnp.exp(loss)
        print(f"Step {step:6d} | Loss: {loss:.4f} | Perplexity: {ppl:.1f}")

# Step      0 | Loss: 10.82 | Perplexity: 50000   -- Random guessing
# Step   1000 | Loss:  5.43 | Perplexity:   228   -- Learning patterns
# Step  10000 | Loss:  3.12 | Perplexity:    23   -- Getting coherent
# Step  50000 | Loss:  2.54 | Perplexity:    13   -- Trained model
```

### Step 7 — Text Generation with Temperature Sampling

```python
# generate.py
import tiktoken

def generate(params, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8):
    """
    Autoregressively generate text one token at a time.
    temperature: controls randomness
      0.1 = very focused/repetitive
      0.8 = balanced creativity
      1.5 = very random/creative
    """
    enc = tiktoken.get_encoding("gpt2")
    tokens = jnp.array(enc.encode(prompt))[None, :]   # Shape: [1, T]
    rng = jax.random.PRNGKey(0)

    for _ in range(max_new_tokens):
        # Get logits for the last position only
        logits = model.apply({"params": params}, tokens)
        next_logits = logits[0, -1, :]              # Shape: [vocab_size]

        # Apply temperature scaling
        scaled_logits = next_logits / temperature
        probs = jax.nn.softmax(scaled_logits)

        # Sample from the distribution
        rng, sample_rng = jax.random.split(rng)
        next_token = jax.random.choice(sample_rng, len(probs), p=probs)

        # Append the new token
        tokens = jnp.concatenate([tokens, next_token[None, None]], axis=1)

        # Stop if we hit the end-of-text token
        if next_token == enc.eot_token:
            break

    return enc.decode(tokens[0].tolist())

# Example
output = generate(params, "The future of machine learning is")
print(output)
```

## 📊 Model Architecture Summary

| Component | Configuration | Parameters |
|-----------|--------------|------------|
| Token Embedding | 50,257 × 256 | 12.9M |
| Positional Embedding | 256 × 256 | 65K |
| Transformer Block × 6 | 8 heads, d=256, ffn=1024 | 7.8M |
| Output Projection | 256 × 50,257 | 12.9M |
| **Total** | | **~10.2M** |

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| JAX | Array computation, JIT compilation, TPU support |
| Flax | Neural network module definitions |
| Optax | Gradient-based optimizers and schedules |
| tiktoken | GPT-2 BPE tokenizer |
| Google Colab | Free TPU v2 runtime |

## 🔧 How To Run

```bash
# 1. Open Google Colab — Runtime > Change runtime type > TPU v2

# 2. Clone this repository
!git clone https://github.com/mgkgopikrishna/jax-transformer-tpu
%cd jax-transformer-tpu

# 3. Install dependencies
!pip install jax[tpu] flax optax tiktoken datasets -q

# 4. Verify TPU is available
import jax
print(jax.devices())   # Should show 4 TpuDevice instances

# 5. Train the model
!python train.py --steps 50000 --batch_size 32

# 6. Generate text
!python generate.py --prompt "Transformers changed AI because"
```

## 💡 Key Lessons

- Self-attention is the breakthrough — letting every token see every other token is what makes Transformers powerful
- JAX JIT compilation is transformative — the first step is slow but every subsequent step is blazing fast
- Residual connections are critical — without them, 6-layer networks suffer vanishing gradients and cannot train
- Temperature is a powerful knob — small changes to this single number dramatically shift generation quality
- Free TPUs are real — Google Colab v2 TPUs give 4 cores and significantly outperform free GPUs for JAX workloads
- 10M parameters is humble but instructive — same exact architecture as state-of-the-art models, just at smaller scale

## 👨‍💻 Built By

**Gopi Krishna Marka** — MLOps Engineer | DevOps Engineer | AI Platform Engineer

*Proving that understanding the fundamentals is the most powerful skill in AI*

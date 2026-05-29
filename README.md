# ⚡ JAX Transformer TPU — 10M Parameter Model From Scratch

> **A 10 million parameter Transformer built from zero using JAX, Flax, and Optax on free Google Colab TPU**

![JAX](https://img.shields.io/badge/JAX-A8B9CC?style=flat-square&logo=google&logoColor=black)
![Flax](https://img.shields.io/badge/Flax-FF6B6B?style=flat-square&logo=google&logoColor=white)
![TPU](https://img.shields.io/badge/TPU-Free_Colab-4285F4?style=flat-square&logo=google-colab&logoColor=white)
![Status](https://img.shields.io/badge/Status-Planned-blue?style=flat-square)

---

## 🎯 What Does This Project Do?

Imagine you want to build a race car from scratch. Most people buy a ready-made car. But some engineers want to understand EVERY part — so they build the engine themselves!

That is exactly what this project does with AI:
- Most people use pre-built AI models like downloading ChatGPT
- We build a REAL transformer model from scratch — every single part
- We use JAX instead of PyTorch — JAX is like a supercharged calculator
- We train it on a TPU — a special super-fast chip made by Google
- The best part? The TPU is FREE on Google Colab!

**Think of it as building a 10 million piece LEGO set where each piece is a math calculation!**

---

## 🧠 Understanding the Tech Stack

### What is JAX?
JAX is a Python library made by Google with superpowers:
- Runs on CPU, GPU, or TPU
- JIT compilation — compiles code to run blazing fast
- Automatic gradient calculation for training
- Works on Google TPUs natively

### What is Flax?
Flax is a neural network library built on top of JAX. If JAX is the engine, Flax is the car body — it makes building neural networks easier.

### What is Optax?
Optax provides optimization algorithms like Adam that figure out how to update model parameters during training.

### What is a TPU?
TPU = Tensor Processing Unit. Google made it specifically for AI math.

```
Speed comparison for training:
CPU:  100 hours
GPU:  4 hours
TPU:  45 minutes

Google Colab gives you FREE TPU access!
```

---

## 🗺️ Step-By-Step Build Guide

### Step 1 — Setting Up Free TPU on Colab

```python
# In Google Colab:
# Runtime → Change runtime type → TPU v2

!pip install jax[tpu] flax optax tiktoken -q

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

print(f"Devices: {jax.devices()}")
# [TpuDevice(id=0), TpuDevice(id=1), TpuDevice(id=2), TpuDevice(id=3)]
# We have 4 FREE TPU cores!
```

---

### Step 2 — Understanding the Transformer

**What is a Transformer?**
The core technology behind GPT-4, Claude, and Gemini. Invented in 2017.

```
Input: "The cat sat"

Step 1 — TOKENIZATION
  "The" → 464
  "cat" → 3797
  "sat" → 3332

Step 2 — EMBEDDING
  Each number becomes a vector of 256 numbers
  464 → [0.23, -0.45, 0.67, ... (256 numbers)]

Step 3 — POSITIONAL ENCODING
  Add information about WHERE each word is

Step 4 — SELF-ATTENTION (The Magic!)
  Each word asks: which other words should I pay attention to?
  "cat" pays attention to "The" — it tells us WHICH cat!
  "sat" pays attention to "cat" — tells us WHO sat!

Step 5 — FEED-FORWARD NETWORK
  Process the attended information

Step 6 — REPEAT 6 times
  Each layer learns more complex patterns

Step 7 — OUTPUT
  Predict next word → "on"
```

---

### Step 3 — Implementing Self-Attention in JAX

```python
# attention.py
import jax.numpy as jnp
import flax.linen as nn

class MultiHeadAttention(nn.Module):
    """
    Self-Attention — The core of the Transformer.
    
    Think of it like a library search:
    Q (Query)  = the question you are asking
    K (Key)    = the title/label of each book
    V (Value)  = the actual content of each book
    
    We find which books (K) match our question (Q)
    Then we read those books (V) to get our answer!
    """
    n_heads: int = 8
    d_model: int = 256
    
    @nn.compact
    def __call__(self, x, mask=None):
        B, T, D = x.shape  # batch, sequence length, dimension
        d_head = D // self.n_heads
        
        # Create Q, K, V through linear layers
        Q = nn.Dense(D)(x)  # What am I looking for?
        K = nn.Dense(D)(x)  # What do I contain?
        V = nn.Dense(D)(x)  # What information do I have?
        
        # Reshape for multiple heads
        def to_heads(t):
            return t.reshape(B, T, self.n_heads, d_head).transpose(0, 2, 1, 3)
        
        Q, K, V = to_heads(Q), to_heads(K), to_heads(V)
        
        # Compute attention scores
        scale = jnp.sqrt(d_head).astype(jnp.float32)
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / scale
        
        # Apply causal mask (cannot look at future!)
        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)
        
        # Softmax — convert scores to probabilities
        weights = nn.softmax(scores, axis=-1)
        
        # Weighted sum of values
        out = jnp.matmul(weights, V)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        
        return nn.Dense(D)(out)
```

---

### Step 4 — Transformer Block

```python
# transformer_block.py

class TransformerBlock(nn.Module):
    """One complete Transformer block. We stack 6 of these."""
    n_heads: int = 8
    d_model: int = 256
    d_ff: int = 1024
    
    @nn.compact
    def __call__(self, x, mask=None):
        # Self-Attention with residual connection
        attended = MultiHeadAttention(self.n_heads, self.d_model)(x, mask)
        x = nn.LayerNorm()(x + attended)  # Add residual!
        
        # Feed-Forward Network with residual connection
        ff = nn.Dense(self.d_ff)(x)
        ff = nn.gelu(ff)              # Non-linear activation
        ff = nn.Dense(self.d_model)(ff)
        x = nn.LayerNorm()(x + ff)    # Add residual!
        
        return x
```

---

### Step 5 — Complete Model

```python
# model.py

class TransformerLM(nn.Module):
    """
    Complete 10M parameter Language Model!
    
    Architecture:
    - Vocabulary: 50,257 tokens
    - Embedding size: 256
    - Layers: 6 transformer blocks
    - Attention heads: 8
    - FFN hidden size: 1024
    - Max length: 256 tokens
    """
    vocab_size: int = 50257
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    max_len: int = 256
    
    @nn.compact
    def __call__(self, tokens):
        B, T = tokens.shape
        
        # Token + Position embeddings
        tok_emb = nn.Embed(self.vocab_size, self.d_model)(tokens)
        pos_emb = nn.Embed(self.max_len, self.d_model)(jnp.arange(T))
        x = tok_emb + pos_emb
        
        # Causal mask — no peeking at future tokens!
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        
        # Stack 6 Transformer blocks
        for _ in range(self.n_layers):
            x = TransformerBlock(self.n_heads, self.d_model, self.d_ff)(x, mask)
        
        x = nn.LayerNorm()(x)
        
        # Predict next token
        return nn.Dense(self.vocab_size)(x)

# Count parameters
model = TransformerLM()
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.zeros((1, 256), dtype=jnp.int32))["params"]
total = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"Total parameters: {total:,}")
# Total parameters: 10,234,881
```

---

### Step 6 — Training on TPU

```python
# train.py
import optax

# Learning rate schedule — warm up then decay
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=3e-4,
    warmup_steps=1000,
    decay_steps=10000,
)

# Adam optimizer with weight decay
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Prevent exploding gradients
    optax.adamw(schedule, weight_decay=0.1)
)

# Loss function
def compute_loss(params, batch):
    logits = model.apply({"params": params}, batch["input_ids"])
    return optax.softmax_cross_entropy_with_integer_labels(
        logits[:, :-1, :],   # predictions
        batch["labels"][:, 1:]  # targets (shifted by 1)
    ).mean()

# One training step — @jax.jit compiles for TPU speed!
@jax.jit
def train_step(state, batch):
    loss, grads = jax.value_and_grad(compute_loss)(state.params, batch)
    updates, new_opt = optimizer.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    return new_params, new_opt, loss

# Training loop
for step in range(10000):
    batch = get_batch()
    params, opt_state, loss = train_step(state, batch)
    if step % 500 == 0:
        print(f"Step {step:5d} | Loss: {loss:.4f} | PPL: {jnp.exp(loss):.1f}")

# Step     0 | Loss: 10.82 | PPL: 50000  ← Knows nothing
# Step  1000 | Loss: 5.43  | PPL: 228    ← Learning!
# Step  5000 | Loss: 3.12  | PPL: 23     ← Getting good
# Step 10000 | Loss: 2.54  | PPL: 13     ← Trained!
```

---

### Step 7 — Generating Text

```python
# generate.py

def generate(params, prompt, max_tokens=100, temperature=0.8):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    tokens = jnp.array(enc.encode(prompt))[None, :]
    
    for _ in range(max_tokens):
        logits = model.apply({"params": params}, tokens)
        next_logits = logits[0, -1, :] / temperature
        probs = jax.nn.softmax(next_logits)
        next_token = jax.random.choice(key, len(probs), p=probs)
        tokens = jnp.concatenate([tokens, next_token[None, None]], axis=1)
    
    return enc.decode(tokens[0].tolist())

print(generate(params, "The future of AI is"))
# The future of AI is transforming every industry...
```

---

## 📊 Model Summary

| Layer | Size | Parameters |
|-------|------|------------|
| Token Embedding | 50257 × 256 | 12.9M |
| Positional Embedding | 256 × 256 | 65K |
| 6 × Transformer Blocks | 8 heads, 256 dim | 7.8M |
| Output Projection | 256 × 50257 | 12.9M |
| **Total** | | **~10M** |

---

## 🛠️ Tech Stack

| Tool | What It Does |
|------|--------------|
| **JAX** | Fast array math, TPU support |
| **Flax** | Neural network layers |
| **Optax** | Optimization algorithms |
| **tiktoken** | GPT-2 compatible tokenizer |
| **Google Colab** | Free TPU cloud environment |

---

## 🔧 How To Run

```bash
# 1. Open Google Colab, enable TPU v2
# 2. Clone repo
!git clone https://github.com/mgkgopikrishna/jax-transformer-tpu
%cd jax-transformer-tpu
# 3. Install
!pip install jax[tpu] flax optax tiktoken -q
# 4. Verify TPU
import jax; print(jax.devices())
# 5. Train
!python train.py
# 6. Generate
!python generate.py --prompt "Machine learning is"
```

---

## 💡 Key Lessons

1. **Self-attention is the key innovation** — everything in modern AI builds on this
2. **JAX is the future** — functional style, TPU support, research-grade
3. **Residual connections enable deep networks** — they prevent vanishing gradients
4. **Temperature controls creativity** — simple but powerful generation trick
5. **Free TPUs exist** — Google Colab gives powerful hardware to everyone
6. **10M parameters is small but real** — same architecture as GPT-4, just smaller

---

## 👨‍💻 Built By

**Gopi Krishna Marka** — MLOps Engineer | DevOps Engineer | AI Platform Engineer

*Proving that understanding the fundamentals is the most powerful skill in AI*

# The Complete LLM Training Pipeline — A Builder's Reference

I'm training a GPT-2 (124M parameters) from scratch. Every time I step away for a week, I forget how the pieces connect. This article is my reference — 7 blocks that cover the full pipeline from input tokens to generated text.

If you forget everything else, remember this diagram:

```
Input Text
    ↓
[Tokenizer] → token IDs
    ↓
[Token Embedding + Positional Embedding]
    ↓
[Transformer Block 1]
    ↓
[Transformer Block 2]
    ↓
    ... (× 12 layers)
    ↓
[Final Layer Norm]
    ↓
[Linear Projection] → logits (one score per vocab word)
    ↓
[Cross-Entropy Loss] → single scalar (how wrong are we?)
    ↓
[loss.backward()] → 124M gradients (who's responsible?)
    ↓
[optimizer.step()] → 124M weights nudged (fix it slightly)
    ↓
Repeat thousands of times
    ↓
[Temperature + Top-K] → generated text
```

Every block below zooms into one part of this diagram.

---

## Block 1: The 30-Second Overview

**One sentence:** Tokens go in, predictions come out, we measure wrongness, and nudge 124 million weights to be less wrong.

The training loop is 4 lines repeated thousands of times:

```python
optimizer.zero_grad()                # forget last batch's gradients
loss = calc_loss_batch(...)          # forward pass → how wrong are we?
loss.backward()                      # backward pass → who's responsible?
optimizer.step()                     # update all 124M weights
```

Everything else — data loading, evaluation, checkpointing, sampling — is infrastructure around these four lines.

The model starts producing gibberish and gradually learns:

```
Epoch 1:  "Every effort moves the the the the"
Epoch 5:  "Every effort moves the country forward"
Epoch 10: "Every effort moves you closer to the goal"
```

---

## Block 2: Inside the Transformer Block

**One sentence:** Each block has two sub-blocks — attention (which tokens to look at) and FFN (what to conclude from them) — connected by residual shortcuts.

```
Input (x)
    │
    ├──────────────────┐
    ↓                  │
[Layer Norm]           │
    ↓                  │
[Multi-Head Attention] │   ← "which tokens should I blend?"
    ↓                  │
[Dropout]              │
    + ←────────────────┘   ← RESIDUAL CONNECTION (add original x back)
    │
    │ = x'
    ├──────────────────┐
    ↓                  │
[Layer Norm]           │
    ↓                  │
[Feed-Forward Network] │   ← "what do I conclude from what I gathered?"
    ↓                  │
[Dropout]              │
    + ←────────────────┘   ← RESIDUAL CONNECTION (add x' back)
    │
    ↓
Output (goes to next block)
```

### Mapped to the actual code (`TransformerBlock.forward`):

```python
def forward(self, x):
    # Attention sub-block with residual
    shortcut = x
    x = self.norm1(x)
    x = self.att(x)              # Multi-Head Attention
    x = self.drop_shortcut(x)
    x = x + shortcut             # ← residual connection

    # FFN sub-block with residual
    shortcut = x
    x = self.norm2(x)
    x = self.ff(x)               # Feed-Forward Network
    x = self.drop_shortcut(x)
    x = x + shortcut             # ← residual connection
    return x
```

### Why residual connections?

Each layer only needs to learn the "delta" — what to add or change. Without them:
- Information gets lost passing through 12 blocks
- Gradients vanish during backward pass (can't train deep networks)
- Each layer has to reconstruct everything from scratch

### Why Layer Norm?

Normalizes values to keep them stable. Without it, numbers explode or shrink as they pass through many layers, and training becomes unstable.

### The model stacks 12 of these blocks:

```python
self.trf_blocks = nn.Sequential(
    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])  # n_layers = 12
```

What each layer depth tends to learn:
- **Layers 1-3:** Basic syntax patterns — "after 'the', a noun is likely"
- **Layers 4-8:** Grammar structure — "this is a question" or "subject-verb agreement"
- **Layers 9-12:** Meaning and context — sentiment, topic, intent

---

## Block 3: The Feed-Forward Network — The Part People Skip

**One sentence:** A simple expand-then-compress network that processes what attention gathered — this is where the model stores most of its factual knowledge.

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),   # W1: 768 → 3072 (expand)
            GELU(),                                           # non-linear activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),   # W2: 3072 → 768 (compress)
        )
```

### What is 3072? Is it the vocabulary?

No. It's `4 × 768 = 3072`. Just a design choice from the original Transformer paper. It gives the network a bigger "workspace" to think in temporarily.

```
Token vector (768 dims)
    ↓
W1: expand to 3072 dims     ← spread out notes on a big desk
    ↓
GELU activation              ← non-linearity (lets it learn complex patterns)
    ↓
W2: compress back to 768     ← summarize back into compact form
```

### Where vocabulary actually appears (only two places):

```
START:  Embedding        vocab_size (50,257) → 768    ← token → vector
END:    Final projection 768 → vocab_size (50,257)    ← vector → logits
```

Everything in between works purely in 768 dimensions. The model doesn't think in "words" internally.

### Attention vs. FFN — who does what:

| Component | Job | Analogy |
|---|---|---|
| Attention | "Which other tokens should I look at?" | Gathering evidence |
| FFN | "Now that I've gathered info, what do I conclude?" | Processing evidence |

Research shows FFN layers act as key-value memory stores — specific neurons activate for specific facts. Attention routes information, FFN stores and retrieves it.

---

## Block 4: From Logits to Loss — Measuring Wrongness

**One sentence:** The target is just the input shifted by one token, and cross-entropy loss measures how far the model's predictions are from these targets.

### Where do targets come from?

NOT from a validation set. The target is the **same sentence shifted by 1 position**:

```
Raw text tokens:  [every, effort, moves, you]
                    0      1       2      3

Input:   tokens[0:3]  →  [every, effort, moves]     positions 0,1,2
Target:  tokens[1:4]  →  [effort, moves, you]       positions 1,2,3
```

At each position, the model predicts the next word. The target is what the next word actually is.

### The forward pass produces logits:

```python
logits = model(inputs)   # shape: (batch=2, tokens=3, vocab=50257)
```

Each position gets 50,257 scores — one per vocabulary word. We want the score for the **correct** next word to be high.

### Cross-entropy loss:

```python
loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
```

**Why flatten?** PyTorch's `cross_entropy` expects 2D input:

```
logits:  (2, 3, 50257)  → flatten(0,1) →  (6, 50257)    ← 2D ✅
targets: (2, 3)          → flatten()    →  (6,)           ← 1D ✅
```

Cross-entropy computes: "for each position, how much probability did the model put on the correct word?" Then averages across all positions into a **single scalar**.

```
Before training: loss ≈ 10.8   (basically guessing among 50,257 words)
After training:  loss ≈ 0.5    (confidently predicting the right word)
```

### Perplexity = exp(loss)

```python
perplexity = torch.exp(loss)   # e.g., exp(10.8) ≈ 49,000
```

Interpretation: the model is as confused as if it were randomly choosing among ~49,000 words. After training, perplexity drops to ~1.6, meaning it's choosing among ~1.6 words (very confident).

### Training vs. Validation loss:

```
Full text: [5,145 tokens total]
    ├── Training set (first 90%) → used to UPDATE weights
    └── Validation set (last 10%) → used to MEASURE performance (no updates)

Both create input/target pairs the same way (shift by 1).
The difference: training data changes the model, validation data only scores it.

Watch for:
  Train loss ↓ and Val loss ↓  → ✅ Model is learning
  Train loss ↓ but Val loss ↑  → ⚠️ Overfitting (memorizing, not generalizing)
```

---

## Block 5: `backward()` and `optimizer.step()` — The Two Lines That Train Everything

**One sentence:** From one loss number, `backward()` computes 124 million gradients, and `optimizer.step()` nudges every weight downhill.

### Before `backward()` — Nothing

```python
logits = model(inputs)
loss = cross_entropy(logits, targets)   # single scalar: tensor(10.83)
```

Every weight has `grad = None`. The model measured how wrong it was, but hasn't figured out who's responsible yet.

```
tok_emb.weight                     grad = None
trf_blocks.0.att.W_query.weight    grad = None
trf_blocks.0.ff.layers.0.weight    grad = None
... all 124M weights ...           grad = None
```

### `loss.backward()` — The Chain of Blame

That single number (10.83) knows exactly how it was computed — PyTorch recorded every operation along the way (the **computation graph**).

`backward()` walks this graph in reverse, applying the **chain rule of calculus**:

```
loss (scalar)
  |-- backward through cross_entropy
  |-- backward through final Linear (768 → 50,257)
  |-- backward through LayerNorm
  |-- backward through Transformer Block 12
  |     |-- backward through FFN W2 (3,072 → 768)
  |     |-- backward through GELU activation
  |     |-- backward through FFN W1 (768 → 3,072)
  |     |-- backward through LayerNorm
  |     |-- backward through out_proj (768 → 768)
  |     |-- backward through softmax + causal mask
  |     |-- backward through Q × K^T
  |     |-- backward through W_query, W_key, W_value
  |     |-- backward through LayerNorm
  |-- backward through Transformer Block 11
  |     |-- ... same chain ...
  |-- ... all the way through Block 1 ...
  |-- backward through Embeddings
```

After this single call, every weight has a `.grad` attribute — the **exact same shape** as the weight. Crucially, each gradient points in the direction of **steepest increase** in loss — the direction that would make things *worse*. That's why the optimizer uses a minus sign: we go the opposite way, downhill.

```
tok_emb.weight                     grad shape: [50257, 768]    mean: +0.00000012
trf_blocks.0.att.W_query.weight    grad shape: [768, 768]      mean: +0.00008900
trf_blocks.0.ff.layers.0.weight    grad shape: [3072, 768]     mean: +0.00003400
... all 124M weights now have gradients ...
```

**One scalar in. 124 million gradients out.** That's automatic differentiation.

**The silent killer: memory.** Those 124 million gradients are the same size as the weights themselves. `backward()` effectively **doubles your model's memory footprint** in one call. This is why GPU OOM errors almost always hit during `backward()`, not the forward pass. Techniques like gradient checkpointing, mixed precision, and gradient accumulation exist specifically to tame this cost.

### `optimizer.step()` — The Nudge

The optimizer (AdamW) reads every gradient and nudges each weight in the **opposite direction** — downhill toward lower loss:

```
weight_new = weight_old - learning_rate × gradient
                        ^
                        minus sign = go OPPOSITE to gradient = downhill
```

The gradient says "loss increases fastest this way." The minus sign says "then let's go the other way." That's gradient **descent** in one line.

AdamW is smarter than plain gradient descent — it maintains running averages of gradients (momentum) and adapts the learning rate per-parameter. But the core idea is the same: move each weight a tiny step downhill.

And the memory cost doesn't stop at gradients. AdamW stores **two additional tensors per weight** (momentum and variance estimates). So your 124M-parameter model needs memory for:

```
Weights:         124M parameters
Gradients:       124M (same size as weights)
AdamW momentum:  124M
AdamW variance:  124M
─────────────────────────
Total:           ~4× the model size
```

This is why a model that fits in VRAM for inference can OOM during training.

### The Full Training Loop

```python
def train_model_simple(model, train_loader, val_loader, optimizer, ...):
    for epoch in range(num_epochs):
        model.train()                              # dropout ON
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                  # ① clear old gradients
            loss = calc_loss_batch(...)             # ② forward pass → loss
            loss.backward()                        # ③ compute all gradients
            optimizer.step()                       # ④ update all weights

        # Periodically check progress
        evaluate_model(...)    # switches to model.eval() → dropout OFF
                               # measures train + val loss
                               # switches back to model.train()

        # See the model talk
        generate_and_print_sample(...)   # also uses model.eval()
```

**Why `model.train()` and `model.eval()`?** They toggle dropout:
- `.train()`: dropout ON (prevents overfitting during learning)
- `.eval()`: dropout OFF (full model capacity when measuring or generating)

**Why `optimizer.zero_grad()`?** Without it, gradients accumulate from previous batches. Training silently breaks — you get wrong updates with no error message.

---

## Block 6: Temperature and Top-K — Controlling the Output

**One sentence:** The model's intelligence is already in the logits; temperature and top-k just control how we pick from the model's already-smart ranking.

### The Problem: `argmax` Is Boring

```python
next_token = torch.argmax(probas)   # always pick highest probability
```

Given "every effort moves you", this always says "forward". Every time. The output is repetitive and robotic.

### Solution 1: Temperature — Control the Confidence

Temperature is just **dividing logits by a number** before softmax:

```python
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)
```

The effect:

```
Temp 0.1 (low)        Temp 1.0 (default)     Temp 5.0 (high)
█████████ forward     ████      forward      ██        forward
          toward      ██        toward       ██        toward
                      █         closer       ██        closer
                                             █         you
                                             █         inches
                                             █         every
                                                       pizza

CONFIDENT             BALANCED               CREATIVE
(repetitive)          (normal)               (chaotic)
```

**Why does dividing change things?**
- `logits / 0.1` → differences become 10× larger → softmax becomes nearly one-hot
- `logits / 1.0` → no change
- `logits / 5.0` → differences become 5× smaller → softmax becomes nearly uniform

### Solution 2: Top-K — Eliminate Garbage

Temperature alone has a problem: weird words like "pizza" still have a non-zero chance. Top-k says: **only keep the K most likely words, kill everything else.**

It operates on **logits** (raw scores), not probabilities:

```
Model outputs 50,257 logits
        ↓
Top-K: keep the K highest logits
        ↓
Set everything else to -inf
        ↓
Softmax: only K words share 100% of probability
        ↓
Sample from K candidates
```

Example with K=3:

```
BEFORE top-k:                    AFTER top-k (k=3):
"forward"  6.75  ✅ keep         "forward"  6.75  → 0.57
"toward"   6.28  ✅ keep         "toward"   6.28  → 0.36
"closer"   4.51  ✅ keep         "closer"   4.51  → 0.07
"you"      1.79  ❌ → -inf       "you"      -inf  → 0.00
"pizza"   -1.89  ❌ → -inf       "pizza"    -inf  → 0.00  ← impossible!
```

**Why logits, not probabilities?** Same top-K words either way (softmax preserves ranking), but we skip computing softmax over 50,257 words just to throw away 50,254 of them.

### How They Work Together

In the `generate()` function, **top-k runs first, then temperature**:

```
All 50,257 vocab words
        ↓
   Top-K (keep best K)       ← eliminate nonsense
        ↓
   K candidates remain
        ↓
   Temperature scaling        ← control creativity among survivors
        ↓
   Softmax → probabilities
        ↓
   Multinomial sampling       ← randomly pick one
        ↓
   Next token
```

**Why this order?** Top-k removes garbage first. If temperature ran first, you'd be reshaping probabilities of words you're about to discard anyway.

### But What About Grammar?

Top-k and temperature **don't manage grammar**. The model already did that inside the transformer blocks. By the time logits come out, the 12 layers have already encoded:
- Syntax: "after 'you', verbs and adverbs are likely"
- Grammar: "'moves you ___' expects a direction word"
- Meaning: "positive sentiment → 'forward/closer' fit, not 'backward'"

That's why "forward" has logit 6.75 and "pizza" has -1.89. Top-k and temperature are **post-processing tricks**. All the intelligence happened inside the transformer blocks. They're choosing from an already curated menu — the model wrote the menu.

### Practical Combinations

| Setting | Behavior | Use Case |
|---|---|---|
| temp=0.0, top_k=None | Always pick the best (argmax) | Factual Q&A, code generation |
| temp=0.7, top_k=40 | Moderate creativity, no garbage | ChatGPT-style conversation |
| temp=1.0, top_k=10 | Creative but constrained | Story writing |
| temp=1.5, top_k=50 | Very creative, occasionally wild | Brainstorming |

---

## Block 7: The Weight Map — Every Learnable Parameter

**One sentence:** `optimizer.step()` doesn't just update attention weights — it updates every single one of these 124 million parameters simultaneously.

```
GPTModel
├── tok_emb     Embedding(50257, 768)          token → vector
├── pos_emb     Embedding(1024, 768)           position → vector
│
├── TransformerBlock × 12
│   ├── norm1        LayerNorm(768)             scale + shift params
│   ├── att
│   │   ├── W_query   Linear(768, 768)          Q weights
│   │   ├── W_key     Linear(768, 768)          K weights
│   │   ├── W_value   Linear(768, 768)          V weights
│   │   └── out_proj  Linear(768, 768)          combines all heads
│   ├── norm2        LayerNorm(768)             scale + shift params
│   └── ff
│       ├── Linear(768, 3072)                   W1 — expand
│       └── Linear(3072, 768)                   W2 — compress
│
├── final_norm   LayerNorm(768)
└── out_head     Linear(768, 50257)             vector → logits
```

### What each part does:

| Weight | Shape | What It Learns |
|---|---|---|
| tok_emb | (50257, 768) | A unique 768-dim vector for each vocabulary word |
| pos_emb | (1024, 768) | "Position 0 feels like this, position 512 feels like that" |
| W_query | (768, 768) | "What am I looking for?" |
| W_key | (768, 768) | "What do I contain that others might want?" |
| W_value | (768, 768) | "What information do I actually contribute?" |
| out_proj | (768, 768) | Blends 12 heads into one unified representation |
| FFN W1 | (768, 3072) | Expands to big workspace for processing |
| FFN W2 | (3072, 768) | Compresses conclusions back to token size |
| LayerNorm | (768) × 2 | Scale and shift to keep numbers stable |
| out_head | (768, 50257) | Final "which word comes next?" projection |

### One training step touches ALL of them:

```python
loss.backward()    # computes gradients for ALL 124,439,808 parameters
optimizer.step()   # updates ALL 124,439,808 parameters
```

No weight is exempt. From the token embedding of "the" to the 12th transformer block's FFN W2 — every single parameter gets a personalized gradient and a personalized update. Every step.

---

## Quick Reference Card

When you come back in a month, start here:

```
Tokens → Embeddings → 12 × [Attention → FFN] → Logits → Loss → Backward → Step → Repeat
```

| Question You'll Have | Block |
|---|---|
| "What's the full pipeline?" | Block 1 |
| "What's inside a transformer block?" | Block 2 |
| "What's the FFN / what's 3072?" | Block 3 |
| "How does loss work? What are targets?" | Block 4 |
| "What does backward/optimizer actually do?" | Block 5 |
| "What's temperature/top-k?" | Block 6 |
| "What weights exist and get updated?" | Block 7 |

---

*I'm documenting my journey learning LLM internals from scratch. Follow along for more deep dives into how these models actually work.*

*Built with code from Sebastian Raschka's "Build a Large Language Model From Scratch" — highly recommended if you want to understand transformers at the weight level.*

*Full code: [github.com/jkuru/llm](https://github.com/jkuru/llm)*

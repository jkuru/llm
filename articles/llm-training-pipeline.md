# The Complete LLM Training Pipeline — A Builder's Reference

I'm training a GPT-2 (124M parameters) from scratch. Every time I step away for a week, I forget how the pieces connect. This article is my reference — 10 blocks that cover the full pipeline from input tokens to generated text.

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

## Block 2: From Text to Context Vector — The Attention Pipeline

**One sentence:** Text becomes token IDs, token IDs become embeddings, embeddings get Q/K/V projections, and multi-head attention produces context vectors that blend information across tokens.

### Step 1: Text → Token IDs (Tokenization)

The tokenizer (BPE — Byte Pair Encoding) splits text into subword pieces and maps each to an integer:

```
"Every effort moves you"
        ↓ tokenizer.encode()
[6109, 3626, 6100, 345]
   ↑      ↑     ↑    ↑
 "Every" "effort" "moves" "you"
```

These are just row indices into an embedding table. The model never sees the text — only numbers.

```python
tokenizer = tiktoken.get_encoding("gpt2")   # GPT-2's tokenizer
token_ids = tokenizer.encode("Every effort moves you")  # → [6109, 3626, 6100, 345]
```

### Step 2: Token IDs → Embeddings (Lookup, Not Computation)

Each token ID is used to look up a row in a learnable embedding table:

```
Token Embedding Table: (50,257 rows × 768 cols)
                        ↑ one row per vocab word

token ID 6109 ("Every")  → row 6109 → [0.12, -0.34, 0.56, ..., 0.78]  (768 dims)
token ID 3626 ("effort") → row 3626 → [0.45, 0.22, -0.11, ..., 0.33]  (768 dims)
token ID 6100 ("moves")  → row 6100 → [-0.09, 0.67, 0.38, ..., -0.21] (768 dims)
token ID 345  ("you")    → row 345  → [0.31, -0.55, 0.19, ..., 0.44]  (768 dims)
```

This is a simple table lookup — not a matrix multiplication. The embeddings are **random at first** and get refined during training via `optimizer.step()`.

### Step 3: Add Positional Embeddings

The model needs to know word **order** (otherwise "dog bites man" = "man bites dog"). A separate embedding table encodes position:

```
Position 0 → [0.08, -0.12, 0.44, ..., 0.31]  (768 dims)
Position 1 → [0.15, 0.33, -0.22, ..., 0.09]  (768 dims)
Position 2 → [-0.05, 0.41, 0.18, ..., -0.27] (768 dims)
Position 3 → [0.22, -0.09, 0.55, ..., 0.13]  (768 dims)
```

Simply **add** them together:

```python
tok_embeds = self.tok_emb(token_ids)                         # (4, 768)
pos_embeds = self.pos_emb(torch.arange(seq_len))             # (4, 768)
x = tok_embeds + pos_embeds                                  # (4, 768)
```

Now each token's vector encodes both **what** the word is and **where** it is.

### Step 4: Q, K, V Projections — Three Different Views of Each Token

Each token gets projected into three different representations using three separate weight matrices:

```python
self.W_query = nn.Linear(768, 768)   # "What am I looking for?"
self.W_key   = nn.Linear(768, 768)   # "What do I contain?"
self.W_value = nn.Linear(768, 768)   # "What info do I contribute?"
```

```
x (4 tokens × 768 dims)
    │
    ├── × W_query → queries  (4 × 768)  "What am I looking for?"
    ├── × W_key   → keys     (4 × 768)  "What do I have that others might want?"
    └── × W_value → values   (4 × 768)  "What do I actually contribute if selected?"
```

**Analogy:** Think of a library.
- **Query** = "I'm looking for books about history" (what this token needs)
- **Key** = "I'm a book labeled 'World War II'" (what this token advertises)
- **Value** = The actual content of the book (what gets passed along if selected)

Q and K determine **who pays attention to whom**. V determines **what information flows**.

### Step 5: Multi-Head Attention — 12 Heads Working in Parallel

Here's the key insight: **we don't run one big attention**. We split Q, K, V into 12 smaller heads, each working independently:

```
Q, K, V: each (4 tokens × 768 dims)
                    ↓ reshape into 12 heads
         (4 tokens × 12 heads × 64 dims)
              ↑              ↑
         num_heads      head_dim = 768/12 = 64
```

```python
# From MultiHeadAttention.forward():
keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)      # (b, 4, 12, 64)
queries = queries.view(b, num_tokens, self.num_heads, self.head_dim) # (b, 4, 12, 64)
values = values.view(b, num_tokens, self.num_heads, self.head_dim)   # (b, 4, 12, 64)

keys = keys.transpose(1, 2)       # (b, 12, 4, 64)  ← heads dimension comes first
queries = queries.transpose(1, 2)  # (b, 12, 4, 64)
values = values.transpose(1, 2)    # (b, 12, 4, 64)
```

**Why 12 heads instead of 1?** Each head can focus on a **different relationship**:

```
Head 1:  might learn to look at the previous word (local syntax)
Head 2:  might learn to look at the subject of the sentence
Head 3:  might learn to look at verbs
Head 7:  might learn to track long-range dependencies
Head 12: might learn punctuation and sentence boundaries
...each head learns its own pattern during training
```

### Step 6: Attention Scores + Causal Masking

For each head, compute: "how much should each token attend to every other token?"

```python
attn_scores = queries @ keys.transpose(2, 3)   # (b, 12, 4, 4)
```

This gives a 4×4 attention matrix per head (each token scores against every other token):

```
              "Every"  "effort"  "moves"  "you"
"Every"     [  0.8      -         -        -   ]
"effort"    [  0.3      0.7       -        -   ]
"moves"     [  0.1      0.5       0.4      -   ]
"you"       [  0.2      0.1       0.6      0.1 ]
```

**Causal masking:** The dashes (`-`) are set to `-inf` before softmax. This prevents tokens from looking at **future** tokens. "Every" can only see itself. "effort" can see "Every" and itself. This is what makes it a **language model** — it can only predict forward, never peek ahead.

```python
mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(mask_bool, -torch.inf)  # future positions → -inf
```

Then scale and softmax to get **attention weights** (rows sum to 1):

```python
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
```

The `/keys.shape[-1]**0.5` is **scaled dot-product** — dividing by sqrt(64) prevents the dot products from getting too large, which would make softmax too sharp.

### Step 7: Context Vector — The Output of Attention

Multiply attention weights × values to get a **weighted blend** of information:

```python
context_vec = attn_weights @ values              # (b, 12, 4, 64)
context_vec = context_vec.transpose(1, 2)        # (b, 4, 12, 64)
context_vec = context_vec.reshape(b, num_tokens, self.d_out)  # (b, 4, 768) — concat all heads
```

Each head produced a 64-dim result. Concatenating 12 heads: 12 × 64 = 768. Back to the original dimension.

### Step 8: Output Projection — Blending the Heads

The 12 heads worked independently. `out_proj` lets them communicate:

```python
context_vec = self.out_proj(context_vec)   # Linear(768, 768)
```

```
Head 1:  [64 dims]  ─┐
Head 2:  [64 dims]   │
Head 3:  [64 dims]   │
...                   ├── concatenate → [768 dims] → out_proj → [768 dims]
Head 12: [64 dims]  ─┘
```

Without `out_proj`, each head's 64-dim chunk stays isolated. `out_proj` blends them into a unified representation.

### The Complete Flow — One Diagram

```
"Every effort moves you"
        ↓
[Tokenizer] → [6109, 3626, 6100, 345]
        ↓
[Token Embedding lookup]  → (4, 768)
        +
[Position Embedding]      → (4, 768)
        ↓
x = combined embedding      (4, 768)
        ↓
    ┌───┼───┐
    ↓   ↓   ↓
  W_Q  W_K  W_V             each: (4, 768)
    ↓   ↓   ↓
  split into 12 heads        each: (12, 4, 64)
    ↓   ↓   ↓
  Q × K^T → attention scores (12, 4, 4)
        ↓
  causal mask (hide future)
        ↓
  softmax → attention weights (12, 4, 4)
        ↓
  weights × V → context per head (12, 4, 64)
        ↓
  concatenate 12 heads → (4, 768)
        ↓
  out_proj → context vector (4, 768)  ← THIS is what goes to the FFN
```

**This context vector is what the rest of the transformer block processes.** It goes through the residual connection, then Layer Norm, then the FFN (Block 3), then another residual connection — and out to the next transformer block.

---

## Block 3: Inside the Transformer Block (The Wrapper)

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

## Block 4: The Feed-Forward Network — The Part People Skip

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

## Block 5: From Hidden State to Logits — How Predictions Are Born

**One sentence:** After 12 transformer blocks, each token is a 768-dim vector that knows everything it needs; the final linear layer converts it into 50,257 scores — one per vocabulary word — and these scores are the logits.

### What comes out of the transformer blocks?

After passing through all 12 transformer blocks, each token has a **768-dimensional hidden state** — a rich vector that encodes syntax, grammar, meaning, and context. But it's still in "model space" (768 dims). We need to convert it to "vocabulary space" (50,257 dims) to actually predict a word.

### The three final steps in `GPTModel.forward()`:

```python
def forward(self, in_idx):
    # ... embeddings + transformer blocks ...
    x = self.trf_blocks(x)           # (batch, 4, 768) — output of 12 blocks
    x = self.final_norm(x)           # (batch, 4, 768) — normalize one last time
    logits = self.out_head(x)        # (batch, 4, 50257) — THIS creates the logits
    return logits
```

| Step | What Happens | Shape |
|---|---|---|
| `trf_blocks(x)` | 12 transformer blocks refine the representation | (batch, 4, 768) |
| `final_norm(x)` | One last Layer Norm to stabilize values | (batch, 4, 768) |
| `out_head(x)` | Linear projection: 768 → 50,257 | (batch, 4, 50,257) |

### What is `out_head`?

```python
self.out_head = nn.Linear(768, 50257, bias=False)
```

It's a single matrix multiplication — a weight matrix of shape **(768 × 50,257)**. That's 768 × 50,257 = **38,597,376 parameters** in this one layer alone (about 31% of the entire model!).

```
hidden state for "you":  [0.42, -0.18, 0.95, ..., 0.33]   (768 dims)
                                    ↓
                         × out_head weight matrix (768 × 50,257)
                                    ↓
logits for "you":        [2.1, -0.5, 1.8, ..., 6.75, ..., -3.2]   (50,257 scores)
                          ↑     ↑                ↑
                         "the" "a"           "forward"
```

Each of the 50,257 output values is a **dot product** between the token's hidden state and one column of the weight matrix. A higher score means the model thinks that vocabulary word is more likely to come next.

### What do logit values actually mean?

Logits are **raw, unnormalized scores**. They can be any number — positive, negative, large, small:

```
Logits for the position after "you" (before training):
  "forward"   6.75    ← model thinks this is most likely
  "toward"    6.28
  "closer"    4.51
  "pizza"    -1.89    ← model thinks this is unlikely
  "the"       2.10
  ... 50,252 more scores ...
```

**They are NOT probabilities yet.** To get probabilities, you apply softmax:

```python
probas = torch.softmax(logits, dim=-1)   # now each row sums to 1.0
```

```
After softmax:
  "forward"   0.44    ← 44% chance
  "toward"    0.28    ← 28% chance
  "closer"    0.05    ← 5% chance
  "pizza"     0.0008  ← 0.08% chance
```

### Why one logit vector per position?

The model produces logits at **every token position**, not just the last one:

```
Input: "every   effort   moves"
         ↓        ↓        ↓
Logits: [50,257] [50,257] [50,257]
         ↓        ↓        ↓
Predicts: ???    ???      ???
Should be: "effort" "moves"  "you"    ← these are the targets
```

During **training**, we use all positions to compute loss (more learning signal per batch).
During **generation**, we only care about the **last position's** logits — that's the next word prediction:

```python
logits = logits[:, -1, :]   # only the last token's predictions
```

### The complete path — from input to logits:

```
"every effort moves"  (text)
        ↓
[16833, 3626, 6100]   (token IDs)
        ↓
Embedding + Position   (3, 768)
        ↓
12 × Transformer Block (3, 768)  ← each block: attention → FFN → residual
        ↓
Final Layer Norm       (3, 768)
        ↓
out_head Linear        (3, 50257)  ← THIS is the logits tensor
        ↓
Position 0 logits: "what comes after 'every'?"    → 50,257 scores
Position 1 logits: "what comes after 'effort'?"   → 50,257 scores
Position 2 logits: "what comes after 'moves'?"    → 50,257 scores
```

---

## Block 6: From Logits to Loss — Measuring Wrongness

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

## Block 7: `backward()` and `optimizer.step()` — The Two Lines That Train Everything

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

## Block 8: Temperature and Top-K — Controlling the Output

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

## Block 9: The Weight Map — Every Learnable Parameter

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

## Block 10: Component Registry — Every Piece at a Glance

**One sentence:** Every component in the pipeline — its name, purpose, what goes in, what comes out, and whether training updates it.

Scan this table when you forget what a piece does. Components are listed in the order data flows through them.

### Embedding Stage

| # | Component | Code | Purpose | Input | Output | Updated by training? |
|---|---|---|---|---|---|---|
| 1 | Tokenizer | `tiktoken.get_encoding("gpt2")` | Split text into subword token IDs | Raw text string | List of integers, e.g. `[6109, 3626, 6100, 345]` | No — fixed BPE vocabulary |
| 2 | Token Embedding | `self.tok_emb = nn.Embedding(50257, 768)` | Look up a 768-dim vector for each token ID | Token IDs `(batch, seq_len)` | `(batch, seq_len, 768)` | **Yes** — each row refined by optimizer |
| 3 | Positional Embedding | `self.pos_emb = nn.Embedding(1024, 768)` | Encode position (word order) as a 768-dim vector | Position indices `(seq_len,)` | `(seq_len, 768)` | **Yes** — each row refined by optimizer |
| 4 | Embedding Addition | `x = tok_embeds + pos_embeds` | Combine word identity + position into one vector | Two tensors `(batch, seq_len, 768)` | `(batch, seq_len, 768)` | No — just addition |
| 5 | Embedding Dropout | `self.drop_emb = nn.Dropout(0.1)` | Randomly zero elements to prevent overfitting | `(batch, seq_len, 768)` | `(batch, seq_len, 768)` | No — no weights, random mask |

### Attention Stage (repeated ×12 blocks)

| # | Component | Code | Purpose | Input | Output | Updated by training? |
|---|---|---|---|---|---|---|
| 6 | Layer Norm 1 | `self.norm1 = LayerNorm(768)` | Normalize values before attention | `(batch, seq_len, 768)` | `(batch, seq_len, 768)` | **Yes** — scale and shift params |
| 7 | W_query | `self.W_query = nn.Linear(768, 768)` | Project each token into "what am I looking for?" | `(batch, seq_len, 768)` | `(batch, seq_len, 768)` | **Yes** — weight matrix |
| 8 | W_key | `self.W_key = nn.Linear(768, 768)` | Project each token into "what do I contain?" | `(batch, seq_len, 768)` | `(batch, seq_len, 768)` | **Yes** — weight matrix |
| 9 | W_value | `self.W_value = nn.Linear(768, 768)` | Project each token into "what info do I contribute?" | `(batch, seq_len, 768)` | `(batch, seq_len, 768)` | **Yes** — weight matrix |
| 10 | Multi-Head Split | `.view(b, seq, 12, 64).transpose(1,2)` | Split 768 dims into 12 heads × 64 dims each | Q, K, V each `(batch, seq_len, 768)` | Q, K, V each `(batch, 12, seq_len, 64)` | No — just reshape |
| 11 | Attention Scores | `queries @ keys.transpose(2, 3)` | Compute how much each token attends to every other | Q `(batch, 12, seq, 64)` × K^T | `(batch, 12, seq_len, seq_len)` | No — matrix multiply of Q and K |
| 12 | Causal Mask | `.masked_fill_(mask, -torch.inf)` | Hide future tokens (can only look backward) | Attention scores `(batch, 12, seq, seq)` | Same shape, future positions = -inf | No — fixed triangular mask |
| 13 | Scaled Softmax | `softmax(scores / sqrt(64))` | Convert scores to attention weights (rows sum to 1) | Masked scores `(batch, 12, seq, seq)` | Attention weights `(batch, 12, seq, seq)` | No — just math |
| 14 | Attention Dropout | `self.dropout(attn_weights)` | Randomly zero attention connections | `(batch, 12, seq, seq)` | `(batch, 12, seq, seq)` | No — no weights |
| 15 | Context Vector | `attn_weights @ values` | Weighted blend of value vectors → what each token "learned" | Weights `(b, 12, seq, seq)` × V `(b, 12, seq, 64)` | `(batch, 12, seq_len, 64)` | No — matrix multiply |
| 16 | Head Concat | `.transpose(1,2).reshape(b, seq, 768)` | Rejoin 12 heads back into one 768-dim vector | `(batch, 12, seq_len, 64)` | `(batch, seq_len, 768)` | No — just reshape |
| 17 | Output Projection | `self.out_proj = nn.Linear(768, 768)` | Blend 12 independent head outputs into unified representation | `(batch, seq_len, 768)` | `(batch, seq_len, 768)` | **Yes** — weight matrix |
| 18 | Attention Dropout | `self.drop_shortcut(x)` | Dropout after attention | `(batch, seq_len, 768)` | `(batch, seq_len, 768)` | No — no weights |
| 19 | Residual Add 1 | `x = x + shortcut` | Add original input back (skip connection) | Attention output + original input | `(batch, seq_len, 768)` | No — just addition |

### Feed-Forward Stage (repeated ×12 blocks)

| # | Component | Code | Purpose | Input | Output | Updated by training? |
|---|---|---|---|---|---|---|
| 20 | Layer Norm 2 | `self.norm2 = LayerNorm(768)` | Normalize values before FFN | `(batch, seq_len, 768)` | `(batch, seq_len, 768)` | **Yes** — scale and shift params |
| 21 | FFN W1 (expand) | `nn.Linear(768, 3072)` | Expand to 4× workspace for complex processing | `(batch, seq_len, 768)` | `(batch, seq_len, 3072)` | **Yes** — weight matrix |
| 22 | GELU Activation | `GELU()` | Non-linear activation (lets model learn complex patterns) | `(batch, seq_len, 3072)` | `(batch, seq_len, 3072)` | No — fixed math function |
| 23 | FFN W2 (compress) | `nn.Linear(3072, 768)` | Compress conclusions back to token size | `(batch, seq_len, 3072)` | `(batch, seq_len, 768)` | **Yes** — weight matrix |
| 24 | FFN Dropout | `self.drop_shortcut(x)` | Dropout after FFN | `(batch, seq_len, 768)` | `(batch, seq_len, 768)` | No — no weights |
| 25 | Residual Add 2 | `x = x + shortcut` | Add pre-FFN input back (skip connection) | FFN output + pre-FFN input | `(batch, seq_len, 768)` | No — just addition |

### Output Stage

| # | Component | Code | Purpose | Input | Output | Updated by training? |
|---|---|---|---|---|---|---|
| 26 | Final Layer Norm | `self.final_norm = LayerNorm(768)` | Normalize one last time before projection | `(batch, seq_len, 768)` | `(batch, seq_len, 768)` | **Yes** — scale and shift params |
| 27 | Output Head | `self.out_head = nn.Linear(768, 50257)` | Project 768-dim hidden state to vocabulary scores (logits) | `(batch, seq_len, 768)` | `(batch, seq_len, 50257)` | **Yes** — 38.6M params (31% of model!) |

### Training Stage

| # | Component | Code | Purpose | Input | Output | Updated by training? |
|---|---|---|---|---|---|---|
| 28 | Softmax | `torch.softmax(logits, dim=-1)` | Convert logits to probabilities (each position sums to 1) | `(batch, seq_len, 50257)` | `(batch, seq_len, 50257)` | No — just math |
| 29 | Cross-Entropy Loss | `F.cross_entropy(logits, targets)` | Measure how wrong predictions are vs. targets | Logits + target token IDs | Single scalar (e.g. `tensor(10.83)`) | No — just measurement |
| 30 | loss.backward() | `loss.backward()` | Walk computation graph backward, compute gradient for every weight | Single loss scalar | `.grad` tensor on all 124M params | No — computes gradients, doesn't change weights |
| 31 | optimizer.step() | `optimizer.step()` | Nudge every weight downhill using its gradient | All `.grad` tensors | All weights slightly updated | **This IS the update** — every learnable weight changes |
| 32 | optimizer.zero_grad() | `optimizer.zero_grad()` | Clear all gradients before next batch | All `.grad` tensors | All `.grad` set to zero | No — just cleanup |

### Generation Stage (inference only, no training)

| # | Component | Code | Purpose | Input | Output | Updated by training? |
|---|---|---|---|---|---|---|
| 33 | Top-K Filtering | `torch.topk(logits, k)` | Keep only K highest logits, set rest to -inf | `(batch, 50257)` logits | `(batch, 50257)` with most set to -inf | No — just filtering |
| 34 | Temperature Scaling | `logits / temperature` | Control confidence: low = sharp, high = uniform | `(batch, 50257)` logits | `(batch, 50257)` rescaled logits | No — just division |
| 35 | Softmax | `torch.softmax(logits, dim=-1)` | Convert scaled logits to probabilities | `(batch, 50257)` logits | `(batch, 50257)` probabilities summing to 1.0 | No — just math |
| 36 | Multinomial Sampling | `torch.multinomial(probs, 1)` | Randomly pick one token weighted by probabilities | `(batch, 50257)` probabilities | Single token ID `(batch, 1)` | No — just random sampling |

### Summary Counts

```
Total components:               36
Learnable (updated by training): 12 types  →  but ×12 blocks = many instances
                                              Total learnable params: 124,439,808

Breakdown of learnable vs. operations:
  Learnable weights:  tok_emb, pos_emb, norm1, W_Q, W_K, W_V, out_proj,
                      norm2, FFN W1, FFN W2, final_norm, out_head
  Pure operations:    tokenizer, addition, dropout, reshape, matmul,
                      mask, softmax, GELU, residual add, loss,
                      backward, optimizer, top-k, temperature, sampling
```

---

## Quick Reference Card

When you come back in a month, start here:

```
Text → Tokens → Embeddings → Q,K,V → Multi-Head Attention → Context Vector → 12 × [Attention → FFN] → Logits → Loss → Backward → Step → Repeat
```

| Question You'll Have | Block |
|---|---|
| "What's the full pipeline?" | Block 1 |
| "How does text become Q, K, V? What are attention heads?" | Block 2 |
| "What's inside a transformer block?" | Block 3 |
| "What's the FFN / what's 3072?" | Block 4 |
| "How are logits created? What do they mean?" | Block 5 |
| "How does loss work? What are targets?" | Block 6 |
| "What does backward/optimizer actually do?" | Block 7 |
| "What's temperature/top-k?" | Block 8 |
| "What weights exist and get updated?" | Block 9 |
| "Quick lookup: name, input, output, trainable?" | Block 10 |

---

## Acknowledgments

This article and the accompanying code would not exist without **Sebastian Raschka** and his excellent book [*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) (Manning, 2024). Sebastian's approach of building every component by hand — from tokenization to attention to the training loop — is what makes the architecture click. If you want to truly understand transformers at the weight level, not just use them, this book is the best resource I've found.

The model architecture, training code, and chapter structure in my notebooks follow Sebastian's implementation from his [LLMs-from-scratch GitHub repository](https://github.com/rasbt/LLMs-from-scratch), licensed under Apache 2.0.

### Additional references that helped along the way:

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar — the visual intuition that makes attention click
- OpenAI's [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — the original architecture this implementation follows
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) — the paper that started it all

---

*I'm documenting my journey learning LLM internals from scratch. Follow along for more deep dives into how these models actually work.*

*Full code: [github.com/jkuru/llm](https://github.com/jkuru/llm)*

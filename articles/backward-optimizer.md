# Two Lines of Code That Train a 124-Million-Parameter LLM

Everyone talks about transformers, attention, and billion-parameter models. But the entire learning process — where the model goes from producing gibberish to generating coherent text — comes down to two lines:

```python
loss.backward()
optimizer.step()
```

That's it. Let me show you what actually happens inside these two lines.

## The Setup

I'm training a GPT-2 (124M parameters) from scratch using Sebastian Raschka's *Build a Large Language Model From Scratch*. The model has:

- Token embeddings (50,257 x 768)
- Positional embeddings (1,024 x 768)
- 12 transformer blocks, each containing:
  - W_query, W_key, W_value (768 x 768 each)
  - Output projection (768 x 768)
  - Feed-forward W1 (768 x 3,072) — expand
  - Feed-forward W2 (3,072 x 768) — compress
  - 2 layer norms
- Final projection (768 x 50,257)

Total: **124,439,808 learnable weights.**

## Before `backward()` — Nothing

I ran a forward pass and got a loss value:

```python
logits = model(inputs)                    # forward pass
loss = cross_entropy(logits, targets)     # single scalar: tensor(10.83)
```

At this point, every weight in the model has `grad = None`. The model computed predictions and measured how wrong it was, but it hasn't figured out *who's responsible* for the error yet.

```
tok_emb.weight                     grad = None
trf_blocks.0.att.W_query.weight    grad = None
trf_blocks.0.ff.layers.0.weight    grad = None
... all 124M weights ...           grad = None
```

## `loss.backward()` — The Chain of Blame

This is where the magic happens. That single loss number (10.83) knows exactly how it was computed — PyTorch recorded every matrix multiplication, every activation function, every addition along the way. This recorded path is called the **computation graph**.

`backward()` walks this graph in reverse, applying the **chain rule of calculus** at every step:

```
loss (scalar)
  |-- backward through cross_entropy
  |-- backward through final Linear (768 -> 50,257)
  |-- backward through LayerNorm
  |-- backward through Transformer Block 12
  |     |-- backward through FFN W2 (3,072 -> 768)
  |     |-- backward through GELU activation
  |     |-- backward through FFN W1 (768 -> 3,072)
  |     |-- backward through LayerNorm
  |     |-- backward through out_proj (768 -> 768)
  |     |-- backward through softmax + causal mask
  |     |-- backward through Q x K^T
  |     |-- backward through W_query, W_key, W_value
  |     |-- backward through LayerNorm
  |-- backward through Transformer Block 11
  |     |-- ... same chain ...
  |-- ... all the way through Block 1 ...
  |-- backward through Embeddings
```

After this single call, every weight now has a `.grad` attribute — a tensor of the **exact same shape** as the weight itself. Crucially, each gradient points in the direction of **steepest increase** in loss — the direction that would make things *worse*. That's why the optimizer uses a minus sign: we go the opposite way, downhill.

**The silent killer: memory.** Those 124 million gradients are the same size as the weights themselves. `backward()` effectively **doubles your model's memory footprint** in one call. This is why GPU out-of-memory (OOM) errors almost always hit during `backward()`, not during the forward pass. Techniques like gradient checkpointing, mixed precision, and gradient accumulation exist specifically to tame this cost.

```
tok_emb.weight                     grad shape: [50257, 768]    mean: +0.00000012
trf_blocks.0.att.W_query.weight    grad shape: [768, 768]      mean: +0.00008900
trf_blocks.0.ff.layers.0.weight    grad shape: [3072, 768]     mean: +0.00003400
... all 124M weights now have gradients ...
```

**One scalar in. 124 million gradients out.** That's the power of automatic differentiation.

## `optimizer.step()` — The Nudge

Now the optimizer (AdamW) reads every gradient and nudges each weight in the **opposite direction** of the gradient — downhill toward lower loss:

```
weight_new = weight_old - learning_rate * gradient
                        ^
                        minus sign = go OPPOSITE to gradient = downhill
```

The gradient says "loss increases fastest this way." The minus sign says "then let's go the other way." That's gradient **descent** in one line.

In practice, AdamW is smarter — it maintains running averages of gradients (momentum) and adapts the learning rate per-parameter. But the core idea is the same: move each weight a tiny step downhill.

And the memory cost doesn't stop at gradients. AdamW stores **two additional tensors per weight** (the momentum and variance estimates). So your 124M-parameter model needs memory for: weights (124M) + gradients (124M) + optimizer state (248M) = roughly **4x the model size**. This is why a model that fits in VRAM for inference can OOM during training.

One call. 124 million weights updated simultaneously.

## Why This Is Remarkable

Think about what just happened:

1. We fed in 3 tokens: "every effort moves"
2. The model produced 50,257 probability scores for the next token at each position
3. We compared those against the correct answers: "effort moves you"
4. We got a single number measuring total wrongness: 10.83
5. From that single number, **we computed a personalized correction for each of the 124 million weights**
6. We applied all corrections at once

Repeat this thousands of times with different text, and the model goes from:

```
Epoch 1:  "Every effort moves the the the the"
Epoch 5:  "Every effort moves the country forward"
Epoch 10: "Every effort moves you closer to the goal"
```

## The Code You Can Run

I added a debug cell to my training notebook that visualizes this entire process. You can see gradients appear from nothing after `backward()`, and verify that every single parameter gets updated.

The full code is in my repo: [github.com/jkuru/llm](https://github.com/jkuru/llm) (see ch05.ipynb, the debug cell after Section 5.1).

## Key Takeaways

- `loss.backward()` doesn't update anything — it only computes gradients
- `optimizer.step()` doesn't compute anything — it only applies the pre-computed gradients
- The separation is intentional: you can inspect, clip, or modify gradients between the two calls
- `optimizer.zero_grad()` clears all gradients before the next batch — without it, gradients accumulate and training breaks
- The entire training loop is just these 4 lines repeated:

```python
optimizer.zero_grad()     # forget last batch
loss = calc_loss(...)     # how wrong are we?
loss.backward()           # who's responsible?
optimizer.step()          # fix it slightly
```

Everything else — data loading, evaluation, checkpointing, sampling strategies — is infrastructure around these four lines.

---

*I'm documenting my journey learning LLM internals from scratch. Follow along for more deep dives into how these models actually work.*

*Built with code from Sebastian Raschka's "Build a Large Language Model From Scratch" — highly recommended if you want to understand transformers at the weight level.*

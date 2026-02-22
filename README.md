# GPT with KV Cache — A Beginner's Walkthrough

A from-scratch implementation of a GPT model with **KV Cache** optimization, built as an educational companion to Sebastian Raschka's [*Build a Large Language Model From Scratch*](https://www.manning.com/books/build-a-large-language-model-from-scratch) (Chapters 1–4).

## What You'll Learn

- How the full GPT-2 architecture works (embeddings, multi-head attention, transformer blocks)
- What the **KV Cache** is and why it makes autoregressive generation fast
- How causal masking changes when a cache is involved
- How system prompts and safety instructions live inside the KV cache in production LLMs

> **Note:** The model uses random weights (not pretrained), so generated text will be gibberish. The goal is understanding the **architecture and caching mechanism**, not producing meaningful output.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/jkuru/llm.git
cd gpt-kv-cache
```

### 2. Install uv (if you don't have it)

```bash
# macOS (Homebrew)
brew install uv

# Linux / macOS (standalone installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Set up the environment

```bash
uv sync
```

That's it — `uv` reads `pyproject.toml`, creates a `.venv`, resolves dependencies, and installs everything.

### 4. Register the Jupyter kernel

```bash
uv run python -m ipykernel install --user --name gpt-kv --display-name "GPT-KV (Python 3.11)"
```

This registers the virtual environment as a Jupyter kernel so the notebook can find torch, tiktoken, and all other dependencies.

### 5. Run the notebook

```bash
uv run jupyter notebook gpt-kv.ipynb
```

Run the cells top-to-bottom with `Shift+Enter`. When prompted, select the **"GPT-KV (Python 3.11)"** kernel (Kernel → Change Kernel).

### Alternative: pip

If you don't use `uv`, you can still set up with pip:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook gpt-kv.ipynb
```

## Project Structure

```
.
├── gpt-kv.ipynb        # Main notebook — the full walkthrough
├── pyproject.toml      # Project metadata and dependencies
├── requirements.txt    # Fallback for pip users
├── LICENSE             # MIT License
├── .gitignore          # Git ignore rules
├── .python-version     # Python version pin for uv
└── README.md           # This file
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- tiktoken
- Jupyter

A GPU is optional — the notebook runs on CPU, though generation will be slower.

## Topics Covered

| Chapter | Concept | Where in the notebook |
|---------|---------|----------------------|
| Ch 1 | How LLMs work at a high level | The generate → predict → append loop |
| Ch 2 | Tokenization (BPE) | `tiktoken` encodes/decodes text |
| Ch 3 | Self-attention mechanism | `MultiHeadAttention` with KV cache |
| Ch 4 | Full GPT architecture | `GPTModel` = embeddings + transformer blocks + output head |

## References

- Sebastian Raschka, [*Build a Large Language Model From Scratch*](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- OpenAI's [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## License

MIT — see [LICENSE](LICENSE) for details.

# picochat: learning log

Goal: build a tiny GPT-style language model from scratch, trainable on a CPU,
**≤ 50M params** (target ~5–15M). The point is to learn how each part works.

## Setup

- Machine: 16-core CPU, 61 GB RAM, no GPU
- Python 3.14, PyTorch 2.12 (CPU), NumPy 2.5
- Approach: PyTorch for tensors + autograd; attention, transformer blocks,
  tokenizer and training loop written by hand (nanoGPT-style)

## Plan

| # | Step | File | Status |
|---|------|------|--------|
| 1 | Project setup + dataset | `data/download.py` | ✅ done |
| 2 | BPE tokenizer from scratch | `tokenizer.py` | ⏳ next |
| 3 | GPT model (hand-written attention) | `model.py` | |
| 4 | Training loop on CPU | `train.py` | |
| 5 | Sampling / generation | `sample.py` | |

Target model config (first run):

| setting | value | note |
|---|---|---|
| vocab_size | 4096 | own BPE; small vocab saves embedding params |
| context (block_size) | 256 tokens | |
| n_layer | 6 | |
| n_embd | 256–384 | |
| n_head | 6–8 | |
| params | ~5–15M | |

---

## Step 1: Dataset

**TinyStories** (Eldan & Li, 2023): short stories written by GPT-4 using only
words a 3–4-year-old would know. The paper shows that even models under 10M
params can learn fluent, grammatical English from it, which makes it well
suited to CPU training.

- `python data/download.py`: validation split only (22.5 MB)
- `python data/download.py --train`: also the full train split (2.2 GB)

Stats (valid split):
- 22.5M characters, 27,631 stories, 91 unique characters
- stories separated by `<|endoftext|>`, which becomes a **special token** so
  the model learns where a story ends

Why start with the small split: 22M chars ≈ 5–6M tokens after BPE, enough to
train the tokenizer and a first model quickly. Switch to the full split once
the pipeline works.

---

## Step 2: BPE tokenizer

_(to do)_

Key ideas to cover:
- why bytes (256 base tokens) instead of characters (any text can be encoded, nothing is unknown)
- pre-splitting into words so merges don't cross word boundaries
- training loop: count adjacent pairs → merge the most frequent → repeat
- encode / decode, special tokens

---

## Glossary

- **token**: an integer ID for a chunk of text (a byte, a subword, or a word)
- **vocab_size**: number of distinct tokens
- **context / block_size**: how many tokens the model sees at once
- **parameter**: a learned number (weight); the model size is the count of these

## Results log

| date | step | config | params | train loss | val loss | notes |
|---|---|---|---|---|---|---|

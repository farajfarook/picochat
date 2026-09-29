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
| 2 | BPE tokenizer from scratch | `tokenizer.py` | ✅ done |
| 3 | GPT model (hand-written attention) | `model.py` | ✅ done |
| 4 | Training loop on CPU | `train.py` | ⏳ next |
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

Run: `python tokenizer.py` trains the tokenizer, saves `tokenizer.json`, and
writes `data/train.bin` + `data/val.bin` (uint16 token ids).

### How BPE works
1. **Start with bytes.** Text in UTF-8 is a sequence of bytes 0–255, giving
   256 base tokens. Any text can be encoded and nothing is ever "unknown".
2. **Count adjacent pairs** across the training text.
3. **Merge the most frequent pair** into a new token id (256, 257, ...).
4. **Repeat** until vocab = 4096 (256 bytes + 3839 merges + 1 special).

### Key ideas
- **Pre-splitting (GPT-2 regex):** text is chopped into word-like chunks first
  and merges only happen *inside* a chunk, so there's no junk like `"dog."`.
  The leading space is part of the word: `" dog"` ≠ `"dog"`.
- **Speed trick:** count each distinct chunk once. 2.4M chunks collapse to
  only ~10k distinct ones, so training takes 26 s in pure Python.
- **Encoding** replays merges in the order they were learned (earliest first).
- **Special token** `<|endoftext|>` = id 4095, marks story boundaries.
- `uint16` storage works because 4096 < 65,536 (2 bytes per token).

### Results
- First merges: `" t"`, `"he"`, `" a"`, `" s"`, `" w"`, then `" the"` (built from the first two), `"nd"`, `"ed"`, `" and"`...
  Common English pieces appear first, which is what you'd expect.
- Later merges are whole words: `" children"`, `" perfume"`, `" church"`.
- Example: `"Once upon a time, there was a little dog named Max."` = 13 tokens,
  one per word/punctuation mark.
- Emoji 😀 isn't in the vocab, so it falls back to its 4 raw bytes.
  That's the benefit of byte-level BPE: nothing breaks.
- **3.99 chars/token.** Dataset = 5.07M train tokens + 0.57M val tokens.

### Why vocab 4096 and not 50k like GPT-2?
The embedding table is `vocab_size × n_embd` params. At n_embd=256:
50k vocab = 12.8M params just for embeddings. 4k vocab = 1M. For a tiny
model on simple text, the params are better spent on transformer layers.

---

## Step 3: GPT model

Run: `python model.py` prints the param breakdown, the untrained loss, CPU
speed, and a sample from the untrained model.

### What a language model is
A **next-token predictor**. Given tokens so far, output a probability for
every one of the 4096 possible next tokens. To generate: pick one, append it, repeat.

### The pipeline
```
token ids (T)
 → token embedding + position embedding    (T, C)   id → vector, position → vector
 → Block × 6                               (T, C)
 → LayerNorm → lm_head                     (T, 4096) score per possible next token
 → softmax                                 probabilities
```

### Inside a block
```
x = x + Attention(LayerNorm(x))   # tokens look at earlier tokens (communicate)
x = x + MLP(LayerNorm(x))         # each token processes on its own (compute)
```

### Key ideas
- **Embedding**: a lookup table from token id to a vector of 384 numbers. The
  model learns these; similar words end up with similar vectors.
- **Position embedding**: attention by itself has no notion of order, so we
  add a learned "I am position 5" vector.
- **Attention (q, k, v)**: each token emits a query ("what am I looking for"),
  a key ("what I contain") and a value ("what I pass on"). score = q·k /
  √head_size → softmax → weighted average of values.
- **Causal mask**: a token can only see itself and earlier tokens, so it
  can't cheat by looking at the answer.
- **Multi-head**: 6 heads of size 64 run in parallel, each free to track a
  different relationship.
  Nobody tells a head what to track. Every head starts random, and training
  adjusts each one only in ways that improve next-token guesses. Specialisations
  like "find the subject" can emerge because they help prediction. We can only
  find out what each head does by inspecting a trained model.
- **MLP**: expand ×4 → GELU → shrink. Per-token processing; stores much of the "knowledge".
- **Residual (`x = x + ...`)**: layers add updates rather than replace, which keeps deep nets trainable.
- **LayerNorm**: keeps the numbers in each vector at a sane scale.
- **Weight tying**: input embedding and output layer share one matrix (saves 1.6M params).
- **Cross-entropy loss**: −log(prob given to the correct next token).
  Random guess = ln(4096) = **8.32**. Perfect = 0.
- **Targets = inputs shifted by one**: one sequence of 256 tokens gives 256
  training examples at once (predict token 1 from 0, token 2 from 0–1, ...).

### Results
| part | params |
|---|---|
| token embedding 4096×384 (shared with output) | 1.57M |
| position embedding 256×384 | 0.10M |
| per block (attn 4C² + MLP 8C²) | 1.77M |
| 6 blocks | 10.6M |
| **total** | **12.3M** |

- Untrained loss **8.41** ≈ random 8.32, as expected.
- CPU speed: ~**5,500 tokens/sec** in training, so one pass over 5M train tokens takes ~15 min.
- Untrained output: `Once upon a time moved coming terrible what dropped something planist popcorn...`
  These are real tokens picked at random; there's no grammar yet.

---

## Glossary

- **token**: an integer ID for a chunk of text (a byte, a subword, or a word)
- **vocab_size**: number of distinct tokens
- **context / block_size**: how many tokens the model sees at once
- **BPE**: byte-pair encoding, which builds a vocab by repeatedly merging frequent pairs
- **special token**: a reserved id with meaning (e.g. end of story), never produced by merges
- **logits**: raw scores per vocab token before softmax
- **softmax**: turns scores into probabilities that sum to 1
- **loss**: how wrong the model is; training = making this go down
- **parameter**: a learned number (weight); the model size is the count of these

## Results log

| date | step | config | params | train loss | val loss | notes |
|---|---|---|---|---|---|---|

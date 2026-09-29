# picochat: learning log

Goal: build a tiny GPT-style language model from scratch, trainable on a CPU,
**≤ 50M params** (target ~5–15M). The point is to learn how each part works.

## Setup

- Machine: 16-core CPU, 61 GB RAM, no GPU
- GPU: RTX 3050 laptop (6 GB). Gaming rig has an RTX 3090 (24 GB)
- Python 3.14, PyTorch with CUDA 12.6, NumPy 2.5, managed by **uv** in a local `.venv`
  - setup: `uv sync`, then prefix every command with `uv run`
- Approach: PyTorch for tensors + autograd; attention, transformer blocks,
  tokenizer and training loop written by hand (nanoGPT-style)

## Plan

| # | Step | File | Status |
|---|------|------|--------|
| 1 | Project setup + dataset | `data/download.py` | ✅ done |
| 2 | BPE tokenizer from scratch | `tokenizer.py` | ✅ done |
| 3 | GPT model (hand-written attention) | `model.py` | ✅ done |
| 4 | Training loop | `train.py` | ✅ done |
| 5 | Sampling / generation | `sample.py` | ✅ done |

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

## Step 4: Training

Run: `uv run python train.py` (options: `--max_iters`, `--batch_size`,
`--resume`, `--device cpu`...). Saves the best model to `ckpt.pt` and the
log to `train_log.csv`.

### The loop
```
repeat:
  x, y = random batch (16 snippets × 256 tokens; y = x shifted by one)
  loss = model(x, y)        # forward: how wrong?
  loss.backward()           # backward: gradient for every param
  clip gradients to 1.0     # safety against one bad batch
  optimizer.step()          # AdamW nudges every param
```

### Key ideas
- **Gradient**: for each parameter, "which direction and how much would
  reduce the loss". `loss.backward()` computes all 12.3M of them (backpropagation).
- **Learning rate (lr)**: how big each nudge is. Warmup 0→1e-3 over 100
  steps, then cosine decay to 1e-4.
- **AdamW**: gradient descent + momentum + a per-parameter step size, plus weight decay (pull toward 0).
- **Train vs val loss**: val is measured on stories the model never trains
  on. If val rises while train falls, the model is memorising (**overfitting**).
- **bfloat16 autocast** on GPU: 16-bit maths for about 2× speed. Weights stay 32-bit.

### Hardware
| device | tokens/sec | 3000 steps |
|---|---|---|
| CPU (16 cores) | ~5,500 | ~40 min |
| RTX 3050 laptop | ~52,000 | **4.2 min** |
| RTX 3090 (estimate) | ~150–250k | ~1 min |

A remote GPU can't be used as `device="cuda"` from another PC. Training does
hundreds of tiny GPU ops per step, and network latency would kill it. You send
the whole *job* to the machine with the GPU instead (SSH / VS Code Remote).
Multi-GPU training (DDP) works because each GPU runs its own full copy and
they only exchange gradients once per step, over very fast links (NVLink
~600 GB/s vs home LAN ~0.1 GB/s).

### Run 1 results (12.3M params, vocab 4096, 3000 steps, 3050)
| iter | train | val |
|---|---|---|
| 0 | 8.38 | 8.38 |
| 250 | 3.54 | 3.52 |
| 500 | 2.98 | 3.04 |
| 1000 | 2.51 | 2.58 |
| 1500 | 2.24 | 2.34 |
| 2000 | 2.10 | 2.21 |
| 2500 | 1.92 | 2.10 |
| 3000 | 1.84 | 2.06 |

Best val **2.034** (iter 2750). Sample at iter 3000:
> Once upon a time, there was a blue bird named Baby. Baby was very big and
> loved to fly. One day, Baby found a big, shiny rock. It was so shiny and
> tasty. Baby said, "Let's use this rock to fill it with pretty

Observations:
- Grammar, names, dialogue in quotes and story structure ("One day...") all
  learned from next-token prediction alone.
- The logic is still weak ("a rock ... so shiny and **tasty**"). The model
  knows which words fit, but not yet what makes sense.
- **Train/val gap is opening** (1.84 vs 2.06). 3000 × 4096 = 12M tokens ≈
  2.4 passes over our 5M-token training set, so it's starting to memorise.
  **More data** (the full 2.2 GB TinyStories ≈ 550M tokens) is the next big lever.

---

## Step 5: Sampling

Run: `uv run python sample.py`, an interactive playground
(`:probs`, `:temps`, `:t`, `:k`, `:n`). One-shot: `--prompt "..."`.

### Generation loop
1. model(prompt) → 4096 scores for the next token
2. softmax → probabilities
3. **pick** one token
4. append it, repeat. Stop at `<|endoftext|>` or max_tokens.

### The two dials
- **temperature** T: divide scores by T before softmax.
  T→0 = greedy (always the top token). T<1 = safer. T>1 = wilder.
- **top_k**: keep only the k most likely tokens, so it never picks a 1-in-10,000 dud.

### What the model believes (run 1)
| after | top guesses |
|---|---|
| `Once upon a` | ` time` 99.9% (the model is certain) |
| `The cat was very` | ` happy` 28%, ` sad` 20%, ` excited` 11%, ` surprised` 8%, ` hungry` 8%... (many good options) |

It's certain where the text is fixed and spread out where there are real
choices. That's why we *sample* rather than always taking the top token.

### Temperature comparison, "Lily went to the park"
- **0 (greedy)**: coherent but safe, and starts repeating ("She wanted to climb the tree. She tried to climb the tree...")
- **0.5**: fluent, small logic slips ("The dog was very big and had many dogs")
- **1.0**: more varied, less sensible ("pick the bird's bird to learn its heart")
- **1.5**: grammar starts breaking down ("She liked Tom the park with her toys")

Sweet spot ≈ **0.7–0.9** with top_k 50.

---

## Step 6: Scaling data

`data/prepare.py` encodes the full 2.2 GB train split in parallel on all CPU
cores (the file is cut at story boundaries, and 16 processes encode pieces at
once). Encoding is text/dictionary work, so it runs on the CPU; GPUs only
help with big uniform number-crunching.

### Run 2: 12M model, full data, 15k steps (61M tokens, every batch new)
| iter | train | val |
|---|---|---|
| 1500 | 2.36 | 2.36 |
| 4500 | 1.96 | 1.95 |
| 9000 | 1.72 | 1.69 |
| 15000 | 1.57 | **1.55** |

- val **2.03 → 1.55** from more data alone, same model
- train ≈ val the whole way, so no memorising
- still slowly improving at the end, so not saturated yet

### Where does it saturate?
Two limits: **model capacity** (a small model runs out of room, and more
training stops helping) and **the data's own randomness** (a floor no model
can go below). Test: train a bigger model on the same data. If it gets
clearly lower, capacity was the limit.

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
- **gradient**: direction and size to nudge a parameter to reduce loss
- **learning rate**: how big the nudges are
- **overfitting**: memorising the training data instead of learning general patterns (val loss stops improving)
- **checkpoint**: saved model weights (`ckpt.pt`)
- **temperature**: sharpens (<1) or flattens (>1) the next-token probabilities
- **top-k**: sample only from the k most likely tokens
- **greedy decoding**: always pick the most likely token (temperature 0)
- **parameter**: a learned number (weight); the model size is the count of these

## Results log

| date | step | config | params | train loss | val loss | notes |
|---|---|---|---|---|---|---|
| run 1 | 3000 | 6L 6H 384C, vocab 4096, B16 T256, lr 1e-3 | 12.3M | 1.84 | 2.06 (best 2.034) | RTX 3050, 4.2 min, valid split only (5M tok) |
| run 2 | 15000 | same model, **full TinyStories** (470M tok) | 12.3M | 1.57 | 1.55 | RTX 3050, ~20 min; train≈val, no overfitting |

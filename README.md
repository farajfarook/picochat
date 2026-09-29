# picochat

A tiny GPT-style language model built from scratch for learning.
12M parameters, trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
in about 4 minutes on a laptop RTX 3050 (or about 40 minutes on CPU).

Everything is hand-written and heavily commented: a byte-level BPE tokenizer,
causal self-attention, transformer blocks, the training loop and sampling.
See [NOTES.md](NOTES.md) for the step-by-step learning log.

```
Once upon a time, there was a blue bird named Baby. Baby was very big and
loved to fly. One day, Baby found a big, shiny rock...
```

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Installs PyTorch with CUDA 12.6
into a local `.venv`; it falls back to CPU if there's no NVIDIA GPU.

```bash
uv sync
uv run python data/download.py      # TinyStories validation split, 22 MB
uv run python tokenizer.py          # train BPE (vocab 4096), encode dataset
uv run python model.py              # param count + sanity checks
uv run python train.py              # train, saving ckpt.pt
uv run python sample.py             # interactive story playground
```

## Files

| file | what |
|---|---|
| `tokenizer.py` | byte-level BPE tokenizer from scratch |
| `play_tokenizer.py` | interactive tokenizer explorer |
| `model.py` | GPT: embeddings, causal self-attention, MLP, blocks |
| `train.py` | training loop: AdamW, warmup + cosine LR, eval, checkpoints |
| `sample.py` | generation with temperature / top-k, next-token probabilities |
| `NOTES.md` | learning log: concepts, decisions, results |

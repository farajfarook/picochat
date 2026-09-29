"""
Step 4: the training loop.

Run:
    python train.py                                   # full run (~40 min on 16-core CPU)
    python train.py --max_iters 50 --eval_interval 25 # quick smoke test (~1 min)
    python train.py --resume                          # continue from ckpt.pt

WHAT TRAINING IS
----------------
Repeat thousands of times:
  1. grab a batch of random 256-token snippets from the training data
  2. FORWARD:  the model guesses the next token at every position -> loss
  3. BACKWARD: PyTorch works out, for each of the 12.3M parameters,
               "if I nudge this number up a little, does the loss go up or down,
               and by how much?"  (that's the *gradient*)
  4. STEP:     nudge every parameter a tiny amount in the direction that
               lowers the loss (the optimizer does this)

Nothing else. No rules about grammar, words or stories. Just "be a bit less
wrong about the next token", millions of times.

WHAT TO WATCH
-------------
  train loss  how wrong it is on data it's learning from
  val loss    how wrong it is on stories it has NEVER seen (the honest score)
  If train keeps dropping but val starts rising, the model is memorising
  instead of learning (overfitting). Time to stop or get more data.

Rough guide to the loss (random guessing = 8.3):
  ~6    learned which tokens are common (" the", ".", " was")
  ~4    short word patterns ("Once upon a time")
  ~2.5  grammatical sentences
  ~2.0  simple coherent stories
"""

import argparse
import csv
import math
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

from model import GPT, GPTConfig
from tokenizer import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")

# ----------------------------------------------------------------- settings
p = argparse.ArgumentParser()
# model size (must match what you want to train; saved into the checkpoint)
p.add_argument("--n_layer", type=int, default=6)
p.add_argument("--n_head", type=int, default=6)
p.add_argument("--n_embd", type=int, default=384)
p.add_argument("--block_size", type=int, default=256)
p.add_argument("--dropout", type=float, default=0.0)
# training
p.add_argument("--batch_size", type=int, default=16)     # snippets per step
p.add_argument("--max_iters", type=int, default=3000)    # total training steps
p.add_argument("--lr", type=float, default=1e-3)         # peak learning rate
p.add_argument("--min_lr", type=float, default=1e-4)     # learning rate at the end
p.add_argument("--warmup_iters", type=int, default=100)
p.add_argument("--weight_decay", type=float, default=0.1)
p.add_argument("--grad_clip", type=float, default=1.0)
# logging / eval
p.add_argument("--log_interval", type=int, default=10)
p.add_argument("--eval_interval", type=int, default=250)
p.add_argument("--eval_iters", type=int, default=20)     # batches used to estimate loss
p.add_argument("--out", default="ckpt.pt")
p.add_argument("--resume", action="store_true")
p.add_argument("--seed", type=int, default=1337)
p.add_argument("--device", default="auto", help="auto | cpu | cuda")
args = p.parse_args()

# GPU if available (and PyTorch was installed with CUDA), otherwise CPU
device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
# on GPU, do the heavy maths in bfloat16 (16-bit numbers): about 2x faster, same results in practice
def autocast():
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)  # CPU: plain 32-bit maths
print(f"device: {device}" + (f" ({torch.cuda.get_device_name()})" if device == "cuda" else ""))

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ------------------------------------------------------------------- data
# np.memmap reads the file lazily from disk, so this works for huge datasets too
train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("data/val.bin", dtype=np.uint16, mode="r")
tok = Tokenizer.load("tokenizer.json")


def get_batch(split):
    """Pick batch_size random snippets. y is x shifted by one token:
    x = [Once,  upon, a,    time]
    y = [upon,  a,    time, ,   ]   <- the answer at each position"""
    data = train_data if split == "train" else val_data
    T = args.block_size
    ix = np.random.randint(0, len(data) - T - 1, args.batch_size)
    x = torch.from_numpy(np.stack([data[i:i + T] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1:i + T + 1] for i in ix]).astype(np.int64))
    return x.to(device), y.to(device)


# ------------------------------------------------------------------ model
start_iter, best_val = 0, float("inf")
if args.resume:
    ckpt = torch.load(args.out, map_location="cpu")
    cfg = GPTConfig(**ckpt["cfg"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    start_iter, best_val = ckpt["iter"], ckpt["best_val"]
    print(f"resumed from {args.out} at iter {start_iter} (best val {best_val:.3f})")
else:
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size,
                    n_layer=args.n_layer, n_head=args.n_head,
                    n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(cfg)
model.to(device)
print(f"model: {model.num_params()/1e6:.1f}M params | {cfg}")
print(f"data: {len(train_data):,} train tokens, {len(val_data):,} val tokens")
tokens_per_step = args.batch_size * args.block_size
print(f"{tokens_per_step:,} tokens/step x {args.max_iters} steps = "
      f"{tokens_per_step*args.max_iters/1e6:.1f}M tokens "
      f"(~{tokens_per_step*args.max_iters/len(train_data):.1f} passes over the data)\n")

# -------------------------------------------------------------- optimizer
# AdamW: a smarter version of "param -= lr * gradient". It keeps a running
# average of each parameter's gradients (momentum) and scales each step by how
# noisy that parameter's gradients have been, so every parameter gets a
# sensibly sized step.
#
# Weight decay: gently pulls weights toward 0 each step so no single weight
# gets huge (a form of regularisation). Applied to the big weight matrices
# only, not to LayerNorm scales/biases (1-D params).
decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": args.weight_decay},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95),
)
if args.resume and "optimizer" in ckpt:
    optimizer.load_state_dict(ckpt["optimizer"])


def get_lr(it):
    """Learning-rate schedule = how big each nudge is over time.
    1. warmup:  ramp 0 -> lr over the first steps (weights are random at the
                start; big steps then can throw training off course)
    2. cosine:  smoothly decay lr -> min_lr (big steps early to learn fast,
                small steps late to fine-tune)"""
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    progress = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))) * (args.lr - args.min_lr)


@torch.no_grad()
def estimate_loss():
    """Average loss over several batches (one batch alone is too noisy)."""
    model.eval()  # turns off dropout
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(args.eval_iters):
            with autocast():
                losses.append(model(*get_batch(split))[1].item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


@torch.no_grad()
def sample(prompt="Once upon a time", n=60):
    model.eval()
    idx = torch.tensor([tok.encode(prompt)], device=device)
    with autocast():
        out = model.generate(idx, n, temperature=0.8, top_k=50, stop_id=tok.eot_id)
    model.train()
    return tok.decode(out[0].tolist()).replace("\n", " ")


def save(it):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "cfg": asdict(cfg), "iter": it, "best_val": best_val}, args.out)


# ----------------------------------------------------------- the loop
log_file = open("train_log.csv", "a", newline="")
log = csv.writer(log_file)
if not args.resume:
    log.writerow(["iter", "train_loss", "val_loss", "lr"])

model.train()
t_start = time.time()
t0 = time.time()
steps_since_log = 0
it = start_iter
try:
    for it in range(start_iter, args.max_iters + 1):
        lr = get_lr(it)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # --- evaluate now and then: the honest score + a sample story
        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss()
            marker = ""
            if losses["val"] < best_val:
                best_val = losses["val"]
                if it > 0:
                    save(it)
                    marker = "  (saved)"
            print(f"\n=== iter {it}: train {losses['train']:.3f} | val {losses['val']:.3f}{marker}")
            print(f"    sample: {sample()}\n")
            log.writerow([it, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{lr:.2e}"])
            log_file.flush()
            if it == args.max_iters:
                break
            t0, steps_since_log = time.time(), 0  # don't count eval time in the speed numbers

        # --- one training step: forward -> backward -> step
        x, y = get_batch("train")
        with autocast():
            _, loss = model(x, y)        # 1. forward: how wrong are we?
        optimizer.zero_grad()            #    (clear gradients from the last step)
        loss.backward()                  # 2. backward: compute gradients
        # gradient clipping: if a rare bad batch produces a huge gradient,
        # scale it down so one step can't wreck the model
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()                 # 3. nudge all parameters
        steps_since_log += 1

        # --- progress line
        if it % args.log_interval == 0:
            dt = (time.time() - t0) / steps_since_log
            t0, steps_since_log = time.time(), 0
            done = it - start_iter + 1
            eta = (time.time() - t_start) / done * (args.max_iters - it)
            print(f"iter {it:5d} | loss {loss.item():.3f} | lr {lr:.2e} | "
                  f"{dt*1000:.0f} ms/step | {tokens_per_step/dt:,.0f} tok/s | "
                  f"eta {eta/60:.0f} min")

except KeyboardInterrupt:
    print("\nstopped by Ctrl+C")

print(f"\ndone at iter {it} in {(time.time()-t_start)/60:.1f} min. best val loss {best_val:.3f}")
print(f"best checkpoint: {args.out}   (log: train_log.csv)")
log_file.close()

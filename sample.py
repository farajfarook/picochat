"""
Step 5: talk to your trained model.

Run:
    uv run python sample.py                       # interactive playground
    uv run python sample.py --prompt "The dragon"  # one-shot, then exit

HOW GENERATION WORKS
--------------------
The model only ever does one thing: given the tokens so far, output a score
(logit) for each of the 4096 possible next tokens. To write a story:

    1. run the model on the prompt -> scores for the next token
    2. turn scores into probabilities (softmax)
    3. PICK one token (how we pick is the interesting part, see below)
    4. append it to the text and go back to 1

HOW TO PICK: the two dials
--------------------------
temperature: divide the scores by T before softmax.
    T < 1  -> sharpens the distribution: likely tokens get even likelier.
              Safe, repetitive, "boring".
    T = 1  -> the model's true probabilities
    T > 1  -> flattens it: unlikely tokens get a real chance.
              Creative, then chaotic, then gibberish.
    T -> 0 -> always pick the single most likely token ("greedy")

top_k: before picking, throw away everything except the k most likely tokens.
    Stops the model from ever choosing a really bad 1-in-10,000 token, while
    keeping some variety among the good ones.
"""

import argparse
import sys

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from tokenizer import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="ckpt.pt")
p.add_argument("--prompt", default=None)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top_k", type=int, default=50)
p.add_argument("--max_tokens", type=int, default=200)
p.add_argument("--seed", type=int, default=None)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = Tokenizer.load("tokenizer.json")
ckpt = torch.load(args.ckpt, map_location=device)
model = GPT(GPTConfig(**ckpt["cfg"])).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
assert model.cfg.vocab_size == tok.vocab_size, "tokenizer.json doesn't match the checkpoint"

settings = {"temperature": args.temperature, "top_k": args.top_k, "max_tokens": args.max_tokens}


def piece(i):
    return "<|endoftext|>" if i == tok.eot_id else repr(tok.decode([i]))


@torch.no_grad()
def next_token_probs(ids):
    """The model's raw probabilities for the next token (temperature 1, no top-k)."""
    x = torch.tensor([ids[-model.cfg.block_size:]], device=device)
    logits, _ = model(x)
    return F.softmax(logits[0, -1].float(), dim=-1)


@torch.no_grad()
def generate(prompt):
    """Stream a story token by token so you can watch it being written."""
    ids = tok.encode(prompt)
    print(prompt, end="", flush=True)
    printed = prompt
    for _ in range(settings["max_tokens"]):
        x = torch.tensor([ids[-model.cfg.block_size:]], device=device)
        logits, _ = model(x)
        logits = logits[0, -1].float()
        T = settings["temperature"]
        if T <= 0:  # greedy
            nxt = int(logits.argmax())
        else:
            logits = logits / T
            k = settings["top_k"]
            if k and k < logits.numel():
                cutoff = torch.topk(logits, k).values[-1]
                logits[logits < cutoff] = float("-inf")
            nxt = int(torch.multinomial(F.softmax(logits, dim=-1), 1))
        if nxt == tok.eot_id:
            print("\n  [end of story]", end="")
            break
        ids.append(nxt)
        # decode the whole text and print only the new part (handles multi-byte chars)
        text = tok.decode(ids)
        print(text[len(printed):], end="", flush=True)
        printed = text
    print("\n")


def show_probs(prompt, n=15):
    """What does the model think comes next? Show its top choices."""
    ids = tok.encode(prompt)
    probs = next_token_probs(ids)
    top = torch.topk(probs, n)
    print(f"\n  after {prompt!r}, the model predicts:")
    for pr, i in zip(top.values.tolist(), top.indices.tolist()):
        bar = "█" * int(pr * 50)
        print(f"    {pr*100:5.1f}%  {piece(i):<16} {bar}")
    print(f"  (top {n} cover {top.values.sum().item()*100:.0f}% of the probability)\n")


def compare_temps(prompt, temps=(0.0, 0.5, 1.0, 1.5)):
    old = settings["temperature"], settings["max_tokens"]
    settings["max_tokens"] = 60
    for T in temps:
        settings["temperature"] = T
        print(f"--- temperature {T}{' (greedy)' if T == 0 else ''} ---")
        generate(prompt)
    settings["temperature"], settings["max_tokens"] = old


HELP = """
Type a story opening and the model continues it. Commands:
  :probs <text>      show the model's top next-token guesses
  :temps <text>      same prompt at temperature 0, 0.5, 1.0, 1.5
  :t <value>         set temperature   (now {temperature})
  :k <value>         set top_k, 0=off  (now {top_k})
  :n <value>         set max tokens    (now {max_tokens})
  :help  :quit
"""

if args.seed is not None:
    torch.manual_seed(args.seed)

print(f"loaded {args.ckpt}: {model.num_params()/1e6:.1f}M params, "
      f"trained {ckpt['iter']} steps, val loss {ckpt['best_val']:.3f}, on {device}")

if args.prompt is not None:
    generate(args.prompt)
    sys.exit()

print(HELP.format(**settings))
while True:
    try:
        line = input("prompt> ")
    except (EOFError, KeyboardInterrupt):
        break
    cmd, _, arg = line.partition(" ")
    try:
        if cmd in (":quit", ":q"):
            break
        elif cmd == ":help":
            print(HELP.format(**settings))
        elif cmd == ":probs":
            show_probs(arg or "Once upon a time")
        elif cmd == ":temps":
            compare_temps(arg or "Once upon a time")
        elif cmd == ":t":
            settings["temperature"] = float(arg)
            print(f"  temperature = {settings['temperature']}\n")
        elif cmd == ":k":
            settings["top_k"] = int(arg)
            print(f"  top_k = {settings['top_k']}\n")
        elif cmd == ":n":
            settings["max_tokens"] = int(arg)
            print(f"  max_tokens = {settings['max_tokens']}\n")
        else:
            generate(line if line else "Once upon a time")
    except ValueError:
        print("  bad value\n")

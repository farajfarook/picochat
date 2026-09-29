"""
Step 3: a tiny GPT, written by hand in PyTorch.

WHAT THE MODEL DOES
-------------------
Input:  a list of token ids, e.g. [424, 434, 258, 394]  ("Once upon a time")
Output: for EACH position, a guess about which token comes NEXT.
        After "Once"             -> probably " upon"
        After "Once upon"        -> probably " a"
        After "Once upon a"      -> probably " time"
That's all a language model is: a next-token predictor. To generate a story
we predict one token, append it, and repeat.

THE PIPELINE (for a sequence of T tokens)
-----------------------------------------
  token ids            (T,)            [424, 434, 258, 394]
      │
  token embedding      (T, C)          each id -> a vector of C numbers (its "meaning")
  + position embedding (T, C)          each position 0..T-1 -> a vector ("where am I")
      │
  Block × n_layer      (T, C)          each block = attention + MLP (see below)
      │
  final LayerNorm      (T, C)
  lm_head (Linear)     (T, vocab)      a score (logit) for every possible next token
      │
  softmax                              scores -> probabilities

INSIDE A BLOCK
--------------
  x = x + Attention(LayerNorm(x))   # tokens LOOK AT EACH OTHER (communicate)
  x = x + MLP(LayerNorm(x))         # each token THINKS on its own (compute)

The "x = x + ..." is a *residual connection*: each layer adds a small update
on top of what's already there instead of replacing it. This keeps training
stable in deep networks.

Shapes used below: B = batch size, T = sequence length, C = n_embd (vector size)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096  # from our tokenizer
    block_size: int = 256   # max context length (tokens the model can see at once)
    n_layer: int = 6        # number of transformer blocks stacked
    n_head: int = 6         # attention heads per block
    n_embd: int = 384       # size of each token's vector (C)
    dropout: float = 0.0    # randomly zero some activations during training (regularisation)


class CausalSelfAttention(nn.Module):
    """
    The heart of the transformer: tokens gather information from earlier tokens.

    Each token produces three vectors:
      q (query): "what am I looking for?"
      k (key):   "what do I contain?"
      v (value): "what will I hand over if you pick me?"

    Token i compares its query with the key of every token j <= i
    (dot product = similarity score). High score = "that token is relevant to
    me". Scores are turned into weights with softmax, and token i receives a
    weighted average of the values.

    Example: in "Tom lost his ball. He was sad", the token " He" can learn
    to put high attention weight on "Tom".

    CAUSAL = a token may only look at itself and the PAST, never the future.
    Otherwise it could cheat by peeking at the very token it's trying to predict.

    MULTI-HEAD = run several small attentions in parallel (n_head of them, each
    of size C / n_head). One head might track "who is the subject", another
    "what was the last punctuation", etc. Their outputs are concatenated.
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_size = cfg.n_embd // cfg.n_head
        # one Linear layer produces q, k and v for all heads at once (3*C outputs)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        # after the heads are concatenated, mix them back together
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        # the causal mask: lower-triangular matrix of ones.
        #   [[1,0,0],
        #    [1,1,0],     row i = which positions token i may look at
        #    [1,1,1]]
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape

        # 1. compute q, k, v for every token: (B, T, 3C) -> three tensors of (B, T, C)
        q, k, v = self.qkv(x).split(C, dim=2)

        # 2. split C into n_head heads of head_size each, and move heads next to batch:
        #    (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # 3. attention scores: every query dotted with every key -> (B, n_head, T, T)
        #    divided by sqrt(head_size) so the numbers don't get huge as vectors grow
        #    (huge scores would make softmax pick one token with ~100% and stop learning)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)

        # 4. causal mask: set scores for future positions to -inf, so after softmax
        #    their weight is exactly 0
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # 5. softmax turns each row of scores into weights that sum to 1
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # 6. weighted average of the values: (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v

        # 7. glue the heads back together: (B, nh, T, hs) -> (B, T, C), then mix
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))

        # NOTE: steps 3-6 are exactly what F.scaled_dot_product_attention(q, k, v,
        # is_causal=True) does, only faster. We write it out by hand to learn it.


class MLP(nn.Module):
    """
    After attention has gathered info from other tokens, each token "thinks"
    about it independently: expand to 4x size, apply a non-linearity (GELU),
    shrink back. Much of the model's factual knowledge ("dogs bark",
    "after 'Once upon a' comes ' time'") ends up stored in these weights.
    """

    def __init__(self, cfg):
        super().__init__()
        self.up = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.down = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # without the non-linearity (GELU), up+down would collapse into one
        # linear layer and the network could only learn straight-line relationships
        return self.dropout(self.down(F.gelu(self.up(x))))


class Block(nn.Module):
    """One transformer block: communicate (attention), then compute (MLP)."""

    def __init__(self, cfg):
        super().__init__()
        # LayerNorm rescales each token's vector to mean 0, std 1 (plus a learned
        # scale/shift). Keeps numbers in a healthy range as they flow through layers.
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # residual: add, don't replace
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # lookup table: id -> vector
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # position -> vector
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # WEIGHT TYING: the input embedding (id -> vector) and the output layer
        # (vector -> score per id) share one matrix. Intuition: a token's
        # meaning is the same whether it's being read or predicted. It saves
        # vocab_size * n_embd params (1.6M here) and usually helps small models.
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2 trick: scale down the init of layers that write into the residual
        # stream, so the sum of n_layer updates doesn't blow up at the start
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        # start with small random weights (std 0.02, as in GPT-2)
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) token ids
        targets: (B, T) the correct NEXT token for each position (optional)
        returns: logits (B, T, vocab_size), and loss if targets given
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} longer than block_size"
        pos = torch.arange(T, device=idx.device)

        x = self.tok_emb(idx) + self.pos_emb(pos)  # (B,T,C) + (T,C) -> (B,T,C)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # CROSS-ENTROPY LOSS = -log(probability the model gave to the correct token),
            # averaged over every position in the batch.
            #   gave the right token 100%  -> loss 0
            #   gave it 1/4096 (a random guess) -> loss ln(4096) = 8.32
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_id=None):
        """Autoregressive generation: predict one token, append it, repeat."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]  # can only see the last block_size tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # only the LAST position's prediction matters
            if top_k is not None:  # keep only the k most likely tokens
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # random draw, weighted by probs
            idx = torch.cat([idx, next_id], dim=1)
            if stop_id is not None and next_id.item() == stop_id:
                break
        return idx


# ---------------------------------------------------------------------------
# `python model.py` builds the model, counts params, checks the loss of an
# untrained model, times one training step, and lets the untrained model "talk".
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, time
    import numpy as np
    from tokenizer import Tokenizer

    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(0)
    cfg = GPTConfig()
    model = GPT(cfg)

    # --- parameter count, broken down ---
    C, V, L = cfg.n_embd, cfg.vocab_size, cfg.n_layer
    print(f"config: {cfg}\n")
    print("parameters:")
    print(f"  token embedding  {V} x {C}            = {V*C:>10,}  (shared with lm_head)")
    print(f"  position emb     {cfg.block_size} x {C}             = {cfg.block_size*C:>10,}")
    per_block = sum(p.numel() for p in model.blocks[0].parameters())
    print(f"  per block        attn 4C² + mlp 8C² + ln = {per_block:>10,}")
    print(f"  x {L} blocks                              = {per_block*L:>10,}")
    print(f"  TOTAL                                    = {model.num_params():>10,}"
          f"  ({model.num_params()/1e6:.1f}M)\n")

    # --- loss of an untrained model on real data ---
    data = np.fromfile("data/val.bin", dtype=np.uint16)
    B, T = 16, cfg.block_size
    starts = np.random.randint(0, len(data) - T - 1, B)
    x = torch.tensor(np.stack([data[s:s + T] for s in starts]).astype(np.int64))
    y = torch.tensor(np.stack([data[s + 1:s + T + 1] for s in starts]).astype(np.int64))
    # note: y is x shifted left by one -> target at position t is the token at t+1
    logits, loss = model(x, y)
    print(f"logits shape: {tuple(logits.shape)}  (batch, time, vocab)")
    print(f"untrained loss: {loss.item():.3f}   (random guessing = ln({V}) = {math.log(V):.3f})\n")

    # --- how fast is one training step on this CPU? ---
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for i in range(4):
        t0 = time.time()
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        dt = time.time() - t0
    print(f"one training step (B={B}, T={T} = {B*T:,} tokens): {dt*1000:.0f} ms "
          f"-> ~{B*T/dt:,.0f} tokens/sec\n")

    # --- the untrained model "talks" (expect gibberish) ---
    tok = Tokenizer.load("tokenizer.json")
    model = GPT(cfg)  # fresh untrained model
    start = torch.tensor([tok.encode("Once upon a time")])
    out = model.generate(start, max_new_tokens=30)
    print("untrained model says:\n  " + tok.decode(out[0].tolist()))

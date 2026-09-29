"""
Encode the FULL TinyStories dataset into token files.

    uv run python data/download.py --train    # 2.2 GB download (once)
    uv run python data/prepare.py             # encode with all CPU cores (~5-10 min)

Output:
    data/train.bin   the full train split   (~2.1M stories, ~470M tokens)
    data/val.bin     the full valid split   (~27k stories, ~5.6M tokens)

This uses the EXISTING tokenizer.json (it doesn't retrain it), so the vocab
stays at 4096 and old checkpoints still decode correctly.

Why not just `tok.encode(open(file).read())` like tokenizer.py does?
  - 2.2 GB of text as one Python string, plus ~550M chunk strings from the
    regex, needs tens of GB of RAM
  - one CPU core would take a long time
So we read the file in pieces (always cutting at a story boundary), encode
the pieces in parallel on every core, and append the results to the .bin file.
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import EOT, Tokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PIECE_BYTES = 16 * 1024 * 1024  # read 16 MB at a time

_tok = None


def _init():
    # each worker process loads its own copy of the tokenizer
    global _tok
    _tok = Tokenizer.load(os.path.join(HERE, "..", "tokenizer.json"))


def _encode(text):
    return np.array(_tok.encode(text), dtype=np.uint16), len(text.encode("utf-8"))


def pieces(path):
    """Yield chunks of the file, each ending exactly at a story boundary."""
    leftover = ""
    with open(path, encoding="utf-8") as f:
        while True:
            block = f.read(PIECE_BYTES)
            if not block:
                break
            block = leftover + block
            cut = block.rfind(EOT)
            if cut == -1:
                leftover = block
                continue
            cut += len(EOT)
            leftover = block[cut:]
            yield block[:cut]
    if leftover.strip():
        yield leftover


def encode_file(src, dst, pool):
    size = os.path.getsize(src)
    t0 = time.time()
    total, done_bytes = 0, 0
    with open(dst, "wb") as out:
        # imap keeps the pieces in their original order
        for ids, nbytes in pool.imap(_encode, pieces(src)):
            ids.tofile(out)
            total += len(ids)
            done_bytes += nbytes
            pct = done_bytes / size * 100
            rate = done_bytes / 1e6 / (time.time() - t0)
            eta = (size - done_bytes) / 1e6 / max(rate, 1e-9)
            print(f"\r  {os.path.basename(dst)}: {pct:5.1f}%  {total/1e6:7.1f}M tokens  "
                  f"{rate:5.1f} MB/s  eta {eta/60:4.1f} min", end="", flush=True)
    print(f"\n  -> {dst}: {total:,} tokens in {(time.time()-t0)/60:.1f} min")
    return total


if __name__ == "__main__":
    train_txt = os.path.join(HERE, "TinyStoriesV2-GPT4-train.txt")
    val_txt = os.path.join(HERE, "TinyStoriesV2-GPT4-valid.txt")
    if not os.path.exists(train_txt):
        sys.exit("missing train file: run `uv run python data/download.py --train` first")

    n = os.cpu_count()
    print(f"encoding with {n} processes")
    with Pool(n, initializer=_init) as pool:
        encode_file(val_txt, os.path.join(HERE, "val.bin"), pool)
        encode_file(train_txt, os.path.join(HERE, "train.bin"), pool)

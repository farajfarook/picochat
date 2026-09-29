"""
Step 2: a Byte-Pair Encoding (BPE) tokenizer, written from scratch.

A neural net can only work with numbers, so we need a way to turn text into a
list of integers (tokens) and back again. BPE is what GPT-2/3/4 use.

THE IDEA
--------
1. Start with bytes. Any text is just a sequence of bytes (0-255) in UTF-8,
   so our starting vocabulary is 256 tokens and no text is ever "unknown".
       "hug" -> [104, 117, 103]
2. Find the most frequent pair of adjacent tokens in the training text,
   e.g. (104 'h', 117 'u'), and give it a brand-new token id, 256.
       "hug" -> [256, 103]
3. Repeat. Each merge adds one token. Frequent chunks ("the", " was",
   " happy") end up as single tokens, and rare words stay split into pieces.
4. Stop when the vocabulary reaches the size we want (4096 here).

Result: common text becomes much shorter sequences (about 4 chars/token instead
of 1), so the model can see more text within its fixed context window.

WHY PRE-SPLIT INTO WORDS?
-------------------------
Without it, BPE would happily merge across words ("dog." or "d the") and waste
vocab on junk. We first chop text into word-like chunks with a regex (the same
one GPT-2 uses) and only merge *inside* chunks. Note that the leading space
stays attached: " dog" is a chunk, so "dog" and " dog" become different tokens.

A SPEED TRICK
-------------
TinyStories has millions of words but only a few tens of thousands of
*distinct* ones. So we count each distinct chunk once ("the" x 900,000) and do
all the pair counting on that small table instead of the full text.
"""

import json
from collections import Counter

import regex as re  # like `re`, but supports \p{L} (= "any letter")

# GPT-2's pre-tokenization pattern. It splits text into:
#   contractions ('s 't 're ...), words with an optional leading space,
#   numbers, punctuation runs, and whitespace.
SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

EOT = "<|endoftext|>"  # special token: "a story ends here"


def get_pair_counts(words):
    """words: dict {tuple_of_token_ids: frequency}.
    Returns Counter {(a, b): how often token a is directly followed by b}."""
    counts = Counter()
    for ids, freq in words.items():
        for pair in zip(ids, ids[1:]):
            counts[pair] += freq
    return counts


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in the sequence `ids` with `new_id`.
    merge((1, 2, 3, 1, 2), (1, 2), 99) -> (99, 3, 99)"""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


class Tokenizer:
    def __init__(self):
        self.merges = {}  # (a, b) -> new_id, in the order they were learned
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes
        self.eot_id = None
        self.pattern = re.compile(SPLIT_PATTERN)
        self._cache = {}  # chunk -> token ids (speeds up encoding a lot)

    @property
    def vocab_size(self):
        return len(self.vocab) + 1  # +1 for the special EOT token

    # ------------------------------------------------------------------ train
    def train(self, text, vocab_size, verbose=True):
        num_merges = vocab_size - 256 - 1  # 256 bytes + merges + 1 special

        # 1. split into chunks, drop EOT markers, count distinct chunks
        chunk_counts = Counter()
        for story in text.split(EOT):
            chunk_counts.update(self.pattern.findall(story))
        # each chunk as a tuple of byte values
        words = {tuple(c.encode("utf-8")): n for c, n in chunk_counts.items()}
        if verbose:
            print(f"{sum(chunk_counts.values()):,} chunks, {len(words):,} distinct")

        # 2. the BPE loop
        for i in range(num_merges):
            pair_counts = get_pair_counts(words)
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.get)  # most frequent pair
            new_id = 256 + i
            # apply the merge to every word that contains the pair
            words = {merge(w, best, new_id): n for w, n in words.items()}
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            if verbose and (i < 20 or i % 250 == 0):
                print(f"merge {i+1:4d}/{num_merges}: {best} -> {new_id} "
                      f"{self.vocab[new_id]!r} (seen {pair_counts[best]:,}x)")

        self.eot_id = len(self.vocab)
        self._cache.clear()

    # ----------------------------------------------------------------- encode
    def _encode_chunk(self, chunk):
        """Encode one pre-split chunk by replaying the merges in the order they
        were learned (earliest merge = lowest id = highest priority)."""
        if chunk in self._cache:
            return self._cache[chunk]
        ids = tuple(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # of all adjacent pairs, find the one that was learned first
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # nothing left to merge
            ids = merge(ids, pair, self.merges[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        """text -> list of token ids. '<|endoftext|>' becomes self.eot_id."""
        out = []
        for i, part in enumerate(text.split(EOT)):
            if i > 0:
                out.append(self.eot_id)
            for chunk in self.pattern.findall(part):
                out.extend(self._encode_chunk(chunk))
        return out

    # ----------------------------------------------------------------- decode
    def decode(self, ids):
        """list of token ids -> text."""
        parts = []
        for i in ids:
            parts.append(EOT.encode() if i == self.eot_id else self.vocab[i])
        # errors="replace": a partial multi-byte character becomes '�' instead of crashing
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------- save/load
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": [[a, b, n] for (a, b), n in self.merges.items()]}, f)

    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path) as f:
            for a, b, n in json.load(f)["merges"]:
                tok.merges[(a, b)] = n
                tok.vocab[n] = tok.vocab[a] + tok.vocab[b]
        tok.eot_id = len(tok.vocab)
        return tok


# ---------------------------------------------------------------------------
# Run `python tokenizer.py` to: train the tokenizer, save it, then encode the
# whole dataset into data/train.bin and data/val.bin (arrays of uint16 ids).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, time
    import numpy as np

    sys.stdout.reconfigure(encoding="utf-8")  # Windows console can't print emoji otherwise

    VOCAB_SIZE = 4096
    text = open("data/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8").read()

    # train the tokenizer on a slice (BPE doesn't need all the text)
    t0 = time.time()
    tok = Tokenizer()
    tok.train(text[:10_000_000], VOCAB_SIZE)
    tok.save("tokenizer.json")
    print(f"trained in {time.time()-t0:.0f}s, vocab_size={tok.vocab_size}")

    # sanity check: encode -> decode must give back the exact same text
    sample = "Once upon a time, there was a little dog named Max. 😀"
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample
    print(f"\n{sample!r}\n-> {len(ids)} tokens: {ids}")
    print("pieces:", [tok.decode([i]) for i in ids])

    # encode the full dataset: first 90% of stories = train, last 10% = val
    t0 = time.time()
    stories = text.split(EOT)
    split = int(len(stories) * 0.9)
    total = 0
    for name, part in [("train", stories[:split]), ("val", stories[split:])]:
        ids = np.array(tok.encode(EOT.join(part)), dtype=np.uint16)  # 4096 < 65536
        ids.tofile(f"data/{name}.bin")
        total += len(ids)
        print(f"{name}: {len(ids):,} tokens")
    print(f"encoded in {time.time()-t0:.0f}s, {len(text)/total:.2f} chars/token")

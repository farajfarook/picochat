"""
Tokenizer playground. Run:  python play_tokenizer.py

Type any text to see how it gets tokenized. Commands:
  :trace <word>   show the merge-by-merge build-up of a word
  :id <n>         what text is token n?
  :find <text>    search the vocab for tokens containing <text>
  :first <n>      show the first n merges learned (default 30)
  :last <n>       show the last n merges learned (default 30)
  :bytes <text>   show the raw UTF-8 bytes of some text
  :quit           exit
"""
import sys
from tokenizer import Tokenizer, merge

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")
tok = Tokenizer.load("tokenizer.json")

# alternating background colours so token boundaries are visible
COLORS = ["\033[48;5;24m", "\033[48;5;94m", "\033[48;5;22m", "\033[48;5;90m", "\033[48;5;58m"]
RESET = "\033[0m"


def piece(i):
    """Readable form of one token (show spaces as · and newlines as ↵)."""
    s = tok.decode([i])
    return s.replace(" ", "·").replace("\n", "↵")


def show_tokens(text):
    ids = tok.encode(text)
    colored = "".join(f"{COLORS[k % len(COLORS)]}{piece(i)}{RESET}" for k, i in enumerate(ids))
    print(f"\n  {colored}")
    print(f"\n  ids:    {ids}")
    print(f"  pieces: {[piece(i) for i in ids]}")
    n_bytes = len(text.encode('utf-8'))
    print(f"  {len(text)} chars, {n_bytes} bytes → {len(ids)} tokens "
          f"({n_bytes / max(len(ids), 1):.2f} bytes/token)\n")


def trace(word):
    ids = tuple(word.encode("utf-8"))
    print(f"\n  start       : {[piece(i) for i in ids]}  ({len(ids)} tokens)")
    while len(ids) >= 2:
        pair = min(zip(ids, ids[1:]), key=lambda p: tok.merges.get(p, float("inf")))
        if pair not in tok.merges:
            break
        new = tok.merges[pair]
        ids = merge(ids, pair, new)
        print(f"  merge #{new - 255:<5}: {[piece(i) for i in ids]}   "
              f"({piece(pair[0])} + {piece(pair[1])} → {piece(new)})")
    print(f"  done: {len(ids)} token(s)\n")


def list_merges(items):
    for (a, b), n in items:
        print(f"  #{n - 255:<5} id {n:<5} {piece(a):>12} + {piece(b):<12} → {piece(n)}")
    print()


print(__doc__)
print(f"Loaded tokenizer: vocab_size = {tok.vocab_size}\n")

while True:
    try:
        line = input("text> ")
    except (EOFError, KeyboardInterrupt):
        break
    cmd, _, arg = line.partition(" ")
    if cmd in (":quit", ":q"):
        break
    elif cmd == ":trace":
        trace(arg if arg.startswith(" ") else " " + arg)  # words mid-sentence have a leading space
    elif cmd == ":id":
        try:
            n = int(arg)
            name = "<|endoftext|>" if n == tok.eot_id else repr(tok.decode([n]))
            print(f"  token {n} = {name}\n")
        except (ValueError, KeyError):
            print(f"  give a number 0..{tok.vocab_size - 1}\n")
    elif cmd == ":find":
        hits = [(i, piece(i)) for i in tok.vocab if arg.lower() in tok.decode([i]).lower()]
        print(f"  {len(hits)} tokens contain {arg!r}:")
        print("  " + ", ".join(f"{i}:{p}" for i, p in hits[:60]) + (" ..." if len(hits) > 60 else "") + "\n")
    elif cmd == ":first":
        list_merges(list(tok.merges.items())[: int(arg or 30)])
    elif cmd == ":last":
        list_merges(list(tok.merges.items())[-int(arg or 30):])
    elif cmd == ":bytes":
        print(f"  {list(arg.encode('utf-8'))}\n")
    elif line:
        show_tokens(line)

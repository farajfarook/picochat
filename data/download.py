"""Step 1: download TinyStories (V2, GPT-4 generated).

valid ~ 22 MB  (plenty to train the tokenizer and a first model)
train ~ 2.2 GB (optional, use with --train for longer runs)
"""
import sys, urllib.request, pathlib

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
FILES = ["TinyStoriesV2-GPT4-valid.txt"]
if "--train" in sys.argv:
    FILES.append("TinyStoriesV2-GPT4-train.txt")

out_dir = pathlib.Path(__file__).parent
for name in FILES:
    dest = out_dir / name
    if dest.exists():
        print(f"already have {name}")
        continue
    print(f"downloading {name} ...")
    urllib.request.urlretrieve(BASE + name, dest)
    print(f"  -> {dest} ({dest.stat().st_size/1e6:.1f} MB)")

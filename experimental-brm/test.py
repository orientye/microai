from reasoning_from_scratch.qwen3 import Qwen3Tokenizer
from pathlib import Path
tokenizer = Qwen3Tokenizer(tokenizer_file_path=Path("qwen3")/"tokenizer-base.json")
ids = tokenizer.encode("中文测试")
for i in ids:
    print(i, "-->", repr(tokenizer.decode([i])))
ids = tokenizer.encode("魑魅魍魉")
for i in ids:
    print(i, "-->", repr(tokenizer.decode([i])))
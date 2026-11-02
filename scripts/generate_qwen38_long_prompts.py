#!/usr/bin/env python3
"""Build the same ShareGPT token-ID fixtures as /opt/1m.md.

Example:
    python3 generate_qwen38_long_prompts.py
    python3 generate_qwen38_long_prompts.py /tmp/prompts.npz --model /models/...
"""

import argparse
import json
from pathlib import Path
import random


DEFAULT_PROMPTS = Path(__file__).resolve().parent / "prompts.npz"
DATASET_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
DATASET_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET_REVISION = "192ab2185289094fc556ec8ce5ce1e8e587154ca"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--model", default="/models/Qwen3.8-Flash-Next-PTPC-FP8")
    parser.add_argument("--dataset-path", type=Path, help="Use a local ShareGPT JSON file.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import numpy as np
    from transformers import AutoTokenizer

    dataset_path = args.dataset_path
    if dataset_path is None:
        from huggingface_hub import hf_hub_download

        dataset_path = Path(
            hf_hub_download(
                DATASET_REPO,
                DATASET_FILE,
                repo_type="dataset",
                revision=DATASET_REVISION,
            )
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with dataset_path.open() as stream:
        data = json.load(stream)
    random.Random(args.seed).shuffle(data)

    lengths = (131072, 262144, 524288, 999648)
    names = ["w12000", "w131072"] + [
        f"L{length}_r{replica}" for length in lengths for replica in (0, 1)
    ]
    sizes = [12000, 131072] + [length for length in lengths for _ in (0, 1)]
    needed = sum(sizes)
    token_stream = np.empty(needed, dtype=np.int32)
    filled = processed = 0
    for first in range(0, len(data), 256):
        batch = data[first : first + 256]
        texts = [
            "\n\n".join(turn["value"] for turn in conversation.get("conversations", []))
            for conversation in batch
        ]
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        processed += len(batch)
        for ids in encoded:
            count = min(len(ids), needed - filled)
            if count:
                token_stream[filled : filled + count] = ids[:count]
                filled += count
            if filled == needed:
                break
        print(f"conversations={processed}, tokens={filled}/{needed}", flush=True)
        if filled == needed:
            break
    if filled != needed:
        raise SystemExit(f"Dataset too small: {filled} tokens, need {needed}.")

    output = args.output.expanduser().resolve()
    if not str(output).endswith(".npz"):
        output = Path(str(output) + ".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.cumsum([0] + sizes)
    np.savez(
        output,
        **{
            name: token_stream[offsets[index] : offsets[index + 1]]
            for index, name in enumerate(names)
        },
    )
    print("wrote", output, "tokens", needed, "max_id", int(token_stream.max()))
    print(json.dumps(dict(zip(names, sizes)), indent=2))


if __name__ == "__main__":
    main()

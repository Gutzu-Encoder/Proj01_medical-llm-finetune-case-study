"""Pull a train/eval split from YuSun-AI/ReasonMed and write it to local JSONL files.
Runs locally (CPU only, no GPU needed) - just needs the `datasets` package and internet.
"""
import json
import os

from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
N_TRAIN = 300
N_EVAL = 30
SEED = 42


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = load_dataset("YuSun-AI/ReasonMed", split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=5000)

    train_rows = []
    eval_rows = []

    for row in ds:
        instr = (row.get("instruction") or "").strip()
        out = (row.get("output") or "").strip()
        if not instr or not out:
            continue
        record = {"question": instr, "reference": out}
        if len(eval_rows) < N_EVAL:
            eval_rows.append(record)
        elif len(train_rows) < N_TRAIN:
            train_rows.append(record)
        if len(train_rows) >= N_TRAIN and len(eval_rows) >= N_EVAL:
            break

    with open(os.path.join(OUT_DIR, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(OUT_DIR, "eval.jsonl"), "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(train_rows)} train rows and {len(eval_rows)} eval rows to {OUT_DIR}")


if __name__ == "__main__":
    main()

"""Independently verify the TRAINING LABELS themselves (not model outputs).
Asks DeepSeek Flash to independently determine the correct answer for each
question and assess whether the ReasonMed reference label agrees and is
internally sound (no leftover self-correction artifacts, no contradictions).
Runs locally - just needs internet + DEEPSEEK_API_KEY (loaded from .env).

Runs requests concurrently (deepseek-flash is a reasoning model - its hidden
reasoning_content eats into max_tokens, so that budget is set generously high
here to avoid truncated/empty responses).
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"


def load_dotenv(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and value and key not in os.environ:
                os.environ[key] = value


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

JUDGE_SYSTEM = """You are an expert medical reviewer fact-checking a training label \
before it is used to fine-tune a model. You will be given a multiple-choice medical \
question and a PROPOSED reference answer (reasoning + final answer), which may or may \
not actually be correct. Do not assume the proposed answer is right.

First, independently determine what you believe the correct answer actually is, using \
your own medical knowledge - ignore the proposed answer while doing this.

Then score the PROPOSED reference on:
- label_correct: 0-5, does the proposed final answer match what you independently \
  believe is correct? (0 = clearly wrong, 5 = clearly correct)
- internal_consistency: 0-5, is the reasoning free of self-contradiction, and free of \
  odd leftover artifacts like references to "previous errors," "error feedback," or \
  correcting an earlier draft that isn't actually shown? (5 = clean, 0 = badly \
  contaminated with such artifacts)
- reasoning_quality: 0-5, is the medical reasoning itself sound and well-justified, \
  independent of whether the final answer is correct?

End your response with ONLY a JSON object on its own, no other text after it, in this \
exact shape:
{"your_independent_answer": "<letter and brief reason>", "label_correct": <0-5>, \
"internal_consistency": <0-5>, "reasoning_quality": <0-5>, "note": "<one short sentence>"}
"""


def judge_one(question, reference, api_key, retries=3):
    user_content = f"QUESTION:\n{question}\n\nPROPOSED REFERENCE ANSWER:\n{reference}\n"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "thinking": {"type": "disabled"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                raise ValueError(f"no JSON found in judge output: {content[:200]}")
            return json.loads(match.group(0))
        except Exception as e:
            if attempt == retries - 1:
                return {"error": str(e)}
            time.sleep(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-file", required=True)
    p.add_argument("--out-file", required=True)
    p.add_argument("--limit", type=int, default=None, help="Only judge the first N rows")
    p.add_argument("--workers", type=int, default=10, help="Concurrent requests")
    args = p.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set the DEEPSEEK_API_KEY environment variable first.")

    rows = []
    with open(args.in_file, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[:args.limit]

    results = [None] * len(rows)
    write_lock = threading.Lock()
    out_f = open(args.out_file, "w", encoding="utf-8")
    done = 0

    def work(i):
        row = rows[i]
        scores = judge_one(row["question"], row["reference"], api_key)
        return i, {**row, "label_review": scores}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, i): i for i in range(len(rows))}
        for future in as_completed(futures):
            i, result = future.result()
            results[i] = result
            done += 1
            with write_lock:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
            print(f"[{done}/{len(rows)}] (row {i}) {result['label_review']}", flush=True)

    out_f.close()

    axes = ["label_correct", "internal_consistency", "reasoning_quality"]
    sums = {a: 0.0 for a in axes}
    counts = {a: 0 for a in axes}
    bad_labels = []
    for r in results:
        s = r["label_review"]
        if "error" in s:
            continue
        for a in axes:
            if a in s:
                sums[a] += s[a]
                counts[a] += 1
        if s.get("label_correct", 5) <= 2:
            bad_labels.append(r["question"][:100])

    print("\n=== Aggregate (mean out of 5) ===")
    for a in axes:
        if counts[a]:
            print(f"{a}: {sums[a] / counts[a]:.2f}  (n={counts[a]})")

    print(f"\n{len(bad_labels)}/{len(results)} labels flagged as likely WRONG (label_correct <= 2):")
    for q in bad_labels:
        print(" -", q)


if __name__ == "__main__":
    main()

"""LLM-judge eval using DeepSeek Flash. Runs locally (just needs internet + DEEPSEEK_API_KEY).
Scores each model response against the ReasonMed reference on 5 axes, 0-5 each.
"""
import argparse
import json
import os
import re
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


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

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"

AXES = ["correctness", "logical_coherence", "faithfulness", "hallucination_free", "omission_free"]

JUDGE_SYSTEM = """You are a strict medical-exam answer grader. You will be given a question, \
a reference answer (reasoning + final answer, treated as ground truth), and a candidate \
model response. Score the candidate response on each axis from 0 (very bad) to 5 (excellent):

- correctness: does the candidate reach the same final answer as the reference?
- logical_coherence: does the reasoning logically support the conclusion, step by step?
- faithfulness: is the reasoning grounded in the actual question/medical facts (not fabricated)?
- hallucination_free: 5 = no fabricated facts/citations/data, 0 = severe hallucination.
- omission_free: 5 = no important reasoning steps or considerations are missing, 0 = major gaps.

Respond with ONLY a JSON object, no other text, in this exact shape:
{"correctness": <0-5>, "logical_coherence": <0-5>, "faithfulness": <0-5>, "hallucination_free": <0-5>, "omission_free": <0-5>, "justification": "<one short sentence>"}
"""


def judge_one(question, reference, response, api_key, retries=3):
    user_content = (
        f"QUESTION:\n{question}\n\nREFERENCE (ground truth):\n{reference}\n\n"
        f"CANDIDATE RESPONSE:\n{response}\n"
    )
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
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
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
    args = p.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set the DEEPSEEK_API_KEY environment variable first.")

    rows = []
    with open(args.in_file, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    results = []
    for i, row in enumerate(rows):
        scores = judge_one(row["question"], row["reference"], row["model_response"], api_key)
        results.append({**row, "scores": scores})
        print(f"[{i + 1}/{len(rows)}] {scores}")

    with open(args.out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sums = {axis: 0.0 for axis in AXES}
    counts = {axis: 0 for axis in AXES}
    for r in results:
        s = r["scores"]
        if "error" in s:
            continue
        for axis in AXES:
            if axis in s:
                sums[axis] += s[axis]
                counts[axis] += 1

    print("\n=== Aggregate (mean out of 5) ===")
    for axis in AXES:
        if counts[axis]:
            print(f"{axis}: {sums[axis] / counts[axis]:.2f}  (n={counts[axis]})")
        else:
            print(f"{axis}: no data")


if __name__ == "__main__":
    main()

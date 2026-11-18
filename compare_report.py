"""Print a before/after comparison table from two judge.py outputs."""
import argparse
import json

AXES = ["correctness", "logical_coherence", "faithfulness", "hallucination_free", "omission_free"]


def load_scores(path):
    sums = {a: 0.0 for a in AXES}
    counts = {a: 0 for a in AXES}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            s = row.get("scores", {})
            if "error" in s:
                continue
            for a in AXES:
                if a in s:
                    sums[a] += s[a]
                    counts[a] += 1
    return {a: (sums[a] / counts[a] if counts[a] else None) for a in AXES}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pre", required=True)
    p.add_argument("--post", required=True)
    args = p.parse_args()

    pre = load_scores(args.pre)
    post = load_scores(args.post)

    print(f"{'axis':<20}{'pre':>8}{'post':>8}{'delta':>8}")
    for a in AXES:
        pv, qv = pre[a], post[a]
        if pv is None or qv is None:
            print(f"{a:<20}{'n/a':>8}{'n/a':>8}")
            continue
        print(f"{a:<20}{pv:>8.2f}{qv:>8.2f}{qv - pv:>+8.2f}")


if __name__ == "__main__":
    main()

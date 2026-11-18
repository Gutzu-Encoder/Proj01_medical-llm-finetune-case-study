# Fine-tuning a local LLM on medical reasoning data — a case study in why it didn't work

นี่คือโปรเจคที่ทำมาเพื่อเรียนรู้เกี่ยวกับการเทรน SML( 4B) ผ่าน SFT ในระดับ 8-24GB cuda เพื่อดูว่าสามารถรันได้ดีเเค่ไหน
- code ถูก gen โดน claude sonnet 5, high effort


No trained model weights are included — the adapter was lost when the rented
GPU pod was terminated before it could be copied off. Everything else
(scripts, data, full before/after outputs, and the label audit) is here.

## Setup

- **Base model:** [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache 2.0, 262K context)
- **Method:** QLoRA (4-bit base, LoRA rank 16, ~21M trainable params / 0.5% of total)
- **Data:** [`YuSun-AI/ReasonMed`](https://huggingface.co/datasets/YuSun-AI/ReasonMed) — 300 train / 30 held-out eval, disjoint, seeded shuffle
- **Hardware:** rented RunPod GPU (RTX 4090 → RTX PRO 4500 Blackwell, 32GB)
- **Judge:** DeepSeek Flash API, scoring 0-5 on correctness / logical coherence / faithfulness / hallucination-free / omission-free
- **Training:** 3 epochs, 114 optimizer steps, ~79 minutes เสียเงินไปสุทธิเกือบ 2 dollars

## Pipeline

```
data_prep.py   → builds data/train.jsonl + data/eval.jsonl from ReasonMed
generate.py    → runs the model (base or +adapter) over eval.jsonl
judge.py       → DeepSeek-judges model responses against the reference
train.py       → QLoRA fine-tune, checkpoints every N steps, live progress log
judge_labels.py → independently fact-checks the TRAINING LABELS themselves
compare_report.py → prints the before/after score table
```

Run order is: `data_prep.py` → `generate.py` (baseline) → `judge.py` → `train.py`
→ `generate.py --adapter ...` (post-train) → `judge.py` → `compare_report.py`.

## Result: before vs. after (mean score out of 5, n=30)

| Axis | Pre-train | Post-train | Δ |
|---|---|---|---|
| correctness | 1.53 | 1.62 | +0.09 |
| logical_coherence | 3.10 | 2.79 | **-0.31** |
| faithfulness | 2.87 | 2.48 | **-0.38** |
| hallucination_free | 3.27 | 2.76 | **-0.51** |
| omission_free | 2.50 | 2.38 | -0.12 |

10 steps เเรกลด accuracy loss จาก 2 -> 0.39 เเละ plateau ในช่วงนั้นตลอดการเทรน
**ดูเหมือนจะดี** เเต่ว่า loss ไม่สามรถ capture เนื้อความ free text ได้จากที่เห็นว่า eval ผ่าน LLM judge พบว่า score ลดทุก metrics
นั่นเพราะว่ามันได้ representation learning สไตล์มากกว่าสิ่งที่เราต้องการคือความถูกต้อง

## What was actually happening (read the full outputs in `data/`)

Pulling up real pre/post response pairs showed the mechanism directly:

- **Structural mimicry was near-perfect.** Post-trained model มีสไตล์การพูดเหมือนกับตัว label (*Restated Question → Key Details →
  Pathophysiologic Basis → Option-by-Option Analysis → Differential
  Comparison → Final Justification → Clinical Pearls*) เเต่ดูเหมือนมันไมไ่ด้ใช้มาคิดคำตอบเลย
- **The content inside that structure was confidently fabricated.**สำหรับคำถามหนึ่งเกี่ยวกับภาวะกระดูกต้นขาหัก (femur fracture) โมเดลที่ผ่านการปรับจูนแล้ว (post-train model) ได้กุเหตุผลทั้งหมดที่ฟังดูน่าเชื่อถือขึ้นมาจากกายวิภาคที่ผิดเพี้ยน (โดยอ้างว่ากระดูกหักรหัส AO/OTA 31 คือกระดูกต้นแขนส่วนต้นหัก หรือ proximal humerus fractures ทั้งที่ความจริงคือกระดูกต้นขาส่วนต้นหัก หรือ proximal femur) — แต่กลับยังเลือกตัวเลือกคำตอบสุดท้ายได้ถูกต้องด้วยการจับแพทเทิร์น (pattern-matching) ไม่ใช่จากการให้เหตุผลที่ถูกต้อง
- **A dataset-generation artifact leaked directly into the model's
  hallucinations.** Some ReasonMed labels contain leftover
  "Corrections from Error Feedback" text (apparently a draft-then-correct
  generation pipeline the dataset authors used). The fine-tuned model
  picked this up and started hallucinating references to an "original
  chain-of-thought" that never existed in its own context.

## The deeper question: was the training data ever any good?

Built `judge_labels.py` to independently re-derive the correct answer for
each training example (ignoring the proposed label) and score the proposed
label on correctness, internal consistency, and reasoning quality.

**Result on all 300 training examples:**

| Axis | Mean / 5 |
|---|---|
| label_correct | 2.84 |
| internal_consistency | 2.76 |
| reasoning_quality | 2.84 |

**143 of 300 labels (47.7%) were flagged as likely wrong.**

### But the judge isn't ground truth either

Spot-checked two flagged cases against actual PubMed literature:

- **Rifampicin + interstitial nephritis** — PubMed has 40+ hits, including
  case reports titled exactly that. The judge was **right**; the dataset
  label was genuinely wrong (and self-contradicted in its own text).
- **Von Willebrand factor's site of synthesis** — two authoritative reviews
  confirm it's endothelial cells and megakaryocytes. The judge said "liver,"
  which appears nowhere in the literature as a VWF source. The judge was
  **wrong** here; the original dataset label ("platelets," closest to the
  megakaryocyte-origin answer among the options given) was actually fine.

**Conclusion: the raw "143/300 wrong" number is an overestimate** — some
fraction of it is the judge's own hallucination on medical facts it didn't
actually know, especially on narrow/regional trivia. The underlying problem
is still real (confirmed by at least one PubMed-verified genuine dataset
error, plus the structurally-detectable artifact contamination), just not
exactly quantifiable from one scoring pass. Any eval/filter built on an LLM
judge needs its own spot-checking before you trust the numbers.

##สิ่งที่ได้จากการทำโปรเจคนี้

1. **Loss going down ≠ the model getting better at the task.** It only
   measures "predicts this exact label," not "is correct" or "is a good
   reasoner."
2. **SFT learns surface form faster than substance.** Document structure
   and tone transfer almost instantly; factual accuracy needs much more
   than 300 examples spread across unrelated topics.
3. **The model inherits everything in the data, including what you didn't
   mean to teach it** — like the self-correction artifacts.
4. **Fluency and correctness are separable, and SFT can push them apart.**
   A model can become more confident and more wrong at the same time.
5. **Training bakes labels directly into weights; it doesn't average out
   noise** the way in-context prompting can.
6. **Breadth vs. depth is a real tradeoff.** Scattered topics teach format;
   repeated coverage of the same facts is what would actually move
   correctness.
7. **An LLM judge is a tool with its own error rate, not an oracle** —
   independently spot-check it before trusting its verdicts, the same way
   you'd spot-check the model you're evaluating.

##ในครั้งหน้าจะทำเพิ่ม

Filter training data by a *verified* subset (not just the raw judge flag),
strip the self-correction artifacts, narrow to one subdomain for repeated
signal instead of broad topic coverage, train fewer epochs given how early
the loss plateaued, and copy every checkpoint off the rented GPU immediately.

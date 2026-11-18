# Pod time log (RTX PRO 4500 Blackwell, 32GB, RunPod)

Times approximate, based on observed file timestamps / log output (UTC).

| Stage | Start | End | Duration |
|---|---|---|---|
| Pod connected, setup (install transformers/peft/bitsandbytes/accelerate/datasets) | ~13:36 | ~13:38 | ~2 min |
| Baseline generation attempt #1 (thinking-mode bug, killed) | ~13:38 | ~14:19 | ~41 min (wasted) |
| Baseline generation attempt #2 (fixed, success) | ~14:19 | ~14:41 | ~14 min |
| Baseline judge eval (local, DeepSeek Flash) | ~14:41 | ~14:43 | ~2 min |
| QLoRA training (300 examples, 3 epochs, 114 steps) | ~14:51 | 16:11 | 79.3 min (confirmed) |
| Kernel install (causal-conv1d, flash-linear-attention) attempt | ~15:28 | ~15:30 | ~2 min, FAILED (ABI/triton version mismatch, abandoned) |
| Post-train generation | 16:1x | TBD | TBD |

**Running total pod time so far (as of training completion, ~16:11 UTC): ~2h 35min**

Still to come: post-train judge eval (~2 min, local), compare report (instant, local).

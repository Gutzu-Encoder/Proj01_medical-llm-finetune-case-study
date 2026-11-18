"""LoRA / QLoRA fine-tune Qwen3.5-4B on the ReasonMed train split.
Runs on the GPU pod.
"""
import argparse
import json
import time

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


class ProgressCallback(TrainerCallback):
    """Flushed, ETA-aware progress printing + a machine-readable progress.jsonl
    (default tqdm output gets lost when stdout is redirected to a file over SSH)."""

    def __init__(self, progress_path):
        self.progress_path = progress_path

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.time()
        open(self.progress_path, "w").close()  # truncate/create fresh
        print(f"training started: {state.max_steps} total optimizer steps", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        elapsed = time.time() - self.start
        step = state.global_step
        total = state.max_steps or 1
        pct = 100.0 * step / total
        avg = elapsed / max(step, 1)
        eta = avg * (total - step)
        print(
            f"step {step}/{total} ({pct:.1f}%) loss={logs['loss']:.4f} "
            f"elapsed={elapsed / 60:.1f}min ETA={eta / 60:.1f}min",
            flush=True,
        )
        with open(self.progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "step": step,
                "total_steps": total,
                "percent": round(pct, 1),
                "loss": logs["loss"],
                "elapsed_min": round(elapsed / 60, 2),
                "eta_min": round(eta / 60, 2),
            }) + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        print(f"training finished in {(time.time() - self.start) / 60:.1f} min", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="Qwen/Qwen3.5-4B")
    p.add_argument("--train-file", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", choices=["lora", "qlora"], default="qlora")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=3072)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--save-steps", type=int, default=10,
                    help="Save a LoRA adapter checkpoint every N optimizer steps")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.mode == "qlora":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_config,
    )

    if args.mode == "qlora":
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    rows = load_jsonl(args.train_file)

    def to_example(row):
        messages = [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["reference"]},
        ]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=False
        )}

    examples = [to_example(r) for r in rows]
    ds = Dataset.from_list(examples)

    def tokenize_fn(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_len,
            padding="max_length",
        )
        out["labels"] = [ids.copy() for ids in out["input_ids"]]
        return out

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=[],
        disable_tqdm=True,
    )

    progress_path = args.out_dir.rstrip("/\\") + "_progress.jsonl"
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        callbacks=[ProgressCallback(progress_path)],
    )
    trainer.train()

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"saved adapter to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

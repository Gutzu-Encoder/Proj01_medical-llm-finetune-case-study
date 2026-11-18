"""Generate model responses for an eval set, with or without a trained LoRA adapter.
Runs on the GPU pod. Produces a JSONL of {question, reference, model_response}.

Writes results incrementally (one line per completed example, flushed immediately)
and prints per-example timing/ETA, so progress is visible even when stdout is
redirected to a file.
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_prompt(tokenizer, question):
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="Qwen/Qwen3.5-4B")
    p.add_argument("--adapter", default=None, help="Path to a trained LoRA adapter to apply")
    p.add_argument("--eval-file", required=True)
    p.add_argument("--out-file", required=True)
    p.add_argument("--quant", choices=["none", "4bit"], default="none")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    quant_config = None
    if args.quant == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    print("loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_config,
    )

    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()
    print("model loaded, starting generation", flush=True)

    rows = []
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    out_f = open(args.out_file, "w", encoding="utf-8")
    start = time.time()

    for i, row in enumerate(rows):
        t0 = time.time()
        prompt = build_prompt(tokenizer, row["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        record = {
            "question": row["question"],
            "reference": row["reference"],
            "model_response": gen_text,
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()

        elapsed_item = time.time() - t0
        elapsed_total = time.time() - start
        avg = elapsed_total / (i + 1)
        eta = avg * (len(rows) - (i + 1))
        print(
            f"[{i + 1}/{len(rows)}] {len(gen_text)} chars in {elapsed_item:.1f}s "
            f"(avg {avg:.1f}s/item, ETA {eta / 60:.1f} min)",
            flush=True,
        )

    out_f.close()
    print(f"done: wrote {len(rows)} responses to {args.out_file}", flush=True)


if __name__ == "__main__":
    main()

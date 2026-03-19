"""
QLoRA fine-tuning for Qwen3.5-0.8B-Instruct on recruitGPT distillation data.

Uses Unsloth for fast LoRA training + HuggingFace TRL SFTTrainer.

Usage:
    python src/train.py --config configs/qlora_qwen3_5_0_8b.yaml

    # Dry-run to verify data loading without training
    python src/train.py --config configs/qlora_qwen3_5_0_8b.yaml --dry_run

Input data format (data/generated/train_clean.jsonl):
    {
        "task": "query_parsing",
        "messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},   // few-shot pairs
            {"role": "assistant", "content": "..."},
            ...
            {"role": "user",      "content": "..."},   // actual input
            {"role": "assistant", "content": "..."}    // label — loss computed here
        ]
    }
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


# ---------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    records = []
    with open(p) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def apply_chat_template(sample: dict, tokenizer) -> dict:
    """
    Convert messages list → single string using the model's chat template.
    The tokenizer's apply_chat_template handles Qwen's <|im_start|> format.
    """
    text = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def build_dataset(data_path: str, tokenizer, max_samples: int = None) -> Dataset:
    records = load_jsonl(data_path)
    if max_samples:
        records = records[:max_samples]

    # Apply chat template to each sample
    formatted = [apply_chat_template(r, tokenizer) for r in records]
    return Dataset.from_list(formatted)


# ---------------------------------------------------------------
# Model loading (Unsloth fast path, falls back to HF)
# ---------------------------------------------------------------

def load_model_and_tokenizer(cfg: dict):
    model_name = cfg["model"]["name"]
    max_seq_length = cfg["model"].get("max_seq_length", 2048)
    load_in_4bit = cfg["model"].get("load_in_4bit", True)
    dtype_str = cfg["model"].get("dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    try:
        from unsloth import FastLanguageModel
        print(f"Loading {model_name} via Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        lora_cfg = cfg["lora"]
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg.get("bias", "none"),
            use_gradient_checkpointing="unsloth",
            random_state=cfg["training"].get("seed", 42),
        )
        return model, tokenizer, "unsloth"

    except ImportError:
        print("Unsloth not available — falling back to HuggingFace PEFT")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import get_peft_model

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            device_map="auto",
        )

        lora_cfg = cfg["lora"]
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model, tokenizer, "hf"


# ---------------------------------------------------------------
# Training
# ---------------------------------------------------------------

def train(cfg: dict, dry_run: bool = False):
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    # Load model
    model, tokenizer, backend = load_model_and_tokenizer(cfg)
    print(f"Backend: {backend}")

    # Load datasets
    print(f"Loading train data: {data_cfg['train_file']}")
    train_dataset = build_dataset(data_cfg["train_file"], tokenizer)
    print(f"  Train samples: {len(train_dataset)}")

    eval_dataset = None
    eval_file = data_cfg.get("eval_file")
    if eval_file and Path(eval_file).exists():
        eval_dataset = build_dataset(eval_file, tokenizer)
        print(f"  Eval samples: {len(eval_dataset)}")
    else:
        print("  No eval set found — skipping evaluation during training")

    if dry_run:
        print("\nDry run complete — data loaded successfully, skipping training.")
        sample = train_dataset[0]["text"]
        print(f"\nSample (first 500 chars):\n{sample[:500]}")
        return

    # Training arguments — read from config
    output_dir = train_cfg["output_dir"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy=train_cfg.get("eval_strategy", "steps") if eval_dataset else "no",
        eval_steps=train_cfg.get("eval_steps", 100) if eval_dataset else None,
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True) and eval_dataset is not None,
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        report_to=train_cfg.get("report_to", "none"),
        seed=train_cfg.get("seed", 42),
        dataloader_num_workers=0,
    )

    # SFTTrainer — loss only on assistant turns
    # DataCollatorForCompletionOnlyLM masks the prompt tokens from the loss.
    # Qwen chat format: assistant turn starts after "<|im_start|>assistant\n"
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=cfg["model"].get("max_seq_length", 2048),
        packing=False,
    )

    print(f"\nStarting training — output: {output_dir}")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Checkpoint saved: {output_dir}")

    # Merge LoRA weights into base model
    merge_cfg = cfg.get("merge", {})
    if merge_cfg.get("merge_and_export", True):
        merged_dir = merge_cfg.get("merged_output_dir", f"{output_dir}/merged")
        print(f"\nMerging LoRA weights → {merged_dir}")
        _merge_and_save(model, tokenizer, merged_dir, backend)


def _merge_and_save(model, tokenizer, output_dir: str, backend: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if backend == "unsloth":
        from unsloth import FastLanguageModel
        model.save_pretrained_merged(
            output_dir,
            tokenizer,
            save_method="merged_16bit",
        )
    else:
        merged = model.merge_and_unload()
        merged.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved: {output_dir}")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for recruitGPT")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--dry_run", action="store_true",
                        help="Load data and model, skip actual training")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap training samples (useful for quick smoke tests)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.max_samples:
        # Inject into config for build_dataset
        cfg["_max_samples"] = args.max_samples

    train(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

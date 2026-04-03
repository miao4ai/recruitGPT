"""
Fine-tune BGE Reranker V2 M3 (cross-encoder) for candidate reranking.

A cross-encoder takes (query, candidate) as a single input and outputs
a relevance score. This is more accurate than bi-encoder (embedding)
retrieval but slower — so it's used to rerank the top-K results from
the embedding stage.

Usage:
    python src/reranker/train_reranker.py --config configs/reranker_finetune.yaml
    python src/reranker/train_reranker.py --config configs/reranker_finetune.yaml --dry_run

Input data format (data/reranker/train.jsonl):
    {
        "query": "Senior Backend Engineer | Skills: Go, PostgreSQL...",
        "candidate": "## Alex Chen\\nSenior Backend Engineer\\n...",
        "score": 0.85
    }

    score: 1.0 = strong match, 0.5 = partial, 0.0 = mismatch
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler


# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------

class RerankerDataset(Dataset):
    """Loads (query, candidate, score) pairs from jsonl."""

    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = []

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        with open(p) as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "query": r["query"],
            "candidate": r["candidate"],
            "score": float(r["score"]),
        }

    def collate_fn(self, batch):
        """Tokenize query-candidate pairs as a single sequence."""
        queries = [b["query"] for b in batch]
        candidates = [b["candidate"] for b in batch]
        scores = torch.tensor([b["score"] for b in batch], dtype=torch.float32)

        # Cross-encoder: tokenize (query, candidate) as a pair
        encoded = self.tokenizer(
            queries, candidates,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {"encoded": encoded, "scores": scores}


# ---------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------

def train(cfg: dict, dry_run: bool = False):
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    model_name = model_cfg["name"]
    max_seq_length = model_cfg.get("max_seq_length", 512)

    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if train_cfg.get("bf16", False) else torch.float32
    model = model.to(device=device, dtype=dtype)
    print(f"  Device: {device}, dtype: {dtype}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load datasets
    train_path = data_cfg["train_file"]
    print(f"Loading training data: {train_path}")
    train_dataset = RerankerDataset(train_path, tokenizer, max_seq_length)
    print(f"  Train samples: {len(train_dataset)}")

    eval_dataset = None
    eval_path = data_cfg.get("eval_file")
    if eval_path and Path(eval_path).exists():
        eval_dataset = RerankerDataset(eval_path, tokenizer, max_seq_length)
        print(f"  Eval samples: {len(eval_dataset)}")
    else:
        print("  No eval set found — skipping evaluation")

    if dry_run:
        print("\nDry run — data and model loaded successfully.")
        sample = train_dataset[0]
        print(f"  Sample query: {sample['query'][:80]}...")
        print(f"  Sample score: {sample['score']}")
        return

    batch_size = train_cfg.get("per_device_train_batch_size", 16)
    eval_batch_size = train_cfg.get("per_device_eval_batch_size", 32)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 2)
    num_epochs = train_cfg.get("num_train_epochs", 3)
    lr = train_cfg.get("learning_rate", 5e-6)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.1)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    logging_steps = train_cfg.get("logging_steps", 10)
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=train_dataset.collate_fn, num_workers=0,
    )

    eval_loader = None
    if eval_dataset:
        eval_loader = DataLoader(
            eval_dataset, batch_size=eval_batch_size, shuffle=False,
            collate_fn=eval_dataset.collate_fn, num_workers=0,
        )

    # Optimizer
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr)

    total_steps = (len(train_loader) // grad_accum) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_scheduler(
        train_cfg.get("lr_scheduler_type", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loss_fn = torch.nn.MSELoss()

    print(f"\nTraining config:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size} x {grad_accum} accum = {batch_size * grad_accum} effective")
    print(f"  Total steps: {total_steps}")
    print(f"  LR: {lr}, warmup: {warmup_steps} steps")
    print()

    model.train()
    global_step = 0
    best_eval_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader):
            encoded = {k: v.to(device) for k, v in batch["encoded"].items()}
            scores = batch["scores"].to(device)

            with torch.autocast(device_type=device.type, dtype=dtype):
                outputs = model(**encoded)
                logits = outputs.logits.squeeze(-1)  # (B,)
                # Sigmoid to map to [0, 1] range
                preds = torch.sigmoid(logits)
                loss = loss_fn(preds, scores) / grad_accum

            loss.backward()
            epoch_loss += loss.item() * grad_accum
            num_batches += 1

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    lr_current = scheduler.get_last_lr()[0]
                    print(f"  [Epoch {epoch+1}/{num_epochs}] Step {global_step}/{total_steps} "
                          f"| loss: {avg_loss:.4f} | lr: {lr_current:.2e}")

        avg_train_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{num_epochs} done | train loss: {avg_train_loss:.4f}")

        # Eval
        if eval_loader:
            eval_loss = evaluate(model, eval_loader, loss_fn, device, dtype)
            print(f"  Eval loss: {eval_loss:.4f}")
        else:
            eval_loss = avg_train_loss

        # Save checkpoint
        ckpt_dir = output_dir / f"checkpoint-epoch-{epoch+1}"
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"  Saved → {ckpt_dir}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_dir = output_dir / "best"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  New best model → {best_dir}")

    # Save final
    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nTraining complete. Final model → {final_dir}")


def evaluate(model, eval_loader, loss_fn, device, dtype) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in eval_loader:
            encoded = {k: v.to(device) for k, v in batch["encoded"].items()}
            scores = batch["scores"].to(device)

            with torch.autocast(device_type=device.type, dtype=dtype):
                outputs = model(**encoded)
                logits = outputs.logits.squeeze(-1)
                preds = torch.sigmoid(logits)
                loss = loss_fn(preds, scores)

            total_loss += loss.item()
            num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune cross-encoder reranker")
    parser.add_argument("--config", type=str, default="configs/reranker_finetune.yaml")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

"""
Fine-tune BGE-M3 embedding model on recruiting triplets.

Uses InfoNCE contrastive loss with in-batch negatives + explicit hard negatives.

Usage:
    python src/embedding/train_embedding.py --config configs/bge_finetune.yaml

    # Dry run — load data and model, skip training
    python src/embedding/train_embedding.py --config configs/bge_finetune.yaml --dry_run

Input data format (data/pairs/train_triplets.jsonl):
    {
        "query": "Senior Backend Engineer | Seniority: senior | Skills: Go, PostgreSQL...",
        "positive": "## Alex Chen\nSenior Backend Engineer\n...",
        "negative": "## Dana Zhou\nProduct Manager\n..."
    }
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_scheduler

from losses import info_nce_loss


# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------

class TripletDataset(Dataset):
    """Loads triplets from jsonl and tokenizes on-the-fly."""

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
            "positive": r["positive"],
            "negative": r["negative"],
        }

    def collate_fn(self, batch):
        """Tokenize a batch of triplets."""
        queries = [b["query"] for b in batch]
        positives = [b["positive"] for b in batch]
        negatives = [b["negative"] for b in batch]

        q_enc = self.tokenizer(
            queries, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        p_enc = self.tokenizer(
            positives, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        n_enc = self.tokenizer(
            negatives, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )

        return {"query": q_enc, "positive": p_enc, "negative": n_enc}


# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------

def encode(model, tokenized, pooling: str = "cls") -> torch.Tensor:
    """Run model forward and extract embeddings."""
    outputs = model(**tokenized)
    if pooling == "cls":
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token
    elif pooling == "mean":
        attention_mask = tokenized["attention_mask"].unsqueeze(-1)
        return (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
    else:
        raise ValueError(f"Unknown pooling: {pooling}")


# ---------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------

def train(cfg: dict, dry_run: bool = False):
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    loss_cfg = cfg.get("loss", {})

    model_name = model_cfg["name"]
    max_seq_length = model_cfg.get("max_seq_length", 512)
    pooling = model_cfg.get("pooling", "cls")

    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if train_cfg.get("bf16", False) else torch.float32
    model = model.to(device=device, dtype=dtype)
    print(f"  Device: {device}, dtype: {dtype}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load dataset
    train_path = data_cfg["train_file"]
    print(f"Loading training data: {train_path}")
    dataset = TripletDataset(train_path, tokenizer, max_seq_length)
    print(f"  Triplets: {len(dataset)}")

    if dry_run:
        print("\nDry run — data and model loaded successfully.")
        sample = dataset[0]
        print(f"  Sample query: {sample['query'][:100]}...")
        return

    batch_size = train_cfg.get("per_device_train_batch_size", 16)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 2)
    num_epochs = train_cfg.get("num_train_epochs", 3)
    lr = train_cfg.get("learning_rate", 1e-5)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.1)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    logging_steps = train_cfg.get("logging_steps", 10)
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    temperature = loss_cfg.get("temperature", 0.02)
    use_in_batch_neg = loss_cfg.get("use_in_batch_negatives", True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=0,
        drop_last=True,  # important for in-batch negatives
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

    total_steps = (len(dataloader) // grad_accum) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_scheduler(
        train_cfg.get("lr_scheduler_type", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Training
    print(f"\nTraining config:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size} x {grad_accum} accum = {batch_size * grad_accum} effective")
    print(f"  Total steps: {total_steps}")
    print(f"  LR: {lr}, warmup: {warmup_steps} steps")
    print(f"  Temperature: {temperature}, in-batch negatives: {use_in_batch_neg}")
    print()

    model.train()
    global_step = 0
    best_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            # Move to device
            q_enc = {k: v.to(device) for k, v in batch["query"].items()}
            p_enc = {k: v.to(device) for k, v in batch["positive"].items()}
            n_enc = {k: v.to(device) for k, v in batch["negative"].items()}

            # Forward
            with torch.autocast(device_type=device.type, dtype=dtype):
                q_emb = encode(model, q_enc, pooling)
                p_emb = encode(model, p_enc, pooling)
                n_emb = encode(model, n_enc, pooling)

                loss = info_nce_loss(
                    q_emb, p_emb, n_emb,
                    temperature=temperature,
                    use_in_batch_negatives=use_in_batch_neg,
                )
                loss = loss / grad_accum

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

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{num_epochs} done | avg loss: {avg_epoch_loss:.4f}")

        # Save checkpoint per epoch
        save_strategy = train_cfg.get("save_strategy", "epoch")
        if save_strategy == "epoch":
            ckpt_dir = output_dir / f"checkpoint-epoch-{epoch+1}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  Saved checkpoint → {ckpt_dir}")

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_dir = output_dir / "best"
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                print(f"  New best model → {best_dir}")

    # Save final model
    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nTraining complete. Final model → {final_dir}")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune BGE-M3 embedding model")
    parser.add_argument("--config", type=str, default="configs/bge_finetune.yaml")
    parser.add_argument("--dry_run", action="store_true",
                        help="Load data and model, skip training")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

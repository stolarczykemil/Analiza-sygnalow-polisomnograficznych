import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from dl import config
from dl.config import (
    BATCH_SIZE,
    EVALS_PER_PASS,
    LOSS_WEIGHTS,
    LR,
    MAX_PASSES,
    MODELS_DIR,
    NUM_WORKERS,
    PATIENCE,
    RESP_CLASSES,
    SEED,
    STAGE_IGNORE_INDEX,
    STAGES,
    VAL_BATCH_SIZE,
    WEIGHT_DECAY,
)
from dl.dataset import PSGEpochDataset
from dl.encoder import PSGEncoder

HEADS = list(LOSS_WEIGHTS)
EVENT_HEADS = ["arousal", "limb", "resp"]
TARGET_KEYS = ["stage", "stage_valid", "arousal", "limb", "resp", "arousal_valid", "limb_valid", "resp_valid"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stage_class_frequencies(dataset):
    stages = np.concatenate([
        record["stage"][record["keep"]][record["stage_valid"][record["keep"]]] for record in dataset.records
    ])
    counts = np.bincount(stages, minlength=len(STAGES))
    return counts, counts / counts.sum()


def stage_class_weights(frequencies):
    weights = 1.0 / np.sqrt(frequencies)
    weights = weights / (frequencies * weights).sum()
    return torch.tensor(weights, dtype=torch.float32)


def stage_loss(logits, target, valid, class_weights):
    target = torch.where(valid > 0, target, torch.full_like(target, STAGE_IGNORE_INDEX))
    per_sample = F.cross_entropy(
        logits.float(), target, weight=class_weights, ignore_index=STAGE_IGNORE_INDEX, reduction="none"
    )
    sample_weights = class_weights[target.clamp(min=0)] * (target != STAGE_IGNORE_INDEX)
    return per_sample.sum() / sample_weights.sum().clamp(min=torch.finfo(torch.float32).tiny)


def event_loss(logits, target, valid):
    per_sample = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    if per_sample.dim() > 1:
        per_sample = per_sample.mean(dim=1)
    return (per_sample * valid).sum() / valid.sum().clamp(min=1.0)


def compute_losses(logits, batch, class_weights):
    losses = {"stage": stage_loss(logits["stage"], batch["stage"], batch["stage_valid"], class_weights)}
    for name in EVENT_HEADS:
        losses[name] = event_loss(logits[name], batch[name], batch[f"{name}_valid"])
    losses["total"] = sum(LOSS_WEIGHTS[name] * losses[name] for name in HEADS)
    return losses


def parameter_groups(model, weight_decay):
    decay, no_decay = [], []
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if isinstance(module, nn.modules.batchnorm._BatchNorm) or name == "bias":
                no_decay.append(parameter)
            else:
                decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def make_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY):
    return torch.optim.AdamW(parameter_groups(model, weight_decay), lr=lr)


def make_scheduler(optimizer, total_steps, lr=LR):
    return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps)


def make_scaler(device):
    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


def make_train_loader(dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, seed=SEED):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def make_val_loader(dataset, num_workers=NUM_WORKERS, batch_size=VAL_BATCH_SIZE):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
    )


def to_device(batch, device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def train_step(model, batch, optimizer, scheduler, scaler, class_weights, device):
    model.train()
    batch = to_device(batch, device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        _, logits = model(batch)
    losses = compute_losses(logits, batch, class_weights)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(losses["total"]).backward()
    scaler.step(optimizer)
    scaler.update()
    if scheduler is not None:
        scheduler.step()
    return {name: value.item() for name, value in losses.items()}


def collect_predictions(model, loader, device, max_batches=None):
    model.eval()
    logits_parts = {name: [] for name in HEADS}
    target_parts = {name: [] for name in TARGET_KEYS}
    with torch.no_grad():
        for batch_number, batch in enumerate(loader):
            if max_batches is not None and batch_number >= max_batches:
                break
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                _, logits = model(batch)
            for name in HEADS:
                logits_parts[name].append(logits[name].float().cpu())
            for name in TARGET_KEYS:
                target_parts[name].append(batch[name].cpu())
    logits = {name: torch.cat(parts) for name, parts in logits_parts.items()}
    targets = {name: torch.cat(parts) for name, parts in target_parts.items()}
    return logits, targets


def confusion_matrix(true, predicted, n_classes=len(STAGES)):
    return np.bincount(n_classes * true + predicted, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def cohen_kappa(confusion):
    total = confusion.sum()
    observed = np.trace(confusion) / total
    expected = (confusion.sum(axis=0) * confusion.sum(axis=1)).sum() / total ** 2
    return float((observed - expected) / (1 - expected)) if expected < 1 else 0.0


def f1_per_class(confusion):
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    return np.divide(2 * true_positive, denominator, out=np.zeros_like(true_positive), where=denominator > 0)


def staging_metrics(stage_logits, stage, stage_valid):
    valid = stage_valid > 0
    true = stage[valid].numpy()
    predicted = stage_logits[valid].argmax(dim=1).numpy()
    confusion = confusion_matrix(true, predicted)
    f1 = f1_per_class(confusion)
    metrics = {
        "kappa": cohen_kappa(confusion),
        "accuracy": float(np.trace(confusion) / max(confusion.sum(), 1)),
        "macro_f1": float(f1.mean()),
    }
    metrics.update({f"f1_{name}": float(value) for name, value in zip(STAGES, f1)})
    return metrics, confusion


def evaluate(model, loader, class_weights, device, max_batches=None):
    logits, targets = collect_predictions(model, loader, device, max_batches)
    losses = compute_losses(logits, targets, class_weights.cpu())
    metrics, confusion = staging_metrics(logits["stage"], targets["stage"], targets["stage_valid"])
    metrics.update({f"val_{name}": value.item() for name, value in losses.items()})
    return metrics, confusion


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(plain(item) for item in value)
    if isinstance(value, slice):
        return [value.start, value.stop]
    if isinstance(value, Path):
        return str(value)
    return value


def run_config(args):
    snapshot = {name: plain(getattr(config, name)) for name in dir(config) if name.isupper()}
    snapshot["args"] = plain(vars(args))
    return snapshot


def log_columns():
    columns = ["step", "pass", "lr"]
    columns += [f"train_{name}" for name in HEADS + ["total"]]
    columns += [f"val_{name}" for name in HEADS + ["total"]]
    columns += ["kappa", "accuracy"] + [f"f1_{name}" for name in STAGES] + ["macro_f1", "patience", "elapsed_s"]
    return columns


def append_log(path, row):
    new_file = not path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=log_columns())
        if new_file:
            writer.writeheader()
        writer.writerow({column: row[column] for column in log_columns()})


def save_confusion(path, confusion):
    pd.DataFrame(confusion, index=STAGES, columns=STAGES).rename_axis("true").to_csv(path)


def save_last(path, model, optimizer, scheduler, scaler, step, patience_counter, best_kappa):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "patience_counter": patience_counter,
        "best_kappa": best_kappa,
    }, path)


def save_best(path, model, step, kappa, snapshot):
    torch.save({"model": model.state_dict(), "step": step, "kappa": kappa, "config": snapshot}, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--val-batches", type=int, default=None)
    return parser.parse_args(argv)


def train(args):
    run_dir = MODELS_DIR / args.run_name
    last_path = run_dir / "last.pt"
    if args.resume and not last_path.exists():
        raise FileNotFoundError(f"No checkpoint to resume: {last_path}")
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    device = select_device()
    train_dataset = PSGEpochDataset("train")
    val_dataset = PSGEpochDataset("val")
    _, frequencies = stage_class_frequencies(train_dataset)
    class_weights = stage_class_weights(frequencies).to(device)

    train_loader = make_train_loader(train_dataset)
    val_loader = make_val_loader(val_dataset)
    steps_per_pass = len(train_loader)
    eval_interval = max(steps_per_pass // EVALS_PER_PASS, 1)
    schedule_steps = MAX_PASSES * steps_per_pass
    total_steps = schedule_steps if args.max_steps is None else min(schedule_steps, args.max_steps)

    model = PSGEncoder().to(device)
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer, schedule_steps)
    scaler = make_scaler(device)
    snapshot = run_config(args)

    step, patience_counter, best_kappa = 0, 0, -math.inf
    if args.resume:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        step = checkpoint["step"]
        patience_counter = checkpoint["patience_counter"]
        best_kappa = checkpoint["best_kappa"]
        set_seed(SEED + step)
        train_loader.generator.manual_seed(SEED + step)
        print(f"resumed from step {step}, best kappa {best_kappa:.4f}, patience {patience_counter}")

    print(f"device {device}, steps per pass {steps_per_pass}, eval every {eval_interval}, total steps {total_steps}")
    train_sums = {name: 0.0 for name in HEADS + ["total"]}
    train_count = 0
    start = time.time()
    stopped = patience_counter >= PATIENCE

    while step < total_steps and not stopped:
        for batch in train_loader:
            losses = train_step(model, batch, optimizer, scheduler, scaler, class_weights, device)
            step += 1
            train_count += 1
            for name in train_sums:
                train_sums[name] += losses[name]

            if step % eval_interval == 0 or step == total_steps:
                metrics, confusion = evaluate(model, val_loader, class_weights, device, args.val_batches)
                if metrics["kappa"] > best_kappa:
                    best_kappa = metrics["kappa"]
                    patience_counter = 0
                    save_best(run_dir / "best.pt", model, step, best_kappa, snapshot)
                    save_confusion(run_dir / "confusion_best.csv", confusion)
                else:
                    patience_counter += 1
                row = {
                    "step": step,
                    "pass": step / steps_per_pass,
                    "lr": optimizer.param_groups[0]["lr"],
                    "patience": patience_counter,
                    "elapsed_s": time.time() - start,
                    **{f"train_{name}": value / train_count for name, value in train_sums.items()},
                    **metrics,
                }
                append_log(run_dir / "log.csv", row)
                save_last(last_path, model, optimizer, scheduler, scaler, step, patience_counter, best_kappa)
                print(f"step {step} pass {row['pass']:.2f} train {row['train_total']:.4f} "
                      f"val {row['val_total']:.4f} kappa {row['kappa']:.4f} patience {patience_counter}")
                train_sums = {name: 0.0 for name in train_sums}
                train_count = 0
                stopped = patience_counter >= PATIENCE

            if stopped or step >= total_steps or step % steps_per_pass == 0:
                break

    print(f"finished at step {step}, best kappa {best_kappa:.4f}")


if __name__ == "__main__":
    train(parse_args())

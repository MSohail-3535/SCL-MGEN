import contextlib
import copy
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import fingerprint, write_json
from .data import BalancedBatchSampler, Collator, TextDataset
from .losses import composite_loss
from .metrics import metrics, select_threshold
from .model import SCLMGEN

LOGGER = logging.getLogger(__name__)


def seed_everything(seed, deterministic):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False


def environment():
    packages = {}
    for name in ["torch", "transformers", "numpy", "scikit-learn", "scipy", "PyYAML", "matplotlib"]:
        packages[name] = importlib.metadata.version(name)
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def autocast_context(device, precision):
    if precision == "fp32":
        return contextlib.nullcontext()
    if device.type != "cuda":
        raise RuntimeError("Reported mixed-precision experiments require CUDA; use an explicitly labelled FP32 test configuration on CPU")
    return torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


@torch.no_grad()
def predict(model, loader, device, precision):
    model.eval()
    outputs = {"labels": [], "probabilities": [], "uncertainty": [], "embeddings": []}
    for batch in loader:
        batch = move_batch(batch, device)
        labels = batch.pop("labels")
        with autocast_context(device, precision):
            output = model(**batch)
        for key, value in [("labels", labels), ("probabilities", output["probabilities"]),
                           ("uncertainty", output["uncertainty"]), ("embeddings", output["embedding"])]:
            outputs[key].append(value.float().cpu().numpy() if key != "labels" else value.cpu().numpy())
    return {key: np.concatenate(values) for key, values in outputs.items()}


def save_predictions(path, records, prediction):
    with Path(path).open("w", encoding="utf-8") as stream:
        for i, record in enumerate(records):
            row = {key: record[key] for key in ["id", "label", "language", "generator", "domain", "source_id"]}
            row.update(probabilities=prediction["probabilities"][i].astype(float).tolist(), uncertainty=float(prediction["uncertainty"][i]))
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def scheduler_for(optimizer, total, settings):
    warmup = int(total * settings["warmup_fraction"])

    def factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1, (step - warmup) / max(1, total - warmup))
        return {"cosine": lambda: 0.5 * (1 + math.cos(math.pi * progress)),
                "linear": lambda: 1 - progress, "constant": lambda: 1}[settings["scheduler"]]()

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def fit_fold(config, partition, output_dir, split_manifest, model=None, datasets=None, collator=None, test_fixture=False):
    config = copy.deepcopy(config)
    seed_everything(config["seed"], config["training"]["deterministic"])
    output_dir = Path(output_dir)
    data_identity = {key: [{field: row[field] for field in ["id", "text_hash", "label", "language", "generator", "domain", "source_id"]}
                          for row in records] for key, records in partition.items()}
    signature = fingerprint({"config": config, "splits": split_manifest, "data": data_identity, "test_fixture": test_fixture})
    complete = output_dir / "run.json"
    if complete.exists():
        old = json.loads(complete.read_text(encoding="utf-8"))
        if old.get("status") == "finished" and old["signature"] == signature:
            LOGGER.info("Reusing finished run %s", output_dir)
            return old
        raise ValueError(f"Existing run has different settings or is incomplete: {output_dir}; choose a new output directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Incomplete run directory: {output_dir}; choose a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = config["training"]
    if settings["precision"] != "fp32" and device.type == "cpu":
        raise RuntimeError("CUDA unavailable for reported FP16 protocol")
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "splits.json", split_manifest)
    write_json(output_dir / "environment.json", environment())
    if datasets is None:
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"], revision=config["model"]["revision"], use_fast=True)
        if not tokenizer.is_fast:
            raise ValueError("Offset-based sentence pooling requires a fast tokenizer")
        tokenizer.truncation_side = "right"
        tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
        datasets = {key: TextDataset(rows, tokenizer, config["model"]["max_length"], config["model"]["streams"]) for key, rows in partition.items()}
        collator = Collator(tokenizer.pad_token_id)
    model = model or SCLMGEN(config)
    model.to(device)
    model.backbone.config.save_pretrained(checkpoint_dir / "backbone_config")
    loaders = {key: DataLoader(data, batch_size=settings["eval_batch"], shuffle=False, collate_fn=collator, num_workers=settings["num_workers"])
               for key, data in datasets.items() if key != "train"}
    weight_value = settings["class_weights"]
    if weight_value == "train_ratio":
        labels = [r["label"] for r in partition["train"]]
        weight_value = [labels.count(1) / labels.count(0), 1.0]
    weights = torch.tensor(weight_value, dtype=torch.float32, device=device)
    history = []
    sample_count = min(config["evaluation"]["embedding_sample"], len(partition["validation"]))
    before = predict(model, loaders["validation"], device, settings["precision"])
    sample = np.random.default_rng(config["seed"]).choice(len(before["labels"]), sample_count, replace=False)
    np.savez_compressed(output_dir / "embeddings_before.npz", embeddings=before["embeddings"][sample], labels=before["labels"][sample],
                        ids=np.array([partition["validation"][i]["id"] for i in sample]))
    start_time = time.time()
    best_f1 = -1.0
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and settings["precision"] == "fp16")
    for stage in (1, 2):
        epochs = settings[f"stage{stage}_epochs"]
        if epochs == 0:
            continue
        generator = torch.Generator().manual_seed(config["seed"])
        if stage == 1:
            sampler = BalancedBatchSampler([r["label"] for r in partition["train"]], settings["stage1_batch"], config["seed"])
            loader = DataLoader(datasets["train"], batch_sampler=sampler, collate_fn=collator, num_workers=settings["num_workers"])
            accumulation = 1
        else:
            loader = DataLoader(datasets["train"], batch_size=settings["stage2_batch"], shuffle=True, generator=generator,
                                collate_fn=collator, num_workers=settings["num_workers"])
            accumulation = settings["stage2_accumulation"]
        if settings["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings[f"stage{stage}_lr"], weight_decay=settings["weight_decay"], betas=tuple(settings["betas"]), eps=settings["epsilon"])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=settings[f"stage{stage}_lr"], weight_decay=settings["weight_decay"])
        scheduler = scheduler_for(optimizer, math.ceil(len(loader) / accumulation) * epochs, settings)
        for epoch in range(epochs):
            model.train()
            model.set_trainable_stage(stage, epoch, settings)
            if stage == 1 and epoch < settings["frozen_epochs"]:
                model.backbone.eval()
            optimizer.zero_grad(set_to_none=True)
            loss_sums, seen = np.zeros(3), 0
            for step, batch in enumerate(loader):
                batch = move_batch(batch, device)
                labels = batch.pop("labels")
                with autocast_context(device, settings["precision"]):
                    output = model(**batch)
                    total, scl, supervised = composite_loss(output, labels, weights, settings, stage)
                if not torch.isfinite(total):
                    raise FloatingPointError(f"Nonfinite loss at stage{stage}, epoch{epoch}, batch{step}")
                window_start = (step // accumulation) * accumulation
                window_end = min(window_start + accumulation, len(loader))
                if stage == 1:
                    coefficient = 1.0
                else:
                    total_in_window = min(len(datasets["train"]), window_end * settings["stage2_batch"]) - window_start * settings["stage2_batch"]
                    coefficient = len(labels) / total_in_window
                scaler.scale(total * coefficient).backward()
                loss_sums += np.array([float(total.detach()), float(scl.detach()), float(supervised.detach())]) * len(labels)
                seen += len(labels)
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= old_scale:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            validation = predict(model, loaders["validation"], device, settings["precision"])
            score = metrics(validation["labels"], validation["probabilities"], validation["uncertainty"], bins=config["evaluation"]["ece_bins"])
            row = {"stage": stage, "epoch": epoch + 1, "loss": float(loss_sums[0] / seen), "scl_loss": float(loss_sums[1] / seen),
                   "supervised_loss": float(loss_sums[2] / seen), "validation_accuracy": score["accuracy"], "validation_macro_f1": score["macro_f1"],
                   "learning_rate": optimizer.param_groups[0]["lr"]}
            history.append(row)
            write_json(output_dir / "history.json", history)
            LOGGER.info("stage=%s epoch=%s loss=%.4f validation_f1=%.4f", stage, epoch + 1, row["loss"], score["macro_f1"])
            if stage == 2 and score["macro_f1"] > best_f1:
                best_f1 = score["macro_f1"]
                torch.save({"model": model.state_dict(), "config": config, "weights": weight_value, "validation_macro_f1": best_f1,
                            "epoch": epoch + 1, "signature": signature}, checkpoint_dir / "best.pt")
        if stage == 1:
            after = predict(model, loaders["validation"], device, settings["precision"])
            np.savez_compressed(output_dir / "embeddings_after.npz", embeddings=after["embeddings"][sample], labels=after["labels"][sample],
                                ids=np.array([partition["validation"][i]["id"] for i in sample]))
    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    validation = predict(model, loaders["validation"], device, settings["precision"])
    threshold = select_threshold(validation["labels"], validation["probabilities"], validation["uncertainty"], config["evaluation"]["minimum_coverage"])
    checkpoint["threshold"] = threshold
    torch.save(checkpoint, checkpoint_dir / "best.pt")
    outputs = {}
    for name, loader in loaders.items():
        prediction = predict(model, loader, device, settings["precision"])
        score = metrics(prediction["labels"], prediction["probabilities"], prediction["uncertainty"], threshold, config["evaluation"]["ece_bins"])
        save_predictions(output_dir / f"predictions_{name}.jsonl", partition[name], prediction)
        write_json(output_dir / f"metrics_{name}.json", score)
        outputs[name] = score
    result = {"status": "finished", "signature": signature, "dataset": config["dataset"], "test_fixture": test_fixture,
              "threshold": threshold, "class_weights": weight_value, "metrics": outputs, "seconds": time.time() - start_time,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "backbone_revision": getattr(model.backbone.config, "_commit_hash", None),
              "truncated": {name: getattr(data, "truncated", 0) for name, data in datasets.items()}}
    write_json(complete, result)
    return result


def load_checkpoint(path, device):
    from transformers import AutoConfig, AutoModel

    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    backbone_config = AutoConfig.from_pretrained(path.parent / "backbone_config")
    backbone = AutoModel.from_config(backbone_config)
    model = SCLMGEN(checkpoint["config"], backbone=backbone).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint

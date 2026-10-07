import copy
import hashlib
import json
from pathlib import Path

import yaml


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if "extends" in config:
        base = load_config(Path(path).parent / config.pop("extends"))
        def merge(target, updates):
            for key, value in updates.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    merge(target[key], value)
                else:
                    target[key] = copy.deepcopy(value)
        merge(base, config)
        config = base
    if config["training"]["stage1_batch"] < 4 or config["training"]["stage1_batch"] % 2:
        raise ValueError("Stage 1 requires an even class-balanced batch size")
    if not 0 < config["evaluation"]["minimum_coverage"] <= 1:
        raise ValueError("Minimum coverage must lie in (0,1]")
    if abs(sum(config["split"]["fractions"]) - 1) > 1e-8:
        raise ValueError("Split fractions must sum to one")
    if config["training"]["stage2_epochs"] < 1:
        raise ValueError("At least one supervised Stage 2 epoch is required")
    if config["training"]["stage2_accumulation"] < 1:
        raise ValueError("Gradient accumulation must be positive")
    settings = config["training"]
    for key, supported in {"precision": {"fp16", "bf16", "fp32"}, "optimizer": {"adamw", "sgd"},
                           "scheduler": {"cosine", "linear", "constant"}, "loss_mode": {"printed", "sensoy"}}.items():
        if settings[key] not in supported:
            raise ValueError(f"Unsupported {key}: {settings[key]}")
    if settings["stage1_epochs"] < 0 or not 0 <= settings["frozen_epochs"] <= settings["stage1_epochs"]:
        raise ValueError("Invalid Stage 1 duration or freeze schedule")
    if settings["stage2_batch"] < 1 or settings["eval_batch"] < 1 or settings["temperature"] <= 0:
        raise ValueError("Batch sizes and contrastive temperature must be positive")
    if not 0 <= settings["warmup_fraction"] < 1:
        raise ValueError("Warm-up fraction must lie in [0,1)")
    if any(value <= 0 for value in config["split"]["fractions"]) or config["folds"] < 2:
        raise ValueError("Positive split fractions and at least two folds required")
    model = config["model"]
    if not model["streams"] or not set(model["streams"]) <= {"word", "sentence", "document", "paragraph", "character"}:
        raise ValueError("Unsupported or empty granularity streams")
    if len(set(model["streams"])) != len(model["streams"]) or model["refinement_blocks"] < 0 or model["projection_layers"] < 1:
        raise ValueError("Invalid stream duplication or module depth")
    if model["evidence_activation"] not in {"softplus", "relu", "elu_plus_one"}:
        raise ValueError("Unsupported evidence activation")
    return config


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

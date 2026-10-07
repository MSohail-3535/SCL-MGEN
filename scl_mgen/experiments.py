import copy
from pathlib import Path

from .config import write_json
from .data import make_folds, read_records
from .training import fit_fold


BACKBONES = {
    "mbert": ("google-bert/bert-base-multilingual-cased", 512, 12, 3072),
    "mdeberta": ("microsoft/mdeberta-v3-base", 512, 12, 3072),
    "xlmr": ("FacebookAI/xlm-roberta-large", 512, 16, 4096),
}


def experiment_variants(config, suite):
    variants = [("full", copy.deepcopy(config), {})]
    if suite in ("ablations", "all"):
        for stream in ["word", "sentence", "document"]:
            variant = copy.deepcopy(config)
            variant["model"]["streams"].remove(stream)
            variants.append((f"without_{stream}", variant, {}))
        variant = copy.deepcopy(config)
        variant["model"]["refinement_blocks"] = 0
        variants.append(("without_refinement", variant, {}))
        variant = copy.deepcopy(config)
        variant["training"]["stage2_epochs"] += variant["training"]["stage1_epochs"]
        variant["training"]["stage1_epochs"] = 0
        variant["training"]["frozen_epochs"] = 0
        variants.append(("without_scl", variant, {}))
        variant = copy.deepcopy(config)
        variant["model"]["evidential"] = False
        variants.append(("without_edl", variant, {}))
        variants.append(("without_multilingual", copy.deepcopy(config), {"languages": ["en"]}))
    if suite in ("transfer", "all"):
        languages = ["en", "es", "ru"] if config["dataset"] == "multitude" else ["en", "zh", "ru", "bg", "id", "ur", "ar"]
        for language in languages:
            variants.append((f"source_{language}", copy.deepcopy(config), {"languages": [language]}))
    if suite in ("backbones", "all"):
        for name, (checkpoint, cap, heads, feedforward) in BACKBONES.items():
            variant = copy.deepcopy(config)
            variant["model"].update(backbone=checkpoint, max_length=cap, heads=heads, feedforward=feedforward)
            variants.append((f"backbone_{name}", variant, {}))
    if suite in ("granularity", "all"):
        for name, streams in {
            "with_character": ["character", "word", "sentence", "document"],
            "with_paragraph": ["word", "sentence", "paragraph", "document"],
            "word_sentence": ["word", "sentence"], "word_document": ["word", "document"], "sentence_document": ["sentence", "document"],
        }.items():
            variant = copy.deepcopy(config)
            variant["model"]["streams"] = streams
            variants.append((name, variant, {}))
    if suite in ("depth", "all"):
        for depth in (1, 3):
            variant = copy.deepcopy(config)
            variant["model"]["refinement_blocks"] = depth
            variants.append((f"depth_{depth}", variant, {}))
    if suite in ("sensitivity", "all"):
        grids = {"stage2_lr": [5e-6, 2e-5], "stage2_batch": [8, 32], "stage2_accumulation": [2, 8],
                 "stage2_epochs": [5, 8], "loss_coefficient": [0.1, 0.5, 2.0], "precision": ["bf16"],
                 "temperature": [0.05, 0.10, 0.25], "stage1_batch": [8, 16], "stage1_lr": [1e-5, 5e-5],
                 "stage1_epochs": [5, 8], "beta": [0.01, 0.05, 0.5], "scheduler": ["linear", "constant"],
                 "optimizer": ["sgd"], "loss_mode": ["sensoy"]}
        for key, values in grids.items():
            for value in values:
                variant = copy.deepcopy(config)
                variant["training"][key] = value
                if key == "stage1_epochs":
                    variant["training"]["frozen_epochs"] = value // 2
                variants.append((f"sensitivity_{key}_{value}", variant, {}))
        for key, values in {"evidence_activation": ["relu", "elu_plus_one"], "projection_layers": [1, 3]}.items():
            for value in values:
                variant = copy.deepcopy(config)
                variant["model"][key] = value
                variants.append((f"sensitivity_{key}_{value}", variant, {}))
        for coverage in (0.80, 0.85, 0.95):
            variant = copy.deepcopy(config)
            variant["evaluation"]["minimum_coverage"] = coverage
            variants.append((f"sensitivity_coverage_{coverage}", variant, {}))
    return variants


def run_experiments(config, suite="main", selected_fold=None, dry_run=False):
    variants = experiment_variants(config, suite)
    plan = [{"name": name, "dataset": variant["dataset"], "filter": filters, "folds": variant["folds"],
             "config": variant, "status": "planned"} for name, variant, filters in variants]
    output = Path(config["output_dir"]) / config["dataset"]
    write_json(output / f"plan_{suite}.json", plan)
    if dry_run:
        return plan
    records = read_records(config["data_path"])
    base_folds = list(make_folds(records, config))
    for name, variant, filters in variants:
        for fold, partition, manifest in base_folds:
            if selected_fold is not None and fold != selected_fold:
                continue
            local = copy.deepcopy(partition)
            if "languages" in filters:
                for key in ("train", "validation", "internal_test"):
                    local[key] = [r for r in local[key] if r["language"] in filters["languages"]]
                    if not local[key]:
                        raise ValueError(f"{name} cannot run: acquired {key} contains no requested source language")
            local_manifest = {"base_fold": fold, "grouped": manifest["grouped"], "filter": filters,
                              "counts": {k: len(v) for k, v in local.items()}, "ids": {k: [r["id"] for r in v] for k, v in local.items()}}
            fit_fold(variant, local, output / name / f"fold_{fold}", local_manifest)
    return plan

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from torch.utils.data import Dataset, Sampler

from .config import write_json


def read_records(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream)) if path.suffix == ".csv" else [json.loads(line) for line in stream if line.strip()]
    records = []
    for index, row in enumerate(raw):
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty or invalid document at record {index}")
        if row["label"] not in (0, 1, "0", "1"):
            raise ValueError(f"Non-binary label at record {index}")
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"Non-binary label at record {index}")
        split = row.get("split")
        if split not in ("train", "test"):
            raise ValueError("Every record needs its authentic train/test partition")
        language = row.get("language")
        if not language:
            raise ValueError("Language metadata is required; it cannot be inferred silently")
        digest = hashlib.sha256(text.encode()).hexdigest()
        source_id = row.get("source_id", row.get("source_ID", ""))
        records.append({
            "id": str(row.get("id") or digest), "text": text, "label": label,
            "language": language, "split": split, "text_hash": digest,
            "generator": row.get("generator", row.get("multi_label", "human" if label == 0 else "unknown")),
            "domain": row.get("domain", row.get("source", "unknown")),
            "source_id": str(source_id) if source_id is not None else "",
        })
    audit_records(records)
    return records


def audit_records(records):
    identifiers, hashes, sources = {}, {}, {}
    for row in records:
        if row["id"] in identifiers:
            raise ValueError(f"Duplicate sample identifier: {row['id']}")
        identifiers[row["id"]] = row["split"]
        digest = row["text_hash"]
        if digest in hashes:
            raise ValueError(f"Duplicate text or cross-partition text leakage: {row['id']}")
        hashes[digest] = row["split"]
        if row["source_id"]:
            key = (row["domain"], row["language"], row["source_id"])
            if key in sources and sources[key] != row["split"]:
                raise ValueError(f"Related source crosses official partitions: {key}")
            sources[key] = row["split"]
    return {
        "samples": len(records), "labels": dict(Counter(str(r["label"]) for r in records)),
        "languages": dict(Counter(r["language"] for r in records)),
        "splits": dict(Counter(r["split"] for r in records)),
        "generators": dict(Counter(r["generator"] for r in records)),
        "with_source_id": sum(bool(r["source_id"]) for r in records),
    }


def prepare(input_path, output_path):
    records = read_records(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    source_hash = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    write_json(output.with_suffix(".audit.json"), {**audit_records(records), "source_sha256": source_hash})
    return records


def strata(rows):
    return np.array([f"{r['language']}:{r['label']}" for r in rows])


def groups(rows):
    return np.array([f"{r['domain']}:{r['language']}:{r['source_id']}" if r["source_id"] else r["text_hash"] for r in rows])


def make_folds(records, config):
    pool = [r for r in records if r["split"] == "train"]
    labels = strata(pool)
    group_ids = groups(pool)
    grouped = config["split"]["group_sources"] and len(set(group_ids)) < len(pool)
    count = config["folds"]
    if min(Counter(labels).values()) < count:
        raise ValueError("Insufficient label-language examples for the requested folds")
    outer = (StratifiedGroupKFold if grouped else StratifiedKFold)(count, shuffle=True, random_state=config["seed"])
    splits = outer.split(np.arange(len(pool)), labels, group_ids) if grouped else outer.split(np.arange(len(pool)), labels)
    fractions = config["split"]["fractions"]
    if grouped and not np.allclose(fractions, [0.70, 0.15, 0.15]):
        raise ValueError("Grouped inner splitting currently implements the documented approximate 70/15/15 protocol only")
    for fold, (development, outer_test) in enumerate(splits):
        if grouped:
            # Grouped folds approximate 70/15/15; 
            inner = StratifiedGroupKFold(7, shuffle=True, random_state=config["seed"])
            rest, internal = next(inner.split(development, labels[development], group_ids[development]))
            remainder = development[rest]
            inner_valid = StratifiedGroupKFold(6, shuffle=True, random_state=config["seed"])
            fit, validation = next(inner_valid.split(remainder, labels[remainder], group_ids[remainder]))
            train_idx, val_idx, internal_idx = remainder[fit], remainder[validation], development[internal]
        else:
            remainder, internal_idx = train_test_split(development, test_size=fractions[2], stratify=labels[development], random_state=config["seed"])
            train_idx, val_idx = train_test_split(remainder, test_size=fractions[1] / (fractions[0] + fractions[1]), stratify=labels[remainder], random_state=config["seed"])
        partition = {"train": [pool[i] for i in train_idx], "validation": [pool[i] for i in val_idx],
                     "internal_test": [pool[i] for i in internal_idx], "outer_test": [pool[i] for i in outer_test],
                     "official_test": [r for r in records if r["split"] == "test"]}
        for name, rows in partition.items():
            if not rows:
                raise ValueError(f"Empty {name} partition in fold {fold}")
            if name != "official_test" and {r["label"] for r in rows} != {0, 1}:
                raise ValueError(f"Fold {fold} {name} lacks a class")
        names = list(partition)
        for i, name in enumerate(names):
            for other in names[i + 1:]:
                if set(groups(partition[name])) & set(groups(partition[other])):
                    raise ValueError(f"Source leakage between {name} and {other}")
        yield fold, partition, {"grouped": grouped, "counts": {k: len(v) for k, v in partition.items()},
                                "ids": {k: [r["id"] for r in v] for k, v in partition.items()}}


def segment_ids(text, offsets, level="sentence"):
    boundary = r"[^\n]+(?:\n|$)" if level == "paragraph" else r"[^.!?。！？؟۔\n]+[.!?。！？؟۔\n]*|[.!?。！？؟۔\n]+"
    spans = [m.span() for m in re.finditer(boundary, text)]
    assigned = []
    for start, end in offsets:
        if start == end:
            assigned.append(-1)
        else:
            assigned.append(next((i for i, (a, b) in enumerate(spans) if a <= start < b), max(0, len(spans) - 1)))
    remap = {value: i for i, value in enumerate(sorted(set(assigned) - {-1}))}
    return [remap.get(value, -1) for value in assigned]


class TextDataset(Dataset):
    def __init__(self, records, tokenizer, max_length, streams):
        self.records = records
        self.items = []
        self.truncated = 0
        for row in records:
            full = tokenizer(row["text"], add_special_tokens=True, return_offsets_mapping=True, truncation=False)
            self.truncated += len(full["input_ids"]) > max_length
            encoded = tokenizer(row["text"], add_special_tokens=True, return_offsets_mapping=True, truncation=True, max_length=max_length)
            offsets = encoded.pop("offset_mapping")
            item = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"],
                    "sentence_ids": segment_ids(row["text"], offsets), "label": row["label"]}
            if "paragraph" in streams:
                item["paragraph_ids"] = segment_ids(row["text"], offsets, "paragraph")
            if "character" in streams:
                item["offsets"] = offsets
                item["text"] = row["text"]
            if max(item["sentence_ids"]) < 0:
                raise ValueError(f"Tokenizer retained no content tokens: {row['id']}")
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, items):
        length = max(len(i["input_ids"]) for i in items)
        batch = {}
        for key in ("input_ids", "attention_mask", "sentence_ids", "paragraph_ids"):
            if key not in items[0]:
                continue
            fill = self.pad_id if key == "input_ids" else (0 if key == "attention_mask" else -1)
            batch[key] = torch.tensor([i[key] + [fill] * (length - len(i[key])) for i in items])
        if "offsets" in items[0]:
            mappings = []
            for item in items:
                pairs = [(t, c) for t, (a, b) in enumerate(item["offsets"]) for c in range(a, b) if not item["text"][c].isspace()]
                mappings.append(pairs)
            width = max(len(p) for p in mappings)
            batch["character_tokens"] = torch.tensor([[t for t, _ in p] + [-1] * (width - len(p)) for p in mappings])
        batch["labels"] = torch.tensor([i["label"] for i in items])
        return batch


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, seed):
        self.indices = [np.flatnonzero(np.array(labels) == c) for c in (0, 1)]
        if any(len(x) == 0 for x in self.indices) or batch_size < 4 or batch_size % 2:
            raise ValueError("Balanced SCL batches need both classes and at least two samples per class")
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0
        self.batches = math.ceil(max(map(len, self.indices)) / (batch_size // 2))

    def __len__(self):
        return self.batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        half = self.batch_size // 2
        sampled = []
        for indices in self.indices:
            needed = self.batches * half
            chunks = [rng.permutation(indices) for _ in range(math.ceil(needed / len(indices)))]
            sampled.append(np.concatenate(chunks)[:needed])
        for step in range(self.batches):
            batch = np.concatenate([x[step * half:(step + 1) * half] for x in sampled])
            rng.shuffle(batch)
            yield batch.tolist()

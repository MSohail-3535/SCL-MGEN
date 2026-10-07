import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from scl_mgen.config import write_json
from scl_mgen.data import Collator, TextDataset, read_records
from scl_mgen.metrics import metrics
from scl_mgen.training import load_checkpoint, predict, save_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate with validation-selected uncertainty threshold")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Choose a new output directory to preserve previous results")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    records = [r for r in read_records(args.data) if r["split"] == args.split]
    if not records:
        parser.error("The requested evaluation partition is empty")
    tokenizer = AutoTokenizer.from_pretrained(Path(args.checkpoint).parent / "tokenizer", use_fast=True)
    data = TextDataset(records, tokenizer, config["model"]["max_length"], config["model"]["streams"])
    loader = DataLoader(data, batch_size=config["training"]["eval_batch"], collate_fn=Collator(tokenizer.pad_token_id))
    prediction = predict(model, loader, device, config["training"]["precision"])
    score = metrics(prediction["labels"], prediction["probabilities"], prediction["uncertainty"], checkpoint["threshold"], config["evaluation"]["ece_bins"])
    write_json(output / "metrics.json", score)
    save_predictions(output / "predictions.jsonl", records, prediction)


if __name__ == "__main__":
    main()

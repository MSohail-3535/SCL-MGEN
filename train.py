import argparse
import logging

from scl_mgen.config import load_config
from scl_mgen.experiments import run_experiments


def main():
    parser = argparse.ArgumentParser(description="Train SCL-MGEN using nested, audited benchmark partitions")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--suite", default="main", choices=["main", "ablations", "transfer", "backbones", "granularity", "depth", "sensitivity", "all"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    if args.fold is not None and not 0 <= args.fold < config["folds"]:
        parser.error("Fold index is outside the configured range")
    plan = run_experiments(config, args.suite, args.fold, args.dry_run)
    print(f"{'Planned' if args.dry_run else 'Processed'} {len(plan)} experiment configurations")


if __name__ == "__main__":
    main()

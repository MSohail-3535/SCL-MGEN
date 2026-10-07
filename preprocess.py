import argparse
import json

from scl_mgen.data import audit_records, prepare


def main():
    parser = argparse.ArgumentParser(description="Validate and prepare authentic benchmark records")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = prepare(args.input, args.output)
    print(json.dumps(audit_records(records), indent=2))


if __name__ == "__main__":
    main()

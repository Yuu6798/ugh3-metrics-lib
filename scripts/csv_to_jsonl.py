import argparse
import csv
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CSV to JSONL")
    parser.add_argument("csv_path")
    parser.add_argument("jsonl_out")
    args = parser.parse_args()

    with (
        open(args.csv_path, newline="", encoding="utf-8") as csv_f,
        open(args.jsonl_out, "w", encoding="utf-8") as out_f,
    ):
        reader = csv.DictReader(csv_f)
        for row in reader:
            json.dump(row, out_f, ensure_ascii=False)
            out_f.write("\n")


if __name__ == "__main__":
    main()

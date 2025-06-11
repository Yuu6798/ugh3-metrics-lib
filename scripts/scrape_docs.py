import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy tech docs scraper")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    docs = [
        {"id": 1, "text": "Sample documentation A"},
        {"id": 2, "text": "Sample documentation B"},
        {"id": 3, "text": "Sample documentation C"},
    ]

    with open(args.out, "w", encoding="utf-8") as fh:
        for doc in docs:
            json.dump(doc, fh, ensure_ascii=False)
            fh.write("\n")


if __name__ == "__main__":
    main()

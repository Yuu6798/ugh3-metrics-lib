import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag dataset with domain")
    parser.add_argument("--inp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--domain", required=True)
    args = parser.parse_args()

    with open(args.inp, "r", encoding="utf-8") as inp_f, open(args.out, "w", encoding="utf-8") as out_f:
        for line in inp_f:
            if not line.strip():
                continue
            obj = json.loads(line)
            obj["domain"] = args.domain
            obj["difficulty"] = 3
            json.dump(obj, out_f, ensure_ascii=False)
            out_f.write("\n")


if __name__ == "__main__":
    main()

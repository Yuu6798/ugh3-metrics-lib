import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple JSONL files")
    parser.add_argument("files", nargs="+", help="source1 source2 ... dest")
    args = parser.parse_args()

    *sources, dest = args.files
    with open(dest, "a", encoding="utf-8") as out_f:
        for src in sources:
            with open(src, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line.rstrip("\n") + "\n")


if __name__ == "__main__":
    main()

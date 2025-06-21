try:
    from comet.download_utils import download_model  # COMET < 2.0
except ModuleNotFoundError:  # pragma: no cover
    try:
        from comet.downloaders.download_utils import download_model  # COMET ≥ 2.0
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "サポートされていない COMET 版です。"
            "対応バージョンをインストールするか、import ロジックを更新してください。"
        ) from exc

import argparse
from typing import List

def download_models(models: List[str]) -> None:
    for model in models:
        print(f"Downloading {model} ...")
        download_model(model)
        print("done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm COMET model cache")
    parser.add_argument("models", nargs="+", help="COMET model names")
    args = parser.parse_args()
    download_models(args.models)


if __name__ == "__main__":
    main()

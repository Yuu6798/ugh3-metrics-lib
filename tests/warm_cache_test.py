import subprocess
import sys
import os
from pathlib import Path
from _pytest.monkeypatch import MonkeyPatch  # type: ignore[import-not-found,unused-ignore]


# prepare fake comet package for subprocess and module import
FAKE_DIR = Path('temp_comet')
(FAKE_DIR / 'comet' / 'downloaders').mkdir(parents=True, exist_ok=True)
code = 'def download_model(name):\n    pass\n'
(FAKE_DIR / 'comet' / 'download_utils.py').write_text(code)
(FAKE_DIR / 'comet' / '__init__.py').write_text('\n')
(FAKE_DIR / 'comet' / 'downloaders' / '__init__.py').write_text('\n')
(FAKE_DIR / 'comet' / 'downloaders' / 'download_utils.py').write_text(code)

sys.path.insert(0, str(FAKE_DIR))

import warm_cache  # noqa: E402


def test_cli_help() -> None:
    env = dict(PYTHONPATH=str(FAKE_DIR))
    result = subprocess.run(
        [sys.executable, 'warm_cache.py', '--help'],
        capture_output=True,
        text=True,
        env={**env, **{k: v for k, v in os.environ.items() if k not in env}},
    )
    assert result.returncode == 0
    assert 'Warm COMET model cache' in result.stdout


def test_download_models(monkeypatch: MonkeyPatch) -> None:
    called: list[str] = []

    def fake_download(model: str) -> None:
        called.append(model)

    monkeypatch.setattr(warm_cache, 'download_model', fake_download)
    warm_cache.download_models(['a', 'b'])
    assert called == ['a', 'b']

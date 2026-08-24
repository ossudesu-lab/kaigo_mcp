"""`.env` からAPIキーを読む。

案1（kaigo_matching）は Vercel と Vite が `.env` を自動で読むので、
そちらの習慣に合わせている。こちらは素の Python なので明示的に読む必要がある。

**エージェント側だけで使う。** MCPサーバー本体（`python -m kaigo_mcp`）は
APIキーを一切必要としないので、そちらからは呼ばない。
サーバーを動かすのに余計な依存を増やしたくない。
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


def load_env(path: Path = ENV_PATH) -> list[str]:
    """`.env` を読んで環境変数に入れる。読み込んだキー名を返す。

    **既に環境変数にあるものは上書きしない。** シェルで一時的に別のキーを
    渡して試す、という使い方を潰さないため。

    python-dotenv を使っていないのは、必要なのが `KEY=value` を読むだけで、
    そのために依存を1つ増やす理由が無いから。
    """
    if not path.exists():
        return []

    loaded: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # `export FOO=bar` と書かれていても読めるようにする（shの習慣）
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # 前後の引用符を1組だけ外す
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded

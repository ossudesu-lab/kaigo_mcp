"""案4（kaigo_gap_analysis）から公開用JSONを取り込む。

このサーバーは自前で集計しない。案4が export_web.py で書き出し、
公開ダッシュボードが配信しているのと同じファイルをコピーして使う。
そうしておけば、エージェントの答えとダッシュボードの数字が食い違わない。

使い方:
    python scripts/sync_data.py
    python scripts/sync_data.py --source /path/to/kaigo_gap_analysis
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

FILES = ("insurers.json", "summary.json")
DEFAULT_SOURCE = Path(__file__).resolve().parent.parent.parent / "kaigo_gap_analysis"
DEST = Path(__file__).resolve().parent.parent / "data"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    src = args.source / "web" / "public" / "data"
    if not src.is_dir():
        print(f"取り込み元が見つからない: {src}", file=sys.stderr)
        print("--source で kaigo_gap_analysis の場所を指定すること。", file=sys.stderr)
        return 1

    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        f = src / name
        if not f.exists():
            print(f"{f} が無い。案4で export_web.py を実行すること。", file=sys.stderr)
            return 1
        shutil.copy2(f, DEST / name)
        print(f"取り込み: {name} ({f.stat().st_size:,} バイト)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

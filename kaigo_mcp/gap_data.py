"""案4（kaigo_gap_analysis）が書き出した保険者データを読み、検索できる形にする。

**再計算はしない。** 読むのは公開ダッシュボードが配信しているのと同じ
`insurers.json` / `summary.json` そのもの。指標の定義は案4の `dataset.py` に1か所だけ置く、
という向こうの方針をこちら側でも守るため。ここでもう一度割り算を書くと、
エージェントの答えとダッシュボードの数字が食い違ったときに、どちらが正しいか分からなくなる。

データの更新は scripts/sync_data.py で取り込み直す。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 主指標の列名。案4の dataset.py と同じ名前をそのまま使う（対応を追いやすくするため）。
METRIC = "tokuyo_per100_kaigo3"


class DataUnavailable(RuntimeError):
    """データファイルが無い、または壊れている。"""


@lru_cache(maxsize=1)
def load_insurers() -> list[dict[str, Any]]:
    """1,571保険者。1件1辞書。"""
    path = DATA_DIR / "insurers.json"
    if not path.exists():
        raise DataUnavailable(
            f"{path} が無い。scripts/sync_data.py で案4から取り込むこと。"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_summary() -> dict[str, Any]:
    """全国の基準値（中央値・四分位など）。"""
    path = DATA_DIR / "summary.json"
    if not path.exists():
        raise DataUnavailable(
            f"{path} が無い。scripts/sync_data.py で案4から取り込むこと。"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _sorted_metric_values() -> list[float]:
    """主指標を持つ保険者の値を昇順で。順位計算に使う。

    欠測（避難区域など）は除く。母数が保険者数と一致しないので、
    順位を返すときは必ずこの母数（total）を添える。
    """
    return sorted(
        r[METRIC] for r in load_insurers() if isinstance(r.get(METRIC), (int, float))
    )


def percentile_of(value: float) -> dict[str, Any]:
    """全国で下から何番目か。値が小さいほど特養が足りない。

    同率を必ず数えて返す。特養定員0の保険者は90件あり、全部が「1位」になる。
    同率数を添えないと、読んだ側が「この保険者だけが全国最下位」と誤読する。
    """
    values = _sorted_metric_values()
    below = sum(1 for v in values if v < value)
    tied = sum(1 for v in values if v == value)
    total = len(values)
    return {
        "rank_from_lowest": below + 1,
        "tied": tied,
        "total": total,
        "percentile": round(below / total * 100, 1),
    }


def _normalize(s: str) -> str:
    return s.strip().replace("　", "").replace(" ", "")


def find_insurers(
    name: str, pref: str | None = None, limit: int = 10
) -> list[dict[str, Any]]:
    """保険者を名前で探す。完全一致を優先し、無ければ部分一致。

    表記ゆれと同名（「府中市」は東京都と広島県にある）があるので、
    1件に決め打ちせず候補を返す。絞り込みは呼び出し側の判断に任せる。
    """
    q = _normalize(name)
    rows = load_insurers()
    if pref:
        p = _normalize(pref)
        rows = [r for r in rows if p in r["insurer_pref"]]

    exact = [r for r in rows if _normalize(r["insurer_name"]) == q]
    if exact:
        return exact[:limit]
    return [r for r in rows if q and q in _normalize(r["insurer_name"])][:limit]


def rank_insurers(
    pref: str | None = None, order: str = "low", limit: int = 10
) -> list[dict[str, Any]]:
    """主指標の順に並べる。order="low" は特養が足りない順。"""
    rows = [
        r for r in load_insurers() if isinstance(r.get(METRIC), (int, float))
    ]
    if pref:
        p = _normalize(pref)
        rows = [r for r in rows if p in r["insurer_pref"]]
    rows.sort(key=lambda r: r[METRIC], reverse=(order == "high"))
    return rows[:limit]

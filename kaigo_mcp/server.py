"""MCPサーバー本体。道具の定義だけを置く。

## 道具の説明文について

説明文はLLMがこの道具を選ぶかどうかの判断材料そのものなので、実装のコメントではなく
**呼び出す側への仕様書**として書く。特に主指標は、数字だけ渡しても大小を判断できない
（24.6が高いのか低いのか分からない）ため、定義と全国値を説明文に含めている。

## 出力のキーを日本語にしている理由

英語キーより1件あたりのトークンが増えるが、`要介護3以上` のような概念は英訳すると
かえって曖昧になる（`heavy_care` では要介護3以上という線引きが消える）。
コストへの影響は Phase 2 で実測して判断する。今は読みやすさを取る。
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from . import gap_data

server = MCPServer(
    name="kaigo-gap",
    version="0.1.0",
    instructions=(
        "日本の介護保険の需給データ（2020年度・全1,571保険者）を引くための道具。"
        "特別養護老人ホームの定員が、要介護3以上の認定者に対してどれだけあるかを調べられる。"
    ),
)

# 主指標の説明。3つの道具すべてで同じ文言を使う。
# ばらばらに書くと、道具によってLLMの解釈が変わる。
_METRIC_DOC = (
    "主指標は「要介護3以上の認定者100人あたりの特養定員数」。"
    "特別養護老人ホームは原則として要介護3以上が入所要件なので、これが充足度の分母になる。"
    "全国値は24.6人・中央値26.7人で、小さいほど特養が不足している。"
)


def _format(row: dict[str, Any]) -> dict[str, Any]:
    """1保険者ぶんを、LLMが読める形に整える。"""
    metric = row.get(gap_data.METRIC)
    out: dict[str, Any] = {
        "都道府県": row["insurer_pref"],
        "保険者": row["insurer_name"],
        "要介護3以上の認定者数": row["要介護3以上"],
        "特養定員": row["cap_tokuyo"],
        "認定者100人あたり特養定員": metric,
    }
    # 広域連合は複数市町村で1保険者。市単独の数字と誤解されないよう明示する。
    if row.get("muni_count", 1) > 1:
        out["備考"] = f"{row['muni_count']}市町村による広域連合"
    if isinstance(metric, (int, float)):
        p = gap_data.percentile_of(metric)
        rank = f"{p['total']}保険者中、少ない順に{p['rank_from_lowest']}位（下位{p['percentile']}%）"
        if p["tied"] > 1:
            rank += f"。同じ値の保険者が{p['tied']}件あり同率"
        out["全国順位"] = rank
    # 特養定員0は全国で90件ある。小規模自治体では珍しくなく、住民は近隣自治体の
    # 施設を利用している。この注記が無いと「全国最下位の危機的地域」と読まれる。
    if metric == 0:
        out["注意"] = (
            "域内に特養が無い（全国で90保険者が該当）。小規模自治体では珍しくなく、"
            "住民は近隣自治体の施設を利用していることが多い。"
            "単独で不足と断じず、周辺自治体と併せて見ること。"
        )
    return out


@server.tool(
    description=(
        "全国の基準値を返す。個別の保険者の数字を評価する前に、まずこれを引くこと。"
        f"{_METRIC_DOC}"
    )
)
def get_national_baseline() -> dict[str, Any]:
    s = gap_data.load_summary()
    return {
        "対象年度": s["year"],
        "保険者数": s["insurers"],
        "市区町村数": s["municipalities"],
        "うち広域連合": s["unions"],
        "全国_認定者100人あたり特養定員": s["nationalPer100"],
        "中央値": s["median"],
        "四分位": {"25%": s["p25"], "75%": s["p75"]},
        "上下1割": {"10%": s["p10"], "90%": s["p90"]},
        "特養定員が0の保険者数": s["zeroCapacity"],
        "要介護3以上の認定者数_全国": s["kaigo3Total"],
        "特養定員_全国": s["nationalCapacity"],
    }


@server.tool(
    description=(
        "市区町村名または保険者名で、その地域の介護需給を引く。"
        "同名の自治体（府中市など）や表記ゆれがあるため複数件返ることがある。"
        "その場合は都道府県を pref に指定して絞り込むか、利用者にどちらか尋ねること。"
        f"{_METRIC_DOC}"
    )
)
def lookup_insurer(name: str, pref: str | None = None) -> dict[str, Any]:
    rows = gap_data.find_insurers(name, pref=pref)
    if not rows:
        return {
            "該当": 0,
            "補足": (
                f"「{name}」に一致する保険者が無い。"
                "介護保険は市区町村または広域連合が保険者なので、"
                "町名・字名・施設名では引けない。市区町村名で試すこと。"
            ),
        }
    result: dict[str, Any] = {"該当": len(rows), "保険者": [_format(r) for r in rows]}
    if len(rows) > 1:
        result["補足"] = "複数一致。pref で都道府県を指定すると絞り込める。"
    return result


@server.tool(
    description=(
        "特養の充足度が低い順（または高い順）に保険者を並べる。"
        "pref を指定すると県内だけで並べる。県内の差は県間の差より大きいことが多い。"
        f"{_METRIC_DOC}"
    )
)
def rank_insurers(
    pref: str | None = None, order: str = "low", limit: int = 10
) -> dict[str, Any]:
    if order not in ("low", "high"):
        return {"エラー": "order は 'low'（不足順）か 'high'（充足順）のみ"}
    # 上限を切るのは、1,571件がそのまま文脈に流れ込むのを防ぐため。
    limit = max(1, min(limit, 50))
    rows = gap_data.rank_insurers(pref=pref, order=order, limit=limit)
    if not rows:
        return {"該当": 0, "補足": f"「{pref}」に一致する都道府県が無い。"}
    return {
        "並び順": "特養が不足している順" if order == "low" else "特養が充足している順",
        "対象": pref or "全国",
        "件数": len(rows),
        "保険者": [_format(r) for r in rows],
    }


def main() -> None:
    server.run(transport="stdio")

"""法令検索のeval。LLMは呼ばない（費用0円）。

retrieval-cases.json の各質問について、検索結果の上位に正解の条が入るかを測る。
単位は条。同じ条の項が複数並んだときは、最初の1件だけを数える（項違いで順位が
膨らむと、方式によって不利になるため）。

指標:
    recall@k … 上位k条のどれかが正解に入った割合
    MRR      … 正解が最初に現れた順位の逆数の平均（1位なら1、3位なら0.33、無ければ0）

使い方:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --misses      # 外れた質問と、実際に出た上位3条を表示
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kaigo_mcp import law_search  # noqa: E402

CASES = ROOT / "retrieval-cases.json"
KS = (1, 3, 5)
# 条に畳む前に取る塊の数。畳んだあとに5条が残るよう多めに取る。
FETCH = 40

_ALIAS = {"法": "介護保険法"}


def _key(law: str, num: str) -> tuple[str, str]:
    """eval-cases の短縮名（法/基準）を、索引側の法令名に合わせる。"""
    if law in _ALIAS:
        return (_ALIAS[law], num)
    return ("指定介護老人福祉施設の人員、設備及び運営に関する基準", num)


def ranked_articles(retriever, query: str, depth: int = 5) -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []
    for chunk, _ in retriever(query, FETCH):
        k = (chunk.law, chunk.article_num)
        if k not in seen:
            seen.append(k)
        if len(seen) >= depth:
            break
    return seen


def evaluate(retriever, cases: list[dict], show_misses: bool) -> dict:
    by_level: dict[int, list[float]] = defaultdict(list)
    hits = {k: defaultdict(list) for k in KS}
    rr_all: list[float] = []
    misses: list[tuple[dict, list]] = []
    for c in cases:
        gold = {_key(*g) for g in c["gold"]}
        got = ranked_articles(retriever, c["q"])
        rank = next((i + 1 for i, a in enumerate(got) if a in gold), None)
        rr = 1 / rank if rank else 0.0
        rr_all.append(rr)
        by_level[c["level"]].append(rr)
        for k in KS:
            ok = 1.0 if rank and rank <= k else 0.0
            hits[k]["all"].append(ok)
            hits[k][c["level"]].append(ok)
        if not rank or rank > 3:
            misses.append((c, got[:3]))
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    out = {
        "n": len(cases),
        "mrr": mean(rr_all),
        "recall": {k: mean(hits[k]["all"]) for k in KS},
        "by_level": {
            lv: {"n": len(by_level[lv]), "mrr": mean(by_level[lv]),
                 **{f"r@{k}": mean(hits[k][lv]) for k in KS}}
            for lv in sorted(by_level)
        },
    }
    if show_misses:
        out["misses"] = misses
    return out


def report(name: str, r: dict) -> None:
    rec = "  ".join(f"r@{k}={r['recall'][k]:.2f}" for k in KS)
    print(f"\n## {name}  (n={r['n']})  {rec}  MRR={r['mrr']:.3f}")
    for lv, d in r["by_level"].items():
        cols = "  ".join(f"r@{k}={d[f'r@{k}']:.2f}" for k in KS)
        print(f"   level{lv} (n={d['n']:>2}): {cols}  MRR={d['mrr']:.3f}")
    for c, got in r.get("misses", []):
        gold = ", ".join(f"{a}{b}" for a, b in c["gold"])
        top = " / ".join(f"{a[:4]}{b}" for a, b in got)
        print(f"   x [{c['id']}] {c['q']}\n       正解: {gold}  →  上位: {top}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--misses", action="store_true")
    args = ap.parse_args()

    cases = [c for c in json.loads(CASES.read_text(encoding="utf-8"))["cases"]
             if c.get("answerable", True)]
    bm25 = law_search.load_index()
    report("BM25（文字2-gram）", evaluate(bm25.search, cases, args.misses))
    dense = law_search.DenseIndex(bm25.chunks)
    report("埋め込み（bge-m3）", evaluate(dense.search, cases, args.misses))
    hybrid = law_search.HybridIndex(bm25, dense)
    report("ハイブリッド（RRF）", evaluate(hybrid.search, cases, args.misses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

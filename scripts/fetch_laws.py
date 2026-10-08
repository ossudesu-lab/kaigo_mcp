"""e-Gov法令APIから対象の法令を取得し、条単位に分割して data/laws.json に書く。

検索の単位を「条」にする理由: 法令は条が引用の最小単位で、評価の正解も
「第○条」で書ける。文字数で機械的に切ると、条番号という正解の手がかりが消える。
項・号は条の本文に含めて、条の中の構造は改行で残す。

対象は本則（MainProvision）のみ。附則は経過措置が中心で、制度の問い合わせに
答える目的では雑音になりやすいので外している。

使い方:
    python scripts/fetch_laws.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

API = "https://laws.e-gov.go.jp/api/2/law_data/{law_id}?law_full_text_format=json"
DEST = Path(__file__).resolve().parent.parent / "data" / "laws.json"

LAWS = {
    "409AC0000000123": "介護保険法",
    "411M50000100039": "指定介護老人福祉施設の人員、設備及び運営に関する基準",
}

# 見出しとして文脈に残すタグ（章・節など）。条の検索結果に所属を付けるため。
_HEADING_TAGS = {"Part", "Chapter", "Section", "Subsection", "Division"}
_HEADING_TITLE = {f"{t}Title" for t in _HEADING_TAGS}
# 前に改行を入れるタグ。項・号の区切りを残す。
_BREAK_TAGS = {"Paragraph", "Item", "Subitem1", "Subitem2", "Subitem3", "TableStruct"}
# 項番号・号の題。直後に空白を入れる対象。
_NUMBER_TAGS = {"ParagraphNum", "ItemTitle", "Subitem1Title", "Subitem2Title", "Subitem3Title"}
# ふりがな。本文には要らない。
_SKIP_TAGS = {"Rt"}


def _text(node: Any) -> str:
    """ノード配下の文字列を連結する。項・号の手前で改行する。"""
    if isinstance(node, str):
        return node
    if not isinstance(node, dict) or node.get("tag") in _SKIP_TAGS:
        return ""
    body = "".join(_text(c) for c in node.get("children", []))
    if node.get("tag") in _BREAK_TAGS:
        return "\n" + body
    # 「一」と本文がくっつくと「一医師…」になる。番号・題の直後に空白を入れる。
    if node.get("tag") in _NUMBER_TAGS:
        return body + " "
    return body


def _find(node: Any, tag: str) -> Any | None:
    if isinstance(node, dict):
        if node.get("tag") == tag:
            return node
        for c in node.get("children", []):
            hit = _find(c, tag)
            if hit is not None:
                return hit
    return None


def _walk(node: Any, trail: list[str], out: list[dict[str, Any]], law: str) -> None:
    if not isinstance(node, dict):
        return
    tag = node.get("tag")
    if tag == "Article":
        caption = _find(node, "ArticleCaption")
        title = _find(node, "ArticleTitle")
        # 検索の単位は項。長い条（定義の条は約6,800文字）を丸ごと1塊にすると、
        # 検索語が薄まり、LLMに渡す文脈も無駄に膨らむ。
        paragraphs = [
            {"num": p["attr"].get("Num", ""), "text": _text(p).strip()}
            for p in node.get("children", [])
            if isinstance(p, dict) and p.get("tag") == "Paragraph"
        ]
        out.append(
            {
                "law": law,
                "article_num": node["attr"].get("Num", ""),
                "article_title": _text(title).strip(),
                "caption": _text(caption).strip("（）() \n"),
                "section": " ".join(trail),
                "paragraphs": paragraphs,
                "text": _text(node).strip(),
            }
        )
        return
    if tag in _HEADING_TAGS:
        heading = _find(node, f"{tag}Title")
        trail = trail + [_text(heading).strip()] if heading else trail
    for c in node.get("children", []):
        _walk(c, trail, out, law)


def fetch(law_id: str) -> dict[str, Any]:
    with urllib.request.urlopen(API.format(law_id=law_id), timeout=60) as r:
        return json.load(r)


def main() -> int:
    articles: list[dict[str, Any]] = []
    meta: list[dict[str, str]] = []
    for law_id, title in LAWS.items():
        data = fetch(law_id)
        main_prov = _find(data["law_full_text"], "MainProvision")
        if main_prov is None:
            print(f"本則が見つからない: {title}", file=sys.stderr)
            return 1
        before = len(articles)
        _walk(main_prov, [], articles, title)
        rev = data["revision_info"]
        meta.append(
            {
                "law_id": law_id,
                "title": title,
                "law_num": data["law_info"]["law_num"],
                "revision": rev["law_revision_id"],
                "enforced": rev["amendment_enforcement_date"],
            }
        )
        print(f"{title}: {len(articles) - before}条（施行 {rev['amendment_enforcement_date']}）")

    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(
        json.dumps({"laws": meta, "articles": articles}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"書き出し: {DEST} ({DEST.stat().st_size:,} バイト)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

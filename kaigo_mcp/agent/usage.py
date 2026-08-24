"""実行のコストを記録・累積する。

案1（kaigo_matching）の `scripts/eval-usage.js` と同じ方針。
あちらは1回ごとのコストは表示していたのに**累積**を出しておらず、
「1回数十円」を1日に何十回も回してクレジットを使い切り、公開中の本番が止まった。

エージェントは1問がループ数回ぶんになるので、単発抽出よりさらに効きやすい。
実行のたびに今回と累計の両方を出す。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import RunRecord

LOG_PATH = Path(__file__).resolve().parent.parent.parent / ".agent-usage.jsonl"

# 100万トークンあたりUSD（2026-08 時点）。
# **単価は変わる。** 合わないと感じたら各社の料金表を確認すること。
# 表に無いモデルは金額を出さず、トークン数だけ示す（誤った金額より良い）。
PRICE: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-5": (5.0, 25.0),
    "Qwen/Qwen3-235B-A22B-Instruct": (0.09, 0.10),
}

USD_JPY = 150  # 目安。正確な請求額ではない。


def is_local(model: str) -> bool:
    """Ollama のモデル名は `名前:タグ` 形式。ローカルは課金されない。"""
    return ":" in model and "/" not in model


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if is_local(model):
        return 0.0
    price = PRICE.get(model)
    if price is None:
        return None
    return input_tokens / 1e6 * price[0] + output_tokens / 1e6 * price[1]


def yen(usd: float) -> float:
    return round(usd * USD_JPY, 2)


def append(record: RunRecord, path: Path = LOG_PATH) -> None:
    usd = cost_usd(record.model, record.input_tokens, record.output_tokens)
    row = {
        "model": record.model,
        "question": record.question,
        "steps": record.steps,
        "tools": record.tool_calls,
        "input": record.input_tokens,
        "output": record.output_tokens,
        "seconds": record.seconds,
        "stopped_by": record.stopped_by,
        "stop_reason": record.stop_reason,
        "usd": usd,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def totals(path: Path = LOG_PATH) -> dict[str, Any]:
    """これまでの累計。単価不明の行は件数だけ数える。"""
    runs = 0
    usd = 0.0
    unpriced = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # 壊れた行で集計を止めない
            runs += 1
            if isinstance(row.get("usd"), (int, float)):
                usd += row["usd"]
            else:
                unpriced += 1
    return {"runs": runs, "usd": usd, "unpriced": unpriced}


def format_cost(record: RunRecord, path: Path = LOG_PATH) -> str:
    usd = cost_usd(record.model, record.input_tokens, record.output_tokens)
    tot = totals(path)
    if usd is None:
        now = "単価不明（トークンのみ記録）"
    elif usd == 0:
        now = "0円（ローカル実行）"
    else:
        now = f"${usd:.4f}（約{yen(usd)}円）"
    after = f"累計 {tot['runs']}回・約${tot['usd']:.2f}（約{yen(tot['usd']):,.0f}円）"
    if tot["unpriced"]:
        after += f"（うち{tot['unpriced']}回は単価不明で未計上）"
    return f"今回 {now}　／　{after}"

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
    # DeepInfra 自身の表示は $0.45/$3.00 だが、集計サイトは $0.54/$3.40 を出している。
    # 食い違うときは**高いほう**を入れる。見積もりが実際より安く出るのが
    # 一番まずい（残高は公開中の本番と共有しているため）。
    "Qwen/Qwen3.5-397B-A17B": (0.54, 3.40),
    # NVIDIA の無料枠。金額は発生しないが**クレジットは減る**（初期1,000程度）。
    # 0円と出るからといって無制限ではないので、回数は意識すること。
    "deepseek-ai/deepseek-v4-flash-0731": (0.0, 0.0),
    "moonshotai/kimi-k3": (0.0, 0.0),
}

USD_JPY = 150  # 目安。正確な請求額ではない。

# 金銭は発生しないが**無制限ではない**エンドポイント。
# ローカル実行と同じ「0円」で括ると、クレジットを使い切るまで気づけない。
# 案1の事故は「見えていない上限がある」ことに気づかなかったのが原因なので、
# 0円の中身を分けて表示する。
FREE_TIER = {
    "deepseek-ai/deepseek-v4-flash-0731": "NVIDIA無料枠（初期1,000クレジット・40req/分）",
    "moonshotai/kimi-k3": "NVIDIA無料枠（初期1,000クレジット・40req/分）",
}


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


# 実績が無いモデルを見積もるときの既定プロファイル。
# 列B（qwen3.5:9b / num_ctx 8192）で「青森県で特養が足りない市町村を3つ挙げて…」を
# 完走させたときの実測値。3ステップぶんの合計。
# 楽な質問（尼崎市）は 3,176 / 601 だったので、これはやや重い側の想定。
DEFAULT_PROFILE = (3702, 1010)


def estimate(model: str, requests: int, path: Path = LOG_PATH) -> dict[str, Any]:
    """これから `requests` 回まわすといくらか。

    同じモデルの実績がログにあればそれを使う。無ければ既定プロファイル。
    **推定であって請求額ではない。** 桁を間違えないためのもの。
    """
    samples = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("model") == model and row.get("input"):
                samples.append((row["input"], row["output"]))

    if samples:
        per_in = sum(s[0] for s in samples) / len(samples)
        per_out = sum(s[1] for s in samples) / len(samples)
        basis = f"このモデルの実績{len(samples)}回の平均"
    else:
        per_in, per_out = DEFAULT_PROFILE
        basis = "既定プロファイル（実績が無いため）"

    usd = cost_usd(model, int(per_in * requests), int(per_out * requests))
    return {
        "requests": requests,
        "per_input": int(per_in),
        "per_output": int(per_out),
        "usd": usd,
        "basis": basis,
    }


def print_preflight(model: str, requests: int, path: Path = LOG_PATH) -> None:
    """課金しうる実行の前に、見積もりと累計を必ず目に入れる。

    案1では1回ごとのコストは表示していたのに**累計**を出しておらず、
    「1回数十円」を何十回も回してクレジットを使い切り、公開中の本番が止まった。
    見えていなかったのは累計のほう。だから両方出す。
    """
    est = estimate(model, requests, path)
    tot = totals(path)
    print(f"── 実行前の見積もり（{model}）")
    print(f"   {requests}回 × 入力{est['per_input']:,} / 出力{est['per_output']:,} トークン")
    print(f"   根拠: {est['basis']}")
    if est["usd"] is None:
        print("   今回の想定: 単価不明。金額は出せない")
    elif is_local(model):
        print("   今回の想定: 0円（ローカル実行・上限なし）")
    elif model in FREE_TIER:
        # 0円だが無制限ではない。ここを混ぜると上限に気づけない。
        print(f"   今回の想定: 0円 — ただし {FREE_TIER[model]} を消費する")
    elif est["usd"] == 0:
        print("   今回の想定: 0円")
    else:
        print(f"   今回の想定: ${est['usd']:.2f}（約{yen(est['usd']):,.0f}円）")
    print(
        f"   これまでの累計: {tot['runs']}回・約${tot['usd']:.2f}"
        f"（約{yen(tot['usd']):,.0f}円）"
    )
    # 警告は当たる相手にだけ出す。全部に出すと読み飛ばされる。
    if est["usd"] and model.startswith("claude-"):
        print("   Anthropicの残高は公開中の本番と共有。回す前に残高を確認すること。")
    elif est["usd"]:
        print("   このモデルは従量課金。回す前に残高を確認すること。")


def format_cost(record: RunRecord, path: Path = LOG_PATH) -> str:
    usd = cost_usd(record.model, record.input_tokens, record.output_tokens)
    tot = totals(path)
    if usd is None:
        now = "単価不明（トークンのみ記録）"
    elif is_local(record.model):
        now = "0円（ローカル実行）"
    elif record.model in FREE_TIER:
        now = f"0円（{FREE_TIER[record.model]}を消費）"
    elif usd == 0:
        now = "0円"
    else:
        now = f"${usd:.4f}（約{yen(usd)}円）"
    after = f"累計 {tot['runs']}回・約${tot['usd']:.2f}（約{yen(tot['usd']):,.0f}円）"
    if tot["unpriced"]:
        after += f"（うち{tot['unpriced']}回は単価不明で未計上）"
    return f"今回 {now}　／　{after}"

"""エージェントのeval。列ごとに正答率・完走率・コストを出す。

## 何を測るか

道具呼び出し単体の成功率（scripts/probe_tool_calling.py）ではなく、
**1問を最後まで走らせた結果**を見る。エージェントで効くのは連続成功率のほう。

判定はすべてプログラムで書けるものに限っている。LLM-as-judge は使わない。
案1で「審判自体の検証を済ませずにCIへ載せると、審判が見つけられていないだけの
見逃しを通してしまう」という理由で保留したのと同じ判断。

## 使い方

    python scripts/eval_agent.py --column A-local-cpu            # 無料
    python scripts/eval_agent.py --column D-claude --yes         # 課金あり
    python scripts/eval_agent.py --column A-local-cpu --runs 3

課金の発生する列は `--yes` が無いと、見積もりを出したところで止まる。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ひらがな・カタカナ。漢字は入れない（中国語の回答と区別できなくなるため）。
KANA = re.compile(r"[぀-ゟ゠-ヿ]")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kaigo_mcp.agent import usage  # noqa: E402
from kaigo_mcp.agent.backends import PRESETS, BackendError, build  # noqa: E402
from kaigo_mcp.agent.loop import DEFAULT_MAX_STEPS, run_agent  # noqa: E402
from kaigo_mcp.agent.types import RunRecord  # noqa: E402

CASES_PATH = ROOT / "eval-cases.json"


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    record: RunRecord | None = None


def check(case: dict, record: RunRecord) -> list[str]:
    """満たせなかった条件を返す。空なら合格。"""
    failures: list[str] = []

    # 答えが出ていない実行は、中身を見るまでもなく不合格。
    # truncated を "end" と数えていたせいで空答えが完走に見えていた事故がある。
    if not record.answered:
        return [f"完走せず（{record.stopped_by}{'/' + record.stop_reason if record.stop_reason else ''}）"]

    answer = record.answer
    called = set(record.tool_calls)

    # 全ケース共通。qwen2.5:7b は迷うと中国語で答えることがある
    # （「架空市は特養が足りてる？」で実際に発生した）。
    # 日本の介護制度を日本語で説明する道具なので、これは内容以前の不合格。
    # 漢字だけでは中国語と区別できないので、かなの有無で見る。
    if not KANA.search(answer):
        failures.append("日本語で答えていない（かなが1文字も無い）")

    for name in case.get("must_call", []):
        if name not in called:
            failures.append(f"道具 {name} を呼んでいない（呼んだ: {sorted(called) or 'なし'}）")

    for text in case.get("must_include", []):
        if text not in answer:
            failures.append(f"「{text}」が答えに無い")

    any_of = case.get("must_include_any", [])
    if any_of and not any(t in answer for t in any_of):
        failures.append(f"{any_of} のどれも答えに無い")

    for text in case.get("must_not_include", []):
        if text in answer:
            failures.append(f"「{text}」が答えに含まれている")

    return failures


async def run_all(
    cases: list[dict], column: str, runs: int, max_steps: int, verbose: bool
) -> list[CaseResult]:
    backend = build(column)
    results: list[CaseResult] = []
    for case in cases:
        for attempt in range(runs):
            record = await run_agent(
                case["question"], backend, max_steps=max_steps, verbose=verbose
            )
            usage.append(record)
            failures = check(case, record)
            results.append(
                CaseResult(
                    case_id=case["id"], passed=not failures, failures=failures, record=record
                )
            )
            mark = "OK" if not failures else "NG"
            print(
                f"  {mark} {case['id']}"
                f"（{attempt + 1}/{runs}）"
                f" {record.steps}ステップ {record.seconds}秒"
                f" 道具{record.tool_calls}"
            )
            for f in failures:
                print(f"       - {f}")
    return results


def summarize(results: list[CaseResult], cases: list[dict], runs: int) -> int:
    print()
    print("=" * 60)
    by_case: dict[str, list[CaseResult]] = {}
    for r in results:
        by_case.setdefault(r.case_id, []).append(r)

    for case in cases:
        rs = by_case.get(case["id"], [])
        ok = sum(1 for r in rs if r.passed)
        # 全周そろって通らないケースは、たまたま通っただけなので分けて見せる。
        flag = "" if ok == len(rs) else ("  ← ゆらぐ" if ok else "")
        print(f"  {ok}/{len(rs)}  {case['id']}{flag}")

    total_runs = len(results)
    passed = sum(1 for r in results if r.passed)
    answered = sum(1 for r in results if r.record and r.record.answered)
    followed = sum(
        1
        for r in results
        if r.record and "get_national_baseline" in r.record.tool_calls
    )
    stable = sum(
        1 for case in cases if all(r.passed for r in by_case.get(case["id"], []))
    )

    print()
    print(f"  正答       {passed}/{total_runs} = {passed / total_runs:.0%}")
    print(f"  完走       {answered}/{total_runs} = {answered / total_runs:.0%}")
    print(f"  指示追従   {followed}/{total_runs} = {followed / total_runs:.0%}")
    print(f"  全周合格   {stable}/{len(cases)} ケース（{runs}周すべて通った）")

    tok_in = sum(r.record.input_tokens for r in results if r.record)
    tok_out = sum(r.record.output_tokens for r in results if r.record)
    secs = sum(r.record.seconds for r in results if r.record)
    model = results[0].record.model if results and results[0].record else "?"
    usd = usage.cost_usd(model, tok_in, tok_out)
    print(f"  トークン   入力 {tok_in:,} / 出力 {tok_out:,}　合計 {secs:.0f}秒")
    if usd is None:
        print("  コスト     単価不明")
    else:
        print(f"  コスト     ${usd:.4f}（約{usage.yen(usd):,.0f}円）")
    return 0 if passed == total_runs else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--column", default="A-local-cpu")
    parser.add_argument("--runs", type=int, default=3, help="1ケースあたりの周回数")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--case", help="このIDのケースだけ走らせる")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--yes", action="store_true", help="課金の発生する列を実際に走らせる"
    )
    args = parser.parse_args()

    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = data["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"ケース {args.case} が無い", file=sys.stderr)
            return 1

    if args.column not in PRESETS:
        print(f"未知の列: {args.column}。使えるのは {list(PRESETS)}", file=sys.stderr)
        return 1
    model = PRESETS[args.column]["model"]
    planned = len(cases) * args.runs

    print(f"列 {args.column}（{model}）")
    print(f"{len(cases)}ケース × {args.runs}周 = {planned}回\n")
    usage.print_preflight(model, planned)
    print()

    # 課金しうる列は、見積もりを見せたうえで明示の同意を要求する。
    # 「1回数十円」を何十回も回して本番を止めた事故が案1で起きている。
    est = usage.estimate(model, planned)
    if est["usd"] and not args.yes:
        print("課金が発生する列。内容を確認して --yes を付けて実行すること。")
        return 2

    try:
        results = asyncio.run(
            run_all(cases, args.column, args.runs, args.max_steps, args.verbose)
        )
    except BackendError as exc:
        print(f"列を作れない: {exc}", file=sys.stderr)
        return 1

    return summarize(results, cases, args.runs)


if __name__ == "__main__":
    raise SystemExit(main())

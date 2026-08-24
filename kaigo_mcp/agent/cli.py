"""手でエージェントを試すためのCLI。

    python -m kaigo_mcp.agent "尼崎市は特養が足りてる？"
    python -m kaigo_mcp.agent --column D-claude "..." --verbose
    python -m kaigo_mcp.agent --list
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from . import usage
from .backends import PRESETS, BackendError, build
from .loop import DEFAULT_MAX_STEPS, run_agent


def _list_columns() -> None:
    print("比較する列:")
    for key, cfg in PRESETS.items():
        print(f"  {key:<14} {cfg['model']}")
        print(f"  {'':<14} {cfg['note']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kaigo_mcp.agent")
    parser.add_argument("question", nargs="?", help="エージェントに聞く質問")
    parser.add_argument(
        "--column",
        default="A-local-cpu",
        help="使う列。既定はローカルCPU（無料）",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--verbose", action="store_true", help="道具の呼び出しを表示")
    parser.add_argument("--list", action="store_true", help="列の一覧")
    args = parser.parse_args(argv)

    if args.list:
        _list_columns()
        return 0
    if not args.question:
        parser.error("質問を指定するか --list を使うこと")

    try:
        backend = build(args.column)
    except BackendError as exc:
        print(f"列 {args.column} を作れない: {exc}", file=sys.stderr)
        return 1

    print(f"列: {args.column}（{backend.model}）")
    record = asyncio.run(
        run_agent(
            args.question, backend, max_steps=args.max_steps, verbose=args.verbose
        )
    )

    print()
    if record.stopped_by == "error":
        print(f"失敗: {record.error}")
    elif record.stopped_by == "max_steps":
        print(f"上限{args.max_steps}ステップに到達して打ち切り。答えは出ていない。")
    elif record.stopped_by == "truncated":
        why = record.stop_reason or "本文が空"
        print(f"生成が途中で切れた（{why}）。答えは出ていない。完走として数えないこと。")
        print("ローカルなら num_ctx が足りていない可能性が高い。")
        if record.answer.strip():
            print()
            print("切れる前の本文:")
            print(record.answer)
    else:
        print(record.answer)

    print()
    print(
        f"ステップ {record.steps}"
        f"　道具 {len(record.tool_calls)}回 {record.tool_calls}"
        f"　{record.seconds}秒"
    )
    print(f"トークン 入力 {record.input_tokens:,} / 出力 {record.output_tokens:,}")
    usage.append(record)
    print(usage.format_cost(record))
    return 0 if record.stopped_by == "end" else 1


if __name__ == "__main__":
    raise SystemExit(main())

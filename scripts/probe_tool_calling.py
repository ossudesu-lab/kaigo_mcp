"""同じ質問を繰り返し、道具呼び出しが**構造化された形で出る率**を測る。

## なぜ必要か

小さいモデルは、道具呼び出しをテキストとして書き出すことがある
（`{"name": ...}` や `</tool_call>` が本文に漏れる）。1回試して成功しても意味がない。
エージェントは多段で回るので、成功率70%のモデルは3ステップで 0.7^3 = 34% しか完走しない。

**単発の精度ではなく、連続成功率が効く。** それを見るための計測。

    python scripts/probe_tool_calling.py --model qwen2.5:7b --runs 5
    python scripts/probe_tool_calling.py --model qwen2.5:7b --runs 5 --temperature 0
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kaigo_mcp.agent.loop import SYSTEM  # noqa: E402

QUESTION = "尼崎市は特養が足りてる？"


async def fetch_tools() -> list[dict]:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "kaigo_mcp"], cwd=str(ROOT)
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in result.tools
            ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="省略するとモデルの既定値。0にすると安定するかを見る",
    )
    args = parser.parse_args()

    tools = asyncio.run(fetch_tools())
    client = OpenAI(base_url=args.base_url, api_key="local")

    temp_label = "既定" if args.temperature is None else str(args.temperature)
    print(f"モデル {args.model} / 温度 {temp_label} / {args.runs}回")
    print(f"質問: {QUESTION}\n")

    ok = 0
    seconds: list[float] = []
    for i in range(args.runs):
        kwargs = {} if args.temperature is None else {"temperature": args.temperature}
        started = time.monotonic()
        res = client.chat.completions.create(
            model=args.model,
            tools=tools,
            max_tokens=300,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": QUESTION},
            ],
            **kwargs,
        )
        seconds.append(time.monotonic() - started)
        message = res.choices[0].message
        if message.tool_calls:
            ok += 1
            names = [c.function.name for c in message.tool_calls]
            print(f"  {i + 1}. OK {names}")
        else:
            leaked = (message.content or "").strip().replace("\n", " ")
            print(f"  {i + 1}. NG テキストに漏れた: {leaked[:90]}")

    rate = ok / args.runs
    print(f"\n道具呼び出し成功率: {ok}/{args.runs} = {rate:.0%}")
    # 多段ループでは連続成功が要る。1回あたりの率だけ見ると実力を過大評価する。
    for steps in (2, 3):
        print(f"  {steps}ステップ連続で成功する確率: {rate ** steps:.0%}")
    print(f"1回あたり {statistics.mean(seconds):.1f}秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

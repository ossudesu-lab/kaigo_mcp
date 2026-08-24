"""エージェントループ。**全列がこの1本を通る。**

道具はMCPサーバー越しに呼ぶ（関数を直接importしない）。
importで済ませるとMCPを通していないことになり、
「MCPサーバーを作った」という主張が実際には検証されないため。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .types import RunRecord, ToolResult, ToolSpec, Turn

ROOT = Path(__file__).resolve().parent.parent.parent

SYSTEM = """あなたは日本の介護保険の需給データに答えるアシスタントです。

必ず道具を使って調べ、推測で数字を答えないこと。データは2020年度のものです。

答える前に get_national_baseline で全国の基準値を確認すること。
個別の数字だけでは、それが高いのか低いのか判断できません。

道具が複数の候補を返したときは、勝手に1つ選ばず、利用者にどちらか尋ねてください。
"""

# ループの上限。エージェントは履歴を積みながら回るので、
# 上限が無いと1問で延々とトークンを消費しうる。コスト事故の最後の壁。
DEFAULT_MAX_STEPS = 6


def _to_specs(mcp_tools: Any) -> list[ToolSpec]:
    return [
        ToolSpec(name=t.name, description=t.description or "", schema=t.input_schema)
        for t in mcp_tools.tools
    ]


async def run_agent(
    question: str,
    backend: Any,
    max_steps: int = DEFAULT_MAX_STEPS,
    verbose: bool = False,
) -> RunRecord:
    """1問を最後まで走らせ、実行記録を返す。"""
    record = RunRecord(
        backend=getattr(backend, "kind", "?"),
        model=backend.model,
        question=question,
    )
    started = time.monotonic()
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "kaigo_mcp"], cwd=str(ROOT)
    )

    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                specs = _to_specs(await session.list_tools())

                history: list[Turn] = [Turn(role="user", text=question)]

                for step in range(max_steps):
                    record.steps = step + 1
                    reply = await asyncio.to_thread(
                        backend.send, SYSTEM, history, specs
                    )
                    record.input_tokens += reply.input_tokens
                    record.output_tokens += reply.output_tokens

                    if not reply.tool_calls:
                        record.answer = reply.text
                        record.stop_reason = reply.stop_reason
                        # 生成が上限で切れると、道具呼び出しも本文も無い形で返る。
                        # これを "end" にすると、答えの出ていない実行が完走と同じ形で
                        # 残り、列どうしの比較が壊れる（列Bの9Bで4回とも起きた。
                        # 誤った引数で該当0になり、考え込んだままコンテキストを
                        # 使い切って、出力3,121トークンで本文が空だった）。
                        if reply.stop_reason == "length" or not reply.text.strip():
                            record.stopped_by = "truncated"
                        else:
                            record.stopped_by = "end"
                        break

                    history.append(
                        Turn(
                            role="assistant",
                            text=reply.text,
                            tool_calls=reply.tool_calls,
                        )
                    )
                    if verbose:
                        for c in reply.tool_calls:
                            print(f"  [{step + 1}] {c.name}({c.arguments})")

                    results: list[ToolResult] = []
                    for call in reply.tool_calls:
                        record.tool_calls.append(call.name)
                        try:
                            out = await session.call_tool(call.name, call.arguments)
                            body = out.structured_content
                            if body is None:
                                body = {
                                    "text": "".join(
                                        getattr(b, "text", "") for b in out.content
                                    )
                                }
                            results.append(
                                ToolResult(
                                    id=call.id,
                                    name=call.name,
                                    content=json.dumps(body, ensure_ascii=False),
                                )
                            )
                        except Exception as exc:
                            # 道具の失敗でループを止めない。モデルに伝えて回復させ、
                            # 回復できるかどうかも比較したい性質のため。
                            results.append(
                                ToolResult(
                                    id=call.id,
                                    name=call.name,
                                    content=f"道具の呼び出しに失敗: {exc}",
                                    is_error=True,
                                )
                            )
                    history.append(Turn(role="tool", tool_results=results))
                else:
                    record.stopped_by = "max_steps"
                    record.answer = ""

    except Exception as exc:
        record.stopped_by = "error"
        record.error = f"{type(exc).__name__}: {exc}"

    record.seconds = round(time.monotonic() - started, 2)
    return record

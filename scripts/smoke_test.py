"""サーバーを別プロセスで起動し、stdio越しに本物のMCP通信を通す。

関数を直接呼ぶテストでは、道具が登録されているか・スキーマが妥当か・
プロセスとして起動できるかを確認できない。ここは必ずクライアント経由で叩く。

使い方:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parent.parent

# (道具名, 引数, 応答に必ず含まれるべきキー)
CASES = [
    ("get_national_baseline", {}, "保険者数"),
    ("lookup_insurer", {"name": "いなべ市"}, "該当"),
    # 同名自治体。1件に決め打ちせず両方返すこと。
    ("lookup_insurer", {"name": "府中市"}, "補足"),
    ("lookup_insurer", {"name": "存在しない町"}, "補足"),
    # 特養定員0。同率90件の注記が付くこと。
    ("rank_insurers", {"pref": "青森県", "order": "low", "limit": 2}, "保険者"),
    ("rank_insurers", {"order": "bogus"}, "エラー"),
    ("search_regulations", {"query": "身体拘束はどんなときならしていい？"}, "条文"),
]

# 検索方式を確かめる追加検証。(引数, 検索方式に含まれるべき語)
LAW_MODE = [({"query": "身体拘束はどんなときならしていい？", "limit": 3}, "埋め込み")]
LAW_MODE_FALLBACK = [({"query": "身体拘束はどんなときならしていい？", "limit": 3}, "BM25")]


async def run(env: dict[str, str] | None, cases, law_mode) -> int:
    """サーバーを1回起動して、道具の応答を確かめる。env は子プロセスへ追加する環境変数。"""
    import os

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kaigo_mcp"],
        cwd=str(ROOT),
        env={**os.environ, **(env or {})},
    )
    failures = 0
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"接続: {init.server_info.name} v{init.server_info.version}")

            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print(f"道具: {names}")
            for t in tools.tools:
                if not t.description:
                    print(f"  NG {t.name}: 説明文が無い（LLMが選べない）")
                    failures += 1

            for name, args, must_have in cases:
                result = await session.call_tool(name, args)
                body = result.structured_content or {}
                ok = must_have in body
                print(f"  {'OK' if ok else 'NG'} {name}({args}) -> {must_have}")
                if not ok:
                    print(json.dumps(body, ensure_ascii=False)[:300])
                    failures += 1

            for args, word in law_mode:
                body = (await session.call_tool("search_regulations", args)).structured_content or {}
                ok = word in body.get("検索方式", "") and len(body.get("条文", [])) == args["limit"]
                print(f"  {'OK' if ok else 'NG'} search_regulations 方式={body.get('検索方式')} (期待: {word})")
                if not ok:
                    print(json.dumps(body, ensure_ascii=False)[:300])
                    failures += 1
    return failures


async def main() -> int:
    print("== 通常（Ollamaあり）")
    failures = await run(None, CASES, LAW_MODE)
    # Ollama に繋がらないとき、道具が落ちず BM25 に退避すること。
    print("\n== Ollama に繋がらない場合")
    failures += await run(
        {"KAIGO_OLLAMA_URL": "http://127.0.0.1:1"}, [("get_national_baseline", {}, "保険者数")], LAW_MODE_FALLBACK
    )
    print(f"\n{'すべて通過' if failures == 0 else f'{failures}件 失敗'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

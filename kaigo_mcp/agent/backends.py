"""モデルの差し替え層。

比較の前提として、**ループは1本しか書かない**（loop.py）。ここがやるのは
中立形の履歴を各社の形に変換して1往復投げ、結果を中立形に戻すことだけ。

Anthropic SDK には MCP 用のツールランナーがあるが使っていない。
片方だけランナー、片方だけ手書きループにすると、列間の差が
「モデルの差」なのか「ループ実装の差」なのか分離できなくなるため。
"""

from __future__ import annotations

import json
import os
from typing import Any

from .types import Reply, ToolCall, ToolSpec, Turn


class BackendError(RuntimeError):
    pass


# ---------------------------------------------------------------- Anthropic


class AnthropicBackend:
    """Anthropic Messages API。

    OpenAI互換側と違い temperature を送らない。Claude 4.6 以降のモデルでは
    サンプリング系のパラメータが廃止されており、送ると 400 で落ちる。
    いま既定にしている Haiku 4.5 は受け付けるが、上位モデルへ差し替えた瞬間に
    壊れる書き方を残したくないため、最初から送らない。
    """

    kind = "anthropic"

    def __init__(self, model: str, max_tokens: int = 4096) -> None:
        import anthropic

        if not (
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        ):
            raise BackendError(
                "ANTHROPIC_API_KEY が未設定。この列は課金が発生するので、"
                "残高を確認してから設定すること。"
            )
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def _tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.schema}
            for t in tools
        ]

    def _messages(self, history: list[Turn]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for turn in history:
            if turn.role == "user":
                out.append({"role": "user", "content": turn.text})
            elif turn.role == "assistant":
                content: list[dict[str, Any]] = []
                if turn.text:
                    content.append({"type": "text", "text": turn.text})
                for c in turn.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": c.id,
                            "name": c.name,
                            "input": c.arguments,
                        }
                    )
                out.append({"role": "assistant", "content": content})
            elif turn.role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": r.id,
                                "content": r.content,
                                **({"is_error": True} if r.is_error else {}),
                            }
                            for r in turn.tool_results
                        ],
                    }
                )
        return out

    def send(self, system: str, history: list[Turn], tools: list[ToolSpec]) -> Reply:
        res = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=self._tools(tools),
            messages=self._messages(history),
        )
        text = "".join(b.text for b in res.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in res.content
            if b.type == "tool_use"
        ]
        return Reply(
            text=text,
            tool_calls=calls,
            input_tokens=res.usage.input_tokens,
            output_tokens=res.usage.output_tokens,
            stop_reason=res.stop_reason or "",
        )


# ------------------------------------------------------------ OpenAI互換
# DeepInfra / OpenRouter / Ollama はすべてこの1クラスで足りる。
# Ollama も /v1/chat/completions を持っているため、ローカルだけ別実装にしなくてよい。


class OpenAICompatBackend:
    kind = "openai_compat"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = 0.0,
    ) -> None:
        from openai import OpenAI

        key = os.environ.get(api_key_env, "") if api_key_env else ""
        if api_key_env and not key:
            raise BackendError(f"{api_key_env} が未設定。")
        self.model = model
        self.max_tokens = max_tokens
        # 温度0が既定。qwen2.5:7b は既定温度だと道具呼び出しが 4/6 しか
        # 構造化された形で出ず、残りは `<tool_call>` の開始タグが壊れて
        # 本文に漏れる（scripts/probe_tool_calling.py で計測）。0にすると 6/6。
        # 多段ループでは連続成功が要るので、67%は3ステップで30%まで落ちる。
        self.temperature = temperature
        # ローカル(Ollama)は認証しないがSDKがキーを要求するのでダミーを入れる。
        self._client = OpenAI(base_url=base_url, api_key=key or "local")

    def _tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.schema,
                },
            }
            for t in tools
        ]

    def _messages(self, system: str, history: list[Turn]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for turn in history:
            if turn.role == "user":
                out.append({"role": "user", "content": turn.text})
            elif turn.role == "assistant":
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": turn.text or None,
                }
                if turn.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": json.dumps(
                                    c.arguments, ensure_ascii=False
                                ),
                            },
                        }
                        for c in turn.tool_calls
                    ]
                out.append(msg)
            elif turn.role == "tool":
                for r in turn.tool_results:
                    out.append(
                        {"role": "tool", "tool_call_id": r.id, "content": r.content}
                    )
        return out

    def send(self, system: str, history: list[Turn], tools: list[ToolSpec]) -> Reply:
        extra = {} if self.temperature is None else {"temperature": self.temperature}
        res = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            tools=self._tools(tools),
            messages=self._messages(system, history),
            **extra,
        )
        choice = res.choices[0]
        calls: list[ToolCall] = []
        for i, c in enumerate(choice.message.tool_calls or []):
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                # 小さいモデルは引数のJSONを壊すことがある。落とさずに記録して先へ進める
                # （どのモデルが何回壊すかが、比較したい数字そのものなので）。
                args = {"__parse_error__": c.function.arguments}
            calls.append(
                ToolCall(id=c.id or f"call_{i}", name=c.function.name, arguments=args)
            )
        usage = res.usage
        return Reply(
            text=choice.message.content or "",
            tool_calls=calls,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            stop_reason=choice.finish_reason or "",
        )


# ------------------------------------------------------------------ 列の定義
# 比較する4列。B と C を同じ Qwen 系で揃えているのは、その差を
# 「モデルサイズだけの効果」として読めるようにするため。

PRESETS: dict[str, dict[str, Any]] = {
    "A-local-cpu": {
        "backend": "openai_compat",
        "model": "qwen2.5:7b",
        "base_url": "http://localhost:11434/v1",
        "note": "メイン機・GPUなし。2024年世代の7B",
    },
    "B-local-gpu": {
        "backend": "openai_compat",
        "model": "qwen3.5:9b",
        "base_url": "http://localhost:11434/v1",
        "note": "ゲーミングノート RTX 5050 8GB。要 ollama pull",
    },
    # 列Bのローカル qwen3.5:9b と**同じ3.5世代**を選んでいる。
    # 当初は Qwen3-235B-A22B を指していたが、世代が3.0で列Bと揃わないうえ、
    # DeepInfra のカタログからも消えていた。9B → 397B なら差がサイズだけになる。
    #
    # 同じモデルを NVIDIA が無料で配っているのでそちらを既定にした
    # （クレカ不要・クレジット制。build.nvidia.com で発行）。
    # DeepInfra は同じモデルを有料で持っているので代替として残す。
    #
    # 注意: このモデルは既定で thinking モードで動く。思考ぶんの出力が増えるので、
    # ローカルの qwen3.5:9b と比べるときは条件が揃っているか確認すること。
    "C-open-api": {
        "backend": "openai_compat",
        "model": "Qwen/Qwen3.5-397B-A17B",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key_env": "DEEPINFRA_API_KEY",
        "note": "列Bと同世代・サイズ違い（9B→397B）。従量課金・18回で約15円",
    },
    # NVIDIA無料枠。**速度が保証されない**ので既定にはしていない。
    # 2026-08-24 に実測: 単発は数秒で返ることもあるが、1往復に168秒かかる回があり、
    # 連続すると全滅する（6回の呼び出しが280秒で終わらなかった）。共有キャパのため。
    # 18回×2〜3往復のevalには使えない。単発の試し打ち用として残す。
    #
    # なお同じ Qwen3.5-397B-A17B は NVIDIA では 2026-07-27 に提供終了している（410）。
    # 無料枠のカタログは入れ替わるので、使う前に必ず /models で実在を確かめること。
    "C-alt-nvidia": {
        "backend": "openai_compat",
        "model": "deepseek-ai/deepseek-v4-flash-0731",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "note": "NVIDIA無料枠。無料だが待たされる。evalには不向き",
    },
    "D-claude": {
        "backend": "anthropic",
        "model": "claude-haiku-4-5",
        "note": "商用API。課金あり",
    },
}


def build(preset: str, max_tokens: int = 4096) -> Any:
    if preset not in PRESETS:
        raise BackendError(f"未知の列: {preset}。使えるのは {list(PRESETS)}")
    cfg = dict(PRESETS[preset])
    cfg.pop("note", None)
    kind = cfg.pop("backend")
    if kind == "anthropic":
        return AnthropicBackend(max_tokens=max_tokens, **cfg)
    return OpenAICompatBackend(max_tokens=max_tokens, **cfg)

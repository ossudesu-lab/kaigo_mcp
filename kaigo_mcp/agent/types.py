"""バックエンド間で共通の型。

会話履歴はここで定義する**中立形**で持ち、各バックエンドが送信時に自分の形へ変換する。
Anthropic の content ブロックや OpenAI の messages をそのまま持ち回らないのは、
どちらか片方に寄せると、もう片方が「変換された履歴」を見ることになり、
比較したときの差がモデルの差なのか変換の差なのか分からなくなるため。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    """MCPサーバーから取得した道具の定義。"""

    name: str
    description: str
    schema: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Turn:
    """中立形の1発言。role は user / assistant / tool のいずれか。"""

    role: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class Reply:
    """バックエンドが1往復で返すもの。"""

    text: str
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int
    stop_reason: str = ""


@dataclass
class RunRecord:
    """1問ぶんの実行記録。列どうしを比べるための材料。"""

    backend: str
    model: str
    question: str
    answer: str = ""
    steps: int = 0
    tool_calls: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    stopped_by: str = ""  # end / truncated / max_steps / error
    stop_reason: str = ""  # 最後の往復でバックエンドが返した理由。truncated の内訳用
    error: str = ""

    @property
    def answered(self) -> bool:
        """答えが出たか。列どうしを比べるとき、数えて良いのはこれが真の実行だけ。"""
        return self.stopped_by == "end"

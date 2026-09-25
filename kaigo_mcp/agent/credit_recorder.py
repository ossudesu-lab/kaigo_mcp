"""credit_watch の記録係（Python版）。

元: credit_watch/recorder/recorder.py（版: 2026-09-25）。直すときは元を直してからコピーし直すこと。
JS版（recorder.js）と同じコマンドを作る。両者が揃っていることは cases.json で確かめている。

Claude を呼んだ直後に、API が返したトークン数を Upstash Redis に足し込む。
記録するのは数字とラベルだけ。プロンプトも応答も、APIキーも送らない。

約束:
1. 本番を絶対に落とさない。record_usage は決して例外を投げず、1.5秒で打ち切る
2. 設定（環境変数）が無ければ何もしない
3. 用途（prod / eval）は環境変数 CW_PURPOSE で決める。呼ばれ方から推測しない
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

TIMEOUT_SEC = 1.5
JST = timezone(timedelta(hours=9))

# API の usage のフィールド名 → 記録の種類
KINDS = (
    ("input_tokens", "in"),
    ("output_tokens", "out"),
    ("cache_creation_input_tokens", "cache_w"),
    ("cache_read_input_tokens", "cache_r"),
)


def day_jst(now: datetime) -> str:
    """日本時間の日付。UTC で切ると朝9時に日付が変わってしまう。"""
    return now.astimezone(JST).strftime("%Y-%m-%d")


def _label(s: str) -> str:
    # 区切り文字 | がラベルに入ると列がずれるので置き換える
    return str(s).replace("|", "_")


def _get(usage: Any, name: str) -> int:
    # SDK のオブジェクト（属性）でも dict でも受け取れるようにする
    v = usage.get(name) if isinstance(usage, Mapping) else getattr(usage, name, None)
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def build_commands(project: str, purpose: str, model: str, usage: Any, now: datetime) -> list[list]:
    key = f"cw:day:{day_jst(now)}"
    prefix = "|".join(_label(x) for x in (project, purpose, model))
    cmds: list[list] = []
    for src, kind in KINDS:
        n = _get(usage, src)
        if n > 0:
            cmds.append(["HINCRBY", key, f"{prefix}|{kind}", n])
    cmds.append(["HINCRBY", key, f"{prefix}|calls", 1])
    return cmds


def record_usage(
    model: str,
    usage: Any,
    *,
    env: Mapping[str, str] | None = None,
    urlopen: Callable = urllib.request.urlopen,
    now: datetime | None = None,
) -> None:
    """記録する。失敗しても例外は投げない。"""
    try:
        env = os.environ if env is None else env
        url, token = env.get("KV_REST_API_URL"), env.get("KV_REST_API_TOKEN")
        project, purpose = env.get("CW_PROJECT"), env.get("CW_PURPOSE")
        if not (url and token and project and purpose and usage is not None):
            return

        body = json.dumps(build_commands(project, purpose, model, usage, now or datetime.now(JST))).encode()
        req = urllib.request.Request(
            url.rstrip("/") + "/pipeline",
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        with urlopen(req, timeout=TIMEOUT_SEC):
            pass
    except Exception as e:
        # 記録の失敗で本番を止めない。抜けた分は週報の照合で気づく
        _warn(type(e).__name__)


def _warn(reason: str) -> None:
    # 失敗を1行だけ残す。鍵やURLは出さない（公開リポジトリの Actions ログは誰でも読める）
    try:
        import sys

        print(f"[credit_watch] 記録に失敗: {reason}", file=sys.stderr)
    except Exception:
        pass

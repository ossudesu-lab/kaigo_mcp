"""法令の検索。data/laws.json（scripts/fetch_laws.py が作る）を項単位で引く。

## 基準線をBM25にした理由

埋め込み検索を足す前に、キーワード検索でどこまで届くかを測る。基準線が無いと、
埋め込みを足して良くなったのか、そもそも簡単な問題だったのかが分からない。

形態素解析器は入れず、文字の2-gramで切る。日本語は空白で単語が切れないので、
辞書なしで動く最も単純な方法を基準線にする。辞書依存の差は、後で比較に載せる。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parent.parent / "data" / "laws.json"

OLLAMA = os.environ.get("KAIGO_OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "bge-m3"

# BM25の標準的な値。この規模ではほとんど動かさなくても効く。
_K1 = 1.5
_B = 0.75

# 1塊の上限。項が号で長く伸びる条（定義など）はここで切る。
_MAX_CHARS = 800


@dataclass(frozen=True)
class Chunk:
    law: str
    article_num: str  # e-Gov形式（"8"、"115_45"）。評価の正解もこの形で書く
    article_title: str  # 第八条
    caption: str  # 見出し
    section: str
    paragraph: str  # 項番号。分割したときは "2" や "2-b"
    text: str

    @property
    def ref(self) -> str:
        """出典の表記。例: 介護保険法 第八条（定義） 第2項"""
        cap = f"（{self.caption}）" if self.caption else ""
        para = f" 第{self.paragraph}項" if self.paragraph else ""
        return f"{self.law} {self.article_title}{cap}{para}"


def _grams(text: str) -> list[str]:
    """空白と記号を除き、文字2-gramにする。1文字しか無ければその1文字。"""
    s = re.sub(r"[\s、。，．・（）()「」『』\[\]【】:：;；]", "", text)
    if len(s) < 2:
        return [s] if s else []
    return [s[i : i + 2] for i in range(len(s) - 1)]


def _split(text: str) -> list[str]:
    """上限を超える項を、号（改行）の切れ目で分ける。"""
    if len(text) <= _MAX_CHARS:
        return [text]
    parts: list[str] = []
    buf = ""
    for line in text.split("\n"):
        if buf and len(buf) + len(line) + 1 > _MAX_CHARS:
            parts.append(buf)
            buf = ""
        # 1行が上限を超えるときは、その行をそのまま1塊にする（途中で切らない）。
        buf = f"{buf}\n{line}" if buf else line
    if buf:
        parts.append(buf)
    return parts


def build_chunks(articles: list[dict[str, Any]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for a in articles:
        for p in a["paragraphs"]:
            pieces = _split(p["text"])
            for i, piece in enumerate(pieces):
                para = p["num"] if len(pieces) == 1 else f"{p['num']}-{chr(ord('a') + i)}"
                # 1条1項しかない条は、項番号を出典に出さない。
                if len(a["paragraphs"]) == 1 and len(pieces) == 1:
                    para = ""
                chunks.append(
                    Chunk(
                        law=a["law"],
                        article_num=a["article_num"],
                        article_title=a["article_title"],
                        caption=a["caption"],
                        section=a["section"],
                        paragraph=para,
                        text=piece,
                    )
                )
    return chunks


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        # 見出しと所属の章も索引に含める。「人員に関する基準」のような語で当たるように。
        self._tf = [
            Counter(_grams(f"{c.article_title}{c.caption}{c.section}{c.text}")) for c in chunks
        ]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / len(self._len)
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = _grams(query)
        scored: list[tuple[int, float]] = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            norm = _K1 * (1 - _B + _B * self._len[i] / self._avg)
            for t in q:
                f = tf.get(t)
                if f:
                    score += self._idf[t] * f * (_K1 + 1) / (f + norm)
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda x: -x[1])
        return [(self.chunks[i], s) for i, s in scored[:k]]


def _doc_text(c: Chunk) -> str:
    return f"{c.article_title}{c.caption} {c.section} {c.text}"


class DenseIndex:
    """埋め込み（Ollamaのbge-m3）によるコサイン類似度検索。

    埋め込みは data/ にキャッシュする。法令の本文が変わったとき（fetch_laws.py の再実行）は
    塊のテキストのハッシュが変わるので、自動で作り直す。
    """

    def __init__(self, chunks: list[Chunk], model: str = EMBED_MODEL, build: bool = True) -> None:
        """build=False のときはキャッシュだけを読む。無ければ FileNotFoundError。

        CPUだけだと全塊の埋め込みに約45分かかる。道具の呼び出し中に黙って始めると
        エージェントが固まるので、サーバー側は build=False にして BM25 へ退避させる。
        """
        import numpy as np

        self.chunks = chunks
        self.model = model
        texts = [_doc_text(c) for c in chunks]
        digest = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]
        cache = DATA.parent / f"embeddings-{model.replace(':', '_')}.npz"
        if cache.exists():
            z = np.load(cache)
            if str(z["digest"]) == digest:
                self._m = z["vectors"]
                return
        if not build:
            raise FileNotFoundError(
                f"{cache.name} が無い（または法令が更新された）。"
                "python scripts/eval_retrieval.py などで一度作ること。"
            )
        vecs = np.array(embed(texts, model), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        np.savez(cache, vectors=vecs, digest=digest)
        self._m = vecs

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        import numpy as np

        q = np.array(embed([query], self.model)[0], dtype=np.float32)
        q /= np.linalg.norm(q)
        sims = self._m @ q
        top = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in top]


def embed(texts: list[str], model: str = EMBED_MODEL, batch: int = 16) -> list[list[float]]:
    """OllamaのAPIで埋め込む。依存を増やさないため標準ライブラリで呼ぶ。"""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        req = urllib.request.Request(
            f"{OLLAMA}/api/embed",
            data=json.dumps({"model": model, "input": texts[i : i + batch]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            out.extend(json.load(r)["embeddings"])
    return out


class HybridIndex:
    """BM25と埋め込みの順位を Reciprocal Rank Fusion で合わせる。

    スコアの尺度が違う（BM25は無制限、コサインは-1〜1）ので、足し算せず順位だけを使う。
    """

    def __init__(self, bm25: BM25Index, dense: DenseIndex, rrf_k: int = 60, depth: int = 50) -> None:
        self.chunks = bm25.chunks
        self._bm25, self._dense, self._rrf_k, self._depth = bm25, dense, rrf_k, depth

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        score: dict[int, float] = {}
        by_id = {id(c): c for c in self.chunks}
        for idx in (self._bm25, self._dense):
            for rank, (c, _) in enumerate(idx.search(query, self._depth), start=1):
                score[id(c)] = score.get(id(c), 0.0) + 1 / (self._rrf_k + rank)
        top = sorted(score.items(), key=lambda x: -x[1])[:k]
        return [(by_id[i], s) for i, s in top]


class Searcher:
    """道具から使う窓口。埋め込みが使えなければ BM25 に退避し、どちらで引いたかを返す。"""

    def __init__(self) -> None:
        self.bm25 = load_index()
        self.dense: DenseIndex | None = None
        self.dense_error: str | None = None
        try:
            self.dense = DenseIndex(self.bm25.chunks, build=False)
        except (FileNotFoundError, ImportError) as e:
            self.dense_error = str(e)

    def search(self, query: str, k: int = 5) -> tuple[list[tuple[Chunk, float]], str]:
        if self.dense is not None:
            try:
                return self.dense.search(query, k), "埋め込み検索（bge-m3）"
            except OSError:  # Ollamaに繋がらない（URLError・接続拒否・タイムアウト）
                pass
        return self.bm25.search(query, k), "キーワード検索（BM25）"


_searcher: Searcher | None = None


def get_searcher() -> Searcher:
    global _searcher
    if _searcher is None:
        _searcher = Searcher()
    return _searcher


def corpus_info() -> list[dict[str, str]]:
    """取得した法令の名称と施行日。道具の応答に載せ、いつの条文かを分かるようにする。"""
    return json.loads(DATA.read_text(encoding="utf-8"))["laws"]


_index: BM25Index | None = None


def load_index() -> BM25Index:
    global _index
    if _index is None:
        if not DATA.exists():
            raise FileNotFoundError(
                f"{DATA} が無い。python scripts/fetch_laws.py で取得すること。"
            )
        articles = json.loads(DATA.read_text(encoding="utf-8"))["articles"]
        _index = BM25Index(build_chunks(articles))
    return _index

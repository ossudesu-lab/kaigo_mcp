# 列Bの計測 運用手順（GPU機で作業する）

手を動かすための覚書。**なぜそうなっているか**は `README.md` を見る。

このマシン（RTX 5050 Laptop 8GB）でやることは**1つだけ**。
「GPUと世代更新で、道具呼び出しがどれだけ安定するか」を測って数字を持ち帰る。
費用は**0円**（ローカル実行のみ）。APIキーは要らない。

---

## 0. 最初の一回だけ

```
git clone https://github.com/ossudesu-lab/kaigo_mcp.git
cd kaigo_mcp
pip install -r requirements.txt
ollama pull qwen3.5:9b
```

動作確認した版（メイン機側）: Python 3.14.4 / mcp 2.0.0 / openai 3.3.1

### Windows で文字化けするとき

PowerShell の既定が cp932 なので、日本語でエラーになることがある。

```
$env:PYTHONIOENCODING = "utf-8"
```

### まず疎通だけ確認する

```
python scripts/smoke_test.py
```

「すべて通過」と出れば MCP サーバーは動いている。ここが通らないうちは先へ進まない。

---

## 1. 道具呼び出しの安定性を測る（本題）

```
python scripts/probe_tool_calling.py --model qwen3.5:9b --runs 6
python scripts/probe_tool_calling.py --model qwen3.5:9b --runs 6 --temperature 0
```

**温度を振った2本を必ず両方走らせる。** メイン機の `qwen2.5:7b` では、
既定温度で 4/6（67%）、温度0で 6/6（100%）だった。9B で同じ癖が出るかが見どころ。

失敗するときは `<tool_call>` の開始タグが壊れて（`olith` `pering` のような
数文字が先頭に付く）呼び出しが本文テキストに漏れる。閉じタグ `</tool_call>` だけが
残るのが目印。スクリプトが該当行を出すので、そのまま控えておく。

### 持ち帰る数字

- 成功率（○/6）を温度ごとに
- 1回あたりの秒数（メイン機のCPUは温度0で 7.0秒）
- 失敗したときに漏れたテキスト

---

## 2. エージェントを通しで走らせる

```
python -m kaigo_mcp.agent --column B-local-gpu --verbose "尼崎市は特養が足りてる？"
```

メイン機（列A・CPU）の結果は **2ステップ / 72.31秒 / 入力1,648・出力171トークン**。
答えは「18.1人で全国平均24.6人を下回る」で、これは正しい（検算済み）。

同じ質問で、ステップ数・秒数・答えの正しさを比べる。

### 見ておきたい落とし穴

**8GB VRAM では、先に頭打ちになるのがモデルの賢さではなく
コンテキスト長かもしれない。** 9B/Q4 が約6.6GB を占めるので、
KVキャッシュ（会話履歴）に使えるのは残り1.4GB ほど。
エージェントは道具の結果を積みながら回るので履歴が伸びる。

`--max-steps` を上げて長く回したときに、
品質が落ちるより先に落ちるようなら、それが記事になる部分。

```
python -m kaigo_mcp.agent --column B-local-gpu --max-steps 8 --verbose "青森県で特養が足りない市町村を3つ挙げて、全国と比べてどうか説明して"
```

---

## 3. 結果の持ち帰り方

`.agent-usage.jsonl` に実行記録が溜まるが、**これは .gitignore されている**
（手元の計測結果なので共有しない、という案1と同じ扱い）。

数字を持ち帰るときは、コンソール出力をコピーするか、明示的に送る:

```
git add -f .agent-usage.jsonl
```

とはいえ普通は、上の「持ち帰る数字」をメモして戻るだけで足りる。

---

## やらないこと

- **列C・列D は走らせない。** どちらも課金が発生する（`DEEPINFRA_API_KEY` /
  `ANTHROPIC_API_KEY`）。Anthropic の残高は公開中の本番アプリと共有なので、
  使い切ると案1のデプロイ済みアプリが止まる。列が揃ってから、まとめて1回だけ回す。
- `data/` の中身を作り直さない。案4（kaigo_gap_analysis）が書き出したものを
  そのまま使う前提。更新は `python scripts/sync_data.py` で、案4のあるマシンから。

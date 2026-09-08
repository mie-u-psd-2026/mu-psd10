# 生成AI活用サンプルアプリ

## Linux / VS Codeでのセットアップ

このリポジトリは、Python 3.12以降とOllamaを使ってローカル開発できます。

```bash
cd mu-psd10
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

`.env` に `GEMINI_API_KEY` を設定しない場合は、Ollamaがフォールバックとして使用されます。Ollamaを使う場合は別ターミナルで起動し、モデルを取得してください。

```bash
ollama serve
ollama pull qwen2.5:1.5b
```

Flaskアプリは次のコマンドで起動します。

```bash
source .venv/bin/activate
python app.py
```

ブラウザで http://localhost:5000/ を開いてください。VS Codeでは、推奨拡張機能をインストールすると、`.venv` が自動選択され、F5でデバッグ起動できます。

# 概要

このアプリは Python とVue.jsを用いて作られた簡易的な生成AI活用アプリです。

- フロントエンドに、Vue.js CDN版を用いています。

- バックエンドに、Python,FlaskとOpenAI APIを用いてローカル起動のOllamaを叩いています。

# 環境
- Vscode
- OpenCode
- ollama

# 開発ツールインストール

- 管理者権限でコマンドプロンプトを起動します。

- 以下のコマンドを実行し、必要なソフトウェアを入手します。

```
winget install --id Microsoft.VisualStudioCode -e --source winget --accept-package-agreements --accept-source-agreements
winget install --id Python.Python.3.13 -e --source winget --accept-package-agreements --accept-source-agreements
winget install --id SST.opencode -e --source winget --accept-package-agreements --accept-source-agreements
winget install --id Ollama.Ollama -e --source winget --accept-package-agreements --accept-source-agreements
start /b ollama serve > NUL 2>&1
timeout /t 3 /nobreak > NUL
ollama pull qwen3.5:0.8b
```

- vscodeを起動し、アクティビティバーの拡張機能から、以下のプラグインをインストールしてください。
  - Python
  - Vue.js Extension Pack

# 環境セットアップ

- Python ライブラリインストール

  以下のコマンドでPythonの利用ライブラリをインストールします。

  ```
  pip install -r requirements.txt
  ```

# 実行方法

- 以下のコマンドでサーバを起動します。

  ```
  python app.py
  ```

- ブラウザで以下のURLにアクセスしてみてください。

  ```
  http://localhost:5000
  ```

# LLM APIの利用について

このアプリでは、通常時はGoogle Gemini APIを使用し、Gemini APIが利用できない場合はローカルのOllamaへ切り替えて処理を続行します。

## Gemini API

Gemini APIを利用する場合は、プロジェクトフォルダに`.env`ファイルを作成し、以下のようにAPIキーを設定してください。

```env
GEMINI_API_KEY=ここにGeminiのAPIキー
```
# 開発の参考資料

## ローカルの Ollama を使う場合（低性能だが利用制限なし）：

- VsCode上でターミナルを開いて、以下を入力します。
```
ollama launch opencode --model=qwen3.5:0.8b
```

## クラウドの無料モデルを使う場合：(中性能、無料枠少ない)

- VsCode上でターミナルを開いて、 opencode と入力します。

- /models と入力し、Free 表示のあるモデルを選択します。（例: DeepSeek V4 Flash Free）

## Google AI Studioを使う場合:(高性能、無料枠多い)

- [Google AI Studio](https://aistudio.google.com/api-keys)を開きます。

- APIキーを作成、を押下し、キー名を適当に命名し、プロジェクトを新規作成します。

- APIキーが表示されるので、クリップボードにコピーしておきます。

- [プロジェクト一覧](https://aistudio.google.com/projects)を開き、新規作成したプロジェクトが無料枠となっていることを確認します。

- VsCode上でターミナルを開いて、 opencode と入力します。

- /connect と入力、プロバイダ一覧が表示されるので、Googleを選択、APIキーに先ほどのAPIキーを貼り付けます。

# AIを用いたコード修正

- opencodeに修正を依頼してみてください。（例：猫語で回答するボタンを追加して ）

- フロントエンド担当者は、html/JavaScriptを追加／修正して画面を構築してください。

- バックエンド担当者は、app.py上にURLとAPIを作成してください。

# 参考リンク

- [Flask](https://flask.palletsprojects.com/en/stable/)

  - Python で書かれた Webアプリケーションサーバ

- [Vue.js](https://vuejs.org/)

  - JavaScript製製のWebフロントエンド フレームワーク

- [Vue.js Tutorial](https://ja.vuejs.org/tutorial/)

  - Vue.jsの入門用チュートリアル
  
- [OpenAI API](https://github.com/openai/openai-python)

  - Pythonから、OpenAI APIを呼び出すライブラリ


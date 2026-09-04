import json
import random
import re
import uuid

from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI

app = Flask(__name__)

# Server-side game state: {game_id: {card_id: answer, ...}}
games = {}

if app.debug:
    @app.after_request
    def add_header(response):
        if request.endpoint == 'static':
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        return response


client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
OLLAMA_MODEL = "qwen2.5-coder:0.5b"


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/send_api', methods=['POST'])
def send_api():
    data = request.get_json()

    if not data or 'text' not in data:
        app.logger.error("Request JSON is missing or does not contain 'text' field.")
        return jsonify({"error": "Missing 'text' in request body"}), 400

    received_text = data['text']
    if not received_text.strip():
        app.logger.error("Received text is empty or whitespace.")
        return jsonify({"error": "Input text cannot be empty"}), 400

    system_prompt = "140字以内で回答してください。"
    if 'context' in data and data['context'] and data['context'].strip():
        system_prompt = data['context'].strip()
        app.logger.info(f"Using custom system prompt from context: {system_prompt}")
    else:
        app.logger.info(f"Using default system prompt: {system_prompt}")

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": received_text}
            ],
            model=OLLAMA_MODEL,
        )

        if chat_completion.choices and chat_completion.choices[0].message:
            processed_text = chat_completion.choices[0].message.content
        else:
            processed_text = "AIから有効な応答がありませんでした。"

        return jsonify({"message": "AIによってデータが処理されました。", "processed_text": processed_text})

    except Exception as e:
        app.logger.error(f"Ollama API call failed: {e}")
        return jsonify({"error": f"AIサービスとの通信中にエラーが発生しました。"}), 500


@app.route('/generate_game', methods=['POST'])
def generate_game():
    data = request.get_json(silent=True) or {}
    theme = data.get('theme', '').strip()
    if not theme:
        return jsonify({"error": "お題を入力してください。"}), 400

    system_prompt = (
        "あなたはJSONのみを出力します。説明文は一切出力しないでください。"
    )
    user_prompt = (
        f"お題「{theme}」から関連する8つの単語を選び、"
        "それぞれについて異なる2つのクイズを生成してください。\n"
        "【条件】\n"
        "- 8つの単語はお題に密接に関連するもの\n"
        "- 各単語について答えがその単語になるクイズを2つ作る\n"
        "- 2つのクイズは質問文が異なるもの\n"
        "- クイズは簡潔に（1文程度）\n"
        "- 以下のJSON形式で出力：\n"
        '{"pairs":[{"pair_id":1,"answer":"単語","questions":["クイズA","クイズB"]}]}\n'
        "- 説明やコードブロック不要、JSONのみ出力"
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=OLLAMA_MODEL,
        )
        raw = chat_completion.choices[0].message.content
    except Exception as e:
        app.logger.error(f"Ollama API call failed: {e}")
        return jsonify({"error": "AIサービスとの通信中にエラーが発生しました。"}), 500

    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        app.logger.error(f"Failed to extract JSON from LLM response: {raw}")
        return jsonify({"error": "ゲームデータの生成に失敗しました。"}), 500

    try:
        game_data = json.loads(json_match.group())
        pairs = game_data.get("pairs", [])
    except (json.JSONDecodeError, KeyError):
        app.logger.error(f"Invalid JSON from LLM: {raw}")
        return jsonify({"error": "ゲームデータの生成に失敗しました。"}), 500

    if len(pairs) != 8:
        app.logger.error(f"Expected 8 pairs, got {len(pairs)}")
        return jsonify({"error": "ゲームデータの生成に失敗しました。"}), 500

    cards = []
    card_id = 1
    for pair in pairs:
        questions = pair.get("questions", [])
        if len(questions) != 2:
            app.logger.error(f"Pair {pair.get('pair_id')} does not have 2 questions")
            return jsonify({"error": "ゲームデータの生成に失敗しました。"}), 500
        answer = pair.get("answer", "")
        for q in questions:
            cards.append({"id": card_id, "answer": answer, "text": q})
            card_id += 1

    if len(cards) != 16:
        app.logger.error(f"Expected 16 cards, got {len(cards)}")
        return jsonify({"error": "ゲームデータの生成に失敗しました。"}), 500

    random.shuffle(cards)

    game_id = uuid.uuid4().hex
    games[game_id] = {c["id"]: c["answer"] for c in cards}

    public_cards = [{"id": c["id"], "text": c["text"]} for c in cards]
    return jsonify({"game_id": game_id, "cards": public_cards})


@app.route('/check_pair', methods=['POST'])
def check_pair():
    data = request.get_json(silent=True) or {}
    game_id = data.get("game_id")
    card_ids = data.get("card_ids", [])

    if not game_id or game_id not in games:
        return jsonify({"error": "ゲームが見つかりません。"}), 400

    if not isinstance(card_ids, list) or len(card_ids) != 2:
        return jsonify({"error": "card_idsには2つのカードIDを指定してください。"}), 400

    id1, id2 = card_ids
    state = games[game_id]

    if id1 not in state or id2 not in state:
        return jsonify({"error": "無効なカードIDです。"}), 400

    if id1 == id2:
        return jsonify({"error": "同じカードは選択できません。"}), 400

    is_pair = state[id1] == state[id2]
    return jsonify({"is_pair": is_pair, "answer": state[id1] if is_pair else None})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

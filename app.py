import json
import random
import re
import uuid

from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI

app = Flask(__name__)

# Server-side game state
# {game_id: {"cards": {card_id: answer}, "matched": set(card_id, ...)}}
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


MAX_WORD_RETRIES = 3


def _call_llm(system_prompt, user_prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=OLLAMA_MODEL,
    )
    return chat_completion.choices[0].message.content


def _validate_words(words, theme):
    if not isinstance(words, list):
        return False
    if len(words) != 8:
        return False
    if any(not isinstance(w, str) or not w.strip() for w in words):
        return False
    if len(set(words)) != 8:
        return False
    return True


def _generate_words(theme):
    system_prompt = (
        "あなたはJSONのみを出力します。他のテキストは一切出力しないでください。"
        "コードブロック（```）も使用しないでください。"
    )
    for attempt in range(MAX_WORD_RETRIES):
        user_prompt = (
            f"お題「{theme}」に関連する8つの単語を抽出してください。\n"
            "【規則】\n"
            "1. お題と関連する具体的な単語を8つ\n"
            "2. 単語は簡潔に（1語〜数語）\n"
            "3. 重複しないこと\n"
            "【出力形式】\n"
            '{"words":["単語1","単語2","単語3","単語4","単語5","単語6","単語7","単語8"]}\n'
            "上記JSONのみ出力してください。"
        )
        try:
            raw = _call_llm(system_prompt, user_prompt)
        except Exception as e:
            app.logger.error(f"Ollama API call failed (attempt {attempt + 1}): {e}")
            continue

        json_match = re.search(r"\{[\s\S]*\}", raw)
        if not json_match:
            app.logger.warning(f"Failed to extract JSON (attempt {attempt + 1}): {raw}")
            continue

        try:
            data = json.loads(json_match.group())
            words = data.get("words", [])
        except (json.JSONDecodeError, KeyError):
            app.logger.warning(f"Invalid JSON (attempt {attempt + 1}): {raw}")
            continue

        if _validate_words(words, theme):
            return words

        app.logger.warning(f"Word validation failed (attempt {attempt + 1}): {words}")

    return None


def _generate_quizzes(words):
    system_prompt = (
        "あなたはJSONのみを出力します。他のテキストは一切出力しないでください。"
        "コードブロック（```）も使用しないでください。"
    )
    user_prompt = (
        f"以下の8つの単語について、各単語に対し答えがその単語になるクイズを2つずつ作ってください。\n"
        f"対象単語：{json.dumps(words, ensure_ascii=False)}\n"
        "【規則】\n"
        "1. 各単語に対しクイズを2つ作る\n"
        "2. 2つのクイズは質問文を変える\n"
        "3. クイズは短く簡潔に\n"
        "【出力形式】\n"
        '{"pairs":[{"pair_id":1,"answer":"単語","questions":["質問A","質問B"]}]}\n'
        "上記JSONのみ出力してください。"
    )
    raw = _call_llm(system_prompt, user_prompt)

    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        app.logger.error(f"Failed to extract JSON from quiz response: {raw}")
        return None

    try:
        data = json.loads(json_match.group())
        pairs = data.get("pairs", [])
    except (json.JSONDecodeError, KeyError):
        app.logger.error(f"Invalid JSON from quiz response: {raw}")
        return None

    return pairs


@app.route('/generate_game', methods=['POST'])
def generate_game():
    data = request.get_json(silent=True) or {}
    theme = data.get('theme', '').strip()
    if not theme:
        return jsonify({"error": "お題を入力してください。"}), 400

    words = _generate_words(theme)
    if words is None:
        return jsonify({"error": "単語の生成に失敗しました。もう一度お試しください。"}), 500

    try:
        pairs = _generate_quizzes(words)
    except Exception as e:
        app.logger.error(f"Ollama API call failed: {e}")
        return jsonify({"error": "AIサービスとの通信中にエラーが発生しました。"}), 500

    if pairs is None or len(pairs) != 8:
        app.logger.error(f"Expected 8 quiz pairs, got {len(pairs) if pairs else 0}")
        return jsonify({"error": "ゲームデータの生成に失敗しました。"}), 500

    cards = []
    card_id = 1
    for pair in pairs:
        questions = pair.get("questions", [])
        if len(questions) != 2:
            app.logger.error(f"Pair {pair.get('pair_id')} does not have 2 questions")
            return jsonify({"error": "ゲームデータの生成に失敗しました。"}), 500
        answer = pair.get("answer", "")
        if not answer.strip():
            app.logger.error(f"Pair {pair.get('pair_id')} has empty answer")
            return jsonify({"error": "ゲームデータの生成に失敗しました。"}), 500
        for q in questions:
            if not q.strip():
                app.logger.error(f"Pair {pair.get('pair_id')} has empty question")
                return jsonify({"error": "ゲームデータの生成に失敗しました。"}), 500
            cards.append({"id": card_id, "answer": answer, "text": q})
            card_id += 1

    if len(cards) != 16:
        app.logger.error(f"Expected 16 cards, got {len(cards)}")
        return jsonify({"error": "ゲームデータの生成に失敗しました。"}), 500

    random.shuffle(cards)

    game_id = uuid.uuid4().hex
    games[game_id] = {
        "cards": {c["id"]: c["answer"] for c in cards},
        "matched": set(),
    }

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

    if id1 not in state["cards"] or id2 not in state["cards"]:
        return jsonify({"error": "無効なカードIDです。"}), 400

    if id1 == id2:
        return jsonify({"error": "同じカードは選択できません。"}), 400

    if id1 in state["matched"] or id2 in state["matched"]:
        return jsonify({"error": "このカードはすでにペアが成立しています。"}), 400

    is_pair = state["cards"][id1] == state["cards"][id2]
    game_cleared = False

    if is_pair:
        state["matched"].add(id1)
        state["matched"].add(id2)
        game_cleared = len(state["matched"]) == 16

    return jsonify({
        "is_pair": is_pair,
        "answer": state["cards"][id1] if is_pair else None,
        "game_cleared": game_cleared,
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

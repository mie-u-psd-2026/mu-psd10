import json
import os
import random
import re
import uuid
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)

# Server-side game state
# {game_id: {"theme": str, "cards": {card_id: answer}, "matched": set, "started_at": str, "moves": int}}
games = {}

# Game history (completed games)
game_history = []

if app.debug:
    @app.after_request
    def add_header(response):
        if request.endpoint == 'static':
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        return response


# --- LLM Configuration ---

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.6-flash"

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "qwen3.5:0.8b"

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

WORDS_SCHEMA = {
    "type": "object",
    "properties": {
        "words": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 8,
            "maxItems": 8,
        }
    },
    "required": ["words"],
}

QUIZ_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pair_id": {"type": "integer"},
                    "answer": {"type": "string"},
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "required": ["pair_id", "answer", "questions"],
            },
            "minItems": 8,
            "maxItems": 8,
        }
    },
    "required": ["pairs"],
}


# --- LLM Call Layer ---

def _call_gemini(prompt, response_schema=None):
    if not gemini_client:
        raise RuntimeError("Gemini APIキーが設定されていません。")
    config = types.GenerateContentConfig(temperature=1.0)
    if response_schema:
        config.response_mime_type = "application/json"
        config.response_json_schema = response_schema
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=config,
    )
    if response_schema:
        return response.parsed
    return response.text


def _call_ollama(prompt):
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    http_client = httpx.Client(timeout=60.0)
    try:
        resp = http_client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
    finally:
        http_client.close()


def _parse_json_from_text(raw):
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        return None
    try:
        return json.loads(json_match.group())
    except (json.JSONDecodeError, KeyError):
        return None


# --- Score ---

def _calculate_score(moves):
    return max(0, 1000 - (moves - 8) * 10)


# --- Validation ---

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


def _validate_quizzes(pairs, words):
    if not isinstance(pairs, list) or len(pairs) != 8:
        return False
    seen_answers = set()
    seen_pair_ids = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            return False
        pair_id = pair.get("pair_id")
        if pair_id in seen_pair_ids:
            return False
        seen_pair_ids.add(pair_id)
        answer = pair.get("answer", "")
        if not isinstance(answer, str) or not answer.strip():
            return False
        questions = pair.get("questions", [])
        if not isinstance(questions, list) or len(questions) != 2:
            return False
        q0, q1 = questions
        if not isinstance(q0, str) or not q0.strip():
            return False
        if not isinstance(q1, str) or not q1.strip():
            return False
        if q0 == q1:
            return False
        if answer in seen_answers:
            return False
        seen_answers.add(answer)
    if set(seen_answers) != set(words):
        return False
    return True


# --- Word Generation ---

def _generate_words(theme):
    prompt = (
        f"お題「{theme}」に関連する8つの単語を抽出してください。\n"
        "【規則】\n"
        "1. お題と関連する具体的な単語を8つ\n"
        "2. 単語は簡潔に（1語〜数語）\n"
        "3. 重複しないこと\n"
    )

    # Try Gemini first
    if gemini_client:
        for attempt in range(MAX_WORD_RETRIES):
            try:
                result = _call_gemini(prompt, response_schema=WORDS_SCHEMA)
                words = result.get("words", []) if isinstance(result, dict) else []
            except Exception as e:
                app.logger.error(f"Gemini API call failed (attempt {attempt + 1}): {e}")
                continue
            if _validate_words(words, theme):
                app.logger.info("Using Gemini API")
                return words
            app.logger.warning(f"Gemini word validation failed (attempt {attempt + 1}): {words}")
        app.logger.warning("Gemini unavailable, falling back to Ollama")

    # Fallback to Ollama
    for attempt in range(MAX_WORD_RETRIES):
        try:
            raw = _call_ollama(prompt)
            data = _parse_json_from_text(raw)
            words = data.get("words", []) if isinstance(data, dict) else []
        except Exception as e:
            app.logger.error(f"Ollama API call failed (attempt {attempt + 1}): {e}")
            continue
        if _validate_words(words, theme):
            app.logger.info(f"Using Ollama model {OLLAMA_MODEL}")
            return words
        app.logger.warning(f"Ollama word validation failed (attempt {attempt + 1}): {words}")

    return None


# --- Quiz Generation ---

def _generate_quizzes(words):
    prompt = (
        f"以下の8つの単語について、各単語に対し答えがその単語になるクイズを2つずつ作ってください。\n"
        f"対象単語：{json.dumps(words, ensure_ascii=False)}\n"
        "【規則】\n"
        "1. 各単語に対しクイズを2つ作る\n"
        "2. 2つのクイズは質問文を変える\n"
        "3. クイズは短く簡潔に\n"
        "4. 2つの問題文は同一にしないこと\n"
    )

    # Try Gemini first
    if gemini_client:
        for attempt in range(MAX_QUIZ_RETRIES):
            try:
                result = _call_gemini(prompt, response_schema=QUIZ_SCHEMA)
                pairs = result.get("pairs", []) if isinstance(result, dict) else []
            except Exception as e:
                app.logger.error(f"Gemini API call failed (attempt {attempt + 1}): {e}")
                continue
            if _validate_quizzes(pairs, words):
                app.logger.info("Using Gemini API")
                return pairs
            app.logger.warning(f"Gemini quiz validation failed (attempt {attempt + 1}): {pairs}")
        app.logger.warning("Gemini unavailable, falling back to Ollama")

    # Fallback to Ollama
    for attempt in range(MAX_QUIZ_RETRIES):
        try:
            raw = _call_ollama(prompt)
            data = _parse_json_from_text(raw)
            pairs = data.get("pairs", []) if isinstance(data, dict) else []
        except Exception as e:
            app.logger.error(f"Ollama API call failed (attempt {attempt + 1}): {e}")
            continue
        if _validate_quizzes(pairs, words):
            app.logger.info(f"Using Ollama model {OLLAMA_MODEL}")
            return pairs
        app.logger.warning(f"Ollama quiz validation failed (attempt {attempt + 1}): {pairs}")

    return None


# --- Game Cleanup ---

GAME_TTL_SECONDS = 3600


def _cleanup_old_games():
    now = datetime.now(timezone.utc)
    expired_ids = []
    for gid, state in games.items():
        try:
            started = datetime.fromisoformat(state["started_at"])
            if (now - started).total_seconds() > GAME_TTL_SECONDS:
                expired_ids.append(gid)
        except (KeyError, ValueError):
            expired_ids.append(gid)
    for gid in expired_ids:
        del games[gid]


# --- Routes ---

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

    prompt = f"{system_prompt}\n\n{received_text}"

    # Try Gemini first
    if gemini_client:
        try:
            response_text = _call_gemini(prompt)
            app.logger.info("Using Gemini API")
            return jsonify({"message": "AIによってデータが処理されました。", "processed_text": response_text})
        except Exception as e:
            app.logger.warning(f"Gemini API failed: {e}")

    # Fallback to Ollama
    try:
        response_text = _call_ollama(prompt)
        app.logger.info(f"Using Ollama model {OLLAMA_MODEL}")
        return jsonify({"message": "AIによってデータが処理されました。", "processed_text": response_text})
    except Exception as e:
        app.logger.error(f"Ollama API call failed: {e}")
        return jsonify({"error": "Gemini APIとOllamaのどちらも利用できないため、応答を生成できませんでした。"}), 500


MAX_WORD_RETRIES = 3
MAX_QUIZ_RETRIES = 3


@app.route('/generate_game', methods=['POST'])
def generate_game():
    _cleanup_old_games()
    data = request.get_json(silent=True) or {}
    theme = data.get('theme', '').strip()
    if not theme:
        return jsonify({"error": "お題を入力してください。"}), 400
    if len(theme) > 50:
        return jsonify({"error": "お題は50文字以内で入力してください。"}), 400

    result = _create_game(theme)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@app.route('/regenerate_game', methods=['POST'])
def regenerate_game():
    _cleanup_old_games()
    data = request.get_json(silent=True) or {}
    theme = data.get('theme', '').strip()
    if not theme:
        return jsonify({"error": "お題を入力してください。"}), 400
    if len(theme) > 50:
        return jsonify({"error": "お題は50文字以内で入力してください。"}), 400

    result = _create_game(theme)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


def _create_game(theme):
    words = _generate_words(theme)
    if words is None:
        return ({"error": "Gemini APIとOllamaのどちらも利用できないため、ゲームを生成できませんでした。"}, 500)

    pairs = _generate_quizzes(words)
    if pairs is None:
        return ({"error": "Gemini APIとOllamaのどちらも利用できないため、ゲームを生成できませんでした。"}, 500)

    cards = []
    card_id = 1
    for pair in pairs:
        answer = pair["answer"]
        for q in pair["questions"]:
            cards.append({"id": card_id, "answer": answer, "text": q})
            card_id += 1

    if len(cards) != 16:
        app.logger.error(f"Expected 16 cards, got {len(cards)}")
        return ({"error": "ゲームデータの生成に失敗しました。"}, 500)

    random.shuffle(cards)

    game_id = uuid.uuid4().hex
    games[game_id] = {
        "theme": theme,
        "cards": {c["id"]: c["answer"] for c in cards},
        "matched": set(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "moves": 0,
        "cleared": False,
    }

    public_cards = [{"id": c["id"], "text": c["text"]} for c in cards]
    return {"game_id": game_id, "cards": public_cards}


@app.route('/check_pair', methods=['POST'])
def check_pair():
    data = request.get_json(silent=True) or {}
    game_id = data.get("game_id")
    card_ids = data.get("card_ids", [])

    if not game_id or game_id not in games:
        return jsonify({"error": "ゲームが見つかりません。"}), 400

    state = games[game_id]

    if state["cleared"]:
        return jsonify({"error": "このゲームはすでに終了しています。"}), 400

    if not isinstance(card_ids, list) or len(card_ids) != 2:
        return jsonify({"error": "card_idsには2つのカードIDを指定してください。"}), 400

    id1, id2 = card_ids

    if id1 not in state["cards"] or id2 not in state["cards"]:
        return jsonify({"error": "無効なカードIDです。"}), 400

    if id1 == id2:
        return jsonify({"error": "同じカードは選択できません。"}), 400

    if id1 in state["matched"] or id2 in state["matched"]:
        return jsonify({"error": "このカードはすでにペアが成立しています。"}), 400

    is_pair = state["cards"][id1] == state["cards"][id2]
    state["moves"] += 1
    game_cleared = False
    score = None

    if is_pair:
        state["matched"].add(id1)
        state["matched"].add(id2)
        game_cleared = len(state["matched"]) == 16

        if game_cleared:
            state["cleared"] = True
            score = _calculate_score(state["moves"])
            cleared_at = datetime.now(timezone.utc)
            started_at = datetime.fromisoformat(state["started_at"])
            elapsed_seconds = int((cleared_at - started_at).total_seconds())
            game_history.append({
                "game_id": game_id,
                "theme": state["theme"],
                "started_at": state["started_at"],
                "cleared_at": cleared_at.isoformat(),
                "moves": state["moves"],
                "score": score,
                "elapsed_seconds": elapsed_seconds,
            })
            if len(game_history) > 50:
                del game_history[:-50]

    return jsonify({
        "is_pair": is_pair,
        "answer": state["cards"][id1] if is_pair else None,
        "game_cleared": game_cleared,
        "score": score,
    })


@app.route('/game_history', methods=['GET'])
def get_game_history():
    return jsonify({"history": list(reversed(game_history))})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

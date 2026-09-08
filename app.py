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
OLLAMA_MODEL = "qwen2.5:1.5b"

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

GAME_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "minItems": 8,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "pair_id": {"type": "integer"},
                    "answer": {"type": "string"},
                    "questions": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "string"},
                    },
                },
                "required": ["pair_id", "answer", "questions"],
            },
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


def _call_ollama(prompt, temperature=0.2):
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
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

def _validate_game_data(data):
    if not isinstance(data, dict):
        app.logger.warning("Validation failed: response is not a dict")
        return False
    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        app.logger.warning("Validation failed: pairs is not a list")
        return False
    if len(pairs) != 8:
        app.logger.warning(f"Validation failed: pairs count is {len(pairs)}, expected 8")
        return False
    seen_pair_ids = set()
    seen_answers = set()
    for i, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            app.logger.warning(f"Validation failed: pair {i} is not a dict")
            return False
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, int) or pair_id < 1 or pair_id > 8:
            app.logger.warning(f"Validation failed: pair {i} has invalid pair_id: {pair_id}")
            return False
        if pair_id in seen_pair_ids:
            app.logger.warning(f"Validation failed: duplicate pair_id {pair_id}")
            return False
        seen_pair_ids.add(pair_id)
        answer = pair.get("answer", "")
        if not isinstance(answer, str) or not answer.strip():
            app.logger.warning(f"Validation failed: pair {i} has empty answer")
            return False
        if answer in seen_answers:
            app.logger.warning(f"Validation failed: duplicate answer '{answer}'")
            return False
        seen_answers.add(answer)
        questions = pair.get("questions", [])
        if not isinstance(questions, list) or len(questions) != 2:
            app.logger.warning(f"Validation failed: pair {i} has {len(questions)} questions, expected 2")
            return False
        q0, q1 = questions
        if not isinstance(q0, str) or not q0.strip():
            app.logger.warning(f"Validation failed: pair {i} question[0] is empty")
            return False
        if not isinstance(q1, str) or not q1.strip():
            app.logger.warning(f"Validation failed: pair {i} question[1] is empty")
            return False
        if q0 == q1:
            app.logger.warning(f"Validation failed: pair {i} has duplicate questions")
            return False
    return True


# --- Game Data Generation (single call) ---

def _generate_game_data(theme):
    prompt = (
        f"お題「{theme}」について、神経衰弱ゲームのデータを生成してください。\n\n"
        "【厳守事項】\n"
        "- JSONのみ出力してください。説明文・Markdown・コードブロックは一切不要です。\n"
        "- pairsは必ず8個生成してください。\n"
        "- 各pairのquestionsは必ず2個にしてください。\n"
        "- answerは8個すべて異なるものにしてください。\n"
        "- 2つのquestionsは異なる表現・聞き方にしてください。\n"
        "- 質問は短く具体的にしてください。曖昧で複数の答えが考えられる質問は避けてください。\n"
        "- お題から大きく外れた内容は生成しないでください。\n"
        "- pair_idは1〜8の整数にしてください。\n"
        "- 余計なフィールドは生成しないでください。\n\n"
        "【出力形式】\n"
        '{"pairs":[{"pair_id":1,"answer":"回答","questions":["質問A","質問B"]}]}\n'
    )

    # Try Gemini (single call, no retry)
    if gemini_client:
        try:
            result = _call_gemini(prompt, response_schema=GAME_SCHEMA)
            if _validate_game_data(result):
                app.logger.info("Using Gemini API")
                return result.get("pairs", [])
            app.logger.warning("Gemini response validation failed")
        except Exception as e:
            app.logger.warning(f"Gemini API failed: {e}")

    # Fallback to Ollama (step-by-step generation)
    return _generate_game_data_with_ollama(theme)


def _generate_game_data_with_ollama(theme):
    app.logger.info(f"Using Ollama model {OLLAMA_MODEL}")

    # Step 1: Generate 8 answers one by one
    answers = []
    for i in range(8):
        answer = _ollama_generate_one_answer(theme, answers)
        if answer is None:
            app.logger.warning(f"Ollama failed to generate answer {i + 1}")
            return None
        answers.append(answer)

    # Step 2: Generate 2 questions per answer (1 question at a time)
    pairs = []
    for i, answer in enumerate(answers):
        q1 = _ollama_generate_question_1(theme, answer)
        if q1 is None:
            app.logger.warning(f"Ollama failed to generate question 1 for answer '{answer}'")
            return None
        q2 = _ollama_generate_question_2(theme, answer, q1)
        if q2 is None:
            app.logger.warning(f"Ollama failed to generate question 2 for answer '{answer}'")
            return None
        pairs.append({
            "pair_id": i + 1,
            "answer": answer,
            "questions": [q1, q2],
        })

    return pairs


MAX_ANSWER_RETRIES = 3
MAX_QUESTION_RETRIES = 3


def _normalize_for_comparison(text):
    import unicodedata
    import string
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(
        str.maketrans("", "", string.punctuation + "　、。！？「」（）")
    )
    return normalized.strip().lower()


def _ollama_generate_one_answer(theme, existing_answers):
    existing_text = ""
    if existing_answers:
        existing_text = (
            f"\nすでに使用した答え：{', '.join(existing_answers)}\n"
            "上記とは異なる答えを生成してください。"
        )
    prompt = (
        f"あなたはクイズゲームの答えを1つだけ生成するAIです。\n\n"
        f"お題：\n「{theme}」\n"
        f"{existing_text}\n"
        f"上記のお題に直接関係し、すでに使用した答えとは異なる具体的な答えを1つだけ生成してください。\n\n"
        "【絶対条件】\n"
        "- 答えは1つだけ\n"
        "- お題に直接関係するもの\n"
        "- 既に使用した答えと同じものは禁止\n"
        "- お題と無関係なものは禁止\n"
        "- 説明は禁止\n"
        "- 前置きは禁止\n"
        "- JSONのみ出力\n\n"
        "【出力形式】\n"
        '{"answer":"単語"}\n'
    )
    for attempt in range(MAX_ANSWER_RETRIES):
        try:
            raw = _call_ollama(prompt, temperature=0.2)
            data = _parse_json_from_text(raw)
            answer = data.get("answer", "") if isinstance(data, dict) else ""
        except Exception as e:
            app.logger.error(f"Ollama answer generation failed (attempt {attempt + 1}): {e}")
            continue
        if (isinstance(answer, str) and answer.strip()
                and answer.strip() not in existing_answers):
            return answer.strip()
        app.logger.warning(
            f"Ollama answer rejected (attempt {attempt + 1}): "
            f"'{answer}' (empty={not answer.strip()}, "
            f"duplicate={answer.strip() in existing_answers})"
        )
    return None


def _ollama_generate_question_1(theme, answer):
    prompt = (
        f"あなたはクイズゲームの問題作成AIです。\n\n"
        f"お題：\n「{theme}」\n\n"
        f"正解：\n「{answer}」\n\n"
        f"上記の「{answer}」が正解になる問題を1問だけ作成してください。\n\n"
        "【絶対条件】\n"
        "- 問題は1問だけ\n"
        "- 正解は必ず「{answer}」\n"
        "- 問題文に「{answer}」という文字を含めない\n"
        f"- お題「{theme}」に直接関係する内容にする\n"
        "- プレイヤー個人の経験を質問する問題は禁止\n"
        "- 正解が客観的に決まる問題にする\n"
        "- 複数の答えが成立する曖昧な問題は禁止\n"
        "- JSONのみ出力\n"
        "- 説明文は禁止\n\n"
        "【出力形式】\n"
        '{"question":"問題文"}\n'
    )
    for attempt in range(MAX_QUESTION_RETRIES):
        try:
            raw = _call_ollama(prompt, temperature=0.2)
            data = _parse_json_from_text(raw)
            q = data.get("question", "") if isinstance(data, dict) else ""
        except Exception as e:
            app.logger.error(
                f"Ollama question 1 generation failed for '{answer}' "
                f"(attempt {attempt + 1}): {e}"
            )
            continue
        if isinstance(q, str) and q.strip():
            return q.strip()
        app.logger.warning(
            f"Ollama question 1 validation failed for '{answer}' "
            f"(attempt {attempt + 1}): '{q}'"
        )
    return None


def _ollama_generate_question_2(theme, answer, existing_question):
    prompt = (
        f"あなたはクイズゲームの問題作成AIです。\n\n"
        f"お題：\n「{theme}」\n\n"
        f"正解：\n「{answer}」\n\n"
        f"既に生成した問題：\n「{existing_question}」\n\n"
        f"上記の「{answer}」が正解になる問題を、既に生成した問題とは異なる内容で1問だけ作成してください。\n\n"
        "【絶対条件】\n"
        "- 問題は1問だけ\n"
        "- 正解は必ず「{answer}」\n"
        "- 問題文に「{answer}」という文字を含めない\n"
        f"- お題「{theme}」に直接関係する内容にする\n"
        "- 既に生成した問題と内容が重複しない\n"
        "- プレイヤー個人の経験を質問する問題は禁止\n"
        "- 正解が客観的に決まる問題にする\n"
        "- 複数の答えが成立する曖昧な問題は禁止\n"
        "- JSONのみ出力\n"
        "- 説明文は禁止\n\n"
        "【出力形式】\n"
        '{"question":"問題文"}\n'
    )
    for attempt in range(MAX_QUESTION_RETRIES):
        try:
            raw = _call_ollama(prompt, temperature=0.2)
            data = _parse_json_from_text(raw)
            q = data.get("question", "") if isinstance(data, dict) else ""
        except Exception as e:
            app.logger.error(
                f"Ollama question 2 generation failed for '{answer}' "
                f"(attempt {attempt + 1}): {e}"
            )
            continue
        if (isinstance(q, str) and q.strip()
                and _normalize_for_comparison(q) != _normalize_for_comparison(existing_question)):
            return q.strip()
        app.logger.warning(
            f"Ollama question 2 rejected for '{answer}' "
            f"(attempt {attempt + 1}): '{q}' "
            f"(duplicate={_normalize_for_comparison(q) == _normalize_for_comparison(existing_question)})"
        )
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
    pairs = _generate_game_data(theme)
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

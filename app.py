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

# Quiz history (generated quiz pairs with answers)
quiz_history = []

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
GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

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

def _call_gemini(prompt, response_schema=None, model_name=None):
    if not gemini_client:
        raise RuntimeError("Gemini APIキーが設定されていません。")
    model = model_name or GEMINI_MODELS[0]
    config = types.GenerateContentConfig(temperature=1.0)
    if response_schema:
        config.response_mime_type = "application/json"
        config.response_json_schema = response_schema
    response = gemini_client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    if response_schema:
        return response.parsed
    return response.text


def _is_gemini_rate_limit_error(exc):
    error_str = str(exc).lower()
    if "429" in error_str:
        return True
    if "resource_exhausted" in error_str:
        return True
    if "quota" in error_str:
        return True
    if "rate limit" in error_str or "rate-limit" in error_str:
        return True
    if "too many requests" in error_str:
        return True
    return False


def _is_gemini_auth_error(exc):
    error_str = str(exc).lower()
    if "401" in error_str:
        return True
    if "403" in error_str:
        return True
    if "permission denied" in error_str:
        return True
    if "unauthorized" in error_str:
        return True
    return False


def _is_gemini_server_error(exc):
    error_str = str(exc).lower()
    if "500" in error_str:
        return True
    if "502" in error_str:
        return True
    if "503" in error_str:
        return True
    if "504" in error_str:
        return True
    if "internal server error" in error_str:
        return True
    if "service unavailable" in error_str:
        return True
    return False


def _call_gemini_model(theme, model_name, used_answers=None):
    prompt = (
        f"お題「{theme}」について、神経衰弱ゲームのデータを生成してください。\n\n"
        "【厳守事項】\n"
        "- JSONのみ出力してください。説明文・Markdown・コードブロックは一切不要です。\n"
        "- pairsは必ず8個生成してください。\n"
        "- 各pairのquestionsは必ず2個にしてください。\n"
        "- answerは8個すべて異なるものにしてください。\n"
        "- 2つのquestionsは異なる表現・聞き方にしてください。\n"
        "- questionsの文中にanswerそのものを含めないでください。答えを問題文から推測できるようにしてください。\n"
        "- 質問は短く具体的にしてください。曖昧で複数の答えが考えられる質問は避けてください。\n"
        "- お題から大きく外れた内容は生成しないでください。\n"
        "- pair_idは1〜8の整数にしてください。\n"
        "- 余計なフィールドは生成しないでください。\n\n"
    )
    
    # 過去に使用済みのanswerがある場合はプロンプトに追加
    if used_answers:
        # 使用済みanswerが大量の場合、最新50件までに制限
        limited_answers = used_answers[-50:] if len(used_answers) > 50 else used_answers
        answers_text = "、".join(limited_answers)
        prompt += (
            "【過去使用済みのanswer】\n"
            f"このお題では過去に以下の答えが使用されています。\n"
            f"{answers_text}\n\n"
            "上記と同じ答えは使用しないでください。新しい答えを生成してください。\n\n"
        )
    
    prompt += (
        "【出力形式】\n"
        '{"pairs":[{"pair_id":1,"answer":"回答","questions":["質問A","質問B"]}]}\n'
    )
    
    result = _call_gemini(prompt, response_schema=GAME_SCHEMA, model_name=model_name)
    if not _validate_game_data(result):
        raise ValueError("Gemini response validation failed")
    
    # 過去のanswerと重複していないかチェック
    if used_answers:
        generated_pairs = result.get("pairs", [])
        for pair in generated_pairs:
            answer = pair.get("answer", "")
            if answer:
                normalized = _normalize_for_comparison(answer)
                if normalized in used_answers:
                    raise ValueError(f"Generated answer '{answer}' conflicts with past answers")
    
    return result.get("pairs", [])


def _generate_game_data_with_gemini(theme):
    # 過去に同じお題で使用したanswerを取得
    used_answers = _get_used_answers_for_theme(theme)
    if used_answers:
        app.logger.info(f"Found {len(used_answers)} used answers for theme '{theme}'")
    
    for model_name in GEMINI_MODELS:
        app.logger.info(f"Using Gemini model: {model_name}")
        try:
            pairs = _call_gemini_model(theme, model_name, used_answers)
            app.logger.info(f"Gemini model {model_name} succeeded")
            app.logger.info(f"LLM generation backend: Gemini {model_name}")
            return pairs
        except Exception as e:
            error_str = str(e).lower()
            # 過去のanswerとの重複による失敗の場合
            if "conflicts with past answers" in error_str:
                app.logger.warning(
                    f"Gemini model {model_name} generated conflicting answers, trying next model"
                )
                continue
            elif _is_gemini_rate_limit_error(e):
                app.logger.warning(
                    f"Gemini model {model_name} rate limit reached, trying next model"
                )
            elif _is_gemini_auth_error(e):
                app.logger.warning(
                    f"Gemini model {model_name} auth error: {e}, skipping remaining Gemini models"
                )
                break
            elif _is_gemini_server_error(e):
                app.logger.warning(
                    f"Gemini model {model_name} server error: {e}, retrying once"
                )
                try:
                    pairs = _call_gemini_model(theme, model_name, used_answers)
                    app.logger.info(f"Gemini model {model_name} succeeded on retry")
                    app.logger.info(f"LLM generation backend: Gemini {model_name}")
                    return pairs
                except Exception as retry_e:
                    if _is_gemini_rate_limit_error(retry_e):
                        app.logger.warning(
                            f"Gemini model {model_name} rate limit on retry, trying next model"
                        )
                    else:
                        app.logger.warning(
                            f"Gemini model {model_name} retry failed: {retry_e}"
                        )
            else:
                app.logger.warning(f"Gemini model {model_name} failed: {e}")
    app.logger.warning("All Gemini models failed, falling back to Ollama")
    return None


def _call_ollama(prompt, temperature=0.2, response_format=None):
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    }
    if response_format == "json":
        payload["format"] = "json"
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
    return max(0, 1000 - (moves - 10) * 30)


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
        if answer.lower() in q0.lower():
            app.logger.warning(f"Validation failed: pair {i} answer found in question[0]")
            return False
        if answer.lower() in q1.lower():
            app.logger.warning(f"Validation failed: pair {i} answer found in question[1]")
            return False
        if _check_personal_expression(q0):
            app.logger.warning(f"Validation failed: pair {i} question[0] contains personal expression")
            return False
        if _check_personal_expression(q1):
            app.logger.warning(f"Validation failed: pair {i} question[1] contains personal expression")
            return False
    return True


# --- Game Data Generation (single call) ---

def _generate_game_data(theme):
    if gemini_client:
        result = _generate_game_data_with_gemini(theme)
        if result is not None:
            return result

    if not gemini_client:
        app.logger.warning("GEMINI_API_KEY is not configured, using Ollama")
    app.logger.info(f"Using Ollama model: {OLLAMA_MODEL}")
    app.logger.info(f"LLM generation backend: Ollama {OLLAMA_MODEL}")
    return _generate_game_data_with_ollama(theme)


def _generate_game_data_with_ollama(theme):
    app.logger.info(f"Using Ollama model {OLLAMA_MODEL}")

    # 過去に同じお題で使用したanswerを取得
    used_answers = _get_used_answers_for_theme(theme)
    if used_answers:
        app.logger.info(f"Found {len(used_answers)} used answers for theme '{theme}' (Ollama)")

    # Step 1: Generate candidate answers (20 at once)
    all_answers = _ollama_generate_answers(theme, used_answers)
    if all_answers is None:
        return None

    # Step 2: Try each answer until we have 8 pairs
    pairs = []
    for answer in all_answers:
        if len(pairs) >= REQUIRED_ANSWERS:
            break

        app.logger.info(f"Ollama answer candidate: {answer}")

        # Step 2a: Generate question candidates
        candidates = _ollama_generate_question_candidates(theme, answer)
        if candidates is None:
            continue

        # Step 2b: Validate and select 2 questions
        selected = _select_questions(candidates, answer)
        if selected is None:
            continue

        pairs.append({
            "pair_id": len(pairs) + 1,
            "answer": answer,
            "questions": selected,
        })
        app.logger.info(f"Ollama pair completed: {answer}")

    if len(pairs) < REQUIRED_ANSWERS:
        app.logger.warning(
            f"Ollama failed to generate {REQUIRED_ANSWERS} pairs, got {len(pairs)}"
        )
        return None

    app.logger.info(f"Ollama successfully generated {len(pairs)}/{REQUIRED_ANSWERS} pairs")
    return pairs


MAX_CANDIDATE_RETRIES = 3
MAX_QUESTION_CANDIDATE_RETRIES = 3
REQUIRED_ANSWERS = 8
CANDIDATE_POOL_SIZE = 20
QUESTION_CANDIDATE_COUNT = 8


def _normalize_for_comparison(text):
    import unicodedata
    import string
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(
        str.maketrans("", "", string.punctuation + "　、。！？「」（）")
    )
    return normalized.strip().lower()


def _get_used_answers_for_theme(theme):
    """quiz_historyから指定テーマの過去のanswer一覧を取得"""
    used_answers = set()
    for record in quiz_history:
        if record["theme"] == theme:
            for pair in record["pairs"]:
                answer = pair.get("answer", "")
                if answer:
                    normalized = _normalize_for_comparison(answer)
                    if normalized:
                        used_answers.add(normalized)
    return list(used_answers)


def _ollama_generate_answers(theme, used_answers=None):
    seen = set()
    candidates = []

    for pool_attempt in range(MAX_CANDIDATE_RETRIES):
        used_text = ""
        if candidates:
            used_text = (
                f"\nすでに生成された候補：{', '.join(candidates)}\n"
                "上記とは異なる候補を生成してください。"
            )
        
        past_text = ""
        if used_answers:
            limited = used_answers[-50:] if len(used_answers) > 50 else used_answers
            past_text = (
                f"\n【過去使用済みの答え】\n"
                f"{'、'.join(limited)}\n"
                "上記の答えは使用しないでください。新しい答えを生成してください。\n"
            )
        
        prompt = (
            f"あなたはクイズゲームの答えを生成するAIです。\n\n"
            f"お題：\n「{theme}」\n"
            f"{used_text}\n"
            f"{past_text}\n"
            f"上記のお題に関連する答えの候補を{CANDIDATE_POOL_SIZE}個生成してください。\n\n"
            "【絶対条件】\n"
            f"- お題「{theme}」に直接関係する具体的な単語または固有名詞\n"
            "- お題と無関係なものは禁止（変数、JSON、リスト、配列、要素、プログラミング用語など）\n"
            "- 日本語で自然な表記にする\n"
            "- 候補はすべて異なるもの\n"
            "- 説明文や前置きは禁止\n"
            "- JSONのみ出力\n\n"
            "【出力形式】\n"
            '{"candidates":["候補1","候補2",...]}'
        )
        try:
            raw = _call_ollama(prompt, temperature=0.2, response_format="json")
            data = _parse_json_from_text(raw)
            new_candidates = data.get("candidates", []) if isinstance(data, dict) else []
        except Exception as e:
            app.logger.error(f"Ollama candidate generation failed (pool attempt {pool_attempt + 1}): {e}")
            continue

        for c in new_candidates:
            if (isinstance(c, str) and c.strip()
                    and c.strip() not in seen
                    and _is_valid_answer(c.strip())):
                normalized = _normalize_for_comparison(c.strip())
                if used_answers and normalized in used_answers:
                    app.logger.info(f"Ollama candidate '{c.strip()}' skipped: already used in past games")
                    continue
                seen.add(c.strip())
                candidates.append(c.strip())

        app.logger.info(f"Ollama candidates collected: {len(candidates)}/{REQUIRED_ANSWERS}")
        if len(candidates) >= REQUIRED_ANSWERS:
            break

    if len(candidates) < REQUIRED_ANSWERS:
        app.logger.warning(
            f"Ollama failed to collect {REQUIRED_ANSWERS} unique answers, "
            f"got {len(candidates)}"
        )
        return None

    return candidates


def _is_valid_answer(answer):
    import unicodedata
    if not answer or len(answer) < 1:
        return False
    if len(answer) > 15:
        return False
    has_cjk = False
    has_kana_or_latin = False
    for ch in answer:
        name = unicodedata.name(ch, "")
        if "CJK" in name:
            has_cjk = True
        if "HIRAGANA" in name or "KATAKANA" in name or ch.isalpha():
            has_kana_or_latin = True
    if has_cjk and not has_kana_or_latin:
        return False
    if answer.isdigit():
        return False
    return True


def _ollama_generate_question_candidates(theme, answer):
    for attempt in range(MAX_QUESTION_CANDIDATE_RETRIES):
        prompt = (
            f"あなたはクイズゲームの問題作成AIです。\n\n"
            f"お題：「{theme}」\n"
            f"正解：「{answer}」\n\n"
            f"正解が「{answer}」となるクイズを{QUESTION_CANDIDATE_COUNT}個作成してください。\n\n"
            "【最も重要】\n"
            "- 問題文に正解の名前を絶対に含めない\n"
            "- 正解の別名・表記も含めない\n"
            "- 正解を含む固有名詞・複合語も使わない\n\n"
            "【問題の条件】\n"
            "- テーマについて一般的な知識を問う質問にする\n"
            "- 正解の特徴・役割・関係を利用して問題を作る\n"
            "- 客観的な知識問題にする\n"
            "- 正解が客観的に1つに決まる問題にする\n"
            "- 複数の答えが成立する曖昧な問題は禁止\n"
            "- 問題文は簡潔で自然な日本語にする\n"
            "- 日本語だけで生成する\n"
            "- 各問題は異なる観点・特徴から出題する\n\n"
            "【禁止事項】\n"
            "- 「あなた」「あなたが」「あなたは」「君」を使用しない\n"
            "- 「好きですか」「知っていますか」「飼っていますか」「持っていますか」は禁止\n"
            "- 「使っていますか」「見たことがありますか」「経験がありますか」は禁止\n"
            "- 「〜したことがありますか」の形式は禁止\n"
            "- プレイヤーの経験・好み・所有物を質問しない\n"
            "- 個人の意見を求める質問は禁止\n\n"
            "【出力形式】\n"
            '{"questions":["問題1","問題2","問題3","問題4","問題5","問題6","問題7","問題8"]}'
        )
        try:
            raw = _call_ollama(prompt, temperature=0.4, response_format="json")
            data = _parse_json_from_text(raw)
            questions = data.get("questions", []) if isinstance(data, dict) else []
        except Exception as e:
            app.logger.error(
                f"Ollama question candidates failed for '{answer}' "
                f"(attempt {attempt + 1}): {e}"
            )
            continue
        if isinstance(questions, list) and len(questions) >= 2:
            app.logger.info(f"Ollama question candidates: {len(questions)}")
            return [q.strip() for q in questions if isinstance(q, str) and q.strip()]
        app.logger.warning(
            f"Ollama question candidates rejected for '{answer}' "
            f"(attempt {attempt + 1}): {questions}"
        )
    return None


def _validate_question(question, answer):
    if not question or not isinstance(question, str):
        return False, "empty"
    q = question.strip()
    if len(q) < 5:
        return False, "too_short"
    if answer in q:
        return False, "contains_answer"
    if answer.lower() in q.lower():
        return False, "contains_answer_case_insensitive"
    if _check_personal_expression(q):
        return False, "personal_expression"
    if not ("？" in q or "?" in q):
        return False, "no_question_mark"
    return True, "ok"


def _select_questions(candidates, answer):
    valid = []
    for q in candidates:
        ok, reason = _validate_question(q, answer)
        if ok:
            app.logger.info(f"Ollama question accepted: '{q}'")
            valid.append(q)
        else:
            app.logger.info(f"Ollama question rejected: '{q}' (reason={reason})")

    if len(valid) < 2:
        return None

    selected = [valid[0]]
    for v in valid[1:]:
        if _normalize_for_comparison(v) != _normalize_for_comparison(selected[0]):
            selected.append(v)
            break

    if len(selected) < 2:
        return None

    app.logger.info(f"Ollama selected questions: {selected}")
    return selected


PERSONAL_EXPRESSIONS = [
    "あなた",
    "好きですか",
    "知っていますか",
    "飼っていますか",
    "持っていますか",
    "使っていますか",
    "見たことがありますか",
    "経験がありますか",
    "したことがありますか",
]


def _check_personal_expression(text):
    for expr in PERSONAL_EXPRESSIONS:
        if expr in text:
            return True
    return False



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

    quiz_history.append({
        "quiz_id": uuid.uuid4().hex,
        "theme": theme,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pairs": pairs,
    })
    if len(quiz_history) > 100:
        del quiz_history[:-100]

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
    elapsed_seconds = None

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
        "moves": state["moves"],
        "elapsed_seconds": elapsed_seconds,
    })


@app.route('/game_history', methods=['GET'])
def get_game_history():
    return jsonify({"history": list(reversed(game_history))})


@app.route('/quiz_history', methods=['GET'])
def get_quiz_history():
    return jsonify({"history": list(reversed(quiz_history))})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

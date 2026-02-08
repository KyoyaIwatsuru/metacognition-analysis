#!/usr/bin/env python3
"""
根拠箇所抽出スクリプト

question_mapping.jsonを読み込み、各問題の本文・解説等を収集し、
LLM（OpenAI GPT-4o）を使用して根拠文を特定する。

変更履歴:
- v2: Excelの本文テキストをそのまま使用（シンプルで確実）
      本文は \n\n\n で分割して「本文1」「本文2」としてLLMに渡す
      LLMはテキストベースで根拠を返す

出力: data/working/evidence_mapping.json
"""

import json
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent.parent.parent
WORKING_DIR = BASE_DIR / "data" / "working"
QUESTION_MAPPING_FILE = WORKING_DIR / "question_mapping.json"
OUTPUT_FILE = WORKING_DIR / "evidence_mapping.json"

# .envファイルを読み込み
ENV_FILE = Path("/Users/kyoya/Laboratory/metacognition/llm/.env")
load_dotenv(ENV_FILE)


def load_question_mapping() -> dict:
    """question_mapping.jsonを読み込み"""
    with open(QUESTION_MAPPING_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def split_passage_text(passage_text: str) -> list[str]:
    """本文を複数のパッセージに分割

    分割ルール:
    - \n\n\n（3つの改行）で分割
    - 各パッセージは独立した文書（例：メール + スケジュール表）
    """
    if not passage_text:
        return []

    # \n\n\n で分割
    parts = passage_text.split('\n\n\n')

    # 空のパートを除去し、前後の空白をトリム
    return [p.strip() for p in parts if p.strip()]


def format_passage_for_prompt(passage_text: str) -> str:
    """本文をプロンプト用にフォーマット

    複数パッセージがある場合は「本文1」「本文2」として表示
    """
    passages = split_passage_text(passage_text)

    if not passages:
        return "(本文なし)"

    if len(passages) == 1:
        return passages[0]

    result = []
    for i, p in enumerate(passages):
        result.append(f"### 本文{i + 1}\n{p}")

    return '\n\n'.join(result)


def format_choices(choices: list) -> str:
    """選択肢をフォーマット"""
    return '\n'.join([f"({c['id'].upper()}) {c['text']}" for c in choices])


def create_extraction_prompt(
    passage_text: str,
    question_text: str,
    choices: str,
    correct_answer: str,
    explanation: str,
    metacognition_feedback: str = None,
    metacognition_memo: str = None
) -> str:
    """根拠抽出用のプロンプトを作成

    本文はExcelから取得したテキストをそのまま使用。
    複数パッセージがある場合は「本文1」「本文2」として表示。
    """

    formatted_passage = format_passage_for_prompt(passage_text)

    # 複数パッセージかどうかを判定
    passages = split_passage_text(passage_text)
    has_multiple_passages = len(passages) > 1

    prompt = f"""あなたは英語読解問題の分析専門家です。以下の問題について、正解の根拠となる文を特定してください。

## 本文
{formatted_passage}

## 設問
{question_text}

## 選択肢
{choices}

## 正解
{correct_answer}

## 解説
{explanation}
"""

    if metacognition_feedback:
        prompt += f"""
## メタ認知フィードバック
{metacognition_feedback}
"""

    if metacognition_memo:
        # JSONの場合はmust_aoiを抽出
        try:
            memo = json.loads(metacognition_memo)
            if 'must_aoi' in memo:
                prompt += f"""
## 重要な着目箇所（ヒント）
{', '.join(memo['must_aoi'])}
"""
        except (json.JSONDecodeError, TypeError):
            pass

    # 出力形式の説明
    if has_multiple_passages:
        passage_index_desc = "passage_index: <本文番号（0から開始、本文1=0, 本文2=1）>,"
    else:
        passage_index_desc = "passage_index: 0,"

    prompt += f"""
## タスク
上記の情報を分析し、正解を導くために必要な根拠となる文を特定してください。

以下のJSON形式で回答してください：
```json
{{
  "evidence_sentences": [
    {{
      {passage_index_desc}
      "type": "primary" または "secondary",
      "text": "<該当する文のテキスト（本文から正確にコピー）>",
      "reason": "<なぜこの文が根拠になるか（日本語で簡潔に）>"
    }}
  ],
  "reasoning": "<全体的な解答の流れ（日本語で簡潔に）>"
}}
```

注意事項：
- primary: 正解を直接示す最も重要な根拠（通常1-2文）
- secondary: 補助的な根拠、文脈理解に必要な文
- 必ず本文中の文のみを参照してください（選択肢の文は含めない）
- textには本文から正確にテキストをコピーしてください（視線分析時の照合に使用します）
"""

    return prompt


def extract_evidence_with_llm(
    passage_text: str,
    question_text: str,
    choices: str,
    correct_answer: str,
    explanation: str,
    metacognition_feedback: str = None,
    metacognition_memo: str = None
) -> dict:
    """LLMを使用して根拠を抽出"""

    client = OpenAI()

    prompt = create_extraction_prompt(
        passage_text=passage_text,
        question_text=question_text,
        choices=choices,
        correct_answer=correct_answer,
        explanation=explanation,
        metacognition_feedback=metacognition_feedback,
        metacognition_memo=metacognition_memo
    )

    response = client.responses.create(
        model="gpt-5.2",
        instructions="あなたは英語読解問題の分析専門家です。JSONフォーマットで回答してください。",
        input=prompt,
        temperature=0.0
    )

    response_text = response.output_text

    # JSONを抽出
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # JSONブロックなしの場合
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {
            'error': 'Failed to parse LLM response',
            'raw_response': response_text
        }


def main():
    """メイン処理"""
    print("Loading question mapping...")
    mapping = load_question_mapping()

    evidence_mapping = {}
    total_questions = 0
    processed_questions = 0

    # 処理対象をカウント
    for passage_id, entry in mapping.items():
        total_questions += len(entry.get('questions', {}))

    print(f"Processing {total_questions} questions...")

    for passage_id, entry in mapping.items():
        # Excel由来の本文テキストを使用
        passage_text = entry.get('passage_text', '')

        evidence_mapping[passage_id] = {
            'passage_id': passage_id,
            'phase': entry['phase'],
            'questions': {}
        }

        for q_id, q_info in entry.get('questions', {}).items():
            processed_questions += 1
            print(f"  [{processed_questions}/{total_questions}] {passage_id}/{q_id}")

            # マッチングに失敗している場合はスキップ
            if not q_info.get('excel_key'):
                evidence_mapping[passage_id]['questions'][q_id] = {
                    'error': 'No Excel match',
                    'question_text': q_info.get('coord_question_text', '')
                }
                continue

            # 本文がない場合もスキップ
            if not passage_text:
                evidence_mapping[passage_id]['questions'][q_id] = {
                    'error': 'No passage text',
                    'question_text': q_info.get('coord_question_text', '')
                }
                continue

            question_text = q_info.get('coord_question_text', '')
            choices = format_choices(q_info.get('choices', []))
            correct_answer = q_info.get('correct_answer', '')
            explanation = q_info.get('explanation', '')
            metacognition_feedback = q_info.get('metacognition_feedback', '')
            metacognition_memo = q_info.get('metacognition_memo', '')

            try:
                result = extract_evidence_with_llm(
                    passage_text=passage_text,
                    question_text=question_text,
                    choices=choices,
                    correct_answer=correct_answer,
                    explanation=explanation,
                    metacognition_feedback=metacognition_feedback,
                    metacognition_memo=metacognition_memo
                )

                evidence_mapping[passage_id]['questions'][q_id] = {
                    'toeic_q': q_info.get('toeic_q'),
                    'question_text': question_text,
                    'correct_answer': correct_answer,
                    **result
                }
            except Exception as e:
                evidence_mapping[passage_id]['questions'][q_id] = {
                    'error': str(e),
                    'question_text': question_text
                }
                print(f"    Error: {e}")

    # 出力
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(evidence_mapping, f, ensure_ascii=False, indent=2)

    print(f"\nOutput: {OUTPUT_FILE}")

    # サマリー
    success_count = 0
    error_count = 0
    for passage_id, entry in evidence_mapping.items():
        for q_id, q_info in entry['questions'].items():
            if 'error' in q_info:
                error_count += 1
            else:
                success_count += 1

    print(f"\nSummary:")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")


if __name__ == '__main__':
    main()

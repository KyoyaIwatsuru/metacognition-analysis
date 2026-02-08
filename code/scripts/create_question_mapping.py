#!/usr/bin/env python3
"""
問題マッピング作成スクリプト

Excelの各行と座標ファイルの対応を自動生成する。
設問文の文字列照合により、Excel行と座標ファイルのquestion_idを対応付ける。

変更履歴:
- v2: passage_structureを削除し、Excel由来の情報のみ保持（根拠特定用）
      座標ファイルとの照合は視線分析時に行う

出力: data/working/question_mapping.json
"""

import json
import re
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "working"
EXCEL_FILE = INPUT_DIR / "問題の分析_with_feedback.xlsx"
COORDINATES_DIR = INPUT_DIR / "B" / "Test"


def normalize_text(text: str) -> str:
    """テキストを正規化（比較用）"""
    if not isinstance(text, str):
        return ""
    # 空白・改行を正規化
    text = re.sub(r'\s+', ' ', text.strip())
    # 全角を半角に
    text = text.replace('．', '.').replace('，', ',')
    return text


def similarity(a: str, b: str) -> float:
    """2つの文字列の類似度を計算"""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def extract_questions_from_coordinates(coord_file: Path) -> list[dict]:
    """座標ファイルから設問情報を抽出"""
    with open(coord_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = []
    coords = data.get('coordinates', {})
    right_panel = coords.get('right_panel', {})

    for q in right_panel.get('questions', []):
        question_text = q.get('question_text', {}).get('text', '')
        choices = []
        for c in q.get('choices', []):
            choice_text = c.get('choice_text', {}).get('text', '')
            choices.append({
                'id': c.get('choice_id', ''),
                'text': choice_text
            })

        questions.append({
            'question_id': q.get('question_id', ''),
            'question_index': q.get('question_index', 0),
            'question_text': question_text,
            'choices': choices
        })

    return questions




def extract_unique_id_from_filename(filename: str) -> str:
    """ファイル名から一意のIDを抽出（タイムスタンプ部分を除去）"""
    # analog_question_tr_01_tr_01_an1_2026-01-24T12-03-07-386Z.json → tr_01_an1
    # question_pre_01_2026-01-24T12-02-15-977Z.json → pre_01

    # タイムスタンプ部分を除去 (YYYY-MM-DD以降を削除)
    base = re.sub(r'_\d{4}-\d{2}-\d{2}T.*\.json$', '', filename)

    # プレフィックスを除去
    if base.startswith('analog_question_'):
        # analog_question_tr_01_tr_01_an1 → tr_01_an1
        parts = base.replace('analog_question_', '').split('_')
        # tr_01_tr_01_an1 → [tr, 01, tr, 01, an1]
        # 重複するtr_01を除去して tr_01_an1 にする
        if len(parts) >= 4:
            return f"{parts[0]}_{parts[1]}_{parts[-1]}"
        return '_'.join(parts)
    elif base.startswith('question_'):
        # question_pre_01 → pre_01
        return base.replace('question_', '')

    return base


def load_coordinate_files() -> dict:
    """全座標ファイルを読み込み"""
    coord_data = {}

    for phase in ['pre', 'post', 'training1', 'training2', 'training3']:
        phase_dir = COORDINATES_DIR / phase / 'coordinates'
        if not phase_dir.exists():
            continue

        for coord_file in phase_dir.glob('*.json'):
            filename = coord_file.name

            # question_*, analog_question_* ファイルを対象
            if filename.startswith('question_') or filename.startswith('analog_question_'):
                questions = extract_questions_from_coordinates(coord_file)

                if questions:
                    # ファイル名から一意のIDを生成
                    unique_id = extract_unique_id_from_filename(filename)
                    if unique_id:
                        coord_data[unique_id] = {
                            'phase': phase,
                            'file': str(coord_file),
                            'questions': questions
                        }

    return coord_data


def load_excel_data() -> list:
    """Excelデータを読み込み（リストで返す、重複問番号に対応）"""
    excel_data = []

    xlsx = pd.ExcelFile(EXCEL_FILE)

    # training シート
    df_training = pd.read_excel(xlsx, sheet_name='training')
    for idx, row in df_training.iterrows():
        toeic_q = row.get('問', None)
        if pd.isna(toeic_q):
            continue

        excel_data.append({
            'excel_key': f'training_{int(toeic_q)}_{idx}',
            'toeic_q': int(toeic_q),
            'sheet': 'training',
            'row_index': idx,
            'metacognition_type': row.get('メタ認知タイプ', ''),
            'passage_text_en': row.get('本文（英語）', ''),
            'passage_text_ja': row.get('本文（日本語）', ''),
            'question_text_en': row.get('設問文（英語）', ''),
            'choices_en': row.get('選択肢（英語）', ''),
            'question_text_ja': row.get('設問文（日本語）', ''),
            'choices_ja': row.get('選択肢（日本語）', ''),
            'correct_answer': row.get('正解', ''),
            'explanation': row.get('解説', ''),
            'metacognition_feedback': row.get('メタ認知フィードバック', ''),
            'metacognition_memo': row.get('メタ認知メモ(JSON)', '')
        })

    # pre-test シート
    df_pre = pd.read_excel(xlsx, sheet_name='pre-test')
    for idx, row in df_pre.iterrows():
        toeic_q = row.get('問', None)
        if pd.isna(toeic_q):
            continue

        excel_data.append({
            'excel_key': f'pre_{int(toeic_q)}_{idx}',
            'toeic_q': int(toeic_q),
            'sheet': 'pre-test',
            'row_index': idx,
            'metacognition_type': row.get('メタ認知タイプ', ''),
            'passage_text_en': row.get('本文（英語）', ''),
            'passage_text_ja': row.get('本文（日本語）', ''),
            'question_text_en': row.get('設問文（英語）', ''),
            'choices_en': row.get('選択肢（英語）', ''),
            'question_text_ja': row.get('設問文（日本語）', ''),
            'choices_ja': row.get('選択肢（日本語）', ''),
            'correct_answer': row.get('正解', ''),
            'explanation': row.get('解説', '')
        })

    # post-test シート
    df_post = pd.read_excel(xlsx, sheet_name='post-test')
    for idx, row in df_post.iterrows():
        toeic_q = row.get('問', None)
        if pd.isna(toeic_q):
            continue

        excel_data.append({
            'excel_key': f'post_{int(toeic_q)}_{idx}',
            'toeic_q': int(toeic_q),
            'sheet': 'post-test',
            'row_index': idx,
            'metacognition_type': row.get('メタ認知タイプ', ''),
            'passage_text_en': row.get('本文（英語）', ''),
            'passage_text_ja': row.get('本文（日本語）', ''),
            'question_text_en': row.get('設問文（英語）', ''),
            'choices_en': row.get('選択肢（英語）', ''),
            'question_text_ja': row.get('設問文（日本語）', ''),
            'choices_ja': row.get('選択肢（日本語）', ''),
            'correct_answer': row.get('正解', ''),
            'explanation': row.get('解説', '')
        })

    return excel_data


def match_questions(excel_data: list, coord_data: dict) -> dict:
    """Excelデータと座標データをマッチング"""
    mapping = {}

    for passage_id, coord_info in coord_data.items():
        phase = coord_info['phase']
        questions = coord_info['questions']

        passage_entry = {
            'passage_id': passage_id,
            'phase': phase,
            'coordinate_file': coord_info['file'],
            'passage_text': None,  # 最初にマッチした設問から取得
            'questions': {}
        }

        for q in questions:
            coord_question_text = q['question_text']
            question_id = q['question_id']

            best_match = None
            best_score = 0

            # Excelの設問文と照合（リストから検索）
            for excel_info in excel_data:
                excel_question_text = excel_info.get('question_text_en', '')
                if not excel_question_text:
                    continue

                score = similarity(coord_question_text, excel_question_text)
                if score > best_score:
                    best_score = score
                    best_match = excel_info

            if best_match and best_score > 0.6:
                # passage_textをExcelから取得（有効な本文が見つかるまで試行）
                passage_text = best_match.get('passage_text_en', '')
                # NaN対策: pandas NaNは float型でself != selfになる
                if isinstance(passage_text, float) and passage_text != passage_text:
                    passage_text = ''
                # 現在のpassage_textが空で、新しい本文が有効なら更新
                if not passage_entry['passage_text'] and passage_text:
                    passage_entry['passage_text'] = passage_text

                passage_entry['questions'][question_id] = {
                    'question_id': question_id,
                    'question_index': q['question_index'],
                    'coord_question_text': coord_question_text,
                    'excel_key': best_match['excel_key'],
                    'toeic_q': best_match['toeic_q'],
                    'excel_question_text': best_match['question_text_en'],
                    'choices': q['choices'],
                    'correct_answer': best_match['correct_answer'],
                    'explanation': best_match.get('explanation', ''),
                    'metacognition_feedback': best_match.get('metacognition_feedback', ''),
                    'metacognition_memo': best_match.get('metacognition_memo', ''),
                    'match_score': best_score
                }
            else:
                passage_entry['questions'][question_id] = {
                    'question_id': question_id,
                    'question_index': q['question_index'],
                    'coord_question_text': coord_question_text,
                    'excel_key': None,
                    'toeic_q': None,
                    'match_score': best_score,
                    'warning': 'No match found or low similarity'
                }

        mapping[passage_id] = passage_entry

    return mapping


def main():
    """メイン処理"""
    print("Loading coordinate files...")
    coord_data = load_coordinate_files()
    print(f"  Found {len(coord_data)} passages with questions")

    print("Loading Excel data...")
    excel_data = load_excel_data()
    print(f"  Found {len(excel_data)} questions in Excel")

    print("Matching questions...")
    mapping = match_questions(excel_data, coord_data)

    # 出力ディレクトリ作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # passage_idをソート（pre_01, pre_02, ..., post_01, ..., tr_01, tr_01_an1, ...）
    def sort_key(passage_id: str):
        # フェーズの優先順位
        phase_order = {'pre': 0, 'post': 1, 'tr': 2}
        parts = passage_id.split('_')
        phase = parts[0]
        phase_num = phase_order.get(phase, 99)

        # 番号を抽出（pre_01 → 1, tr_01_an2 → 1.2）
        try:
            num = int(parts[1])
        except (IndexError, ValueError):
            num = 0

        # 類題番号（an1=0.1, an2=0.2, an3=0.3, 本問=0）
        sub_num = 0
        if len(parts) >= 3 and parts[2].startswith('an'):
            try:
                sub_num = int(parts[2][2:]) * 0.1
            except ValueError:
                pass

        return (phase_num, num, sub_num)

    sorted_mapping = {k: mapping[k] for k in sorted(mapping.keys(), key=sort_key)}

    # JSON出力
    output_file = OUTPUT_DIR / "question_mapping.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_mapping, f, ensure_ascii=False, indent=2)

    print(f"\nOutput: {output_file}")

    # サマリー表示
    total_questions = 0
    matched_questions = 0
    low_match_questions = []

    for passage_id, entry in mapping.items():
        for q_id, q_info in entry['questions'].items():
            total_questions += 1
            if q_info.get('excel_key'):
                matched_questions += 1
                if q_info.get('match_score', 0) < 0.8:
                    low_match_questions.append({
                        'passage_id': passage_id,
                        'question_id': q_id,
                        'score': q_info.get('match_score', 0)
                    })
            else:
                low_match_questions.append({
                    'passage_id': passage_id,
                    'question_id': q_id,
                    'score': q_info.get('match_score', 0),
                    'warning': 'No match'
                })

    print("\nSummary:")
    print(f"  Total questions: {total_questions}")
    print(f"  Matched: {matched_questions}")
    print(f"  Unmatched/Low confidence: {len(low_match_questions)}")

    if low_match_questions:
        print("\nLow confidence matches:")
        for item in low_match_questions[:10]:
            print(f"  - {item['passage_id']}/{item['question_id']}: score={item['score']:.2f}")
        if len(low_match_questions) > 10:
            print(f"  ... and {len(low_match_questions) - 10} more")


if __name__ == '__main__':
    main()

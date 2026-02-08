#!/usr/bin/env python3
"""
根拠テキストと座標ファイルの照合スクリプト

evidence_mapping.json の根拠テキストを座標ファイルの sentence と照合し、
視線分析で使用できる座標情報を付与する。

処理フロー:
1. evidence_mapping.json から根拠テキストを取得
2. 対応する座標ファイルを読み込み
3. 根拠テキストと座標ファイルの sentence.text を照合
   - 完全一致 → そのsentenceの座標を使用
   - 部分一致 → 含まれるsentenceを全て取得
   - 不一致 → 類似度で最も近いsentenceを探す
4. マッチ結果を evidence_with_coordinates.json に出力

出力: data/working/evidence_with_coordinates.json
"""

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
WORKING_DIR = BASE_DIR / "data" / "working"
EVIDENCE_MAPPING_FILE = WORKING_DIR / "evidence_mapping.json"
QUESTION_MAPPING_FILE = WORKING_DIR / "question_mapping.json"
OUTPUT_FILE = WORKING_DIR / "evidence_with_coordinates.json"


@dataclass
class SentenceInfo:
    """座標ファイルから抽出した文の情報"""
    passage_index: int
    paragraph_index: int
    sentence_index: int
    text: str
    lines: list[dict]
    words: list[dict] = field(default_factory=list)
    source_type: str = "sentence"  # "sentence", "metadata", "table_header", "table_cell"


def normalize_text(text: str) -> str:
    """テキストを正規化（比較用）

    - 連続する空白を1つに
    - 前後の空白を除去
    - ダブルクォートの統一（"" → "")
    """
    if not text:
        return ""
    # 連続する空白を1つに
    text = re.sub(r'\s+', ' ', text)
    # 前後の空白を除去
    text = text.strip()
    # ダブルクォートの統一
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    return text


def calculate_similarity(text1: str, text2: str) -> float:
    """2つのテキストの類似度を計算（0-1）"""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio()


def is_substring_match(evidence_text: str, sentence_text: str) -> bool:
    """evidence_text が sentence_text に部分文字列として含まれるか判定"""
    e = normalize_text(evidence_text)
    s = normalize_text(sentence_text)
    return e in s


def is_sentence_in_evidence(sentence_text: str, evidence_text: str) -> bool:
    """sentence_text が evidence_text に部分文字列として含まれるか判定"""
    e = normalize_text(evidence_text)
    s = normalize_text(sentence_text)
    return s in e


def extract_sentences_from_coordinate(coord_data: dict) -> list[SentenceInfo]:
    """座標ファイルから全ての文・メタデータ・テーブルデータを抽出"""
    sentences = []

    coordinates = coord_data.get('coordinates', {})
    left_panel = coordinates.get('left_panel', {})
    passages = left_panel.get('passages', [])

    for passage in passages:
        passage_idx = passage.get('passage_index', 0)

        # 1. パラグラフ内のセンテンス
        paragraphs = passage.get('paragraphs', [])
        for paragraph in paragraphs:
            para_idx = paragraph.get('paragraph_index', 0)
            para_sentences = paragraph.get('sentences', [])

            for sent in para_sentences:
                sentences.append(SentenceInfo(
                    passage_index=passage_idx,
                    paragraph_index=para_idx,
                    sentence_index=sent.get('sentence_index', 0),
                    text=sent.get('text', ''),
                    lines=sent.get('lines', []),
                    words=sent.get('words', []),
                    source_type='sentence'
                ))

        # 2. タイトル・サブタイトル
        title = passage.get('title')
        if title:
            title_text = title.get('text', '')
            title_lines = title.get('lines', [])
            if title_text:
                sentences.append(SentenceInfo(
                    passage_index=passage_idx,
                    paragraph_index=-6,
                    sentence_index=0,
                    text=title_text,
                    lines=title_lines,
                    source_type='title'
                ))

        subtitle = passage.get('subtitle')
        if subtitle:
            subtitle_text = subtitle.get('text', '')
            subtitle_lines = subtitle.get('lines', [])
            if subtitle_text:
                sentences.append(SentenceInfo(
                    passage_index=passage_idx,
                    paragraph_index=-7,
                    sentence_index=0,
                    text=subtitle_text,
                    lines=subtitle_lines,
                    source_type='subtitle'
                ))

        # 4. メタデータ（ヘッダー情報など）
        metadata = passage.get('metadata', [])
        if metadata:
            for meta_idx, meta in enumerate(metadata):
                value_text = meta.get('value_text', '')
                value_lines = meta.get('value', [])
                if value_text:
                    sentences.append(SentenceInfo(
                        passage_index=passage_idx,
                        paragraph_index=-1,  # メタデータはパラグラフに属さない
                        sentence_index=meta_idx,
                        text=value_text,
                        lines=value_lines if isinstance(value_lines, list) else [],
                        source_type='metadata'
                    ))

        # 5. テーブルデータ
        table = passage.get('table')
        if table:
            # テーブルヘッダー
            headers = table.get('headers', [])
            for h_idx, header in enumerate(headers):
                header_text = header.get('text', '')
                if header_text:
                    # ヘッダーの座標はbboxまたはlines
                    header_lines = []
                    if 'bbox' in header:
                        header_lines = [header['bbox']]
                    elif 'lines' in header:
                        header_lines = header['lines']

                    sentences.append(SentenceInfo(
                        passage_index=passage_idx,
                        paragraph_index=-2,  # テーブルヘッダー
                        sentence_index=h_idx,
                        text=header_text,
                        lines=header_lines,
                        source_type='table_header'
                    ))

            # テーブル行のセル - 2つの形式に対応
            # 形式1: rows配列（各rowにcells配列）
            rows = table.get('rows', [])
            if rows:
                for row_idx, row in enumerate(rows):
                    cells = row.get('cells', [])
                    row_texts = []
                    row_lines = []

                    for cell_idx, cell in enumerate(cells):
                        cell_text = cell.get('text', '')
                        if cell_text:
                            row_texts.append(cell_text)
                            cell_lines = []
                            if 'bbox' in cell:
                                cell_lines = [cell['bbox']]
                            elif 'lines' in cell:
                                cell_lines = cell['lines']
                            row_lines.extend(cell_lines)

                            # 個別セルも追加
                            sentences.append(SentenceInfo(
                                passage_index=passage_idx,
                                paragraph_index=-3,  # テーブルセル
                                sentence_index=row_idx * 100 + cell_idx,
                                text=cell_text,
                                lines=cell_lines,
                                source_type='table_cell'
                            ))

                    # 行全体を連結
                    if row_texts:
                        _add_table_row_entries(sentences, passage_idx, row_idx, row_texts, row_lines)

            # 形式2: cells配列（フラット、row_indexとcell_indexで識別）
            cells = table.get('cells', [])
            if cells and not rows:
                # row_indexでグループ化
                rows_dict: dict[int, list] = {}
                for cell in cells:
                    row_idx = cell.get('row_index', 0)
                    if row_idx not in rows_dict:
                        rows_dict[row_idx] = []
                    rows_dict[row_idx].append(cell)

                # 各行を処理
                for row_idx in sorted(rows_dict.keys()):
                    row_cells = sorted(rows_dict[row_idx], key=lambda c: c.get('cell_index', 0))
                    row_texts = []
                    row_lines = []

                    for cell in row_cells:
                        cell_text = cell.get('text', '')
                        cell_idx = cell.get('cell_index', 0)
                        if cell_text:
                            row_texts.append(cell_text)
                            cell_lines = []
                            if 'bbox' in cell:
                                cell_lines = [cell['bbox']]
                            elif 'lines' in cell:
                                cell_lines = cell['lines']
                            row_lines.extend(cell_lines)

                            # 個別セルも追加
                            sentences.append(SentenceInfo(
                                passage_index=passage_idx,
                                paragraph_index=-3,
                                sentence_index=row_idx * 100 + cell_idx,
                                text=cell_text,
                                lines=cell_lines,
                                source_type='table_cell'
                            ))

                    # 行全体を連結
                    if row_texts:
                        _add_table_row_entries(sentences, passage_idx, row_idx, row_texts, row_lines)

    return sentences


def _add_table_row_entries(
    sentences: list[SentenceInfo],
    passage_idx: int,
    row_idx: int,
    row_texts: list[str],
    row_lines: list[dict]
) -> None:
    """テーブル行のエントリを追加（複数の区切り形式）"""
    # スラッシュ区切り
    row_text_slash = ' / '.join(row_texts)
    sentences.append(SentenceInfo(
        passage_index=passage_idx,
        paragraph_index=-4,
        sentence_index=row_idx,
        text=row_text_slash,
        lines=row_lines,
        source_type='table_row'
    ))
    # スペース区切り
    row_text_space = '  '.join(row_texts)
    sentences.append(SentenceInfo(
        passage_index=passage_idx,
        paragraph_index=-5,
        sentence_index=row_idx,
        text=row_text_space,
        lines=row_lines,
        source_type='table_row'
    ))


def match_evidence_to_sentences(
    evidence_text: str,
    evidence_passage_index: int,
    sentences: list[SentenceInfo],
    similarity_threshold: float = 0.7
) -> list[dict]:
    """根拠テキストを座標ファイルの文と照合

    Returns:
        マッチした文のリスト。各要素は以下を含む:
        - passage_index, paragraph_index, sentence_index
        - text: 座標ファイルの文テキスト
        - lines: 座標情報
        - match_type: "exact" | "partial" | "contains" | "fuzzy"
        - similarity: 類似度スコア
    """
    matches = []
    evidence_norm = normalize_text(evidence_text)

    # まず同じパッセージ内の文のみを対象にする
    same_passage_sentences = [s for s in sentences if s.passage_index == evidence_passage_index]

    # もし同じパッセージ内に文がなければ、全ての文を対象にする
    target_sentences = same_passage_sentences if same_passage_sentences else sentences

    for sent in target_sentences:
        sent_norm = normalize_text(sent.text)

        # 完全一致
        if evidence_norm == sent_norm:
            matches.append({
                'passage_index': sent.passage_index,
                'paragraph_index': sent.paragraph_index,
                'sentence_index': sent.sentence_index,
                'text': sent.text,
                'lines': sent.lines,
                'source_type': sent.source_type,
                'match_type': 'exact',
                'similarity': 1.0
            })
            continue

        # 根拠テキストが座標の文に含まれる（座標の文の方が長い）
        if is_substring_match(evidence_text, sent.text):
            similarity = len(evidence_norm) / len(sent_norm) if sent_norm else 0
            matches.append({
                'passage_index': sent.passage_index,
                'paragraph_index': sent.paragraph_index,
                'sentence_index': sent.sentence_index,
                'text': sent.text,
                'lines': sent.lines,
                'source_type': sent.source_type,
                'match_type': 'partial',
                'similarity': similarity
            })
            continue

        # 座標の文が根拠テキストに含まれる（根拠テキストの方が長い）
        if is_sentence_in_evidence(sent.text, evidence_text):
            matches.append({
                'passage_index': sent.passage_index,
                'paragraph_index': sent.paragraph_index,
                'sentence_index': sent.sentence_index,
                'text': sent.text,
                'lines': sent.lines,
                'source_type': sent.source_type,
                'match_type': 'contains',
                'similarity': 1.0
            })
            continue

    # 完全一致・部分一致がなければ、類似度で検索
    if not matches:
        best_match = None
        best_similarity = 0.0

        for sent in target_sentences:
            similarity = calculate_similarity(evidence_text, sent.text)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = sent

        if best_match and best_similarity >= similarity_threshold:
            matches.append({
                'passage_index': best_match.passage_index,
                'paragraph_index': best_match.paragraph_index,
                'sentence_index': best_match.sentence_index,
                'text': best_match.text,
                'lines': best_match.lines,
                'source_type': best_match.source_type,
                'match_type': 'fuzzy',
                'similarity': best_similarity
            })

    return matches


def load_coordinate_file(coord_path: str) -> dict | None:
    """座標ファイルを読み込み"""
    path = Path(coord_path)
    if not path.exists():
        return None

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """メイン処理"""
    print("Loading evidence mapping...")
    with open(EVIDENCE_MAPPING_FILE, 'r', encoding='utf-8') as f:
        evidence_mapping = json.load(f)

    print("Loading question mapping...")
    with open(QUESTION_MAPPING_FILE, 'r', encoding='utf-8') as f:
        question_mapping = json.load(f)

    output = {}
    stats = {
        'total_evidence': 0,
        'exact_match': 0,
        'partial_match': 0,
        'contains_match': 0,
        'fuzzy_match': 0,
        'no_match': 0,
        'coord_file_missing': 0,
        'error': 0
    }

    for passage_id, evidence_entry in evidence_mapping.items():
        print(f"\nProcessing {passage_id}...")

        # 座標ファイルのパスを取得
        q_entry = question_mapping.get(passage_id, {})
        coord_path = q_entry.get('coordinate_file')

        if not coord_path:
            print(f"  Warning: No coordinate file for {passage_id}")
            output[passage_id] = {
                'passage_id': passage_id,
                'phase': evidence_entry.get('phase'),
                'error': 'No coordinate file path',
                'questions': {}
            }
            continue

        # 座標ファイルを読み込み
        coord_data = load_coordinate_file(coord_path)
        if not coord_data:
            print(f"  Warning: Coordinate file not found: {coord_path}")
            stats['coord_file_missing'] += 1
            output[passage_id] = {
                'passage_id': passage_id,
                'phase': evidence_entry.get('phase'),
                'coordinate_file': coord_path,
                'error': 'Coordinate file not found',
                'questions': {}
            }
            continue

        # 座標ファイルから文を抽出
        sentences = extract_sentences_from_coordinate(coord_data)
        print(f"  Found {len(sentences)} sentences in coordinate file")

        output[passage_id] = {
            'passage_id': passage_id,
            'phase': evidence_entry.get('phase'),
            'coordinate_file': coord_path,
            'questions': {}
        }

        # 各設問の根拠を処理
        for q_id, q_info in evidence_entry.get('questions', {}).items():
            evidence_sentences = q_info.get('evidence_sentences', [])

            if 'error' in q_info:
                output[passage_id]['questions'][q_id] = {
                    'question_text': q_info.get('question_text', ''),
                    'correct_answer': q_info.get('correct_answer', ''),
                    'error': q_info['error'],
                    'evidence_sentences': []
                }
                continue

            matched_evidence = []

            for ev in evidence_sentences:
                stats['total_evidence'] += 1
                ev_text = ev.get('text', '')
                ev_passage_idx = ev.get('passage_index', 0)
                ev_type = ev.get('type', 'primary')
                ev_reason = ev.get('reason', '')

                # 照合
                matches = match_evidence_to_sentences(
                    evidence_text=ev_text,
                    evidence_passage_index=ev_passage_idx,
                    sentences=sentences
                )

                if matches:
                    # 統計更新
                    for m in matches:
                        match_type = m.get('match_type', 'unknown')
                        if match_type == 'exact':
                            stats['exact_match'] += 1
                        elif match_type == 'partial':
                            stats['partial_match'] += 1
                        elif match_type == 'contains':
                            stats['contains_match'] += 1
                        elif match_type == 'fuzzy':
                            stats['fuzzy_match'] += 1

                    matched_evidence.append({
                        'original_text': ev_text,
                        'type': ev_type,
                        'reason': ev_reason,
                        'matched_sentences': matches
                    })
                else:
                    stats['no_match'] += 1
                    matched_evidence.append({
                        'original_text': ev_text,
                        'type': ev_type,
                        'reason': ev_reason,
                        'matched_sentences': [],
                        'warning': 'No match found'
                    })

            output[passage_id]['questions'][q_id] = {
                'question_text': q_info.get('question_text', ''),
                'correct_answer': q_info.get('correct_answer', ''),
                'toeic_q': q_info.get('toeic_q'),
                'reasoning': q_info.get('reasoning', ''),
                'evidence_sentences': matched_evidence
            }

    # 出力
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"\n{'='*50}")
    print("Statistics:")
    print(f"  Total evidence sentences: {stats['total_evidence']}")
    print(f"  Exact match: {stats['exact_match']}")
    print(f"  Partial match: {stats['partial_match']}")
    print(f"  Contains match: {stats['contains_match']}")
    print(f"  Fuzzy match: {stats['fuzzy_match']}")
    print(f"  No match: {stats['no_match']}")
    print(f"  Coordinate file missing: {stats['coord_file_missing']}")

    # マッチ率を計算
    total_matched = stats['exact_match'] + stats['partial_match'] + stats['contains_match'] + stats['fuzzy_match']
    if stats['total_evidence'] > 0:
        match_rate = total_matched / stats['total_evidence'] * 100
        print(f"\n  Match rate: {match_rate:.1f}% ({total_matched}/{stats['total_evidence']})")

        # 完全/部分一致率
        good_match = stats['exact_match'] + stats['partial_match'] + stats['contains_match']
        good_rate = good_match / stats['total_evidence'] * 100
        print(f"  Good match rate (exact/partial/contains): {good_rate:.1f}% ({good_match}/{stats['total_evidence']})")


if __name__ == '__main__':
    main()

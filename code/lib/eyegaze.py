import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from scipy.stats import multivariate_normal
from dataclasses import dataclass, field
from typing import List, Dict


# =============================================================================
# フェーズ設定
# =============================================================================

@dataclass
class PhaseConfig:
    """実験フェーズの設定"""
    phase_name: str                      # "pre", "training1", etc.
    phase_type: str                      # "simple" (単一イベント) or "multi" (複数イベント)
    segment_event_types: List[str]       # セグメント境界となるイベントタイプ
    image_mapping: Dict[str, str] = field(default_factory=dict)  # event_type -> image_number
    use_formula_mapping: bool = True     # True: 式計算, False: dict参照
    image_offset: int = 2                # 式計算時のオフセット (passage番号 + offset)
    extract_event_type: bool = False     # セグメントにevent_typeを含めるか
    extract_analog_id: bool = False      # セグメントにanalog_idを含めるか


# Training フェーズ用の画像マッピング
_TRAINING_IMAGE_MAPPING = {
    'phase_intro_enter': '002',        # intro画面
    'question_screen_open': '003',     # 問題画面
    'reflection1_open': '004',         # reflection1画面
    'training_explanation_open': '005', # 解説
    'analog_intro_enter': '006',       # 類題intro画面
    'analog_question_open_an1': '007', # 類題1
    'analog_explanation_open_an1': '008', # 類題1解説
    'analog_question_open_an2': '009', # 類題2
    'analog_explanation_open_an2': '010', # 類題2解説
    'analog_question_open_an3': '011', # 類題3
    'analog_explanation_open_an3': '012', # 類題3解説
    'reflection2_open': '013',         # reflection2画面
    'phase_complete_enter': '014',     # complete画面
}

# フェーズ設定辞書
PHASE_CONFIGS = {
    "pre": PhaseConfig(
        phase_name="pre",
        phase_type="simple",
        segment_event_types=["question_screen_open"],
        image_mapping={},
        use_formula_mapping=True,
        image_offset=2,
        extract_event_type=False,
        extract_analog_id=False,
    ),
    "post": PhaseConfig(
        phase_name="post",
        phase_type="simple",
        segment_event_types=["question_screen_open"],
        image_mapping={},
        use_formula_mapping=True,
        image_offset=2,
        extract_event_type=False,
        extract_analog_id=False,
    ),
    "training1": PhaseConfig(
        phase_name="training1",
        phase_type="multi",
        segment_event_types=[
            'phase_intro_enter',
            'question_screen_open',
            'reflection1_open',
            'training_explanation_open',
            'analog_intro_enter',
            'analog_question_open',
            'analog_explanation_open',
            'reflection2_open',
            'phase_complete_enter'
        ],
        image_mapping=_TRAINING_IMAGE_MAPPING,
        use_formula_mapping=False,
        extract_event_type=True,
        extract_analog_id=True,
    ),
    "training2": PhaseConfig(
        phase_name="training2",
        phase_type="multi",
        segment_event_types=[
            'phase_intro_enter',
            'question_screen_open',
            'reflection1_open',
            'training_explanation_open',
            'analog_intro_enter',
            'analog_question_open',
            'analog_explanation_open',
            'reflection2_open',
            'phase_complete_enter'
        ],
        image_mapping=_TRAINING_IMAGE_MAPPING,
        use_formula_mapping=False,
        extract_event_type=True,
        extract_analog_id=True,
    ),
    "training3": PhaseConfig(
        phase_name="training3",
        phase_type="multi",
        segment_event_types=[
            'phase_intro_enter',
            'question_screen_open',
            'reflection1_open',
            'training_explanation_open',
            'analog_intro_enter',
            'analog_question_open',
            'analog_explanation_open',
            'reflection2_open',
            'phase_complete_enter'
        ],
        image_mapping=_TRAINING_IMAGE_MAPPING,
        use_formula_mapping=False,
        extract_event_type=True,
        extract_analog_id=True,
    ),
}


# =============================================================================
# Fixation検出
# =============================================================================


def isMinimumFixation(X, Y, mfx):
    if max([max(X) - min(X), max(Y) - min(Y)]) < mfx:
        return True
    return False


def detectFixations(
        times, X, Y, P=None,
        min_concat_gaze_count=9,
        min_fixation_size=50,
        max_fixation_size=80):
    """
    視線データからfixation（注視点）を検出する

    Parameters:
    -----------
    times : array-like
        タイムスタンプ配列
    X, Y : array-like
        視線座標配列
    P : array-like, optional
        瞳孔径配列（Noneの場合は出力に含まない）

    Returns:
    --------
    np.array
        各行: [timestamp, x, y, duration, saccade_length, saccade_angle, saccade_speed, pupil_diameter]
        - timestamp: fixation開始時刻
        - x, y: fixation位置（平均座標）
        - duration: fixation継続時間
        - saccade_length: 前のfixationからの距離（ピクセル）
        - saccade_angle: 前のfixationからの角度（度、0°=右、90°=上、-90°=下）
        - saccade_speed: saccade速度（ピクセル/秒）
        - pupil_diameter: fixation中の平均瞳孔径（Pが指定された場合のみ）
        ※最初のfixationはsaccade情報が0
    """

    fixations = []
    last_fixation = None  # [timestamp, x, y, duration]
    i = 0
    j = 0
    while max([i, j]) < len(times)-min_concat_gaze_count:
        X_ = list(X[i:i+min_concat_gaze_count])
        Y_ = list(Y[i:i+min_concat_gaze_count])
        P_ = list(P[i:i+min_concat_gaze_count]) if P is not None else None
        if isMinimumFixation(X_, Y_, min_fixation_size):
            begin = times[i]
            j = i + min_concat_gaze_count
            end = times[j - 1]  # 初期値: 最初のfixation範囲の最後
            c = 0
            while(c < min_concat_gaze_count and j < len(times)):
                X_.append(X[j])
                Y_.append(Y[j])
                if P_ is not None:
                    P_.append(P[j])
                if max([max(X_) - min(X_), max(Y_) - min(Y_)]) > max_fixation_size:
                    # X[j]Y[j] is out of max_fixation_size
                    if c == 0:
                        # X[j]Y[j] will be next minimum fixation
                        i = j
                    X_.pop()
                    Y_.pop()
                    if P_ is not None:
                        P_.pop()
                    c += 1
                else:
                    c = 0
                    end = times[j]  # 範囲内の時だけendを更新
                j += 1

            # fixation情報
            fx = np.mean(X_)
            fy = np.mean(Y_)
            fp = np.mean(P_) if P_ is not None else None
            duration = end - begin

            # saccade情報（前のfixationとの関係）
            if last_fixation is not None:
                # 前のfixation終了時刻から現在のfixation開始時刻までの時間
                delta_t = begin - (last_fixation[0] + last_fixation[3])
                delta_x = fx - last_fixation[1]
                delta_y = fy - last_fixation[2]
                saccade_length = math.sqrt(delta_x * delta_x + delta_y * delta_y)
                saccade_angle = math.degrees(math.atan2(delta_y, delta_x))
                saccade_speed = saccade_length / delta_t if delta_t > 0 else 0
            else:
                # 最初のfixationはsaccade情報なし
                saccade_length = 0
                saccade_angle = 0
                saccade_speed = 0

            if fp is not None:
                fixations.append([begin, fx, fy, duration, saccade_length, saccade_angle, saccade_speed, fp])
            else:
                fixations.append([begin, fx, fy, duration, saccade_length, saccade_angle, saccade_speed])
            last_fixation = [begin, fx, fy, duration]
            i = i - 1
            j = j - 1

        i += 1
    return np.array(fixations)


def plotScanPath(
        X, Y, durations, figsize=(30, 15),
        bg_image="", save_path="", halfPage=False,
        duration_scale=1):
    """
    スキャンパスを描画する

    Parameters:
    -----------
    duration_scale : float
        durationのスケーリング係数（デフォルト: 1）
        fixationデータ（秒単位）の場合は1000を指定すると適切な円サイズになる
    """
    plt.figure(figsize=figsize)
    if bg_image != "":
        img = mpimg.imread(bg_image)
        plt.imshow(img)
        if halfPage:
            plt.xlim(150, 1000)
        else:
            plt.xlim(0, len(img[0]))
        plt.ylim(len(img), 0)
    scale = float(figsize[0]) / 40.0

    # durationをスケーリング（秒→描画用サイズ）
    scaled_durations = durations * duration_scale

    plt.plot(X, Y, "-", c="blue", linewidth=scale, zorder=1, alpha=0.8)
    plt.scatter(X, Y, scaled_durations*scale, c="b", zorder=2, alpha=0.3)
    plt.scatter(X[0], Y[0], scaled_durations[0]*scale, c="g", zorder=2, alpha=0.6)
    plt.scatter(X[-1], Y[-1], scaled_durations[-1]*scale, c="r", zorder=2, alpha=0.6)

    if save_path != "":
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        plt.close()


def plotHeatmap(
        X, Y, durations, figsize=(30, 15),
        bg_image="", save_path="", data_save_path=""):
    plt.figure(figsize=figsize)
    if bg_image != "":
        img = mpimg.imread(bg_image)
        plt.imshow(img)
        plt.xlim(0, len(img[0]))
        plt.ylim(len(img), 0)

    gx, gy = np.meshgrid(np.arange(0, len(img[0])), np.arange(0, len(img)))
    values = np.zeros((len(img), len(img[0])))
    for i in range(len(X)):
        pos = np.dstack([gx, gy])
        rv = multivariate_normal([X[i], Y[i]], [[50**2, 0], [0, 50**2]])
        values += rv.pdf(pos) * durations[i] / 2.0
    values = values/np.max(values)

    masked = np.ma.masked_where(values < 0.05, values)
    cmap = cm.jet
    cmap.set_bad('white', 1.)
    plt.imshow(masked, alpha=0.4, cmap=cmap)

    if save_path != "":
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        plt.close()

    if data_save_path != "":
        np.savetxt(data_save_path, values, delimiter=",", fmt="%f")


def plotGazeCorrectionComparison(original_data, corrected_data, bg_image="",
                                  figsize=(24, 12), title=""):
    """
    補正前後の視線データを並べて比較表示する

    Parameters:
    -----------
    original_data : np.array
        補正前データ [[timestamp, gaze_x, gaze_y, pupil_diameter], ...]
    corrected_data : np.array
        補正後データ [[timestamp, gaze_x, gaze_y, pupil_diameter], ...]
    bg_image : str
        背景画像パス
    figsize : tuple
        図のサイズ
    title : str
        図のタイトル
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, data, label in [(axes[0], original_data, "Original"),
                            (axes[1], corrected_data, "Corrected")]:
        if bg_image != "":
            img = mpimg.imread(bg_image)
            ax.imshow(img)
            ax.set_xlim(0, len(img[0]))
            ax.set_ylim(len(img), 0)

        X = data[:, 1]
        Y = data[:, 2]

        ax.plot(X, Y, "-", c="blue", linewidth=0.5, alpha=0.5)
        ax.scatter(X, Y, s=5, c="blue", alpha=0.3)
        ax.scatter(X[0], Y[0], s=50, c="green", alpha=0.8, label="Start")
        ax.scatter(X[-1], Y[-1], s=50, c="red", alpha=0.8, label="End")
        ax.set_title(label)
        ax.legend()

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    return fig


def readEventLog(jsonl_path, event_type="question_screen_open"):
    """
    JSONLイベントログから特定イベントのタイムスタンプとメタデータを抽出

    Parameters:
    -----------
    jsonl_path : str
        イベントログファイルのパス (.jsonl)
    event_type : str
        抽出するイベントタイプ (デフォルト: "question_screen_open")

    Returns:
    --------
    list of dict
        各イベントの情報 [{"timestamp": float, "passage_id": str, ...}, ...]
        タイムスタンプはローカルタイム基準のUnixエポック（秒単位）に変換済み
    """
    import json
    from datetime import datetime

    events = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get('event') == event_type:
                # ISO 8601 (UTC) -> ローカルタイム基準のUnixタイムスタンプ
                iso_timestamp = data['timestamp']
                dt_utc = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
                # ローカルタイムに変換
                dt_local = dt_utc.astimezone()
                # タイムスタンプ（ローカルタイムとして解釈）
                unix_timestamp = dt_local.timestamp()

                events.append({
                    'timestamp': unix_timestamp,
                    'passage_id': data.get('passage_id'),
                    'raw_data': data
                })

    return events


def correctGazeForHeadPosition(df,
                               correction_factor_x=500.0,
                               correction_factor_y=200.0,
                               calibration_center_x=0.5,
                               calibration_center_y=0.5):
    """
    頭部位置のずれに基づいて視線座標を補正する（trackbox座標版）

    Parameters:
    -----------
    df : pandas.DataFrame
        視線データ（gaze_x, gaze_y, trackbox座標、validityカラムを含む）
    correction_factor_x : float
        X軸補正係数（ピクセル/trackbox単位）。デフォルト: 500.0
    correction_factor_y : float
        Y軸補正係数（ピクセル/trackbox単位）。デフォルト: 200.0
    calibration_center_x : float
        キャリブレーション時の頭部X位置（trackbox座標）。デフォルト: 0.5
    calibration_center_y : float
        キャリブレーション時の頭部Y位置（trackbox座標）。デフォルト: 0.5

    Returns:
    --------
    pandas.DataFrame
        補正済み座標（corrected_gaze_x, corrected_gaze_y）を追加したDataFrame
    """
    df = df.copy()

    # trackbox座標カラム名（pandasが重複カラムを自動リネーム）
    left_trackbox_x_col = 'left_gaze_origin_in_trackbox_coordinate_system'
    left_trackbox_y_col = 'left_gaze_origin_in_trackbox_coordinate_system.1'
    right_trackbox_x_col = 'right_gaze_origin_in_trackbox_coordinate_system'
    right_trackbox_y_col = 'right_gaze_origin_in_trackbox_coordinate_system.1'

    # 両目の平均trackbox座標を計算
    left_valid = df['left_gaze_origin_validity'] == 1
    right_valid = df['right_gaze_origin_validity'] == 1
    both_valid = left_valid & right_valid

    # 頭部位置オフセットを計算
    head_x = np.full(len(df), np.nan)
    head_y = np.full(len(df), np.nan)

    # 両目有効: 平均を使用
    head_x[both_valid] = (df.loc[both_valid, left_trackbox_x_col].values +
                          df.loc[both_valid, right_trackbox_x_col].values) / 2
    head_y[both_valid] = (df.loc[both_valid, left_trackbox_y_col].values +
                          df.loc[both_valid, right_trackbox_y_col].values) / 2

    # 左目のみ有効
    left_only = left_valid & ~right_valid
    head_x[left_only] = df.loc[left_only, left_trackbox_x_col].values
    head_y[left_only] = df.loc[left_only, left_trackbox_y_col].values

    # 右目のみ有効
    right_only = right_valid & ~left_valid
    head_x[right_only] = df.loc[right_only, right_trackbox_x_col].values
    head_y[right_only] = df.loc[right_only, right_trackbox_y_col].values

    # オフセット計算（キャリブレーション中央からのずれ）
    head_offset_x = head_x - calibration_center_x
    head_offset_y = head_y - calibration_center_y

    # 補正適用
    # X軸: trackbox_x > 0.5（右寄り）= 頭が左 = UCS X負 → 逆の関係なので「-」で補正
    # Y軸: trackbox_y > 0.5（下寄り）= 頭が下 = UCS Y負 → 逆の関係だが画面Yも逆なので「+」で補正
    df['head_offset_x'] = head_offset_x
    df['head_offset_y'] = head_offset_y
    df['corrected_gaze_x'] = df['gaze_x'] - correction_factor_x * head_offset_x
    df['corrected_gaze_y'] = df['gaze_y'] + correction_factor_y * head_offset_y  # Y軸は+

    # NaN（無効データ）の場合は元の値を使用
    nan_mask = np.isnan(head_offset_x)
    df.loc[nan_mask, 'corrected_gaze_x'] = df.loc[nan_mask, 'gaze_x']
    df.loc[nan_mask, 'corrected_gaze_y'] = df.loc[nan_mask, 'gaze_y']

    return df


def correctGazeGeometric(df,
                         calibration_head_x=0.0,
                         calibration_head_y=0.0,
                         use_average_reference=False,
                         correct_y=False,
                         screen_width_mm=509.0,
                         screen_height_mm=287.0,
                         screen_width_px=1920,
                         screen_height_px=1080):
    """
    mm単位の頭部位置を使った幾何学的視差補正

    頭部位置（mm）と画面からの距離を考慮して、各サンプルごとに
    視差による視線ずれを補正します。

    Parameters:
    -----------
    df : pandas.DataFrame
        視線データ（全カラム含む）
    calibration_head_x : float
        キャリブレーション時の頭部X位置（mm）。デフォルト: 0.0（画面中央）
    calibration_head_y : float
        キャリブレーション時の頭部Y位置（mm）。デフォルト: 0.0
    use_average_reference : bool
        Trueの場合、calibration_head_x/yを無視してデータの平均頭部位置を基準にする。
        デフォルト: False
    correct_y : bool
        Y軸も補正するか。デフォルト: False（X軸のみ補正）
    screen_width_mm : float
        画面の物理幅（mm）。デフォルト: 509.0（23インチモニタ相当）
    screen_height_mm : float
        画面の物理高さ（mm）。デフォルト: 287.0
    screen_width_px : int
        画面の横解像度。デフォルト: 1920
    screen_height_px : int
        画面の縦解像度。デフォルト: 1080

    Returns:
    --------
    pandas.DataFrame
        補正済み座標（corrected_gaze_x, corrected_gaze_y）を追加したDataFrame
    """
    df = df.copy()

    # mm単位カラム名（pandasが重複カラムを自動リネーム）
    # left_gaze_origin_in_user_coordinate_system: x, .1: y, .2: z
    left_origin_x_col = 'left_gaze_origin_in_user_coordinate_system'
    left_origin_y_col = 'left_gaze_origin_in_user_coordinate_system.1'
    left_origin_z_col = 'left_gaze_origin_in_user_coordinate_system.2'
    right_origin_x_col = 'right_gaze_origin_in_user_coordinate_system'
    right_origin_y_col = 'right_gaze_origin_in_user_coordinate_system.1'
    right_origin_z_col = 'right_gaze_origin_in_user_coordinate_system.2'

    left_valid = df['left_gaze_origin_validity'] == 1
    right_valid = df['right_gaze_origin_validity'] == 1
    both_valid = left_valid & right_valid

    # 頭部位置（mm）を取得
    head_x_mm = np.full(len(df), np.nan)
    head_y_mm = np.full(len(df), np.nan)
    head_z_mm = np.full(len(df), np.nan)

    # 両目有効: 平均を使用
    head_x_mm[both_valid] = (df.loc[both_valid, left_origin_x_col].values +
                             df.loc[both_valid, right_origin_x_col].values) / 2
    head_y_mm[both_valid] = (df.loc[both_valid, left_origin_y_col].values +
                             df.loc[both_valid, right_origin_y_col].values) / 2
    head_z_mm[both_valid] = (df.loc[both_valid, left_origin_z_col].values +
                             df.loc[both_valid, right_origin_z_col].values) / 2

    # 左目のみ有効
    left_only = left_valid & ~right_valid
    head_x_mm[left_only] = df.loc[left_only, left_origin_x_col].values
    head_y_mm[left_only] = df.loc[left_only, left_origin_y_col].values
    head_z_mm[left_only] = df.loc[left_only, left_origin_z_col].values

    # 右目のみ有効
    right_only = right_valid & ~left_valid
    head_x_mm[right_only] = df.loc[right_only, right_origin_x_col].values
    head_y_mm[right_only] = df.loc[right_only, right_origin_y_col].values
    head_z_mm[right_only] = df.loc[right_only, right_origin_z_col].values

    # 基準位置の決定
    if use_average_reference:
        # データの平均頭部位置を基準にする（変動分のみ補正）
        ref_x = np.nanmean(head_x_mm)
        ref_y = np.nanmean(head_y_mm)
    else:
        ref_x = calibration_head_x
        ref_y = calibration_head_y

    # 頭部移動量（mm）
    delta_x_mm = head_x_mm - ref_x
    delta_y_mm = head_y_mm - ref_y

    # ピクセル/mm変換係数
    px_per_mm_x = screen_width_px / screen_width_mm
    px_per_mm_y = screen_height_px / screen_height_mm

    # 幾何学的視差補正
    # 頭が左に移動（delta_x < 0）すると、視線は右にずれて見える
    # 補正: 視線を左に移動させる = gaze_x を減らす
    # 視差量は head_z（距離）に反比例する
    # 基準距離を550mmとして、それより近いと補正量が増え、遠いと減る
    reference_z = 550.0  # キャリブレーション時の基準距離（mm）

    # 距離比率（近いほど大きい補正）
    z_ratio = reference_z / np.where(head_z_mm > 0, head_z_mm, reference_z)

    # 補正量（ピクセル）
    # 頭が右（delta_x > 0）→ Tobiiの計算した視線は実際より左にずれる → 補正で右に戻す
    # 頭が左（delta_x < 0）→ Tobiiの計算した視線は実際より右にずれる → 補正で左に戻す
    # X軸: Tobiiと画面は同じ方向（右が正）なので、delta_xと同じ方向に補正
    correction_x_px = delta_x_mm * px_per_mm_x * z_ratio

    # 補正適用
    df['head_x_mm'] = head_x_mm
    df['head_y_mm'] = head_y_mm
    df['head_z_mm'] = head_z_mm
    df['correction_x_px'] = correction_x_px
    df['corrected_gaze_x'] = df['gaze_x'] + correction_x_px  # delta_xと同じ方向

    # Y軸補正（オプション）
    if correct_y:
        correction_y_px = delta_y_mm * px_per_mm_y * z_ratio
        df['correction_y_px'] = correction_y_px
        # Y軸は座標系が反転（Tobii: 上が正、画面: 下が正）なので逆方向に補正
        df['corrected_gaze_y'] = df['gaze_y'] - correction_y_px  # delta_yと逆方向
    else:
        df['corrected_gaze_y'] = df['gaze_y']  # Y補正なし

    # NaN（無効データ）の場合は元の値を使用
    nan_mask = np.isnan(head_x_mm)
    df.loc[nan_mask, 'corrected_gaze_x'] = df.loc[nan_mask, 'gaze_x']
    df.loc[nan_mask, 'corrected_gaze_y'] = df.loc[nan_mask, 'gaze_y']

    return df


def segmentGazeDataByEvents(gaze_csv_path, events, end_timestamp=None,
                            apply_head_correction=False,
                            correction_method="geometric",
                            # geometric方式用
                            use_average_reference=False,
                            correct_y=False,
                            calibration_head_x=0.0,
                            calibration_head_y=0.0,
                            screen_width_mm=509.0,
                            screen_height_mm=287.0,
                            screen_width_px=1920,
                            screen_height_px=1080,
                            # trackbox方式用
                            correction_factor_x=500.0,
                            correction_factor_y=200.0,
                            calibration_center_x=0.5,
                            calibration_center_y=0.5):
    """
    視線データをイベントログに基づいてセグメント化

    Parameters:
    -----------
    gaze_csv_path : str
        tobii_pro_gaze.csvのパス
    events : list of dict
        readEventLog()の戻り値
    end_timestamp : float, optional
        最後のセグメントの終了タイムスタンプ（例: phase_complete_enterの時刻）
        指定しない場合は視線データの最後まで
    apply_head_correction : bool
        頭部位置補正を適用するか（デフォルト: False）
    correction_method : str
        補正方式: "geometric"（mm単位、距離考慮）または "trackbox"（従来方式）
    use_average_reference : bool
        平均頭部位置を基準にするか（geometric方式用）。デフォルト: False
    correct_y : bool
        Y軸も補正するか（geometric方式用）。デフォルト: False
    --- geometric方式用 ---
    calibration_head_x : float
        キャリブレーション時の頭部X位置（mm）。デフォルト: 0.0
    calibration_head_y : float
        キャリブレーション時の頭部Y位置（mm）。デフォルト: 0.0
    screen_width_mm : float
        画面の物理幅（mm）。デフォルト: 509.0
    screen_height_mm : float
        画面の物理高さ（mm）。デフォルト: 287.0
    screen_width_px : int
        画面の横解像度。デフォルト: 1920
    screen_height_px : int
        画面の縦解像度。デフォルト: 1080

    --- trackbox方式用 ---
    correction_factor_x : float
        X軸補正係数。デフォルト: 500.0
    correction_factor_y : float
        Y軸補正係数。デフォルト: 200.0
    calibration_center_x : float
        キャリブレーション時の頭部X位置（trackbox座標）。デフォルト: 0.5
    calibration_center_y : float
        キャリブレーション時の頭部Y位置（trackbox座標）。デフォルト: 0.5

    Returns:
    --------
    list of dict
        各セグメントのデータ
        [{"passage_id": "pre_01",
          "data": np.array([[timestamp, gaze_x, gaze_y, pupil_diameter], ...]),
          "start_time": float,
          "end_time": float
        }, ...]
    """
    import pandas as pd

    # 読み込むカラムを決定
    base_cols = ['#timestamp', 'gaze_x', 'gaze_y', 'pupil_diameter']
    if apply_head_correction:
        # 全カラム読み込み（重複カラム名があるため、usecolsは使用しない）
        df = pd.read_csv(gaze_csv_path)
    else:
        df = pd.read_csv(gaze_csv_path, usecols=base_cols)

    # NaN行を削除（基本カラムのみでチェック）
    df = df.dropna(subset=['gaze_x', 'gaze_y', 'pupil_diameter'])

    # タイムスタンプをミリ秒→秒に変換し、9時間（32400秒）を足す
    # （視線データのタイムスタンプはUTCだが、イベントログはローカルタイム基準のため）
    df['timestamp_sec'] = df['#timestamp'] * 0.001 + 32400

    # 頭部位置補正を適用
    if apply_head_correction:
        if correction_method == "geometric":
            df = correctGazeGeometric(df,
                                      calibration_head_x=calibration_head_x,
                                      calibration_head_y=calibration_head_y,
                                      use_average_reference=use_average_reference,
                                      correct_y=correct_y,
                                      screen_width_mm=screen_width_mm,
                                      screen_height_mm=screen_height_mm,
                                      screen_width_px=screen_width_px,
                                      screen_height_px=screen_height_px)
        else:  # trackbox
            df = correctGazeForHeadPosition(df,
                                            correction_factor_x=correction_factor_x,
                                            correction_factor_y=correction_factor_y,
                                            calibration_center_x=calibration_center_x,
                                            calibration_center_y=calibration_center_y)
        gaze_x_col = 'corrected_gaze_x'
        gaze_y_col = 'corrected_gaze_y'
    else:
        gaze_x_col = 'gaze_x'
        gaze_y_col = 'gaze_y'

    segments = []

    for i in range(len(events)):
        start_event = events[i]
        start_time = start_event['timestamp']

        # 次のイベントまで、または最後のイベントなら終端イベントまで
        if i + 1 < len(events):
            end_time = events[i + 1]['timestamp']
        elif end_timestamp is not None:
            end_time = end_timestamp  # phase_complete_enterなど
        else:
            end_time = df['timestamp_sec'].max() + 1  # フォールバック: データの最後まで

        # 該当範囲のデータを抽出
        mask = (df['timestamp_sec'] >= start_time) & (df['timestamp_sec'] < end_time)
        segment_df = df[mask]

        if len(segment_df) > 0:
            # numpy配列に変換 [timestamp, gaze_x, gaze_y, pupil_diameter]
            data = np.vstack((
                segment_df['timestamp_sec'].values,
                segment_df[gaze_x_col].values,
                segment_df[gaze_y_col].values,
                segment_df['pupil_diameter'].values
            )).T

            segments.append({
                'passage_id': start_event['passage_id'],
                'data': data,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time
            })

    return segments


def passageIdToImageNumber(passage_id):
    """
    passage_id (例: "pre_01") を画像番号 (例: "003") に変換

    Parameters:
    -----------
    passage_id : str
        例: "pre_01", "training1_03"

    Returns:
    --------
    str
        ゼロパディングされた画像番号 (例: "003", "004")
    """
    # "pre_01" -> ["pre", "01"]
    parts = passage_id.split('_')
    phase = parts[0]  # "pre", "training1", etc.
    number = int(parts[-1])

    # preフェーズの場合: 001=home, 002=intro, 003-010=pre_01-08, 011=complete
    if phase == "pre":
        image_number = number + 2
    else:
        # 他のフェーズも同様にオフセットが必要な場合は追加
        image_number = number + 2

    return str(image_number).zfill(3)


# =============================================================================
# 統合データ読み込み関数
# =============================================================================

def segmentGazeDataUnified(gaze_csv_path, events, end_timestamp, phase_config,
                           apply_head_correction=False,
                           correction_method="geometric",
                           # geometric方式用
                           use_average_reference=False,
                           correct_y=False,
                           calibration_head_x=0.0,
                           calibration_head_y=0.0,
                           screen_width_mm=509.0,
                           screen_height_mm=287.0,
                           screen_width_px=1920,
                           screen_height_px=1080,
                           # trackbox方式用
                           correction_factor_x=500.0,
                           correction_factor_y=200.0,
                           calibration_center_x=0.5,
                           calibration_center_y=0.5):
    """
    統合セグメント化関数（全フェーズ対応）

    Parameters:
    -----------
    gaze_csv_path : str
        tobii_pro_gaze.csvのパス
    events : list of dict
        readEventLog()またはreadEventLogMultiple()の戻り値
    end_timestamp : float or None
        最後のセグメントの終了タイムスタンプ
    phase_config : PhaseConfig
        フェーズ設定

    Returns:
    --------
    list of dict
        各セグメントのデータ（統一構造）
    """
    import pandas as pd

    # 読み込むカラムを決定
    base_cols = ['#timestamp', 'gaze_x', 'gaze_y', 'pupil_diameter']
    if apply_head_correction:
        df = pd.read_csv(gaze_csv_path)
    else:
        df = pd.read_csv(gaze_csv_path, usecols=base_cols)

    df = df.dropna(subset=['gaze_x', 'gaze_y', 'pupil_diameter'])
    df['timestamp_sec'] = df['#timestamp'] * 0.001 + 32400

    # 頭部位置補正
    if apply_head_correction:
        if correction_method == "geometric":
            df = correctGazeGeometric(df,
                                      calibration_head_x=calibration_head_x,
                                      calibration_head_y=calibration_head_y,
                                      use_average_reference=use_average_reference,
                                      correct_y=correct_y,
                                      screen_width_mm=screen_width_mm,
                                      screen_height_mm=screen_height_mm,
                                      screen_width_px=screen_width_px,
                                      screen_height_px=screen_height_px)
        else:
            df = correctGazeForHeadPosition(df,
                                            correction_factor_x=correction_factor_x,
                                            correction_factor_y=correction_factor_y,
                                            calibration_center_x=calibration_center_x,
                                            calibration_center_y=calibration_center_y)
        gaze_x_col = 'corrected_gaze_x'
        gaze_y_col = 'corrected_gaze_y'
    else:
        gaze_x_col = 'gaze_x'
        gaze_y_col = 'gaze_y'

    segments = []

    for i, event in enumerate(events):
        start_time = event['timestamp']

        # 終了時刻の決定
        if i + 1 < len(events):
            end_time = events[i + 1]['timestamp']
        elif end_timestamp is not None:
            end_time = end_timestamp
        else:
            end_time = df['timestamp_sec'].max() + 1

        # データ抽出
        mask = (df['timestamp_sec'] >= start_time) & (df['timestamp_sec'] < end_time)
        segment_df = df[mask]

        if len(segment_df) == 0:
            continue

        data = np.vstack((
            segment_df['timestamp_sec'].values,
            segment_df[gaze_x_col].values,
            segment_df[gaze_y_col].values,
            segment_df['pupil_diameter'].values
        )).T

        # 画像番号の決定
        if phase_config.use_formula_mapping:
            image_number = passageIdToImageNumber(event.get('passage_id', ''))
        else:
            event_type = event.get('event_type', '')
            image_number = phase_config.image_mapping.get(event_type, '000')

        # セグメント構造
        segment = {
            'data': data,
            'image_number': image_number,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'passage_id': event.get('passage_id'),
        }

        # フェーズ設定に基づいて追加フィールド
        if phase_config.extract_event_type:
            segment['event_type'] = event.get('event_type')
        if phase_config.extract_analog_id:
            segment['analog_id'] = event.get('analog_id')

        segments.append(segment)

    return segments


def readTobiiData(eye_tracking_dir, event_log_path, phase="pre",
                  apply_head_correction=False,
                  correction_method=None,
                  # geometric方式用
                  use_average_reference=False,
                  correct_y=False,
                  calibration_head_x=0.0,
                  calibration_head_y=0.0,
                  screen_width_mm=509.0,
                  screen_height_mm=287.0,
                  screen_width_px=1920,
                  screen_height_px=1080,
                  # trackbox方式用
                  correction_factor_x=500.0,
                  correction_factor_y=200.0,
                  calibration_center_x=0.5,
                  calibration_center_y=0.5):
    """
    統合データ読み込み関数（全フェーズ対応）

    Parameters:
    -----------
    eye_tracking_dir : str
        eye_trackingディレクトリ（tobii_pro_gaze.csvと背景画像を含む）
    event_log_path : str
        events.jsonlのパス
    phase : str
        フェーズ名（"pre", "training1", "training2", "posttest"）
    apply_head_correction : bool
        頭部位置補正を適用するか
    correction_method : str or None
        補正方式。Noneの場合はフェーズのデフォルトを使用

    Returns:
    --------
    list of dict
        各セグメントの情報（統一構造）
    """
    import os

    # フェーズ設定を取得
    if phase not in PHASE_CONFIGS:
        raise ValueError(f"Unknown phase '{phase}'. Available: {list(PHASE_CONFIGS.keys())}")
    phase_config = PHASE_CONFIGS[phase]

    # 補正方式のデフォルト
    if correction_method is None:
        correction_method = "trackbox" if phase.startswith("training") else "geometric"

    # イベント読み込み
    if phase_config.phase_type == "simple":
        events = readEventLog(event_log_path, phase_config.segment_event_types[0])
    else:
        events = readEventLogMultiple(event_log_path, phase_config.segment_event_types)

    # 終了イベントのタイムスタンプ
    end_events = readEventLog(event_log_path, "phase_complete_enter")
    end_timestamp = end_events[0]['timestamp'] if end_events else None

    # セグメント化
    gaze_csv = os.path.join(eye_tracking_dir, "tobii_pro_gaze.csv")
    segments = segmentGazeDataUnified(
        gaze_csv, events, end_timestamp, phase_config,
        apply_head_correction=apply_head_correction,
        correction_method=correction_method,
        use_average_reference=use_average_reference,
        correct_y=correct_y,
        calibration_head_x=calibration_head_x,
        calibration_head_y=calibration_head_y,
        screen_width_mm=screen_width_mm,
        screen_height_mm=screen_height_mm,
        screen_width_px=screen_width_px,
        screen_height_px=screen_height_px,
        correction_factor_x=correction_factor_x,
        correction_factor_y=correction_factor_y,
        calibration_center_x=calibration_center_x,
        calibration_center_y=calibration_center_y
    )

    # 背景画像パスを追加
    for segment in segments:
        img_num = segment['image_number']
        segment['image_path'] = os.path.join(eye_tracking_dir, f"{img_num}_back.png")

    return segments


# =============================================================================
# AOI (Area of Interest) 分析関連関数
# =============================================================================

def loadCoordinates(json_path):
    """
    座標JSONファイルを読み込む

    Parameters:
    -----------
    json_path : str
        座標JSONファイルのパス

    Returns:
    --------
    dict
        座標データ
    """
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extractAOIs(coordinates, levels=None):
    """
    座標データからAOI (Area of Interest) を抽出

    Parameters:
    -----------
    coordinates : dict
        loadCoordinates()で読み込んだ座標データ
    levels : list of str, optional
        抽出するレベル。デフォルト: ["paragraph", "sentence", "word", "choice"]
        - "paragraph": 段落レベル
        - "sentence": 文レベル
        - "word": 単語レベル
        - "choice": 選択肢レベル
        - "question": 問題文レベル

    Returns:
    --------
    list of dict
        AOIリスト。各要素:
        {
            "id": "para_0_sent_1",
            "level": "sentence",
            "text": "Could you please...",
            "bbox": {"x": 295, "y": 113, "width": 348, "height": 19},
            "parent_ids": {"paragraph": "para_0"}
        }
    """
    if levels is None:
        levels = ["paragraph", "sentence", "word", "choice"]

    aois = []
    coords = coordinates.get('coordinates', coordinates)

    # 左パネル（本文）の処理
    left_panel = coords.get('left_panel', {})
    passages = left_panel.get('passages', [])

    # trainingのような言語別座標の場合、日本語版を優先
    if not passages:
        passages = left_panel.get('passages_ja', left_panel.get('passages_en', []))

    for passage in passages:
        for para in passage.get('paragraphs', []):
            para_idx = para.get('paragraph_index', 0)
            para_id = f"para_{para_idx}"

            # 段落レベル
            if "paragraph" in levels:
                # 段落のbboxは全ての行を包含する領域
                para_lines = para.get('lines', [])
                if para_lines:
                    bboxes, encompassing_bbox, is_multiline = _lines_to_bboxes(para_lines)
                    aois.append({
                        "id": para_id,
                        "level": "paragraph",
                        "text": para.get('text', '')[:50] + ('...' if len(para.get('text', '')) > 50 else ''),
                        "full_text": para.get('text', ''),
                        "bbox": encompassing_bbox,
                        "bboxes": bboxes,
                        "is_multiline": is_multiline,
                        "parent_ids": {}
                    })

            for sent in para.get('sentences', []):
                sent_idx = sent.get('sentence_index', 0)
                sent_id = f"{para_id}_sent_{sent_idx}"

                # 文レベル
                if "sentence" in levels:
                    sent_lines = sent.get('lines', [])
                    if sent_lines:
                        bboxes, encompassing_bbox, is_multiline = _lines_to_bboxes(sent_lines)
                        aois.append({
                            "id": sent_id,
                            "level": "sentence",
                            "text": sent.get('text', ''),
                            "bbox": encompassing_bbox,
                            "bboxes": bboxes,
                            "is_multiline": is_multiline,
                            "parent_ids": {"paragraph": para_id}
                        })

                # 単語レベル
                if "word" in levels:
                    for word in sent.get('words', []):
                        word_idx = word.get('word_index', 0)
                        word_id = f"{sent_id}_word_{word_idx}"
                        bbox = word.get('bbox', {})
                        aois.append({
                            "id": word_id,
                            "level": "word",
                            "text": word.get('text', ''),
                            "bbox": bbox,
                            "parent_ids": {"paragraph": para_id, "sentence": sent_id}
                        })

    # 右パネル（問題・選択肢）の処理
    right_panel = coords.get('right_panel', {})
    questions = right_panel.get('questions', [])

    for q in questions:
        q_idx = q.get('question_index', 0)
        q_id = f"question_{q_idx}"

        # 問題文レベル
        if "question" in levels:
            q_text = q.get('question_text', {})
            q_lines = q_text.get('lines', [])
            if q_lines:
                bboxes, encompassing_bbox, is_multiline = _lines_to_bboxes(q_lines)
                aois.append({
                    "id": q_id,
                    "level": "question",
                    "text": q_text.get('text', ''),
                    "bbox": encompassing_bbox,
                    "bboxes": bboxes,
                    "is_multiline": is_multiline,
                    "parent_ids": {}
                })

        # 選択肢レベル
        if "choice" in levels:
            for choice in q.get('choices', []):
                choice_id = f"{q_id}_choice_{choice.get('choice_id', '')}"
                choice_bbox = choice.get('choice_bbox', {})
                choice_text = choice.get('choice_text', {})
                aois.append({
                    "id": choice_id,
                    "level": "choice",
                    "text": choice_text.get('text', ''),
                    "bbox": choice_bbox,
                    "parent_ids": {"question": q_id}
                })

    return aois


def _point_in_bboxes(x, y, bboxes):
    """
    点が複数のbboxのいずれかに含まれるか判定

    Parameters:
    -----------
    x, y : float
        チェックする座標
    bboxes : list of dict
        bboxのリスト [{x, y, width, height}, ...]

    Returns:
    --------
    bool
        いずれかのbboxに含まれればTrue
    """
    for bbox in bboxes:
        if (bbox['x'] <= x <= bbox['x'] + bbox['width'] and
            bbox['y'] <= y <= bbox['y'] + bbox['height']):
            return True
    return False


def findAOIForPoint(x, y, aois, level=None):
    """
    座標がどのAOIに含まれるか判定

    Parameters:
    -----------
    x : float
        X座標（ピクセル）
    y : float
        Y座標（ピクセル）
    aois : list of dict
        extractAOIs()で抽出したAOIリスト
    level : str, optional
        特定のレベルのみを検索（例: "sentence"）。
        Noneの場合は全レベルから最小のAOIを返す

    Returns:
    --------
    dict or None
        マッチしたAOI。マッチしない場合はNone
    """
    matches = []

    for aoi in aois:
        if level is not None and aoi['level'] != level:
            continue

        # multiline AOIの場合は個別のbboxをチェック
        if aoi.get('is_multiline') and 'bboxes' in aoi:
            if _point_in_bboxes(x, y, aoi['bboxes']):
                matches.append(aoi)
        else:
            # 単一行または後方互換性
            bbox = aoi['bbox']
            if (bbox['x'] <= x <= bbox['x'] + bbox['width'] and
                bbox['y'] <= y <= bbox['y'] + bbox['height']):
                matches.append(aoi)

    if not matches:
        return None

    # 複数マッチした場合は最小面積のAOIを返す（より具体的な要素）
    # multiline AOIの場合はbboxesの合計面積を使用
    def get_area(aoi):
        if aoi.get('is_multiline') and 'bboxes' in aoi:
            return sum(b['width'] * b['height'] for b in aoi['bboxes'])
        return aoi['bbox']['width'] * aoi['bbox']['height']

    return min(matches, key=get_area)


def matchFixationsToAOIs(fixations, aois):
    """
    全fixationに対してAOIをマッチング

    Parameters:
    -----------
    fixations : np.array or pd.DataFrame
        fixationデータ。カラム: [timestamp, x, y, duration, ...]
    aois : list of dict
        extractAOIs()で抽出したAOIリスト

    Returns:
    --------
    list of dict
        各fixationにAOI情報を付加したリスト
    """
    import pandas as pd

    # DataFrameに変換
    if isinstance(fixations, np.ndarray):
        df = pd.DataFrame(fixations, columns=[
            'timestamp', 'x', 'y', 'duration',
            'saccade_length', 'saccade_angle', 'saccade_speed', 'pupil_diameter'
        ])
    else:
        df = fixations.copy()

    results = []

    for _, row in df.iterrows():
        x, y = row['x'], row['y']

        # 各レベルでマッチング
        word_aoi = findAOIForPoint(x, y, aois, level="word")
        sent_aoi = findAOIForPoint(x, y, aois, level="sentence")
        para_aoi = findAOIForPoint(x, y, aois, level="paragraph")
        choice_aoi = findAOIForPoint(x, y, aois, level="choice")
        question_aoi = findAOIForPoint(x, y, aois, level="question")

        result = {
            'timestamp': row['timestamp'],
            'x': x,
            'y': y,
            'duration': row['duration'],
            'pupil_diameter': row.get('pupil_diameter', np.nan),
            # 各レベルのAOI情報
            'word_id': word_aoi['id'] if word_aoi else None,
            'word_text': word_aoi['text'] if word_aoi else None,
            'sentence_id': sent_aoi['id'] if sent_aoi else None,
            'sentence_text': sent_aoi['text'] if sent_aoi else None,
            'paragraph_id': para_aoi['id'] if para_aoi else None,
            'choice_id': choice_aoi['id'] if choice_aoi else None,
            'choice_text': choice_aoi['text'] if choice_aoi else None,
            'question_id': question_aoi['id'] if question_aoi else None,
        }
        results.append(result)

    return results


def computeAOIStatistics(matched_fixations, level="sentence"):
    """
    各AOIの注視統計を計算

    Parameters:
    -----------
    matched_fixations : list of dict
        matchFixationsToAOIs()の戻り値
    level : str
        集計するレベル（"word", "sentence", "paragraph", "choice"）

    Returns:
    --------
    pd.DataFrame
        各AOIの統計:
        - aoi_id: AOIのID
        - total_duration: 総注視時間
        - fixation_count: 注視回数
        - first_fixation_time: 最初の注視時刻
        - mean_duration: 平均注視時間
        - revisits: 再訪問回数
    """
    import pandas as pd
    from collections import defaultdict

    id_key = f"{level}_id"
    text_key = f"{level}_text" if level != "paragraph" else None

    # AOIごとにfixationを集計
    aoi_data = defaultdict(list)

    for fix in matched_fixations:
        aoi_id = fix.get(id_key)
        if aoi_id is not None:
            aoi_data[aoi_id].append(fix)

    # 統計計算
    stats = []
    for aoi_id, fixations in aoi_data.items():
        durations = [f['duration'] for f in fixations]
        timestamps = [f['timestamp'] for f in fixations]

        # 再訪問回数（連続しないfixationのグループ数 - 1）
        revisits = 0
        prev_idx = -2
        for fix in sorted(fixations, key=lambda x: x['timestamp']):
            fix_idx = matched_fixations.index(fix)
            if fix_idx != prev_idx + 1:
                revisits += 1
            prev_idx = fix_idx
        revisits = max(0, revisits - 1)

        text = fixations[0].get(text_key, '') if text_key else ''

        stats.append({
            'aoi_id': aoi_id,
            'level': level,
            'text': text[:50] + ('...' if len(text) > 50 else '') if text else '',
            'total_duration': sum(durations),
            'fixation_count': len(fixations),
            'mean_duration': np.mean(durations),
            'first_fixation_time': min(timestamps),
            'revisits': revisits
        })

    return pd.DataFrame(stats).sort_values('first_fixation_time')


# =============================================================================
# Training フェーズ対応関数
# =============================================================================

# Training フェーズのイベントタイプと画像番号のマッピング
TRAINING_EVENT_TO_IMAGE = {
    'phase_intro_enter': '002',        # intro画面
    'question_screen_open': '003',     # 問題画面
    'reflection1_open': '004',         # reflection1画面
    'training_explanation_open': '005', # 解説
    'analog_intro_enter': '006',       # 類題intro画面
    'analog_question_open_an1': '007', # 類題1
    'analog_explanation_open_an1': '008', # 類題1解説
    'analog_question_open_an2': '009', # 類題2
    'analog_explanation_open_an2': '010', # 類題2解説
    'analog_question_open_an3': '011', # 類題3
    'analog_explanation_open_an3': '012', # 類題3解説
    'reflection2_open': '013',         # reflection2画面
    'phase_complete_enter': '014',     # complete画面
}


def readEventLogMultiple(jsonl_path, event_types):
    """
    JSONLイベントログから複数のイベントタイプを抽出

    Parameters:
    -----------
    jsonl_path : str
        イベントログファイルのパス (.jsonl)
    event_types : list of str
        抽出するイベントタイプのリスト

    Returns:
    --------
    list of dict
        各イベントの情報（タイムスタンプ順にソート）
    """
    import json
    from datetime import datetime

    events = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get('event') in event_types:
                iso_timestamp = data['timestamp']
                dt_utc = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
                dt_local = dt_utc.astimezone()
                unix_timestamp = dt_local.timestamp()

                # analog_id を含むイベントは event_type を拡張
                event_type = data['event']
                analog_id = data.get('analog_id', '')
                if analog_id:
                    # tr_01_an1 -> an1
                    analog_suffix = analog_id.split('_')[-1]
                    event_type = f"{event_type}_{analog_suffix}"

                events.append({
                    'timestamp': unix_timestamp,
                    'event_type': event_type,
                    'passage_id': data.get('passage_id'),
                    'analog_id': analog_id,
                    'raw_data': data
                })

    # タイムスタンプ順にソート
    events.sort(key=lambda x: x['timestamp'])
    return events


def segmentGazeDataForTraining(gaze_csv_path, events,
                                apply_head_correction=False,
                                correction_method="trackbox",
                                correction_factor_x=500.0,
                                correction_factor_y=200.0):
    """
    Training フェーズ用の視線データセグメント化

    Parameters:
    -----------
    gaze_csv_path : str
        tobii_pro_gaze.csvのパス
    events : list of dict
        readEventLogMultiple()の戻り値
    apply_head_correction : bool
        頭部位置補正を適用するか
    correction_method : str
        補正方式
    correction_factor_x : float
        X軸補正係数
    correction_factor_y : float
        Y軸補正係数

    Returns:
    --------
    list of dict
        各セグメントのデータ
    """
    import pandas as pd

    df = pd.read_csv(gaze_csv_path)
    df = df.dropna(subset=['gaze_x', 'gaze_y', 'pupil_diameter'])
    df['timestamp_sec'] = df['#timestamp'] * 0.001 + 32400

    if apply_head_correction:
        if correction_method == "trackbox":
            df = correctGazeForHeadPosition(df, correction_factor_x, correction_factor_y)
        gaze_x_col = 'corrected_gaze_x'
        gaze_y_col = 'corrected_gaze_y'
    else:
        gaze_x_col = 'gaze_x'
        gaze_y_col = 'gaze_y'

    segments = []

    for i in range(len(events)):
        start_event = events[i]
        start_time = start_event['timestamp']

        # 次のイベントまで
        if i + 1 < len(events):
            end_time = events[i + 1]['timestamp']
        else:
            end_time = df['timestamp_sec'].max() + 1

        mask = (df['timestamp_sec'] >= start_time) & (df['timestamp_sec'] < end_time)
        segment_df = df[mask]

        if len(segment_df) > 0:
            data = np.vstack((
                segment_df['timestamp_sec'].values,
                segment_df[gaze_x_col].values,
                segment_df[gaze_y_col].values,
                segment_df['pupil_diameter'].values
            )).T

            # 画像番号を取得
            event_type = start_event['event_type']
            image_number = TRAINING_EVENT_TO_IMAGE.get(event_type, '000')

            segments.append({
                'event_type': event_type,
                'passage_id': start_event['passage_id'],
                'analog_id': start_event.get('analog_id'),
                'image_number': image_number,
                'data': data,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time
            })

    return segments


def plotAOIWithGaze(image_path, aois, fixations, level="sentence",
                    save_path=None, figsize=(16, 9)):
    """
    背景画像にAOI領域とfixationを重ねて可視化

    Parameters:
    -----------
    image_path : str
        背景画像のパス
    aois : list of dict
        extractAOIs()で抽出したAOIリスト
    fixations : np.array or pd.DataFrame
        fixationデータ
    level : str
        表示するAOIレベル（"word", "sentence", "paragraph", "choice"）
    save_path : str, optional
        保存先パス
    figsize : tuple
        図のサイズ
    """
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    _, ax = plt.subplots(figsize=figsize)

    # 背景画像
    if image_path and os.path.exists(image_path):
        img = plt.imread(image_path)
        ax.imshow(img)

    # AOI領域を描画
    colors = {
        'word': 'blue',
        'sentence': 'green',
        'paragraph': 'orange',
        'choice': 'purple',
        'question': 'red'
    }

    for aoi in aois:
        if aoi['level'] != level:
            continue

        color = colors.get(level, 'gray')

        # multiline AOIの場合は各行を個別に描画
        if aoi.get('is_multiline') and 'bboxes' in aoi:
            for bbox in aoi['bboxes']:
                rect = patches.Rectangle(
                    (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                    linewidth=1, edgecolor=color,
                    facecolor='none', alpha=0.7
                )
                ax.add_patch(rect)
        else:
            bbox = aoi['bbox']
            rect = patches.Rectangle(
                (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                linewidth=1, edgecolor=color,
                facecolor='none', alpha=0.7
            )
            ax.add_patch(rect)

    # fixationを描画
    if isinstance(fixations, np.ndarray):
        fx, fy, fdur = fixations[:, 1], fixations[:, 2], fixations[:, 3]
    else:
        fx, fy, fdur = fixations['x'], fixations['y'], fixations['duration']

    # 注視時間に応じたサイズ
    sizes = np.array(fdur) * 500

    ax.scatter(fx, fy, s=sizes, c='red', alpha=0.5, edgecolors='darkred')

    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)  # Y軸反転
    ax.set_title(f'AOI ({level}) with Fixations')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =============================================================================
# 全要素AOI抽出・AOI率計算
# =============================================================================

def _lines_to_bbox(lines):
    """
    lines配列からbboxを計算するヘルパー関数

    Parameters:
    -----------
    lines : list of dict
        [{x, y, width, height}, ...]

    Returns:
    --------
    dict or None
        {"x": min_x, "y": min_y, "width": w, "height": h} or None
    """
    if not lines:
        return None
    min_x = min(line['x'] for line in lines)
    min_y = min(line['y'] for line in lines)
    max_x = max(line['x'] + line['width'] for line in lines)
    max_y = max(line['y'] + line['height'] for line in lines)
    return {'x': min_x, 'y': min_y, 'width': max_x - min_x, 'height': max_y - min_y}


def _lines_to_bboxes(lines):
    """
    lines配列から個別のbboxリストと包含bboxを計算するヘルパー関数

    複数行にまたがるテキスト要素を正確に表現するため、
    各行の個別bboxと、全行を包含するbbox（後方互換性用）を返す。

    同じ視覚的行にある要素（例：丸数字①とテキスト本体）がある場合、
    丸数字（widthが小さい方）は除外してテキスト本体のみを残す。

    Parameters:
    -----------
    lines : list of dict
        [{x, y, width, height}, ...]

    Returns:
    --------
    tuple (bboxes, encompassing_bbox, is_multiline)
        bboxes: list of dict - 各視覚的行のbbox（丸数字は除外）
        encompassing_bbox: dict - 全行を包含するbbox (後方互換性用)
        is_multiline: bool - 複数の視覚的行にまたがるかどうか
    """
    if not lines:
        return None, None, False

    # Y座標でソート
    sorted_lines = sorted(lines, key=lambda l: l['y'])

    # Y座標が重なっている行をグループ化
    visual_lines = []
    current_group = [sorted_lines[0]]

    for line in sorted_lines[1:]:
        # 現在のグループの最大Y（下端）を計算
        group_max_y = max(l['y'] + l['height'] for l in current_group)
        # 新しい行のY（上端）がグループの下端より小さければ重なっている
        if line['y'] < group_max_y:
            # 同じ視覚的行なのでグループに追加
            current_group.append(line)
        else:
            # 新しい視覚的行を開始
            visual_lines.append(current_group)
            current_group = [line]
    visual_lines.append(current_group)

    # 各視覚的行のグループから、widthが最大のもの（テキスト本体）だけを採用
    # （丸数字①はwidthが小さいので除外される）
    bboxes = []
    for group in visual_lines:
        # widthが最大のlineを採用
        main_line = max(group, key=lambda l: l['width'])
        bboxes.append({
            'x': main_line['x'],
            'y': main_line['y'],
            'width': main_line['width'],
            'height': main_line['height']
        })

    # 包含bbox (フィルタリング後のbboxesから計算)
    min_x = min(b['x'] for b in bboxes)
    min_y = min(b['y'] for b in bboxes)
    max_x = max(b['x'] + b['width'] for b in bboxes)
    max_y = max(b['y'] + b['height'] for b in bboxes)
    encompassing_bbox = {'x': min_x, 'y': min_y, 'width': max_x - min_x, 'height': max_y - min_y}

    # 視覚的行が複数あるかどうか
    is_multiline = len(visual_lines) > 1

    return bboxes, encompassing_bbox, is_multiline


def parseImageFilename(image_path):
    """
    画像ファイル名からAOI抽出パラメータを解析

    Parameters:
    -----------
    image_path : str
        画像ファイルのパス（例: '005_2_ja_back.png'）

    Returns:
    --------
    dict or None
        {
            'image_num': '005',           # 画像番号
            'target_locale': 'ja',        # 'en' or 'ja'
            'target_question': 1,         # 0-indexed Q番号
            'target_analog': 1,           # 0-indexed 類題番号
        }
        パターンにマッチしない場合はNone
    """
    import re
    filename = os.path.basename(image_path)

    # パターン: NNN[_N][_ja]_back.png
    # NNN: 画像番号（3桁）
    # _N: オプション。Q番号または類題番号（2, 3など）
    # _ja: オプション。日本語の場合
    pattern = r'^(\d{3})(?:_(\d))?(?:_(ja))?_back\.png$'
    match = re.match(pattern, filename)

    if not match:
        return None

    image_num = match.group(1)
    variant_num = match.group(2)  # '2', '3', or None
    lang = match.group(3)         # 'ja' or None

    target_locale = 'ja' if lang == 'ja' else 'en'
    # variant_numは0-indexedに変換（'2' -> 1, '3' -> 2）
    # Noneの場合は0（デフォルト: Q1/類題1）
    variant_index = int(variant_num) - 1 if variant_num else 0

    return {
        'image_num': image_num,
        'target_locale': target_locale,
        'target_question': variant_index,
        'target_analog': variant_index,
    }


def extractAllAOIs(coordinates, levels=None,
                   target_locale=None, target_question=None, target_analog=None):
    """
    座標JSONから全ての要素をAOIとして抽出

    extractAOIs()と異なり、instruction, metadata, ui要素も含む
    全てのページタイプ（intro, complete, question, explanation, reflection等）に対応

    Parameters:
    -----------
    coordinates : dict
        loadCoordinates()で読み込んだ座標データ
    levels : list of str, optional
        抽出するAOIレベルのリスト。Noneの場合は全レベルを抽出。
        指定可能:
        - 'instruction': 問題文の指示
        - 'paragraph': 段落
        - 'sentence': 文
        - 'question': 設問
        - 'choice': 選択肢
        - 'timer': タイマー (ui_components内 or header内)
        - 'ui': submit_button, confirm_button, button等
        - 'header': locale_tabs, question_tabs, analog_tabs
        - 'reflection': reflection_form
        - 'title': passage title
        - 'subtitle': passage subtitle (例: "By Erika Eaton")
        - 'intro': intro/complete画面のtitle, description
        - 'analog': analogs内の要素 (reflection2用)
        - 'metadata': passage内のメタデータ (sender, time, date, recipient, subject等)
        - 'table': テーブル要素 (headers, cells)
        - 'explanation': 解説テキスト (training_explanation, analog_explanation画面)
        - 'metacog': メタ認知フィードバック (B群のtraining_explanation画面)
        例: levels=['instruction', 'sentence', 'question', 'choice', 'timer', 'ui']
    target_locale : str, optional
        抽出する言語。'en':英語のみ, 'ja':日本語のみ, None:自動判定
        (explanation/reflection2ではデフォルトで英語のみ)
    target_question : int, optional
        抽出するQ番号（0-indexed）。None:自動判定
        (explanation/reflection2ではデフォルトでQ1のみ)
    target_analog : int, optional
        抽出する類題番号（0-indexed）。None:自動判定
        (explanation/reflection2ではデフォルトで類題1のみ)

    Returns:
    --------
    list of dict
        AOIリスト。各要素:
        {
            "id": "para_0_sent_1",
            "level": "sentence",
            "text": "...",
            "bbox": {"x": 295, "y": 113, "width": 348, "height": 19}
        }
    """
    # デフォルトは全レベル
    if levels is None:
        levels = ['instruction', 'paragraph', 'sentence', 'question', 'choice',
                  'timer', 'ui', 'header', 'reflection', 'title', 'subtitle', 'intro', 'analog',
                  'metadata', 'table', 'explanation', 'metacog']
    aois = []
    coords = coordinates.get('coordinates', coordinates)

    # page_typeを取得（explanation/reflection2画面の判定用）
    page_type = coords.get('page_type', '')
    passage_id = coords.get('passage_id', '')
    # explanation画面とreflection2画面はデフォルトで英語のみ、Q1のみ表示
    is_en_only_page_default = 'explanation' in page_type or page_type == 'reflection2'
    is_explanation = 'explanation' in page_type
    is_reflection2 = page_type == 'reflection2'

    # target_localeが指定された場合はそれを使用、なければデフォルト動作
    if target_locale is not None:
        include_en = (target_locale == 'en')
        include_ja = (target_locale == 'ja')
    elif is_en_only_page_default:
        # explanation/reflection2画面でも、passages_jaがあれば日本語を優先
        left_panel = coords.get('left_panel', {})
        if left_panel.get('passages_ja'):
            include_en = False
            include_ja = True
        else:
            include_en = True
            include_ja = False
    else:
        include_en = True
        include_ja = True

    # 後方互換性のため維持
    is_en_only_page = not include_ja

    # === ヘッダー ===
    header = coords.get('header', {})

    # locale_tabs (EN/JA切り替えタブ)
    if 'header' in levels:
        for tab in header.get('locale_tabs', []):
            bbox = tab.get('bbox')
            if bbox:
                aois.append({
                    'id': f"locale_tab_{tab.get('locale', '')}",
                    'level': 'header',
                    'text': tab.get('locale', ''),
                    'bbox': bbox
                })

        # question_tabs (Q1/Q2切り替えタブ)
        for tab in header.get('question_tabs', []):
            bbox = tab.get('bbox')
            if bbox:
                aois.append({
                    'id': f"question_tab_{tab.get('question_index', '')}",
                    'level': 'header',
                    'text': f"Q{tab.get('question_index', 0) + 1}",
                    'bbox': bbox
                })

        # analog_tabs (類題切り替えタブ - reflection2で使用)
        for tab in header.get('analog_tabs', []):
            bbox = tab.get('bbox')
            if bbox:
                aois.append({
                    'id': f"analog_tab_{tab.get('analog_index', '')}",
                    'level': 'header',
                    'text': f"An{tab.get('analog_index', 0) + 1}",
                    'bbox': bbox
                })

    # header内のtimer (analog_questionで使用)
    if 'timer' in levels:
        header_timer = header.get('timer', {})
        if header_timer:
            if 'x' in header_timer and 'y' in header_timer:
                bbox = header_timer
            elif 'position' in header_timer:
                bbox = header_timer['position']
            else:
                bbox = None
            if bbox:
                aois.append({
                    'id': 'timer',
                    'level': 'timer',
                    'text': 'Timer',
                    'bbox': bbox
                })

    # === トップレベル要素（intro/complete/analog_intro画面用）===
    if 'intro' in levels:
        # intro/complete画面のtitle
        intro_title = coords.get('title', {})
        if intro_title:
            if 'lines' in intro_title:
                bboxes, bbox, is_multiline = _lines_to_bboxes(intro_title['lines'])
            elif 'x' in intro_title:
                bbox = intro_title
                bboxes = [bbox]
                is_multiline = False
            else:
                bbox = None
                bboxes = None
                is_multiline = False
            if bbox:
                aois.append({
                    'id': 'intro_title',
                    'level': 'intro',
                    'text': intro_title.get('text', 'Title')[:50],
                    'bbox': bbox,
                    'bboxes': bboxes,
                    'is_multiline': is_multiline
                })

        # intro/complete画面のdescription
        intro_desc = coords.get('description', {})
        if intro_desc:
            if 'lines' in intro_desc:
                bboxes, bbox, is_multiline = _lines_to_bboxes(intro_desc['lines'])
            elif 'x' in intro_desc:
                bbox = intro_desc
                bboxes = [bbox]
                is_multiline = False
            else:
                bbox = None
                bboxes = None
                is_multiline = False
            if bbox:
                aois.append({
                    'id': 'intro_description',
                    'level': 'intro',
                    'text': intro_desc.get('text', 'Description')[:50],
                    'bbox': bbox,
                    'bboxes': bboxes,
                    'is_multiline': is_multiline
                })

    # intro/complete/analog_intro画面のbutton (uiレベルとして扱う)
    if 'ui' in levels:
        intro_button = coords.get('button', {})
        if intro_button:
            if 'x' in intro_button:
                bbox = intro_button
            else:
                bbox = None
            if bbox:
                aois.append({
                    'id': 'intro_button',
                    'level': 'ui',
                    'text': intro_button.get('text', 'Button')[:30],
                    'bbox': bbox
                })

    # analog_intro画面のinstruction (トップレベル)
    if 'instruction' in levels:
        top_instruction = coords.get('instruction', {})
        if top_instruction and 'left_panel' not in coords:  # left_panelがない場合のみ
            if 'lines' in top_instruction:
                bbox = _lines_to_bbox(top_instruction['lines'])
            elif 'x' in top_instruction:
                bbox = top_instruction
            else:
                bbox = None
            if bbox:
                aois.append({
                    'id': 'instruction',
                    'level': 'instruction',
                    'text': top_instruction.get('text', '')[:50],
                    'bbox': bbox
                })

    # === 左パネル ===
    left_panel = coords.get('left_panel', {})

    # instruction (通常版)
    def extract_instruction(instr, suffix=''):
        if instr and 'instruction' in levels:
            if 'position' in instr:
                bbox = instr['position']
            elif 'lines' in instr:
                bbox = _lines_to_bbox(instr['lines'])
            else:
                bbox = None
            if bbox:
                aois.append({
                    'id': f'instruction{suffix}',
                    'level': 'instruction',
                    'text': instr.get('text', ''),
                    'bbox': bbox
                })

    extract_instruction(left_panel.get('instruction'))
    # 英語instructionの抽出（target_localeで制御）
    if include_en:
        extract_instruction(left_panel.get('instruction_en'), '_en')
    # 日本語instructionの抽出（target_localeで制御）
    if include_ja:
        extract_instruction(left_panel.get('instruction_ja'), '_ja')

    # passages内のparagraph, sentence, title, metadata
    def extract_passages(passages, suffix=''):
        for psg_idx, passage in enumerate(passages):
            # passage metadata (sender, time, date, recipient, subject, etc.)
            if 'metadata' in levels:
                for m_idx, meta in enumerate(passage.get('metadata') or []):
                    mtype = meta.get('metadata_type', 'unknown')

                    # label部分のbbox（例: "From:", "To:", "Date:"など）
                    label_lines = meta.get('label', [])
                    if label_lines:
                        bbox = _lines_to_bbox(label_lines)
                        if bbox:
                            aois.append({
                                'id': f'psg_{psg_idx}_meta_{mtype}_label_{m_idx}{suffix}',
                                'level': 'metadata',
                                'text': meta.get('label_text', '')[:30],
                                'bbox': bbox
                            })

                    # value部分のbbox（例: "Rex Martinez", "[10:16 A.M.]"など）
                    value_lines = meta.get('value', [])
                    if value_lines:
                        bbox = _lines_to_bbox(value_lines)
                        if bbox:
                            aois.append({
                                'id': f'psg_{psg_idx}_meta_{mtype}_{m_idx}{suffix}',
                                'level': 'metadata',
                                'text': meta.get('value_text', '')[:50],
                                'bbox': bbox
                            })

            # passage title
            title = passage.get('title', {})
            if title and 'title' in levels:
                if 'position' in title:
                    bbox = title['position']
                elif 'lines' in title:
                    bbox = _lines_to_bbox(title['lines'])
                else:
                    bbox = None
                if bbox:
                    aois.append({
                        'id': f'title_{psg_idx}{suffix}',
                        'level': 'title',
                        'text': title.get('text', '')[:50],
                        'bbox': bbox
                    })

            # passage subtitle
            subtitle = passage.get('subtitle', {})
            if subtitle and 'subtitle' in levels:
                if 'position' in subtitle:
                    bbox = subtitle['position']
                elif 'lines' in subtitle:
                    bbox = _lines_to_bbox(subtitle['lines'])
                else:
                    bbox = None
                if bbox:
                    aois.append({
                        'id': f'subtitle_{psg_idx}{suffix}',
                        'level': 'subtitle',
                        'text': subtitle.get('text', '')[:50],
                        'bbox': bbox
                    })

            # passage table (headers and cells)
            # passages_enにはtable（英語）とtableHidden（日本語）の両方がある
            # suffixに応じて適切なtableを使用
            if suffix == '_en':
                table = passage.get('table')
            elif suffix == '_ja':
                # passages_jaにはtableがないので、passages_enのtableHiddenを使うことはできない
                # この場合はスキップ（passages_enの処理でtableHiddenを出力する）
                table = passage.get('table')  # passages_jaの場合はnullになる
            else:
                table = passage.get('table')
            if table and 'table' in levels:
                # table headers
                for h_idx, header in enumerate(table.get('headers', [])):
                    bbox = header.get('bbox')
                    if bbox:
                        aois.append({
                            'id': f'table_{psg_idx}_header_{h_idx}{suffix}',
                            'level': 'table',
                            'text': header.get('text', '')[:30],
                            'bbox': bbox
                        })

                # table cells
                for cell in table.get('cells', []):
                    row_idx = cell.get('row_index', 0)
                    cell_idx = cell.get('cell_index', 0)
                    bbox = cell.get('bbox')
                    if bbox:
                        aois.append({
                            'id': f'table_{psg_idx}_cell_{row_idx}_{cell_idx}{suffix}',
                            'level': 'table',
                            'text': cell.get('text', '')[:30],
                            'bbox': bbox
                        })

            # tableHidden（日本語版table）の処理 - passages_enにのみ存在
            # include_jaがtrueで、suffix='_en'の場合のみ処理
            tableHidden = passage.get('tableHidden')
            if tableHidden and 'table' in levels and suffix == '_en' and include_ja:
                for h_idx, header in enumerate(tableHidden.get('headers', [])):
                    bbox = header.get('bbox')
                    if bbox:
                        aois.append({
                            'id': f'table_{psg_idx}_header_{h_idx}_ja',
                            'level': 'table',
                            'text': header.get('text', '')[:30],
                            'bbox': bbox
                        })
                for cell in tableHidden.get('cells', []):
                    row_idx = cell.get('row_index', 0)
                    cell_idx = cell.get('cell_index', 0)
                    bbox = cell.get('bbox')
                    if bbox:
                        aois.append({
                            'id': f'table_{psg_idx}_cell_{row_idx}_{cell_idx}_ja',
                            'level': 'table',
                            'text': cell.get('text', '')[:30],
                            'bbox': bbox
                        })

            for p_idx, para in enumerate(passage.get('paragraphs', [])):
                # paragraph の bbox
                if 'position' in para:
                    bbox = para['position']
                    bboxes = [bbox]
                    is_multiline = False
                elif 'lines' in para:
                    bboxes, bbox, is_multiline = _lines_to_bboxes(para['lines'])
                else:
                    bbox = None
                    bboxes = None
                    is_multiline = False
                if bbox and 'paragraph' in levels:
                    aois.append({
                        'id': f'psg_{psg_idx}_para_{p_idx}{suffix}',
                        'level': 'paragraph',
                        'text': para.get('text', '')[:50],
                        'bbox': bbox,
                        'bboxes': bboxes,
                        'is_multiline': is_multiline
                    })

                # sentence の bbox
                for s_idx, sent in enumerate(para.get('sentences', [])):
                    if 'position' in sent:
                        bbox = sent['position']
                        bboxes = [bbox]
                        is_multiline = False
                    elif 'lines' in sent:
                        bboxes, bbox, is_multiline = _lines_to_bboxes(sent['lines'])
                    else:
                        bbox = None
                        bboxes = None
                        is_multiline = False
                    if bbox and 'sentence' in levels:
                        aois.append({
                            'id': f'psg_{psg_idx}_para_{p_idx}_sent_{s_idx}{suffix}',
                            'level': 'sentence',
                            'text': sent.get('text', '')[:50],
                            'bbox': bbox,
                            'bboxes': bboxes,
                            'is_multiline': is_multiline
                        })

    extract_passages(left_panel.get('passages', []))
    # 英語passagesの抽出（target_localeで制御）
    if include_en:
        extract_passages(left_panel.get('passages_en', []), '_en')
    # 日本語passagesの抽出（target_localeで制御）
    if include_ja:
        extract_passages(left_panel.get('passages_ja', []), '_ja')

    # 日本語モードでpassages_enのtableHiddenを処理（tableHiddenはpassages_enにのみ存在）
    if include_ja and not include_en and 'table' in levels:
        for psg_idx, passage in enumerate(left_panel.get('passages_en', [])):
            tableHidden = passage.get('tableHidden')
            if tableHidden:
                for h_idx, header in enumerate(tableHidden.get('headers', [])):
                    bbox = header.get('bbox')
                    if bbox:
                        aois.append({
                            'id': f'table_{psg_idx}_header_{h_idx}_ja',
                            'level': 'table',
                            'text': header.get('text', '')[:30],
                            'bbox': bbox
                        })
                for cell in tableHidden.get('cells', []):
                    row_idx = cell.get('row_index', 0)
                    cell_idx = cell.get('cell_index', 0)
                    bbox = cell.get('bbox')
                    if bbox:
                        aois.append({
                            'id': f'table_{psg_idx}_cell_{row_idx}_{cell_idx}_ja',
                            'level': 'table',
                            'text': cell.get('text', '')[:30],
                            'bbox': bbox
                        })

    # === 左パネルのanalogs (reflection2で使用) ===
    if 'analog' in levels:
        # analogsの抽出（explanation/reflection2ページでのみtarget_analogで制限）
        analogs_list = left_panel.get('analogs', [])
        analogs_to_process = []
        if is_en_only_page_default:
            # explanation/reflection2ページでのみ制限を適用
            if target_analog is not None:
                # 特定の類題のみ抽出（インデックスを保持）
                if target_analog < len(analogs_list):
                    analogs_to_process = [(target_analog, analogs_list[target_analog])]
                else:
                    analogs_to_process = []
            else:
                # デフォルト動作: 類題1のみ（インデックスを保持）
                if analogs_list:
                    analogs_to_process = [(0, analogs_list[0])]
        else:
            # 他のページでは全analogsを抽出（インデックスを保持）
            analogs_to_process = list(enumerate(analogs_list))

        for an_idx, analog in analogs_to_process:
            # analog内のinstruction（reflection2で使用）
            if 'instruction' in levels:
                if include_en:
                    instr_en = analog.get('instruction_en')
                    if instr_en:
                        if 'lines' in instr_en:
                            bbox = _lines_to_bbox(instr_en['lines'])
                        else:
                            bbox = None
                        if bbox:
                            aois.append({
                                'id': f'analog_{an_idx}_instruction_en',
                                'level': 'instruction',
                                'text': instr_en.get('text', '')[:50],
                                'bbox': bbox
                            })
                if include_ja:
                    instr_ja = analog.get('instruction_ja')
                    if instr_ja:
                        if 'lines' in instr_ja:
                            bbox = _lines_to_bbox(instr_ja['lines'])
                        else:
                            bbox = None
                        if bbox:
                            aois.append({
                                'id': f'analog_{an_idx}_instruction_ja',
                                'level': 'instruction',
                                'text': instr_ja.get('text', '')[:50],
                                'bbox': bbox
                            })

            # analogs内のpassages_en/passages_ja（target_localeで制御）
            passages_keys = []
            if include_en:
                passages_keys.append('passages_en')
            if include_ja:
                passages_keys.append('passages_ja')
            for passages_key in passages_keys:
                suffix = '_en' if 'en' in passages_key else '_ja'
                for psg_idx, passage in enumerate(analog.get(passages_key, [])):
                    # passage metadata
                    if 'metadata' in levels:
                        for m_idx, meta in enumerate(passage.get('metadata') or []):
                            mtype = meta.get('metadata_type', 'unknown')
                            label_lines = meta.get('label', [])
                            if label_lines:
                                bbox = _lines_to_bbox(label_lines)
                                if bbox:
                                    aois.append({
                                        'id': f'analog_{an_idx}_psg_{psg_idx}_meta_{mtype}_label_{m_idx}{suffix}',
                                        'level': 'metadata',
                                        'text': meta.get('label_text', '')[:30],
                                        'bbox': bbox
                                    })
                            value_lines = meta.get('value', [])
                            if value_lines:
                                bbox = _lines_to_bbox(value_lines)
                                if bbox:
                                    aois.append({
                                        'id': f'analog_{an_idx}_psg_{psg_idx}_meta_{mtype}_{m_idx}{suffix}',
                                        'level': 'metadata',
                                        'text': meta.get('value_text', '')[:50],
                                        'bbox': bbox
                                    })

                    # passage title
                    title = passage.get('title', {})
                    if title and 'title' in levels:
                        if 'position' in title:
                            bbox = title['position']
                        elif 'lines' in title:
                            bbox = _lines_to_bbox(title['lines'])
                        else:
                            bbox = None
                        if bbox:
                            aois.append({
                                'id': f'analog_{an_idx}_title_{psg_idx}{suffix}',
                                'level': 'title',
                                'text': title.get('text', '')[:50],
                                'bbox': bbox
                            })

                    # passage subtitle
                    subtitle = passage.get('subtitle', {})
                    if subtitle and 'subtitle' in levels:
                        if 'position' in subtitle:
                            bbox = subtitle['position']
                        elif 'lines' in subtitle:
                            bbox = _lines_to_bbox(subtitle['lines'])
                        else:
                            bbox = None
                        if bbox:
                            aois.append({
                                'id': f'analog_{an_idx}_subtitle_{psg_idx}{suffix}',
                                'level': 'subtitle',
                                'text': subtitle.get('text', '')[:50],
                                'bbox': bbox
                            })

                    # passage table
                    table = passage.get('table')
                    if table and 'table' in levels:
                        for h_idx, header in enumerate(table.get('headers', [])):
                            bbox = header.get('bbox')
                            if bbox:
                                aois.append({
                                    'id': f'analog_{an_idx}_table_{psg_idx}_header_{h_idx}{suffix}',
                                    'level': 'table',
                                    'text': header.get('text', '')[:30],
                                    'bbox': bbox
                                })
                        for cell in table.get('cells', []):
                            row_idx = cell.get('row_index', 0)
                            cell_idx = cell.get('cell_index', 0)
                            bbox = cell.get('bbox')
                            if bbox:
                                aois.append({
                                    'id': f'analog_{an_idx}_table_{psg_idx}_cell_{row_idx}_{cell_idx}{suffix}',
                                    'level': 'table',
                                    'text': cell.get('text', '')[:30],
                                    'bbox': bbox
                                })

                    # tableHidden（日本語版table）の処理
                    tableHidden = passage.get('tableHidden')
                    if tableHidden and 'table' in levels and suffix == '_en' and include_ja:
                        for h_idx, header in enumerate(tableHidden.get('headers', [])):
                            bbox = header.get('bbox')
                            if bbox:
                                aois.append({
                                    'id': f'analog_{an_idx}_table_{psg_idx}_header_{h_idx}_ja',
                                    'level': 'table',
                                    'text': header.get('text', '')[:30],
                                    'bbox': bbox
                                })
                        for cell in tableHidden.get('cells', []):
                            row_idx = cell.get('row_index', 0)
                            cell_idx = cell.get('cell_index', 0)
                            bbox = cell.get('bbox')
                            if bbox:
                                aois.append({
                                    'id': f'analog_{an_idx}_table_{psg_idx}_cell_{row_idx}_{cell_idx}_ja',
                                    'level': 'table',
                                    'text': cell.get('text', '')[:30],
                                    'bbox': bbox
                                })

                    # paragraphs（sentenceも含めて処理するため、paragraph or sentenceレベルが必要な場合に実行）
                    if 'paragraph' in levels or 'sentence' in levels:
                        for p_idx, para in enumerate(passage.get('paragraphs', [])):
                            # paragraph AOIの追加（paragraphレベルが指定されている場合のみ）
                            if 'paragraph' in levels:
                                if 'position' in para:
                                    bbox = para['position']
                                    bboxes = [bbox]
                                    is_multiline = False
                                elif 'lines' in para:
                                    bboxes, bbox, is_multiline = _lines_to_bboxes(para['lines'])
                                else:
                                    bbox = None
                                    bboxes = None
                                    is_multiline = False
                                if bbox:
                                    aois.append({
                                        'id': f'analog_{an_idx}_psg_{psg_idx}_para_{p_idx}{suffix}',
                                        'level': 'paragraph',
                                        'text': para.get('text', '')[:50],
                                        'bbox': bbox,
                                        'bboxes': bboxes,
                                        'is_multiline': is_multiline
                                    })

                            # sentences
                            if 'sentence' in levels:
                                for s_idx, sent in enumerate(para.get('sentences', [])):
                                    if 'position' in sent:
                                        bbox = sent['position']
                                        bboxes = [bbox]
                                        is_multiline = False
                                    elif 'lines' in sent:
                                        bboxes, bbox, is_multiline = _lines_to_bboxes(sent['lines'])
                                    else:
                                        bbox = None
                                        bboxes = None
                                        is_multiline = False
                                    if bbox:
                                        aois.append({
                                            'id': f'analog_{an_idx}_psg_{psg_idx}_para_{p_idx}_sent_{s_idx}{suffix}',
                                            'level': 'sentence',
                                            'text': sent.get('text', '')[:50],
                                            'bbox': bbox,
                                            'bboxes': bboxes,
                                            'is_multiline': is_multiline
                                        })

    # === 右パネル ===
    right_panel = coords.get('right_panel', {})

    # questions, choices（explanationページのみtarget_questionで制限、reflection2は全Q表示）
    questions_list = right_panel.get('questions', [])
    questions_to_process = []
    if is_explanation:
        # explanationページでのみQ制限を適用
        if target_question is not None:
            # 特定のQのみ抽出（インデックスを保持）
            if target_question < len(questions_list):
                questions_to_process = [(target_question, questions_list[target_question])]
            else:
                questions_to_process = []
        else:
            # デフォルト動作: Q1のみ（インデックスを保持）
            if questions_list:
                questions_to_process = [(0, questions_list[0])]
    else:
        # reflection2/pre/postなど他のページでは全questionsを抽出（インデックスを保持）
        questions_to_process = list(enumerate(questions_list))

    # training2の類題2,3(tr_02_an2, tr_02_an3)判定用
    analog_id = coords.get('analog_id', '')

    for q_idx, question in questions_to_process:
        # question の bbox
        # explanation画面ではquestion_text_en/question_text_jaが使用される（target_localeで制御）
        # pre/post画面ではquestion_textまたはpositionを使用

        # 共通のposition（全言語で同じ位置を使う場合）
        if 'position' in question:
            common_bbox = question['position']
            common_bboxes = [common_bbox]
            common_is_multiline = False
        elif 'question_bbox' in question:
            common_bbox = question['question_bbox']
            common_bboxes = [common_bbox]
            common_is_multiline = False
        else:
            common_bbox = None
            common_bboxes = None
            common_is_multiline = False

        # question_text_en（英語設問テキスト）
        q_text_en = question.get('question_text_en', {})
        if include_en and q_text_en and 'question' in levels:
            if 'lines' in q_text_en:
                bboxes, bbox, is_multiline = _lines_to_bboxes(q_text_en['lines'])
            elif common_bbox:
                bbox, bboxes, is_multiline = common_bbox, common_bboxes, common_is_multiline
            else:
                bbox = None
            if bbox:
                aois.append({
                    'id': f'question_{q_idx}_en',
                    'level': 'question',
                    'text': q_text_en.get('text', '')[:50],
                    'bbox': bbox,
                    'bboxes': bboxes,
                    'is_multiline': is_multiline
                })

        # question_text_ja（日本語設問テキスト）
        q_text_ja = question.get('question_text_ja', {})
        if include_ja and q_text_ja and 'question' in levels:
            if 'lines' in q_text_ja:
                bboxes, bbox, is_multiline = _lines_to_bboxes(q_text_ja['lines'])
            elif common_bbox:
                bbox, bboxes, is_multiline = common_bbox, common_bboxes, common_is_multiline
            else:
                bbox = None
            if bbox:
                aois.append({
                    'id': f'question_{q_idx}_ja',
                    'level': 'question',
                    'text': q_text_ja.get('text', '')[:50],
                    'bbox': bbox,
                    'bboxes': bboxes,
                    'is_multiline': is_multiline
                })

        # question_text（言語なしのフォールバック、pre/postなど）
        q_text = question.get('question_text', {})
        if not q_text_en and not q_text_ja and q_text and 'question' in levels:
            if 'lines' in q_text:
                bboxes, bbox, is_multiline = _lines_to_bboxes(q_text['lines'])
            elif common_bbox:
                bbox, bboxes, is_multiline = common_bbox, common_bboxes, common_is_multiline
            else:
                bbox = None
            if bbox:
                aois.append({
                    'id': f'question_{q_idx}',
                    'level': 'question',
                    'text': q_text.get('text', question.get('text', ''))[:50],
                    'bbox': bbox,
                    'bboxes': bboxes,
                    'is_multiline': is_multiline
                })

        # choices の bbox（choice_bboxを使用）
        for c_idx, choice in enumerate(question.get('choices', [])):
            if 'choice_bbox' in choice:
                bbox = dict(choice['choice_bbox'])  # コピーを作成
            elif 'bbox' in choice:
                bbox = dict(choice['bbox'])
            elif 'position' in choice:
                bbox = dict(choice['position'])
            else:
                bbox = None

            # training2の類題2(tr_02_an2)の解説画面のQ2で、日本語モードの場合のみy座標を20上にずらす
            # （英語Q2は設問が2行、日本語Q2は1行のため、選択肢位置がずれる）
            if (bbox and include_ja and not include_en
                and is_explanation and analog_id in ('tr_02_an2', 'tr_02_an3') and q_idx == 1):
                bbox['y'] = bbox['y'] - 20

            if bbox and 'choice' in levels:
                aois.append({
                    'id': f'question_{q_idx}_choice_{c_idx}',
                    'level': 'choice',
                    'text': choice.get('choice_text_en', {}).get('text', choice.get('text', ''))[:50],
                    'bbox': bbox
                })

        # explanation (training_explanation, analog_explanation画面)
        if 'explanation' in levels:
            explanation = question.get('explanation', {})
            if explanation and 'lines' in explanation:
                bbox = _lines_to_bbox(explanation['lines'])
                if bbox:
                    # training2の類題2(tr_02_an2)の解説画面のQ2で、日本語モードの場合のみy座標を20上にずらす
                    if (include_ja and not include_en
                        and is_explanation and analog_id in ('tr_02_an2', 'tr_02_an3') and q_idx == 1):
                        bbox = dict(bbox)
                        bbox['y'] = bbox['y'] - 20
                    aois.append({
                        'id': f'question_{q_idx}_explanation',
                        'level': 'explanation',
                        'text': explanation.get('text', '')[:50],
                        'bbox': bbox
                    })

        # metacog_feedback (メタ認知フィードバック - B群のtraining_explanation画面)
        if 'metacog' in levels:
            metacog = question.get('metacog_feedback', {})
            if metacog and 'lines' in metacog:
                bbox = _lines_to_bbox(metacog['lines'])
                if bbox:
                    # training2の類題2(tr_02_an2)の解説画面のQ2で、日本語モードの場合のみy座標を20上にずらす
                    if (include_ja and not include_en
                        and is_explanation and analog_id in ('tr_02_an2', 'tr_02_an3') and q_idx == 1):
                        bbox = dict(bbox)
                        bbox['y'] = bbox['y'] - 20
                    aois.append({
                        'id': f'question_{q_idx}_metacog',
                        'level': 'metacog',
                        'text': metacog.get('text', '')[:50],
                        'bbox': bbox
                    })

    # ui_components内のtimer
    ui_components = right_panel.get('ui_components', {})
    timer = ui_components.get('timer', {})
    if timer and 'timer' in levels:
        # 直接座標が入っている場合
        if 'x' in timer and 'y' in timer:
            bbox = timer
        elif 'position' in timer:
            bbox = timer['position']
        elif 'lines' in timer:
            bbox = _lines_to_bbox(timer['lines'])
        else:
            bbox = None
        if bbox:
            aois.append({
                'id': 'timer',
                'level': 'timer',
                'text': 'Timer',
                'bbox': bbox
            })

    # reflection_form（right_panel直下、またはanalogs内から取得）
    reflection_form = right_panel.get('reflection_form', {})
    reflection_form_analog_id = ''

    # right_panel直下にない場合、analogs内から取得を試みる
    if not reflection_form:
        rp_analogs_temp = right_panel.get('analogs', [])
        if rp_analogs_temp:
            # target_analogが指定されていればそれを使用、なければ最初のanalogを使用
            target_idx = target_analog if target_analog is not None and target_analog < len(rp_analogs_temp) else 0
            if target_idx < len(rp_analogs_temp):
                reflection_form = rp_analogs_temp[target_idx].get('reflection_form', {})
                reflection_form_analog_id = rp_analogs_temp[target_idx].get('analog_id', '')

    # training2のreflection2の類題2,3で日本語モードの場合、-20オフセット適用
    is_reflection_form_tr02_an2_ja = (is_reflection2 and passage_id == 'tr_02'
                                      and reflection_form_analog_id in ('tr_02_an2', 'tr_02_an3')
                                      and include_ja and not include_en)

    if reflection_form and 'reflection' in levels:
        # prompt
        prompt = reflection_form.get('prompt', {})
        if prompt:
            if 'lines' in prompt:
                bbox = _lines_to_bbox(prompt['lines'])
            elif 'position' in prompt:
                bbox = prompt['position']
            else:
                bbox = None
            if bbox:
                # training2のreflection2の類題2で日本語モードは-20オフセット
                if is_reflection_form_tr02_an2_ja:
                    bbox = dict(bbox)
                    bbox['y'] = bbox['y'] - 20
                aois.append({
                    'id': 'reflection_prompt',
                    'level': 'reflection',
                    'text': prompt.get('text', '')[:50],
                    'bbox': bbox
                })

        # textarea
        textarea = reflection_form.get('textarea', {})
        if textarea and 'x' in textarea:
            # training2のreflection2の類題2で日本語モードは-20オフセット
            if is_reflection_form_tr02_an2_ja:
                textarea = dict(textarea)
                textarea['y'] = textarea['y'] - 20
            aois.append({
                'id': 'reflection_textarea',
                'level': 'reflection',
                'text': 'textarea',
                'bbox': textarea
            })

    # === 右パネルのanalogs (reflection2で使用) ===
    # reflection2画面: 類題1のみ、Q1-3すべて、英語のみ（デフォルト）
    # explanation画面: 類題1のみ、Q1のみ、英語のみ（デフォルト）
    rp_analogs_list = right_panel.get('analogs', [])
    rp_analogs_to_process = []
    if is_en_only_page_default:
        # explanation/reflection2ページでのみ制限を適用
        if target_analog is not None:
            # 特定の類題のみ抽出（インデックスを保持）
            if target_analog < len(rp_analogs_list):
                rp_analogs_to_process = [(target_analog, rp_analogs_list[target_analog])]
            else:
                rp_analogs_to_process = []
        else:
            # デフォルト動作: 類題1のみ（インデックスを保持）
            if rp_analogs_list:
                rp_analogs_to_process = [(0, rp_analogs_list[0])]
    else:
        # 他のページでは全analogsを抽出（インデックスを保持）
        rp_analogs_to_process = list(enumerate(rp_analogs_list))

    for an_idx, analog in rp_analogs_to_process:
        # training2のreflection2の類題2,3(tr_02_an2, tr_02_an3)で日本語モードの場合、Q2以降に-20オフセット適用
        rp_analog_id = analog.get('analog_id', '')
        is_tr02_an2_ja = (is_reflection2 and passage_id == 'tr_02'
                         and rp_analog_id in ('tr_02_an2', 'tr_02_an3')
                         and include_ja and not include_en)

        # analogs内のquestions（explanationページのみtarget_questionで制限、reflection2は全Q表示）
        questions_list = analog.get('questions', [])
        questions_to_process = []
        if is_explanation:
            # explanationページでのみQ制限を適用
            if target_question is not None:
                # 特定のQのみ抽出（インデックスを保持）
                if target_question < len(questions_list):
                    questions_to_process = [(target_question, questions_list[target_question])]
                else:
                    questions_to_process = []
            else:
                # デフォルト動作: Q1のみ（インデックスを保持）
                if questions_list:
                    questions_to_process = [(0, questions_list[0])]
        else:
            # reflection2/他のページではQ1-3すべて（制限なし、インデックスを保持）
            questions_to_process = list(enumerate(questions_list))

        for q_idx, question in questions_to_process:
            # 多言語対応
            q_text_en = question.get('question_text_en', {})
            q_text_ja = question.get('question_text_ja', {})

            # question_text_en
            # question_text_en（target_localeで制御）
            if q_text_en and include_en and 'question' in levels:
                if 'lines' in q_text_en:
                    bboxes, bbox, is_multiline = _lines_to_bboxes(q_text_en['lines'])
                else:
                    bbox = None
                    bboxes = None
                    is_multiline = False
                if bbox:
                    aois.append({
                        'id': f'analog_{an_idx}_question_{q_idx}_en',
                        'level': 'question',
                        'text': q_text_en.get('text', '')[:50],
                        'bbox': bbox,
                        'bboxes': bboxes,
                        'is_multiline': is_multiline
                    })

            # question_text_ja（target_localeで制御）
            if q_text_ja and include_ja and 'question' in levels:
                if 'lines' in q_text_ja:
                    bboxes, bbox, is_multiline = _lines_to_bboxes(q_text_ja['lines'])
                else:
                    bbox = None
                    bboxes = None
                    is_multiline = False
                if bbox:
                    # training2のreflection2の類題2でQ3以降は-20オフセット（設問文）
                    if is_tr02_an2_ja and q_idx >= 2:
                        bbox = dict(bbox)
                        bbox['y'] = bbox['y'] - 20
                        # multilineの場合、各行のbboxも修正
                        if bboxes:
                            bboxes = [dict(b) for b in bboxes]
                            for b in bboxes:
                                b['y'] = b['y'] - 20
                    aois.append({
                        'id': f'analog_{an_idx}_question_{q_idx}_ja',
                        'level': 'question',
                        'text': q_text_ja.get('text', '')[:50],
                        'bbox': bbox,
                        'bboxes': bboxes,
                        'is_multiline': is_multiline
                    })

            # choices
            if 'choice' in levels:
                for c_idx, choice in enumerate(question.get('choices', [])):
                    # bboxがある場合はそれを使用
                    choice_bbox = choice.get('bbox')
                    if choice_bbox:
                        # training2のreflection2の類題2でQ2以降は-20オフセット
                        if is_tr02_an2_ja and q_idx >= 1:
                            choice_bbox = dict(choice_bbox)
                            choice_bbox['y'] = choice_bbox['y'] - 20
                        aois.append({
                            'id': f'analog_{an_idx}_question_{q_idx}_choice_{c_idx}',
                            'level': 'choice',
                            'text': choice.get('choice_text_en', {}).get('text', '')[:50],
                            'bbox': choice_bbox
                        })
                    else:
                        # bboxがない場合はchoice_text_en/jaから抽出
                        choice_text_en = choice.get('choice_text_en', {})
                        choice_text_ja = choice.get('choice_text_ja', {})
                        choice_text = choice.get('choice_text', {})

                        # target_localeで言語を制御
                        langs_to_process = []
                        if include_en:
                            langs_to_process.append(('en', choice_text_en))
                        if include_ja:
                            langs_to_process.append(('ja', choice_text_ja))
                        langs_to_process.append(('', choice_text))  # フォールバック用

                        for lang, ct in langs_to_process:
                            if ct and 'lines' in ct:
                                bbox = _lines_to_bbox(ct['lines'])
                                # training2のreflection2の類題2でQ2以降は-20オフセット
                                if is_tr02_an2_ja and q_idx >= 1:
                                    bbox = dict(bbox)
                                    bbox['y'] = bbox['y'] - 20
                                suffix = f'_{lang}' if lang else ''
                                aois.append({
                                    'id': f'analog_{an_idx}_question_{q_idx}_choice_{c_idx}{suffix}',
                                    'level': 'choice',
                                    'text': ct.get('text', '')[:50],
                                    'bbox': bbox
                                })

    # === メタデータ (旧形式対応) ===
    metadata = coords.get('metadata', {})
    old_timer = metadata.get('timer', {})
    if old_timer and 'timer' in levels:
        if 'position' in old_timer:
            bbox = old_timer['position']
        elif 'lines' in old_timer:
            bbox = _lines_to_bbox(old_timer['lines'])
        elif 'x' in old_timer:
            bbox = old_timer
        else:
            bbox = None
        if bbox:
            # 重複チェック
            if not any(a['id'] == 'timer' for a in aois):
                aois.append({
                    'id': 'timer',
                    'level': 'timer',
                    'text': 'Timer',
                    'bbox': bbox
                })

    # === フッター ===
    footer = coords.get('footer', {})

    # submit_button
    submit_button = footer.get('submit_button', {})
    if submit_button and 'ui' in levels:
        if 'position' in submit_button:
            bbox = submit_button['position']
        elif 'lines' in submit_button:
            bbox = _lines_to_bbox(submit_button['lines'])
        elif 'x' in submit_button and 'y' in submit_button:
            bbox = submit_button
        else:
            bbox = None
        if bbox:
            aois.append({
                'id': 'submit_button',
                'level': 'ui',
                'text': submit_button.get('text', 'Submit'),
                'bbox': bbox
            })

    # confirm_button (analog_questionで使用)
    confirm_button = footer.get('confirm_button', {})
    if confirm_button and 'ui' in levels:
        if 'x' in confirm_button and 'y' in confirm_button:
            bbox = confirm_button
        else:
            bbox = None
        if bbox:
            aois.append({
                'id': 'confirm_button',
                'level': 'ui',
                'text': confirm_button.get('text', 'Confirm'),
                'bbox': bbox
            })

    return aois


def computeAllAOIRate(fixations, aois):
    """
    全AOIを対象にFixationのAOI内率を計算

    Parameters:
    -----------
    fixations : np.ndarray
        Fixationデータ (N, 8)。[:, 1]がX座標、[:, 2]がY座標
    aois : list of dict
        extractAllAOIs()で抽出したAOIリスト

    Returns:
    --------
    dict
        {
            "rate": 0.567,           # AOI内率（0〜1）
            "fixations_in_aoi": 250, # AOI内のFixation数
            "total_fixations": 441   # 全Fixation数
        }
    """
    if len(fixations) == 0:
        return {"rate": 0.0, "fixations_in_aoi": 0, "total_fixations": 0}

    in_aoi = 0
    for fix in fixations:
        x, y = fix[1], fix[2]
        for aoi in aois:
            # multiline AOIの場合は個別bboxをチェック
            if aoi.get('is_multiline') and 'bboxes' in aoi:
                if _point_in_bboxes(x, y, aoi['bboxes']):
                    in_aoi += 1
                    break
            else:
                bbox = aoi['bbox']
                if (bbox['x'] <= x <= bbox['x'] + bbox['width'] and
                    bbox['y'] <= y <= bbox['y'] + bbox['height']):
                    in_aoi += 1
                    break

    return {
        "rate": in_aoi / len(fixations),
        "fixations_in_aoi": in_aoi,
        "total_fixations": len(fixations)
    }


# =============================================================================
# 視線補正（スケーリング + オフセット）
# =============================================================================

def applyScalingAndOffset(fixations, scale_x=1.0, scale_y=1.0,
                          offset_x=0, offset_y=0,
                          center_x=960, center_y=540):
    """
    スケーリングとオフセットを視線座標に適用

    変換式:
    x' = scale_x * (x - center_x) + center_x + offset_x
    y' = scale_y * (y - center_y) + center_y + offset_y

    Parameters:
    -----------
    fixations : np.ndarray
        Fixationデータ (N, 8)。[:, 1]がX座標、[:, 2]がY座標
    scale_x, scale_y : float
        X/Y方向のスケーリング係数（1.0=等倍）
    offset_x, offset_y : float
        X/Y方向のオフセット（ピクセル）
    center_x, center_y : float
        スケーリングの基準点（通常は画面中心）

    Returns:
    --------
    np.ndarray
        補正後のFixationデータ
    """
    corrected = fixations.copy()
    corrected[:, 1] = scale_x * (fixations[:, 1] - center_x) + center_x + offset_x
    corrected[:, 2] = scale_y * (fixations[:, 2] - center_y) + center_y + offset_y
    return corrected


def estimateOffsetWithScaling(fixations, aois,
                               search_range_x=(-30, 30),
                               search_range_y=(-50, 50),
                               scale_range=(0.90, 1.10),
                               offset_step=10,
                               scale_step=0.02,
                               center_x=960, center_y=540,
                               verbose=True):
    """
    スケーリング + オフセットの最適パラメータをグリッドサーチで推定

    Parameters:
    -----------
    fixations : np.ndarray
        Fixationデータ (N, 8)
    aois : list of dict
        AOIリスト
    search_range_x, search_range_y : tuple
        オフセットの探索範囲 (min, max) ピクセル
    scale_range : tuple
        スケーリングの探索範囲 (min, max)
    offset_step : int
        オフセットの探索刻み幅（ピクセル）
    scale_step : float
        スケーリングの探索刻み幅
    center_x, center_y : float
        スケーリングの基準点
    verbose : bool
        進捗表示するか

    Returns:
    --------
    dict
        {
            "best_offset_x": float,
            "best_offset_y": float,
            "best_scale_x": float,
            "best_scale_y": float,
            "best_rate": float,
            "search_results": list  # (offset_x, offset_y, scale_x, scale_y, rate)
        }
    """
    if len(fixations) == 0:
        return {
            "best_offset_x": 0.0,
            "best_offset_y": 0.0,
            "best_scale_x": 1.0,
            "best_scale_y": 1.0,
            "best_rate": 0.0,
            "search_results": []
        }

    results = []
    best_rate = 0.0
    best_params = (0, 0, 1.0, 1.0)

    # スケーリング値のリスト
    scale_values = np.arange(scale_range[0], scale_range[1] + scale_step, scale_step)
    offset_x_values = range(search_range_x[0], search_range_x[1] + 1, offset_step)
    offset_y_values = range(search_range_y[0], search_range_y[1] + 1, offset_step)

    total_iterations = len(scale_values) * len(scale_values) * len(offset_x_values) * len(offset_y_values)

    if verbose:
        print(f"探索空間: {len(scale_values)} scales × {len(offset_x_values)} X offsets × {len(offset_y_values)} Y offsets")
        print(f"合計: {total_iterations} 通り")

    iteration = 0
    for scale_x in scale_values:
        for scale_y in scale_values:
            for offset_x in offset_x_values:
                for offset_y in offset_y_values:
                    # 変換適用
                    corrected_fix = applyScalingAndOffset(
                        fixations, scale_x, scale_y, offset_x, offset_y,
                        center_x, center_y
                    )

                    # AOI内率を計算
                    rate_info = computeAllAOIRate(corrected_fix, aois)
                    rate = rate_info['rate']

                    results.append((offset_x, offset_y, scale_x, scale_y, rate))

                    if rate > best_rate:
                        best_rate = rate
                        best_params = (offset_x, offset_y, scale_x, scale_y)

                    iteration += 1

        # 進捗表示（scale_xごと）
        if verbose:
            print(f"  scale_x={scale_x:.2f} 完了 ({iteration}/{total_iterations})")

    return {
        "best_offset_x": float(best_params[0]),
        "best_offset_y": float(best_params[1]),
        "best_scale_x": float(best_params[2]),
        "best_scale_y": float(best_params[3]),
        "best_rate": best_rate,
        "search_results": results
    }

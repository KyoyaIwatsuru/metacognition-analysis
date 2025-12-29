import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from scipy.stats import multivariate_normal


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


def readTobiiDataWithEventLog(eye_tracking_dir, event_log_path,
                                event_type="question_screen_open",
                                end_event_type="phase_complete_enter",
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
    イベントログベースで視線データを読み込み・セグメント化

    Parameters:
    -----------
    eye_tracking_dir : str
        eye_trackingディレクトリ（tobii_pro_gaze.csvと背景画像を含む）
    event_log_path : str
        events.jsonlのパス
    event_type : str
        セグメント化に使うイベントタイプ
    end_event_type : str
        最後のセグメントの終了イベントタイプ（デフォルト: "phase_complete_enter"）
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
        各セグメントの情報
        [{"passage_id": str,
          "data": np.array,
          "image_path": str,
          "image_number": str}, ...]
    """
    import os

    # イベント読み込み
    events = readEventLog(event_log_path, event_type)

    # 終了イベントのタイムスタンプを取得
    end_events = readEventLog(event_log_path, end_event_type)
    end_timestamp = end_events[0]['timestamp'] if end_events else None

    # 視線データセグメント化（tobii_pro_gaze.csvを使用）
    gaze_csv = os.path.join(eye_tracking_dir, "tobii_pro_gaze.csv")
    segments = segmentGazeDataByEvents(
        gaze_csv, events, end_timestamp,
        apply_head_correction=apply_head_correction,
        correction_method=correction_method,
        # geometric方式用
        use_average_reference=use_average_reference,
        correct_y=correct_y,
        calibration_head_x=calibration_head_x,
        calibration_head_y=calibration_head_y,
        screen_width_mm=screen_width_mm,
        screen_height_mm=screen_height_mm,
        screen_width_px=screen_width_px,
        screen_height_px=screen_height_px,
        # trackbox方式用
        correction_factor_x=correction_factor_x,
        correction_factor_y=correction_factor_y,
        calibration_center_x=calibration_center_x,
        calibration_center_y=calibration_center_y
    )

    # 背景画像パスを追加
    for segment in segments:
        img_num = passageIdToImageNumber(segment['passage_id'])
        segment['image_number'] = img_num
        segment['image_path'] = os.path.join(eye_tracking_dir, f"{img_num}_back.png")

    return segments

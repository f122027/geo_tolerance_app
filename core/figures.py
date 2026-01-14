import numpy as np
import matplotlib.pyplot as plt

def make_tolerance_figure(
    h_nominal=20.0,
    pos_tol=0.2,
    para_tol=0.1,
    flat_tol=0.1,
    slope_amount=0.2,   # 左端と右端の高さ差 [mm]
    amplitude=0.06,     # 凹凸の最大値−最小値（振幅）[mm]
    center_offset=0.0,  # 実測平均中心のオフセット量
    show_real=True,     # 実測形状の表示/非表示
    show_pos=True,
    show_para=True,
    show_flat=True,
):
    """
    位置度・平行度・平面度の公差域 + ランダムな実測形状 を描画し，
    それぞれの公差を満たしているかどうかも返す
    """

    length = 50  # x方向の長さ

    # ===== 1 枚だけのプロット領域 =====
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    # データム A（y=0 の基準線）
    ax.hlines(0, 0, length, colors="black", linewidth=2)

    # --- 実測形状（傾き + ランダム凹凸） ---
    xs = np.linspace(0, length, 300)

    # 傾き：左右端で slope_amount [mm] 差が出るように
    slope = slope_amount / length
    base_line = h_nominal + slope * (xs - length / 2)

    # 凹凸ノイズ
    noise = np.random.uniform(-amplitude / 2, amplitude / 2, xs.size)
    ys = base_line + noise

    #  実測平均中心を意図的にずらす処理 
    ys = ys + center_offset

    # 凡例用ハンドル
    handles = []
    labels = []

    # 実測形状（表示ONのときだけ描画）
    if show_real:
        real_line, = ax.plot(xs, ys, color="black", linewidth=1.2, label="実測形状")
        handles.append(real_line)
        labels.append("実測形状")

    # 真の位置（水平破線）※常に描画して凡例にも入れる
    true_line = ax.hlines(
        h_nominal, 0, length,
        colors="gray", linestyles="dashed", linewidth=1.2
    )
    handles.append(true_line)
    labels.append(f"真の位置 {h_nominal:.1f}")

    y_min = ys.min()
    y_max = ys.max()

    # =========================
    # 公差域の描画
    # =========================

    # 位置度（真の位置まわりの水平帯：データム基準）
    if show_pos:
        y1_pos = h_nominal - pos_tol / 2
        y2_pos = h_nominal + pos_tol / 2
        h_pos = ax.fill_between(
            [0, length], y1_pos, y2_pos,
            color="red", alpha=0.25, edgecolor="red"
        )
        handles.append(h_pos)
        labels.append(f"位置度 ±{pos_tol/2:.3g}（データム基準）")
        y_min = min(y_min, y1_pos)
        y_max = max(y_max, y2_pos)
    else:
        y1_pos = h_nominal - pos_tol / 2
        y2_pos = h_nominal + pos_tol / 2

    # 平行度（実測平均高さまわりの水平帯：データムに平行）
    actual_center = ys.mean()
    y1_para = actual_center - para_tol / 2
    y2_para = actual_center + para_tol / 2

    if show_para:
        h_para = ax.fill_between(
            [0, length], y1_para, y2_para,
            color="blue", alpha=0.25, edgecolor="blue"
        )
        ax.hlines(actual_center, 0, length,
                  colors="blue", linestyles="dashed")
        handles.append(h_para)
        labels.append(f"平行度 ±{para_tol/2:.3g}（データムに平行）")
        y_min = min(y_min, y1_para)
        y_max = max(y_max, y2_para)

    # 平面度（傾き自由：実測形状にフィットした帯）
    a, b = np.polyfit(xs, ys, 1)
    fit_y = a * xs + b
    y1_flat = fit_y - flat_tol / 2
    y2_flat = fit_y + flat_tol / 2

    if show_flat:
        h_flat = ax.fill_between(
            xs, y1_flat, y2_flat,
            color="green", alpha=0.25, edgecolor="green"
        )
        handles.append(h_flat)
        labels.append(f"平面度 ±{flat_tol/2:.3g}（傾き自由）")

    # y_min, y_max は判定に関係なく平面度帯も反映
    y_min = min(y_min, y1_flat.min())
    y_max = max(y_max, y2_flat.max())

    # =========================
    # y軸スケール（余白0.5倍）
    # =========================
    pad = max((y_max - y_min) * 0.5, 0.1)
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_ylabel("高さ [mm]（公差域拡大）")
    ax.set_xlabel("部品の長さ方向")
    ax.grid(True, axis="y", linestyle=":")

    # =========================
    # 判定ロジック
    # =========================
    pos_ok = np.all((ys >= y1_pos) & (ys <= y2_pos))
    para_ok = np.all((ys >= y1_para) & (ys <= y2_para))
    flat_deviation = np.abs(ys - fit_y)
    flat_ok = flat_deviation.max() <= flat_tol / 2

    results = {
        "position": pos_ok,
        "parallel": para_ok,
        "flat": flat_ok,
    }

    # =========================
    # 凡例（x軸の下）
    # =========================
    fig.subplots_adjust(bottom=0.30)
    if handles:  # 何か1つでもあれば凡例を出す
        ax.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=min(len(handles), 4),
            frameon=True
        )

    return fig, results


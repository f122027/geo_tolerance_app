import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.rcParams['font.family'] = 'Meiryo'
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ===== 日本語フォントの設定 =====
font_path = "fonts/NotoSansJP-Regular.otf"  # プロジェクト内のフォント
jp_font = fm.FontProperties(fname=font_path)


from PIL import Image, ImageDraw, ImageFont

from PIL import Image, ImageDraw, ImageFont

def make_straightness_gdt_image(tol_value: float) -> Image.Image:
    """真直度公差値を画像に描く"""

    # ベース画像
    base = Image.open("images/straightness_base.png").convert("RGBA")
    img = base.copy()
    draw = ImageDraw.Draw(img)

    # 表示テキスト
    text = f"{tol_value:.2f}"

    # フォント（メイリオ）
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 40)
    except:
        font = ImageFont.load_default()

    # Pillow 最新版では textsize() は使えない
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # テキスト位置
    x = 900
    y = 65

    # 背景の白塗り
    padding = 10
    draw.rectangle(
        [x - padding, y - padding, x + w + padding, y + h + padding],
        fill="white"
    )

    # 新しい値を書く
    draw.text((x, y), text, fill="black", font=font)

    return img


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
    それぞれの公差を満たしているかどうかも返す（1段構成）。
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
            frameon=True,
            prop=jp_font  
        )

    return fig, results





def make_annotated_image(
    image_path,
    scale=0.45,
    **kwargs,
):
    """
    新しい図面画像に a, b, c (位置度・平行度・平面度) を描き込む。
    a: 位置度公差
    b: 平行度公差
    c: 平面度公差
    """

    # --- 画像読み込み ---
    img = Image.open(image_path).convert("RGBA")

    # --- 画像リサイズ ---
    if scale != 1.0:
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return img



st.set_page_config(page_title="幾何公差ビジュアル教材", layout="wide")

# =========================
# ページごとの表示関数
# =========================
def page_home():
    st.title("幾何公差")
    st.subheader("メニュー")

    st.markdown(
        """
        このアプリでは、幾何公差を理解するための図を表示します。

        左の **「メニュー」** から見たい項目を選んでください。

        
        """
    )


def show_gdt_callout(kind: str):
    """公差ごとの図面上の指示例（簡易版）を表示する"""
    st.markdown("---")
    st.markdown("#### 図面での指示例")

    if kind == "straightness":
        st.markdown(
            """
            **例：軸の外形真直度 0.05**

            ```text
            Φ10 ───────────── 軸

                 ┌────┬───────┐
                 │  | │  0.05 │   ← 真直度記号（縦棒）と公差値
                 └────┴───────┘
            ```

            - 特徴：単独形体（データム参照なし）の真直度
            - 図面では、対象となる線や軸の寸法のそばに
              **幾何特性記号 + 公差値** をフレームで指示します。
            """
        )
    
    else:
        st.info("この公差種別の図示例はまだ登録されていません。")


def page_straightness():
    st.title("真直度のイメージ")

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("真直度とは？")
        st.write(
            """
            **真直度** は，
            「直線形体が幾何学的に正しい直線からどれだけずれているか」を表した幾何公差です。

            - 許容：公差域内（平行二平面間）に実際の線が入っていれば OK
            """
        )

        tol = st.slider("真直度公差（mm）", 0.00, 0.20, 0.05, 0.01)
        noise = st.slider("直線形状のゆがみ", 0.00, 0.20, 0.08, 0.01)

        st.write(f"- 公差：±{tol/2:.3f} mm とみなして表示")
        st.write(f"- 実際の線にランダムなゆがみ（±{noise/2:.3f} mm程度）を付加")

        # ▼ ここで図面画像を生成＆表示 ▼
        img_col1, img_col2 = st.columns(2)

        with img_col1:
            # スライダー連動の図面画像
            img = make_straightness_gdt_image(tol)
            st.image(
                img,
                caption="図面での真直度指示（値はスライダーと連動）",
                use_container_width=True,
            )

        with img_col2:
            st.image(
                "images/straightness_zone.jpg",
                caption="真直度の公差域のイメージ",
                use_container_width=True,
            )

    with col_right:
        x = np.linspace(0, 100, 200)
        ideal_y = np.zeros_like(x)
        actual_y = np.random.uniform(-noise/2, noise/2, size=x.shape)

        lower = -tol / 2
        upper =  tol / 2

        fig, ax = plt.subplots()
        ax.plot(x, ideal_y, linestyle="--", label="理想直線")
        ax.plot(x, actual_y, label="実際の線")
        ax.fill_between(x, lower, upper, alpha=0.2, label="公差域")

        out_idx = np.where((actual_y < lower) | (actual_y > upper))[0]
        if len(out_idx) > 0:
            ax.scatter(x[out_idx], actual_y[out_idx], marker="x", label="公差アウト", zorder=3)

        ax.set_xlabel("長さ方向 [mm]")
        ax.set_ylabel("偏差 [mm]")
        ax.grid(True)
        ax.legend(prop=jp_font)

        st.pyplot(fig)


def page_roundness():
    st.title("真円度のイメージ")

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("真円度とは？")
        st.write(
            """
            **真円度** は，
            「断面の形状が，理想的な円からどれくらいズレてよいか」
            を指定する幾何公差です。

            - 理想：完全な円
            - 許容：同心の2円の間に実形状が入っていれば OK
            """
        )

        tol = st.slider("真円度公差（mm）", 0.00, 0.20, 0.05, 0.01)
        noise = st.slider("凹凸の最大値−最小値（振幅）[mm]", 0.00, 0.20, 0.08, 0.01)
        mag = st.slider("半径方向の表示倍率（見た目のみ）", 10.0, 200.0, 80.0, 10.0)

        st.write(f"- 公差：円半径の ±{tol/2:.3f} mm とみなして判定しています。")
        st.write(f"- 図では半径方向のズレを **{mag:.0f} 倍** に誇張して表示しています。")

    show_gdt_callout("roundness")

    with col_right:
        # 角度（0〜2π）
        theta = np.linspace(0, 2 * np.pi, 400)

        # 名目半径
        R0 = 10.0  # [mm]

        # 理想半径（評価用）
        ideal_r = np.ones_like(theta) * R0

        # 実際の半径（評価用）…ノイズをそのまま加える
        actual_r = R0 + np.random.uniform(-noise / 2, noise / 2, size=theta.shape)

        # 真円度公差（評価用）
        lower = R0 - tol / 2
        upper = R0 + tol / 2

        # 判定は実寸の半径で行う
        inside = (actual_r >= lower) & (actual_r <= upper)
        outside = ~inside

        # -------------------------------
        # 表示用：半径方向だけ倍率 mag をかける
        # -------------------------------
        r_center_vis = R0  # 図の中心となる半径（見た目用）

        # 「理想半径からのズレ」を mag 倍にして表示用半径に変換
        dev = actual_r - R0                 # 実際のズレ [mm]
        actual_r_vis = r_center_vis + dev * mag

        lower_vis = r_center_vis - (tol / 2) * mag
        upper_vis = r_center_vis + (tol / 2) * mag

        # 座標変換（表示用）
        x_ideal = R0 * np.cos(theta)
        y_ideal = R0 * np.sin(theta)

        x_actual = actual_r_vis * np.cos(theta)
        y_actual = actual_r_vis * np.sin(theta)

        fig, ax = plt.subplots(figsize=(5, 5))

        # -----------------------------
        # 先に 真円度公差帯（透明色リング） を描く
        # -----------------------------
        circle_tol = plt.Circle(
            (0, 0),
            upper_vis,
            color="green",
            alpha=0.15,   # 透明度
            fill=True,
            zorder=0,
        )
        ax.add_patch(circle_tol)

        circle_inner = plt.Circle(
            (0, 0),
            lower_vis,
            color="white",   # 背景色でくり抜き
            alpha=1.0,
            fill=True,
            zorder=0,
        )
        ax.add_patch(circle_inner)

        # 公差帯の境界線
        ax.plot(
            upper_vis * np.cos(theta), upper_vis * np.sin(theta),
            color="green", linestyle="--", linewidth=1, zorder=1
        )
        ax.plot(
            lower_vis * np.cos(theta), lower_vis * np.sin(theta),
            color="green", linestyle="--", linewidth=1, zorder=1
        )

        # 理想円
        ax.plot(x_ideal, y_ideal, linestyle="--", color="blue",
                label="理想円", zorder=2)

        # ★ 実際の輪郭：公差内・外すべて線で描く
        ax.plot(
            x_actual,
            y_actual,
            color="orange",
            linewidth=1.2,
            label="実際の輪郭（表示誇張）",
            zorder=3,
        )

        # ★ 公差アウト点だけ赤い × を重ねて強調
        x_out = actual_r_vis[outside] * np.cos(theta[outside])
        y_out = actual_r_vis[outside] * np.sin(theta[outside])
        ax.scatter(
            x_out,
            y_out,
            marker="x",
            color="red",
            label="公差アウト（表示誇張）",
            zorder=4,
        )

        ax.set_aspect("equal", "box")
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.grid(True)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3 ,prop=jp_font)
        fig.subplots_adjust(bottom=0.22)

        st.pyplot(fig, use_container_width=True)



def page_true_position():
    st.title("位置度（2D）のイメージ")

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("位置度とは？")
        st.write(
            """
            **位置度** は，
            「穴や軸の中心が，理論的に正確な位置からどれくらいズレてよいか」
            を指定する幾何公差です。

            - 理想：図面で指示された理論上正確な位置  
            - 許容：公差円の中に軸の中心が入れば OK
            """
        )

        tol = st.slider("位置度公差（直径） [mm]", 0.0, 1.0, 0.4, 0.05)
        n_points = st.slider("穴の個数（サンプル数）", 1, 50, 20)
        mag = st.slider("位置ずれの表示倍率（見た目のみ）", 1.0, 50.0, 10.0, 1.0)

        # ★ ばらつきは固定（評価用の実寸 mm）
        sigma = 0.15
        st.write(f"- 公差：直径 {tol:.2f} mm の円内に入ればOK")
        st.write(f"- 図ではズレを **{mag:.0f} 倍** に誇張表示しています")

    show_gdt_callout("position")

    with col_right:
        # 理想位置（20,20）
        x0, y0 = 20.0, 20.0

        # 穴中心軸を正規分布からサンプル（評価用実寸）
        xs = np.random.normal(x0, sigma, n_points)
        ys = np.random.normal(y0, sigma, n_points)

        radius = tol / 2

        # 合否判定（実寸）
        dist = np.sqrt((xs - x0)**2 + (ys - y0)**2)
        inside = dist <= radius
        outside = ~inside

        # --------------------
        # 表示用：ズレだけ mag 倍
        # --------------------
        dx = xs - x0
        dy = ys - y0
        xs_vis = x0 + dx * mag
        ys_vis = y0 + dy * mag

        radius_vis = radius * mag

        fig, ax = plt.subplots(figsize=(5, 5))

        # --------------------
        # データム（A/B を逆に変更）
        # --------------------
        # データム A → X=0（垂直）
        ax.axvline(0.0, color="black", linewidth=2)

        # データム B → Y=0（水平）
        ax.axhline(0.0, color="black", linewidth=2)

        # --------------------
        # 公差円（誇張表示）
        # --------------------
        circle = plt.Circle(
            (x0, y0),
            radius_vis,
            fill=False,
            linestyle="--",
            color="green",
            alpha=0.8,
            label="位置度公差域（表示誇張）"
        )
        ax.add_artist(circle)

        # 穴中心軸（点）
        ax.scatter(xs_vis[inside], ys_vis[inside],
                   marker="o", color="tab:blue",
                   label="穴の中心軸（合格・表示誇張）")

        ax.scatter(xs_vis[outside], ys_vis[outside],
                   marker="x", color="red",
                   label="穴の中心軸（不合格・表示誇張）")

        # 理想位置
        ax.scatter([x0], [y0], marker="+", s=130,
                   color="black", label="理想位置")

        # 表示範囲調整
        margin = radius_vis * 1.5
        ax.set_xlim(min(0, xs_vis.min()) - margin, max(xs_vis.max(), x0 + radius_vis) + margin)
        ax.set_ylim(min(0, ys_vis.min()) - margin, max(ys_vis.max(), y0 + radius_vis) + margin)

        ax.set_aspect("equal", "box")

        # 軸ラベル
        ax.set_xlabel("データム B からの距離 [mm]")
        ax.set_ylabel("データム A からの距離 [mm]")

        ax.grid(True)

        # データム文字
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(0, ylim[1], "データム B", ha="left", va="top", fontsize=10)
        ax.text(xlim[1], 0, "データム A", ha="right", va="bottom", fontsize=10)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2,prop=jp_font)
        fig.subplots_adjust(bottom=0.25)

        st.pyplot(fig, use_container_width=True)






def page_composite_tolerance():
    st.title("複合幾何公差（複合位置度など）のまとめ")

    st.subheader("複合位置度とは？（ざっくり説明）")
    st.write(
        """
        **複合位置度（Composite Position Tolerance）** は，
        1つの幾何公差フレームの中に **2段（またはそれ以上）** の位置度を組み合わせて，
        「パターン全体の位置」と「パターン内部の相対位置・姿勢」を
        別々に管理するための公差方式です。

        - 上段：パターン全体の位置（データム A|B|C に対して）
        - 下段：パターン内部のピッチ・姿勢など

        詳しい定義や条件は，右側の表（Excel で作成した一覧）を参照してください。
        """
    )

    st.markdown("---")

    st.subheader("複合幾何公差一覧（Excel から読み込み）")

    # ここであなたの Excel ファイル名に合わせて変更してください
    excel_file = "複合位置公差12.xlsm"  # 例：ファイル名

    try:
        df = pd.read_excel(excel_file)
        st.write(f"読み込んだファイル：`{excel_file}`")
        st.dataframe(df, use_container_width=True)
    except FileNotFoundError:
        st.error(
            f"Excel ファイル `{excel_file}` が見つかりませんでした。\n\n"
            "app.py と同じフォルダに配置するか、ファイル名をコード内で修正してください。"
        )
    except Exception as e:
        st.error(f"Excel の読み込み中にエラーが発生しました: {e}")

    st.markdown("---")
    st.write(
        """
        表の各行をもとに，個別のケースを可視化したい場合は，
        例えば「行を選ぶためのセレクトボックス」を追加し，
        そこから上段・下段の公差値やデータム条件を読み取って，
        図を自動生成することもできます（発展案）。
        """
    )

def page_flatness_3d():
    st.title("平面度の 3D モデル")

    # 左右2カラムレイアウト
    col_left, col_right = st.columns([1, 1.4])

    # ============================
    # ▼ 左側（説明＋スライダー）
    # ============================
    with col_left:
        st.subheader("平面度とは？")
        st.write(
            """
            **平面度** は，
            「ある面が理想平面からどれだけズレてよいか」を表す幾何公差です。

            - 評価計算：実際の高さ（mm）
            - 表示：Z 方向だけ **100倍** に誇張して 3D 表示

            """
        )

        # ※ 面のサイズスライダーは削除（固定サイズにします）
        roughness = st.slider("高さ振幅[mm]", 0.00, 0.20, 0.05, 0.01)
        tol = st.slider("平面度公差 [mm]", 0.00, 0.20, 0.10, 0.01)

        # 表示スケール（Z方向のみ拡大）
        scale_z = 100.0
        st.write(f"※ 3D表示では高さ方向を {scale_z:.0f} 倍に誇張しています。")

        # 平面度のGDT記号などの補足
        show_gdt_callout("flatness")

    # ============================
    # ▼ 右側（3D グラフ）
    # ============================
    with col_right:

        # --------------------------
        # 固定サイズ 80mm × 80mm
        # --------------------------
        size = 80

        # ---- 面データ生成 ----
        n = 40
        x = np.linspace(-size / 2, size / 2, n)
        y = np.linspace(-size / 2, size / 2, n)
        X, Y = np.meshgrid(x, y)

        # 実際の高さ（評価用の真の Z [mm]）
        Z_true = np.random.uniform(-roughness / 2, roughness / 2, size=X.shape)

        # ---- 平面度の簡易評価（実寸）----
        max_dev = float(np.max(np.abs(Z_true)))
        flatness = max_dev * 2  # 上下対称とみなして 2*max_dev

        st.write(f"最大偏差（実寸）：{max_dev:.5f} mm")
        st.write(f"平面度（簡易）≈ 2 × 最大偏差 = {flatness:.5f} mm")

        if flatness <= tol:
            st.success("✅ この面は平面度公差 **以内** です（実寸判定：合格）")
        else:
            st.error("❌ この面は平面度公差を **超えています**（実寸判定：不合格）")

        # ---- 表示用に高さを誇張 ----
        Z_vis = Z_true * scale_z
        tol_vis = tol * scale_z

        # ---- 公差外判定（実寸で判定）----
        tol_half = tol / 2.0  # 平面度公差帯を ±tol/2 とみなす
        mask_bad = (np.abs(Z_true) > tol_half)

        # ---- Surface用の2値カラー（公差外=1、公差内=0）----
        surface_color = np.where(mask_bad, 1, 0)

        # ---- Plotly で 3D 表示 ----
        fig = go.Figure()

        # 1) 実際の凸凹面（公差外=赤、公差内=グレー）
        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Z_vis,
                surfacecolor=surface_color,
                colorscale=[
                    [0.0, "lightgrey"],  # 公差内
                    [1.0, "red"],        # 公差外
                ],
                cmin=0,
                cmax=1,
                showscale=False,
                opacity=1.0,
                name="実際の面（公差外=赤）",
            )
        )

        # 2) 理想平面（Z=0）
        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=np.zeros_like(Z_vis),
                showscale=False,
                opacity=0.18,
                colorscale=[[0, "blue"], [1, "blue"]],
                name="理想平面",
            )
        )

        # 3) 公差上限平面（+tol/2）
        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=np.ones_like(Z_vis) * (tol_vis / 2),
                showscale=False,
                opacity=0.15,
                colorscale=[[0, "green"], [1, "green"]],
                name="公差上限（表示用）",
            )
        )

        # 4) 公差下限平面（-tol/2）
        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=np.ones_like(Z_vis) * (-tol_vis / 2),
                showscale=False,
                opacity=0.15,
                colorscale=[[0, "green"], [1, "green"]],
                name="公差下限（表示用）",
            )
        )

        fig.update_layout(
            scene=dict(
                xaxis_title="X [mm]",
                yaxis_title="Y [mm]",
                zaxis_title="高さ [mm]（表示は100倍スケール）",
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, b=0, t=30),
        )

        st.plotly_chart(fig, use_container_width=True)


def page_composite_position_random():
    st.title("複合位置公差")

    st.subheader("複合位置公差とは？")
    st.write(
         """
        - 複合位置公差は，形体相互の位置度とデータムからの位置度とに
        - 異なる公差値を与える公差公式です。
        - 位置の要求は比較的緩いが、姿勢の公差は厳しい場合に使用されます。
        - ・上段：形体グループの位置度
        - ・下段：個々の形体相互の位置度
         """
    )



    show_gdt_callout("composite")

    R_upper = 5.0   # 上段 公差域半径
    R_lower = 2.0   # 下段 公差域半径
    max_offset = R_upper - R_lower  # 中心間距離の上限 = 3

    st.write("""
    - 上段公差域：半径 5
    - 下段公差域：半径 2
    - 穴中心間距離： 30
    - 上段の指示：三平面データム系に対して、Φ5の円筒公差域に入っていることを規制
    - 下段の指示：4つの円筒(Φ2)はデータムAに対して垂直であり、円筒の軸間の距離は30を規制
    - 上段の位置度を満足する範囲で、4つの穴パターンは姿勢を変動できる
    """)

    st.info("ページを再読み込みすると、毎回ランダムな配置が生成されます。")

    # 共通のランダム生成関数（剛体移動パターン）
    def generate_pattern(upper_centers, offsets, max_offset, search_angle=10):
        """
        upper_centers: (N,2) 上段の名目中心
        offsets:       (N,2) パターン形状（穴1原点からの相対位置）
        max_offset:    下段中心と上段中心の距離上限
        """
        for _ in range(10_000):
            # 回転角（例: -search_angle〜+search_angle° の範囲でランダム）
            theta = np.deg2rad(np.random.uniform(-search_angle, search_angle))
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]])

            # 基準となる穴1の位置を、上段穴1のまわり半径 max_offset の円内でランダム生成
            r = np.random.uniform(0, max_offset)
            phi = np.random.uniform(0, 2*np.pi)
            shift = np.array([r*np.cos(phi), r*np.sin(phi)])
            base = upper_centers[0] + shift

            # 下段パターン全体を生成
            lower_centers = (offsets @ R.T) + base  # (N,2)

            # 各穴について「上段円の中に下段円が収まるか」をチェック
            ok = True
            for i in range(len(upper_centers)):
                dist = np.linalg.norm(lower_centers[i] - upper_centers[i])
                if dist > max_offset:
                    ok = False
                    break

            if ok:
                return lower_centers, np.rad2deg(theta)

        return None, None  # 見つからなかった場合

    # ============================
    # 1. 正方形4穴パターン
    # ============================
    st.subheader("① 正方形 4穴パターン")

    # 上段（名目）パターンの中心（Excel 図と同じ 5,35,35,5 の四角）
    upper_square = np.array([
        [ 5.0,  5.0],  # 1
        [35.0,  5.0],  # 2
        [35.0, 35.0],  # 3
        [ 5.0, 35.0],  # 4
    ])

    # パターン形状（穴1 を原点としたときの相対位置）
    offsets_square = np.array([
        [0.0,  0.0],
        [30.0, 0.0],
        [30.0, 30.0],
        [0.0,  30.0],
    ])

    lower_square, theta_sq = generate_pattern(upper_square, offsets_square, max_offset)

    if lower_square is None:
        st.error("正方形パターンの条件を満たす配置が見つかりませんでした。")
    else:
        

        fig1, ax1 = plt.subplots()

        # 上段：大きい円と中心
        for (cx, cy) in upper_square:
            circle = plt.Circle((cx, cy), R_upper, fill=False, color="tab:blue")
            ax1.add_patch(circle)
            ax1.plot(cx, cy, "o", color="tab:blue")

        # 下段：小さい円と中心
        for (cx, cy) in lower_square:
            circle = plt.Circle((cx, cy), R_lower, fill=False, color="tab:orange")
            ax1.add_patch(circle)
            ax1.plot(cx, cy, "o", color="tab:orange")

        # 上段パターンを線で結ぶ
        ax1.plot(
            np.append(upper_square[:, 0], upper_square[0, 0]),
            np.append(upper_square[:, 1], upper_square[0, 1]),
            "-",
            label="上段パターン",
            color="tab:blue",
        )

        # 下段パターンを線で結ぶ
        ax1.plot(
            np.append(lower_square[:, 0], lower_square[0, 0]),
            np.append(lower_square[:, 1], lower_square[0, 1]),
            "-",
            label="下段パターン",
            color="tab:orange",
        )

        ax1.set_aspect("equal", "box")
        ax1.set_xlim(0, 45)
        ax1.set_ylim(0, 45)
        ax1.grid(True)
        ax1.legend()

        st.pyplot(fig1)



    


def page_combined_tolerance():

    st.header("位置度・平行度・平面度の公差域と実測形状")

    # --- 図面イメージ：タイトル直下に小さく表示 ---
    IMAGE_COMBINED = "images/combined_tolerance.png"
    annotated_img = make_annotated_image(
        image_path=IMAGE_COMBINED,
        scale=0.30,   # 画像自体を少し縮小
    )
    st.image(
        annotated_img,
        caption="位置度(a)・平行度(b)・平面度(c) の図面イメージ",
        width=350,
    )

    # --- 公差階層ルールのON/OFF ---
    enforce_rules = st.checkbox(
        "公差の階層（位置度 ≥ 平行度 ≥ 平面度）を自動で守る", value=True
    )

    # --- 公差スライダー（まずは生値を取得） ---
    st.subheader("公差値の設定")
    pos_raw  = st.slider("位置度公差 a [mm]", 0.01, 1.00, 0.20, 0.01)
    para_raw = st.slider("平行度公差 b [mm]", 0.01, 1.00, 0.10, 0.01)
    flat_raw = st.slider("平面度公差 c [mm]", 0.01, 1.00, 0.05, 0.01)

    pos_tol, para_tol, flat_tol = pos_raw, para_raw, flat_raw
    warn_msgs = []

    # --- 階層制御 ---
    if enforce_rules:
        if para_tol < flat_tol:
            warn_msgs.append("c（平面度）は b（平行度）以下である必要があります。c を自動調整しました。")
            flat_tol = para_tol
        if pos_tol < para_tol:
            warn_msgs.append("b（平行度）は a（位置度）以下である必要があります。b を自動調整しました。")
            para_tol = pos_tol
    else:
        if not (pos_tol >= para_tol >= flat_tol):
            warn_msgs.append(
                "※現在の公差値は a ≥ b ≥ c の階層を満たしていません。\n"
                "この状態は図面解釈が不自然になります。"
            )

    # --- 公差ガイド表示 ---
    if warn_msgs:
        for m in warn_msgs:
            st.error(m)
    else:
        st.success("公差階層：a ≥ b ≥ c が成立しています。")

    st.write(f"**使用する公差値**： a={pos_tol:.2f}, b={para_tol:.2f}, c={flat_tol:.2f}")

    # --- 実測形状パラメータ ---
    st.subheader("実測形状のパラメータ")
    slope_amount = st.slider(
        "傾き量（左右端の高さ差）[mm]",
        0.000, 1.000, 0.200, 0.001, format="%.3f"
    )
    amplitude = st.slider(
        "凹凸の最大値−最小値（振幅）[mm]",
        0.000, 1.000, 0.060, 0.001, format="%.3f"
    )

    # 実測平均中心のオフセット量
    center_offset = st.slider(
        "実測平均中心のオフセット量 [mm]（＋で上方向、−で下方向にずらす）",
        -1.000, 1.000, 0.000, 0.001, format="%.3f"
    )

    # --- 公差帯 & 実測形状表示のON/OFF ---
    st.subheader("表示する要素")
    cols = st.columns(4)
    show_real = cols[0].checkbox("実測形状", True)
    show_pos  = cols[1].checkbox("位置度",   True)
    show_para = cols[2].checkbox("平行度",   True)
    show_flat = cols[3].checkbox("平面度",   True)

    # --- 実測形状 + 公差帯のグラフ ---
    fig, results = make_tolerance_figure(
        h_nominal=20.0,
        pos_tol=pos_tol,
        para_tol=para_tol,
        flat_tol=flat_tol,
        slope_amount=slope_amount,
        amplitude=amplitude,
        center_offset=center_offset,
        show_real=show_real,
        show_pos=show_pos,
        show_para=show_para,
        show_flat=show_flat,
    )

    st.pyplot(fig, use_container_width=True)

    # --- 判定結果 ---
    st.subheader("判定結果（実測形状が公差域を満たしているか）")
    c1, c2, c3 = st.columns(3)

    with c1:
        if results["position"]:
            st.success("位置度：OK")
        else:
            st.error("位置度：NG")

    with c2:
        if results["parallel"]:
            st.success("平行度：OK")
        else:
            st.error("平行度：NG")

    with c3:
        if results["flat"]:
            st.success("平面度：OK")
        else:
            st.error("平面度：NG")

    # =====================================================
    # 条件付きサンプル自動生成ボタン
    # =====================================================
    st.subheader("条件を満たす例（自動生成）")
    st.write(
        "- 位置度・平行度・平面度を **すべて満たす**\n"
        "- 傾き・振幅・オフセットが **すべて 0 ではない**\n"
        "ような実測形状を、自動で探して表示します。"
    )

    if st.button("条件を満たす例を自動生成"):
        example_fig = None
        example_params = None
        max_trials = 80  # ランダム探索の試行回数

        for _ in range(max_trials):
            # 0 にはならないよう、ある程度の範囲でランダムに設定
            slope_try = float(np.random.uniform(0.02, 0.50))   # 傾き量 [mm]
            amp_try   = float(np.random.uniform(0.01, 0.30))   # 振幅 [mm]
            offset_try = float(np.random.uniform(-0.30, 0.30)) # オフセット [mm]
            if abs(offset_try) < 0.01:
                continue  # ほぼゼロは避ける

            fig_try, res_try = make_tolerance_figure(
                h_nominal=20.0,
                pos_tol=pos_tol,
                para_tol=para_tol,
                flat_tol=flat_tol,
                slope_amount=slope_try,
                amplitude=amp_try,
                center_offset=offset_try,
                show_real=True,     # サンプルでは全部表示
                show_pos=True,
                show_para=True,
                show_flat=True,
            )

            # 位置度・平行度・平面度すべてOKか
            if res_try["position"] and res_try["parallel"] and res_try["flat"]:
                example_fig = fig_try
                example_params = (slope_try, amp_try, offset_try)
                break

        if example_fig is not None:
            st.pyplot(example_fig, use_container_width=True)
            s, a_, o = example_params
            st.info(
                f"この例では、位置度・平行度・平面度をすべて満たしています。\n\n"
                f"- 傾き量（左右端の高さ差）: **{s:.3f} mm**\n"
                f"- 凹凸振幅: **{a_:.3f} mm**\n"
                f"- 実測平均中心オフセット: **{o:.3f} mm**"
            )
        else:
            st.warning(
                "現在の公差値では、指定した条件を満たす例が見つかりませんでした。\n"
                "公差 a, b, c を少し緩めてから、もう一度お試しください。"
            )

    # =====================================================
    # ここから下：段階的に公差を満たす過程の可視化
    # =====================================================
        


    
    


# =========================
# メイン：メニュー
# =========================
menu = st.sidebar.radio(
    "メニュー",
    [
        "ホーム",
        "真直度",
        "真円度",
        "位置度（2D）",
        "位置度・平行度・平面度の公差域の比較",
        "平面度（3D）",
        "複合幾何公差",
    ],
)


if menu == "ホーム":
    page_home()
elif menu == "真直度":
    page_straightness()
elif menu == "真円度":
    page_roundness()
elif menu == "位置度（2D）":
    page_true_position()
elif menu == "位置度・平行度・平面度の公差域の比較":
    page_combined_tolerance()
elif menu == "平面度（3D）":
    page_flatness_3d()
elif menu == "複合幾何公差":
    page_composite_position_random()

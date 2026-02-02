from core.image_helpers import make_mmc_drawing_example_image
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import textwrap


def page_mmc_straightness():
    st.header("最大実体公差方式（MMC）× 真直度（固定条件）")

    # -----------------------------
    # 固定条件（ユーザー指定）
    # -----------------------------
    L = 100.0                 # 長さ方向固定
    D_NOM = 5.0               # サイズ公差 5 ± 0.2
    D_MIN = 4.8
    D_MMC = 5.2               # 軸なので MMC = 最大径
    STR_MMC = 0.3             # 幾何公差（MMCでの最大実体公差） φ0.3

    # -----------------------------
    # デフォルト値（初回表示用）
    # -----------------------------
    if "mmc_d_actual" not in st.session_state:
        st.session_state.mmc_d_actual = float(D_MMC)
    if "mmc_amp_mag" not in st.session_state:
        st.session_state.mmc_amp_mag = 3.0

    # =============================
    # レイアウト：左=説明/結果/入力、右=図
    # =============================
    colL, colR = st.columns([1.0, 1.2], gap="large")

    # -----------------------------
    # 左：説明（折りたたみ）＋計算結果＋スライダー
    # -----------------------------
    with colL:
        st.markdown("## 最大実体公差方式（MMC）とは")

        with st.expander("説明（開く）", expanded=False):
            st.markdown(textwrap.dedent(
                """
最大実体公差方式（MMC）とは、  
部品が最も材料が多い状態（最大実体）のときに最も厳しい幾何公差を与え、  
実寸がそこから離れるほど、追加の公差（ボーナス公差）を認める考え方です。  
相手部品とのはめ合いを達成したい場合に用いることで、部品を経済的に制作することができます。

最大実体（MMC）とは、部品が **最も材料が多い状態**を指します。  
軸の場合は **直径が最大**のときが MMC（右図の場合は φ5.2）です。

MMC 指示（例：真直度 φ0.3(M)）では、  
- 実寸が **MMCから離れる（細くなる）ほど**  
- 追加で使える公差（ボーナス公差）が増えます

つまり、**細いほど真直度の許容が広がる**、という考え方です。
"""
            ))

        
        d_actual = st.session_state.mmc_d_actual
        amp_mag = st.session_state.mmc_amp_mag

        # -----------------------------
        # 計算（session_stateの現値で表示）
        # -----------------------------
        bonus = max(0.0, D_MMC - d_actual)
        straight_total = STR_MMC + bonus  # 合計 真直度（φ）

        st.markdown("### 計算結果")
        st.write(f"- 基準寸法: φ{D_NOM:.1f} / サイズ公差: 5±0.2（= {D_MIN:.1f}〜{D_MMC:.1f}）")
        st.write(f"- MMC 真直度指示（固定）: φ{STR_MMC:.1f}")
        st.write(f"- 実寸直径: **φ{d_actual:.2f}**")
        st.write(f"- ボーナス: **φ{bonus:.2f}**（= 5.2 − 実寸）")
        st.write(f"- 合計 真直度: **φ{straight_total:.2f}**（= 0.3 + ボーナス）")

        st.markdown("### 入力")
        st.session_state.mmc_d_actual = st.slider(
            "実際の直径（実寸） d [mm]",
            float(D_MIN), float(D_MMC),
            float(st.session_state.mmc_d_actual),
            0.01,
            key="mmc_d_actual_slider",
        )

        st.markdown("### 表示の強調（見た目用）")
        st.session_state.mmc_amp_mag = st.slider(
            "曲がり・公差域の見た目強調倍率",
            1.0, 8.0,
            float(st.session_state.mmc_amp_mag),
            1.0,
            key="mmc_amp_mag_slider",
        )

       

    # -----------------------------
    # 右：図（図面例 + 公差域図 + 動的公差線図）
    # -----------------------------
    with colR:
        # ★ 右側の計算は、最新の slider 値を使う
        d_actual = float(st.session_state.mmc_d_actual)
        amp_mag = float(st.session_state.mmc_amp_mag)

        bonus = max(0.0, D_MMC - d_actual)
        straight_total = STR_MMC + bonus  # 合計 真直度（φ）

        st.subheader("図面での MMC 指示例")
        mmc_img = make_mmc_drawing_example_image(scale=0.6)
        st.image(mmc_img, caption="真直度 φ0.3(M) の図面指示例", use_container_width=True)

        st.subheader("図：公差域と最大曲がり例")

        x = np.linspace(0, L, 600)

        YMIN, YMAX = 0.0, 5.5  # 表示範囲固定

        half_tol = straight_total / 2.0
        half_tol_vis = half_tol * amp_mag

        # 実寸半径（表示用）
        r_vis = d_actual / 2.0

        # 条件を満たす中心線を作る
        C = r_vis
        A = YMAX - 2.0 * r_vis

        # 中心線（表示用）
        y_axis_vis = C + A * np.sin(np.pi * x / L)

        # 軸形体（外形）
        y_top = y_axis_vis + r_vis
        y_bot = y_axis_vis - r_vis

        # 中心線（強調用）
        A_emph = A * amp_mag
        y_axis_emph = C + A_emph * np.sin(np.pi * x / L)

        fig, ax = plt.subplots(figsize=(8.0, 3.6))

        # 公差域（線）
        y_upper = C + A_emph / 2 + half_tol_vis
        y_lower = C + A_emph / 2 - half_tol_vis
        ax.plot([0, L], [y_upper, y_upper], "--", linewidth=2, label="公差域 上限")
        ax.plot([0, L], [y_lower, y_lower], "--", linewidth=2, label="公差域 下限")

        # 公差域（塗り）
        ax.fill_between([0, L], [y_lower, y_lower], [y_upper, y_upper],
                        alpha=0.15, label="公差域（塗り）")

        # 軸線（最大曲がり例）
        ax.plot(x, y_axis_emph, "-", linewidth=2.5, label="軸線（最大曲がり例）")

        # 軸形体
        ax.fill_between(x, y_bot, y_top, alpha=0.25, label="軸形体")
        ax.plot(x, y_top, "-", linewidth=1.0)
        ax.plot(x, y_bot, "-", linewidth=1.0)

        ax.set_xlim(0, L)
        ax.set_ylim(YMIN, YMAX)
        ax.set_xlabel("長さ方向 x")
        ax.set_ylabel("最大実体実行サイズ(5.5)")
        ax.grid(True)

        ax.text(
            0.02 * L, 5.35,
            f"合計真直度 φ{straight_total:.2f}（表示×{amp_mag:.0f}）",
            ha="left", va="top",
        )

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
        fig.subplots_adjust(bottom=0.28)

        st.pyplot(fig, use_container_width=True)

        # ③ 動的公差線図
        st.subheader("動的公差線図")

        d_line = np.linspace(D_MIN, D_MMC, 200)
        t_line = STR_MMC + np.maximum(0.0, D_MMC - d_line)

        fig2, ax2 = plt.subplots(figsize=(7.4, 3.6))

        ax2.fill_between(
            d_line, 0, t_line,
            alpha=0.15,
            label="合計真直度の取りうる範囲"
        )

        ax2.plot(d_line, t_line, "-", linewidth=2, label="合計真直度（φ）")
        ax2.scatter([d_actual], [straight_total], s=90, zorder=3, label="現在点")

        ax2.set_xlim(D_MIN, D_MMC)

        ymax = max(t_line) * 1.05
        ax2.set_ylim(0, ymax)
        ax2.set_yticks(np.arange(0, ymax + 0.001, 0.1))

        ax2.set_xlabel("サイズ(軸直径) d [mm]")
        ax2.set_ylabel("真直度 (合計) [mm]")
        ax2.grid(True)

        ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
        fig2.subplots_adjust(bottom=0.28)

        st.pyplot(fig2, use_container_width=True)

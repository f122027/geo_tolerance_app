import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from app_pages.home import show_gdt_callout

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

        # 実際の輪郭：公差内・外すべて線で描く
        ax.plot(
            x_actual,
            y_actual,
            color="orange",
            linewidth=1.2,
            label="実際の輪郭（表示誇張）",
            zorder=3,
        )

        # 公差アウト点だけ赤い × を重ねて強調
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

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
        fig.subplots_adjust(bottom=0.22)

        st.pyplot(fig, use_container_width=True)


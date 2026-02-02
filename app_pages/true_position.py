import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# from app_pages.home import show_gdt_callout
from core.image_helpers import make_true_position_drawing_example_image


def page_true_position():
    st.title("位置度（2D）のイメージ")

    # =========================
    # 上段：説明（左）｜図面例（右）
    # =========================
    col_desc, col_img = st.columns([1.0, 1.0], gap="large")

    with col_desc:
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

       

    with col_img:
        st.subheader("図面例")
        img = make_true_position_drawing_example_image(scale=0.15)
        st.image(img)  
        

    st.divider()

    # =========================
    # 下段：スライダー（左）｜グラフ（右）
    # =========================
    col_ctrl, col_plot = st.columns([1.0, 1.2], gap="large")

    with col_ctrl:
        st.subheader("操作（スライダー）")

        tol = st.slider("位置度公差（直径） [mm]", 0.0, 1.0, 0.4, 0.05)
        n_points = st.slider("穴の個数（サンプル数）", 1, 50, 20)
        mag = st.slider("位置ずれの表示倍率（見た目のみ）", 1.0, 50.0, 10.0, 1.0)

        sigma = 0.15
        st.caption(f"公差：直径 {tol:.2f} mm の円内に入ればOK")
        st.caption(f"図ではズレを **{mag:.0f} 倍** に誇張表示しています")

    with col_plot:
        # 理想位置（20,20）
        x0, y0 = 20.0, 20.0

        # 穴中心軸を正規分布からサンプル（評価用実寸）
        xs = np.random.normal(x0, sigma, n_points)
        ys = np.random.normal(y0, sigma, n_points)

        radius = tol / 2

        # 合否判定（実寸）
        dist = np.sqrt((xs - x0) ** 2 + (ys - y0) ** 2)
        inside = dist <= radius
        outside = ~inside

        # 表示用：ズレだけ mag 倍
        dx = xs - x0
        dy = ys - y0
        xs_vis = x0 + dx * mag
        ys_vis = y0 + dy * mag
        radius_vis = radius * mag

        fig, ax = plt.subplots(figsize=(5, 5))

        # データム A → X=0（垂直）
        ax.axvline(0.0, color="black", linewidth=2)
        # データム B → Y=0（水平）
        ax.axhline(0.0, color="black", linewidth=2)

        circle = plt.Circle(
            (x0, y0),
            radius_vis,
            fill=False,
            linestyle="--",
            color="green",
            alpha=0.8,
            label="位置度公差域（表示誇張）",
        )
        ax.add_artist(circle)

        ax.scatter(xs_vis[inside], ys_vis[inside], marker="o", color="tab:blue",
                   label="穴の中心軸（合格・表示誇張）")
        ax.scatter(xs_vis[outside], ys_vis[outside], marker="x", color="red",
                   label="穴の中心軸（不合格・表示誇張）")

        ax.scatter([x0], [y0], marker="+", s=130, color="black", label="理想位置")

        margin = radius_vis * 1.5
        ax.set_xlim(min(0, xs_vis.min()) - margin, max(xs_vis.max(), x0 + radius_vis) + margin)
        ax.set_ylim(min(0, ys_vis.min()) - margin, max(ys_vis.max(), y0 + radius_vis) + margin)
        ax.set_aspect("equal", "box")

        ax.set_xlabel("データム B からの距離 [mm]")
        ax.set_ylabel("データム A からの距離 [mm]")
        ax.grid(True)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(0, ylim[1], "データム B", ha="left", va="top", fontsize=10)
        ax.text(xlim[1], 0, "データム A", ha="right", va="bottom", fontsize=10)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
        fig.subplots_adjust(bottom=0.25)

        st.pyplot(fig, use_container_width=True)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.patches import Patch

from core.image_helpers import make_roundness_drawing_example_image


def _fig_to_png_bytes(fig, dpi: int = 140) -> bytes:
    """
    Matplotlib Figure を固定ピクセルの PNG bytes に変換する。
    bbox_inches="tight" を使うと毎回サイズが微妙に変わる原因になるので使わない。
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def page_roundness():
    st.title("真円度のイメージ")

    # =========================
    # 上段：説明（左）｜図面例（右）
    # =========================
    col_desc, col_img = st.columns([1.0, 1.0], gap="large")

    with col_desc:
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

    with col_img:
        st.subheader("図面例（真円度）")

        # 図面例は固定（スライダー非連動）
        gdt_img = make_roundness_drawing_example_image(scale=0.40)

        # 画像の表示幅を固定（ここが振動止めに効く）
        st.image(gdt_img, width=520)


    st.divider()

    # =========================
    # 下段：スライダー（左）｜グラフ（右）
    # =========================
    col_ctrl, col_plot = st.columns([1.0, 1.2], gap="large")

    with col_ctrl:
        st.subheader("操作（スライダー）")

        tol = st.slider("真円度公差（mm）", 0.00, 0.20, 0.05, 0.01)
        noise = st.slider("凹凸の最大値−最小値（振幅）[mm]", 0.00, 0.20, 0.08, 0.01)
        mag = st.slider("半径方向の表示倍率（見た目のみ）", 10.0, 80.0, 40.0, 5.0)

        seed = 0  # 乱数固定

        st.caption(f"判定は、円半径の **±{tol/2:.3f} mm** の範囲に入るかで行います。")
        st.caption(f"図では半径方向のズレを **{mag:.0f} 倍** に誇張して表示しています。")

    with col_plot:
        st.subheader("グラフ")

        theta = np.linspace(0, 2 * np.pi, 400)
        R0 = 10.0  # [mm]

        rng = np.random.default_rng(seed)
        actual_r = R0 + rng.uniform(-noise / 2, noise / 2, size=theta.shape)

        # 評価用 公差域
        lower = R0 - tol / 2
        upper = R0 + tol / 2
        outside = ~((actual_r >= lower) & (actual_r <= upper))

        # 表示用（誇張）
        dev = actual_r - R0
        actual_r_vis = R0 + dev * mag
        lower_vis = R0 - (tol / 2) * mag
        upper_vis = R0 + (tol / 2) * mag

        x_ideal = R0 * np.cos(theta)
        y_ideal = R0 * np.sin(theta)

        x_actual = actual_r_vis * np.cos(theta)
        y_actual = actual_r_vis * np.sin(theta)

        fig, ax = plt.subplots(figsize=(5.2, 5.2))

        # 公差帯リング
        ax.add_patch(plt.Circle((0, 0), upper_vis, color="green", alpha=0.15, fill=True, zorder=0))
        ax.add_patch(plt.Circle((0, 0), lower_vis, color="white", alpha=1.0, fill=True, zorder=0))

        ax.plot(upper_vis * np.cos(theta), upper_vis * np.sin(theta),
                color="green", linestyle="--", linewidth=1, zorder=1)
        ax.plot(lower_vis * np.cos(theta), lower_vis * np.sin(theta),
                color="green", linestyle="--", linewidth=1, zorder=1)

        ax.plot(x_ideal, y_ideal, linestyle="--", color="blue", label="理想円", zorder=2)
        ax.plot(x_actual, y_actual, color="orange", linewidth=1.2,
                label="実際の輪郭（表示誇張）", zorder=3)

        x_out = actual_r_vis[outside] * np.cos(theta[outside])
        y_out = actual_r_vis[outside] * np.sin(theta[outside])
        ax.scatter(x_out, y_out, marker="x", color="red",
                   label="公差アウト（表示誇張）", zorder=4)

        ax.set_aspect("equal", "box")
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.grid(True)

        # 凡例は「図の外（下）」
        fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.22)
        tol_zone_handle = Patch(facecolor="green", alpha=0.15, edgecolor="green", label="公差域")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(tol_zone_handle)
        labels.append("公差域")

        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.06),
            ncol=4,          
            fontsize=9,
            framealpha=0.9
        )


        st.pyplot(fig)

       

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from core.image_helpers import make_straightness_gdt_image

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
                "images/公差域真直度.jpg",
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
        ax.legend()

        st.pyplot(fig)


import streamlit as st
import numpy as np
import plotly.graph_objects as go

from core.image_helpers import make_flatness_gdt_image, make_flatness_gdt_image_2

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

        # 図面イメージ（公差値入りのPNGを 1〜2枚横並び）
        st.markdown("#### 図面イメージ")

        col_i1, col_i2 = st.columns(2)

        # --- 左側の図（例：図面のGDT指示） ---
        with col_i1:
            # 公差値を書き込んだ平面度図面画像（例：flatness_example.png）
            img1 = make_flatness_gdt_image(
                tol_value=tol,
                scale=0.45,  # 画像の表示倍率（お好みで調整）
            )
            st.image(
                img1,
                caption="図1：平面度の図面指示例",
                use_container_width=True,
            )

        # --- 右側の図（例：公差域説明図など） ---
        with col_i2:
            # 同じ関数を使っても良いし，
            # 別の画像用に make_flatness_gdt_image_2() を作ってもOK。
            img2 = make_flatness_gdt_image_2(
                tol_value=tol,
                scale=0.45,
            )
            st.image(
                img2,
                caption="図2：平面度公差域のイメージ図",
                use_container_width=True,
            )


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


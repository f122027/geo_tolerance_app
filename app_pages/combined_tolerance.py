import streamlit as st
from core.image_helpers import make_annotated_image
from core.figures import make_tolerance_figure


def page_combined_tolerance():
    st.header("位置度・平行度・平面度の公差域と実測形状")

    # =====================================================
    # Row 1: 左=説明 / 右=図面例
    # =====================================================
    colL1, colR1 = st.columns([1.2, 1.0], gap="large")

    with colL1:
        st.caption("同一の実測形状でも、公差の種類によって評価条件（拘束）が異なります。")
        st.markdown(
            "- 位置度：真の位置 20 を基準に **20±a/2** の範囲\n"
            "- 平行度：**データム面Aに平行**な範囲（拘束あり）\n"
            "- 平面度：**姿勢自由**の最小範囲（傾き自由）\n"
            "\n"
            "**拘束が強いほど必要な許容幅は大きくなりやすい**ため、一般に 位置度 ≥ 平行度 ≥ 平面度 と設定します。"
        )

    with colR1:
        IMAGE_COMBINED = "images/combined_tolerance.png"
        annotated_img = make_annotated_image(
            image_path=IMAGE_COMBINED,
            scale=0.30,
        )
        st.image(
            annotated_img,
            caption="位置度(a)・平行度(b)・平面度(c) の図面イメージ",
            width=350,
        )

    st.divider()

    # =====================================================
    # Row 2: 左=公差値設定 / 右=実測形状パラメータ
    # =====================================================
    colL2, colR2 = st.columns([1.2, 1.0], gap="large")

    with colL2:
        st.subheader("公差値の設定")
        pos_tol = st.slider("位置度公差 a [mm]", 0.01, 1.00, 0.20, 0.01)
        para_tol = st.slider("平行度公差 b [mm]", 0.01, 1.00, 0.10, 0.01)
        flat_tol = st.slider("平面度公差 c [mm]", 0.01, 1.00, 0.05, 0.01)


        if not (pos_tol >= para_tol >= flat_tol):
            st.warning(
                "現在の公差値は一般的な階層（a ≥ b ≥ c）を満たしていません。\n"
            )
        else:
            st.success("公差階層：a ≥ b ≥ c が成立しています。")

        st.write(f"**使用する公差値**： a={pos_tol:.2f}, b={para_tol:.2f}, c={flat_tol:.2f}")

    with colR2:
        st.subheader("実測形状のパラメータ")
        slope_amount = st.slider(
            "傾き量（左右端の高さ差）[mm]",
            0.000, 0.200, 0.030, 0.001, format="%.3f"
        )
        amplitude = st.slider(
            "凹凸の最大値−最小値（振幅）[mm]",
            0.000, 0.500, 0.030, 0.001, format="%.3f"
        )
        center_offset = st.slider(
            "実測平均中心のオフセット量 [mm]（＋で上方向、−で下方向にずらす）",
            -1.000, 1.000, 0.040, 0.001, format="%.3f"
        )

    st.divider()

    # =====================================================
    # Row 3: 左=表示する要素 / 右=グラフ
    # =====================================================
    
    colL3, colR3 = st.columns([0.4, 2.0], gap="large")

    with colL3:
        st.subheader("表示する要素")

        
        show_real = st.checkbox("実測形状", True)
        show_pos = st.checkbox("位置度", True)
        show_para = st.checkbox("平行度", True)
        show_flat = st.checkbox("平面度", True)

        st.caption("表示のON/OFFで、同一形状に対する評価条件の違いを比較できます。")

    # --- グラフ生成（右列に描画） ---
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

    with colR3:
        st.subheader("グラフ")
        st.pyplot(fig, use_container_width=True)

    st.divider()

    # =====================================================
    # Row 4: 左=判定結果 / 右=補足
    # =====================================================
    colL4, colR4 = st.columns([1.2, 1.0], gap="large")

    with colL4:
        st.subheader("判定結果（実測形状が公差域を満たしているか）")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.success("位置度：OK") if results["position"] else st.error("位置度：NG")
        with c2:
            st.success("平行度：OK") if results["parallel"] else st.error("平行度：NG")
        with c3:
            st.success("平面度：OK") if results["flat"] else st.error("平面度：NG")

    with colR4:
        st.caption("※判定の根拠は、右のグラフで「実測形状が公差域内に収まっているか」で確認できます。")

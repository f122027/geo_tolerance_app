import streamlit as st
from core.image_helpers import make_annotated_image
from core.figures import make_tolerance_figure

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


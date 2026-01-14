import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import textwrap

def page_mmc():
    st.title("最大実体公差方式（MMC）のイメージ")

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("MMC（最大実体）って何？")
        st.write(
            """
**最大実体（MMC）** は「材料がいちばん多い状態」のことです。

- **穴（内径形体）**：直径が **いちばん小さい** ときが MMC
- **軸（外径形体）**：直径が **いちばん大きい** ときが MMC

そして **位置度などの幾何公差**を MMC 条件で指示すると、

- MMC から “サイズがゆるむ” 分だけ、**ボーナス公差（追加の位置度）**がもらえる  
という効果が出ます（初心者が一番 “得した感” を直感で掴めるポイント）。
            """
        )

        st.markdown("### 例：穴の位置度（MMC）")

        # ---- 入力（穴の例）----
        nominal = st.slider("穴の基準寸法 D [mm]", 1.0, 50.0, 10.0, 0.5)
        size_tol = st.slider("寸法公差（±） [mm]", 0.0, 1.0, 0.10, 0.01)

        # 穴の許容範囲：D - tol ～ D + tol
        hole_min = nominal - size_tol  # 穴の最小（= MMC）
        hole_max = nominal + size_tol  # 穴の最大

        base_pos = st.slider("位置度公差（MMC条件の指示値：直径） [mm]", 0.0, 2.0, 0.20, 0.01)

        actual_d = st.slider("実際の穴径（実寸） [mm]", float(hole_min), float(hole_max), float(nominal), 0.01)

        # ---- MMC（穴）とボーナス ----
        mmc = hole_min  # 穴の MMC は最小径
        bonus = max(0.0, actual_d - mmc)  # 穴が大きいほどボーナス増
        total_pos = base_pos + bonus      # 合計位置度（直径）

        st.markdown("### 計算結果")
        st.write(f"- 穴の MMC（最小径）: **{mmc:.3f} mm**")
        st.write(f"- ボーナス公差: **{bonus:.3f} mm**（= 実穴径 − MMC）")
        st.write(f"- 合計位置度（直径）: **{total_pos:.3f} mm**（= 指示値 + ボーナス）")

        # ---- バーチャルコンディション（ゲージの考え方）----
        # 穴（内径形体）の VC は「MMC - 位置度（直径）」で表すのが一般的
        # ※ ここでは “ゲージピン径の上限” として直感表示
        vc = mmc - base_pos
        if vc < 0:
            st.warning("※ VC = MMC − 位置度 が負になっています（数値設定が現実的でない可能性があります）")
        st.write(f"- バーチャルコンディション（穴のVC）: **{vc:.3f} mm**（目安：ゲージピン径）")

        st.info(
            "ポイント：穴が大きくなるほど（材料が減るほど）ボーナス公差が増えて、"
            "位置ずれの許容が広がります。"
        )

        # 表示倍率
        mag = st.slider("位置ずれの表示倍率（見た目のみ）", 1.0, 50.0, 10.0, 1.0)

        # 実測点（1点）をランダム生成するボタン
        if "mmc_point" not in st.session_state:
            st.session_state["mmc_point"] = (0.0, 0.0)

        if st.button("ランダムな実測中心を生成（例）"):
            # 合計公差に対して、わざと内外が出るくらいの範囲で生成
            span = max(total_pos / 2 * 1.4, 0.05)
            dx = float(np.random.uniform(-span, span))
            dy = float(np.random.uniform(-span, span))
            st.session_state["mmc_point"] = (dx, dy)

    with col_right:
        # 真位置（ここでは原点）
        x0, y0 = 0.0, 0.0

        dx, dy = st.session_state.get("mmc_point", (0.0, 0.0))

        # 判定は “実寸” で行う
        r_allow = (total_pos / 2.0)
        dist = float(np.sqrt(dx**2 + dy**2))
        ok = dist <= r_allow

        # 表示は誇張
        xs_vis = x0 + dx * mag
        ys_vis = y0 + dy * mag
        r_vis = r_allow * mag

        fig, ax = plt.subplots(figsize=(5, 5))

        # データムは省略して、真位置中心＋公差円で直感優先
        circle = plt.Circle(
            (x0, y0),
            r_vis,
            fill=False,
            linestyle="--",
            color="green",
            linewidth=2,
            label="合計位置度公差域（表示誇張）"
        )
        ax.add_artist(circle)

        # 真位置
        ax.scatter([x0], [y0], marker="+", s=160, color="black", label="真位置")

        # 実測中心
        ax.scatter([xs_vis], [ys_vis], marker="o" if ok else "x", s=120, color=("tab:blue" if ok else "red"),
                   label=("実測中心（OK）" if ok else "実測中心（NG）"))

        ax.set_aspect("equal", "box")
        margin = max(r_vis * 1.6, 1.0)
        ax.set_xlim(x0 - margin, x0 + margin)
        ax.set_ylim(y0 - margin, y0 + margin)
        ax.grid(True)
        ax.set_xlabel("X 方向のズレ [mm]（見た目は倍率）")
        ax.set_ylabel("Y 方向のズレ [mm]（見た目は倍率）")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
        fig.subplots_adjust(bottom=0.22)

        st.pyplot(fig, use_container_width=True)

        st.subheader("合否")
        if ok:
            st.success(f"✅ 合格：中心ズレ {dist:.3f} mm ≤ 許容半径 {r_allow:.3f} mm")
        else:
            st.error(f"❌ 不合格：中心ズレ {dist:.3f} mm > 許容半径 {r_allow:.3f} mm")


    st.header("最大実体公差方式（MMC）× 真直度（サイズ連動の理解）")

    # -----------------------------
    # 入力（スライダー）
    # -----------------------------
    colA, colB = st.columns(2)

    with colA:
        feature_type = st.radio("形体の種類", ["穴（内径形体）", "軸（外径形体）"], horizontal=True)
        nominal = st.slider("基準寸法（直径）D [mm]", 1.0, 100.0, 5.0, 0.1)
        size_pm = st.slider("サイズ公差（±）[mm]", 0.0, 2.0, 0.2, 0.01)

        d_min = nominal - size_pm
        d_max = nominal + size_pm

        # MMC側の直径（穴=最小、軸=最大）
        if feature_type == "穴（内径形体）":
            d_mmc = d_min
            d_lmc = d_max
        else:
            d_mmc = d_max
            d_lmc = d_min

        d_actual = st.slider("実際の直径（実寸）[mm]", float(d_min), float(d_max), float(nominal), 0.01)

    with colB:
        straight_mmc = st.slider("真直度（MMC条件での指示値）[mm]", 0.0, 2.0, 0.3, 0.01)
        L = st.slider("表示する長さ L（見た目用）", 10.0, 200.0, 80.0, 5.0)
        waves = st.slider("曲がりの山数（例）", 1, 6, 2, 1)
        mag = st.slider("表示倍率（ズレを強調）", 1.0, 50.0, 8.0, 1.0)

    # -----------------------------
    # MMCボーナス（真直度を“増やす”）
    # -----------------------------
    if feature_type == "穴（内径形体）":
        bonus = max(0.0, d_actual - d_mmc)     # 穴は大きいほど材料が減る→ボーナス増
        sign = +1
    else:
        bonus = max(0.0, d_mmc - d_actual)     # 軸は細いほど材料が減る→ボーナス増
        sign = -1

    straight_total = straight_mmc + bonus

    st.markdown("### 計算結果（サイズ連動）")
    st.write(f"- MMC 直径: **{d_mmc:.3f} mm**")
    st.write(f"- 実寸直径: **{d_actual:.3f} mm**")
    st.write(f"- ボーナス公差: **{bonus:.3f} mm**")
    st.write(f"- 合計 真直度: **{straight_total:.3f} mm**（= 指示値 + ボーナス）")

    # -----------------------------
    # ① 公差域 + ② 最大曲がり例（同じ図で表示）
    # -----------------------------
    st.subheader("① 真直度の公差域　② 最大の曲がり例（公差域に接する）")

    x = np.linspace(0, L, 400)
    tol = straight_total
    half = (tol / 2.0) * mag  # 見た目の強調
    # 最大曲がり例：公差域の上下に接する（振幅=half）
    y_bend = half * np.sin(2 * np.pi * waves * x / L)

    fig1, ax1 = plt.subplots(figsize=(7, 3.3))
    ax1.plot([0, L], [0, 0], "-", label="基準（中心線）")
    ax1.plot([0, L], [+half, +half], "--", label="公差域 上限")
    ax1.plot([0, L], [-half, -half], "--", label="公差域 下限")
    ax1.plot(x, y_bend, "-", linewidth=2, label="最大曲がり例（接線）")

    ax1.set_xlabel("長さ方向 x（見た目用）")
    ax1.set_ylabel("中心線のズレ（倍率適用）")
    ax1.grid(True)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig1.subplots_adjust(bottom=0.28)
    st.pyplot(fig1, use_container_width=True)

    # -----------------------------
    # ③ 動的公差線図 + 点（スライダー連動で移動）
    # -----------------------------
    st.subheader("③ 動的公差線図（サイズ ↔ 合計真直度）＋ 現在点")

    d_line = np.linspace(d_min, d_max, 200)

    if feature_type == "穴（内径形体）":
        # y = straight_mmc + (d - d_mmc)
        y_line = straight_mmc + np.maximum(0.0, d_line - d_mmc)
    else:
        # y = straight_mmc + (d_mmc - d)
        y_line = straight_mmc + np.maximum(0.0, d_mmc - d_line)

    fig2, ax2 = plt.subplots(figsize=(6.8, 3.6))
    ax2.plot(d_line, y_line, "-", label="合計真直度（サイズ連動）")
    ax2.scatter([d_actual], [straight_total], s=90, label="現在点（スライダー）")

    ax2.set_xlabel("直径（実寸）D [mm]")
    ax2.set_ylabel("合計 真直度 [mm]")
    ax2.grid(True)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig2.subplots_adjust(bottom=0.28)
    st.pyplot(fig2, use_container_width=True)


    st.header("最大実体公差方式（MMC）× 真直度（軸：基準寸法固定）")

    # =============================
    # 固定値（基準寸法は固定）
    # =============================
    D_MMC = 20.0          # 最大実体サイズ（軸なので最大径）
    D_LMC = 19.8          # 最小実体サイズ（例として固定）
    STR_MMC = 0.1         # MMCでの真直度指示（φ0.1 のイメージ）
    L = 120.0             # 図の長さ（見た目用）

    # =============================
    # スライダー（直径だけ動かす）
    # =============================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("入力（直径だけ動かす）")
        st.write(f"- 基準寸法（固定）: **φ{D_MMC:.1f}**")
        st.write(f"- 最小実体サイズ（固定）: **φ{D_LMC:.1f}**")
        st.write(f"- MMC 真直度指示（固定）: **φ{STR_MMC:.1f}**")

        d_actual = st.slider("実際の直径（実寸）[mm]", float(D_LMC), float(D_MMC), float(D_MMC), 0.01)

        waves = st.slider("曲がりの山数（例）", 1, 6, 2, 1)

    with col2:
        st.subheader("表示調整（見た目のみ）")
        mag = st.slider("真直度の表示倍率（ズレ強調）", 1.0, 80.0, 20.0, 1.0)
        r_scale = st.slider("軸の太さの表示倍率（太すぎるのを抑える）", 0.01, 0.30, 0.08, 0.01)

    # =============================
    # MMCボーナス（軸なので：細いほどボーナス増）
    # =============================
    bonus = max(0.0, D_MMC - d_actual)
    straight_total = STR_MMC + bonus  # φで増える

    # 最大実体実効サイズ（機能ゲージの考え方）
    # 軸：実効サイズ = 実寸直径 + 許容真直度（φ）
    eff = d_actual + straight_total

    st.markdown("### 計算結果")
    st.write(f"- 実寸直径: **φ{d_actual:.2f}**")
    st.write(f"- ボーナス真直度: **φ{bonus:.2f}**（= MMC − 実寸）")
    st.write(f"- 合計 真直度: **φ{straight_total:.2f}**（= φ{STR_MMC:.2f} + ボーナス）")
    st.write(f"- 最大実体実効サイズ（一定になる値）: **φ{eff:.2f}**")

    # =============================
    # ①②：公差域 + 最大曲がり例 + 軸形体 + 軸線
    # =============================
    st.subheader("① 真直度の公差域　② 最大曲がり例（軸形体・軸線・公差域を表示）")

    x = np.linspace(0, L, 600)

    # 軸線の真直度：公差域は φT の円筒 → 側面図では中心線に対し ±(T/2)
    half_tol = (straight_total / 2.0) * mag

    # “最大曲がり例”：公差域の上下にちょうど接する軸線（振幅=half_tol）
    y_axis = half_tol * np.sin(2 * np.pi * waves * x / L)

    # 軸の外形（側面図の簡易表現）：軸線 ± 半径
    # 半径はそのままだと大きすぎるので r_scale で抑えて表示
    r_vis = (d_actual / 2.0) * r_scale
    y_top = y_axis + r_vis
    y_bot = y_axis - r_vis

    fig1, ax1 = plt.subplots(figsize=(8.0, 3.6))

    # 公差域（軸線の真直度）
    ax1.plot([0, L], [+half_tol, +half_tol], "--", linewidth=2, label="公差域 上限（±T/2）")
    ax1.plot([0, L], [-half_tol, -half_tol], "--", linewidth=2, label="公差域 下限（±T/2）")

    # 基準の理想軸線（直線）
    ax1.plot([0, L], [0, 0], "-", linewidth=1.5, label="理想軸線（直線）")

    # 実際の軸線（最大曲がり例）
    ax1.plot(x, y_axis, "-", linewidth=2.5, label="軸線（最大曲がり例）")

    # 軸形体（外形）
    ax1.fill_between(x, y_bot, y_top, alpha=0.25, label="軸の形体（側面の簡易表示）")
    ax1.plot(x, y_top, "-", linewidth=1.0)
    ax1.plot(x, y_bot, "-", linewidth=1.0)

    ax1.set_xlabel("長さ方向 x（見た目用）")
    ax1.set_ylabel("偏差（倍率適用）")
    ax1.grid(True)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig1.subplots_adjust(bottom=0.28)

    st.pyplot(fig1, use_container_width=True)

    # =============================
    # ③：動的公差線図 + 現在点（写真の概念：実効サイズ一定）
    # =============================
    st.subheader("③ 動的公差線図（実寸直径 ↔ 合計真直度）＋ 現在点")

    d_line = np.linspace(D_LMC, D_MMC, 200)
    # 軸：T = STR_MMC + (D_MMC - d)
    t_line = STR_MMC + np.maximum(0.0, D_MMC - d_line)

    fig2, ax2 = plt.subplots(figsize=(7.4, 3.6))
    ax2.plot(d_line, t_line, "-", label="合計真直度 T（φ）")
    ax2.scatter([d_actual], [straight_total], s=90, label="現在点（スライダー）")

    # 参考：実効サイズ一定の線も薄く表示（視覚的に効きます）
    eff_line = d_line + t_line
    # 右軸にするのが本当は分かりやすいけど、まずは注記で見せる（簡単版）
    ax2.set_title(f"最大実体実効サイズ φ{eff:.2f}（実寸 + 許容真直度）")

    ax2.set_xlabel("実寸直径 d [mm]")
    ax2.set_ylabel("合計真直度 T [mm]")
    ax2.grid(True)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig2.subplots_adjust(bottom=0.28)

    st.pyplot(fig2, use_container_width=True)

def page_mmc_straightness():
    st.header("最大実体公差方式（MMC）× 真直度（固定条件）")

    # =============================
    # レイアウト：左=説明、右=図
    # =============================
    colL, colR = st.columns([1.0, 1.2], gap="large")

    # -----------------------------
    # 固定条件（ユーザー指定）
    # -----------------------------
    L = 100.0                 # 長さ方向固定
    D_NOM = 5.0               # サイズ公差 5 ± 0.2
    D_MIN = 4.8
    D_MMC = 5.2               # 軸なので MMC = 最大径
    STR_MMC = 0.3             # 幾何公差（MMCでの最大実体公差） φ0.3

    # -----------------------------
    # 左：説明＋スライダー＋計算結果
    # -----------------------------
    with colL:
        st.markdown("## 最大実体公差方式（MMC）とは")
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

        st.markdown("### 入力")
        d_actual = st.slider(
            "実際の直径（実寸） d [mm]",
            float(D_MIN), float(D_MMC), float(D_MMC), 0.01
        )

        bonus = max(0.0, D_MMC - d_actual)
        straight_total = STR_MMC + bonus  # 合計 真直度（φ）

        st.markdown("### 計算結果")
        st.write(f"- 基準寸法: φ{D_NOM:.1f} / サイズ公差: 5±0.2（= {D_MIN:.1f}〜{D_MMC:.1f}）")
        st.write(f"- MMC 真直度指示（固定）: φ{STR_MMC:.1f}")
        st.write(f"- 実寸直径: **φ{d_actual:.2f}**")
        st.write(f"- ボーナス: **φ{bonus:.2f}**（= 5.2 − 実寸）")
        st.write(f"- 合計 真直度: **φ{straight_total:.2f}**（= 0.3 + ボーナス）")

        st.markdown("### 表示の強調（見た目用）")
        amp_mag = st.slider("曲がり・公差域の見た目強調倍率", 1.0, 8.0, 3.0, 1.0)

    # -----------------------------
    # 右：図（①② + ③）
    # -----------------------------
    with colR:
        st.subheader("図：公差域と最大曲がり例")

        x = np.linspace(0, L, 600)

        YMIN, YMAX = 0.0, 5.5  # 表示範囲固定

        half_tol = straight_total / 2.0
        half_tol_vis = half_tol * amp_mag

        # 実寸半径（まずはそのまま表示に使う）
        r_vis = d_actual / 2.0

        # 条件を満たす中心線を作る
        # 条件1) x=0,100で下側が0  → C - r = 0 → C = r
        # 条件2) x=50で上側が5.5   → C + A + r = 5.5 → A = 5.5 - 2r
        C = r_vis
        A = YMAX - 2.0 * r_vis

        # 中心線（表示用）
        y_axis_vis = C + A * np.sin(np.pi * x / L)

        # 軸形体（外形）
        y_top = y_axis_vis + r_vis
        y_bot = y_axis_vis - r_vis

        #中心線（強調用）
        A_emph = A * amp_mag
        y_axis_emph = C + A_emph * np.sin(np.pi * x / L)

        fig, ax = plt.subplots(figsize=(8.0, 3.6))

        # 公差域（線）
        y_upper = C+A_emph/2 + half_tol_vis
        y_lower = C+A_emph/2 - half_tol_vis
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







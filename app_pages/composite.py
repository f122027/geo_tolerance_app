import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from app_pages.home import show_gdt_callout

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


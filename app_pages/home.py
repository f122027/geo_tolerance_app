import streamlit as st
import textwrap

def page_home():
    st.title("幾何公差")
    st.subheader("メニュー")

    st.markdown(
        """
        このアプリでは、幾何公差を理解するための図を表示します。

        左の **「メニュー」** から見たい項目を選んでください。

        本アプリの概要および実装内容は、GitHubのREADMEにまとめています。
        st.markdown("▶ [GitHub（README）はこちら](https://github.com/f122027/geo_tolerance_app)")
        """
    )

def show_gdt_callout(kind: str):
    """公差ごとの図面上の指示例（簡易版）を表示する"""
    st.markdown("---")
    st.markdown("#### 図面での指示例")

    if kind == "straightness":
        st.markdown(
            """
            **例：軸の外形真直度 0.05**

            ```text
            Φ10 ───────────── 軸

                 ┌────┬───────┐
                 │  | │  0.05 │   ← 真直度記号（縦棒）と公差値
                 └────┴───────┘
            ```

            - 特徴：単独形体（データム参照なし）の真直度
            - 図面では、対象となる線や軸の寸法のそばに
              **幾何特性記号 + 公差値** をフレームで指示します。
            """
        )
    
    else:
        st.info("この公差種別の図示例はまだ登録されていません。")


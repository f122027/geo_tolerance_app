import streamlit as st
from core.config import setup_fonts

from app_pages.home import page_home
from app_pages.straightness import page_straightness
from app_pages.roundness import page_roundness
from app_pages.true_position import page_true_position
from app_pages.combined_tolerance import page_combined_tolerance
from app_pages.flatness_3d import page_flatness_3d
from app_pages.composite import page_composite_position_random
from app_pages.mmc import page_mmc_straightness

setup_fonts()
st.set_page_config(page_title="幾何公差ビジュアル教材", layout="wide")

menu = st.sidebar.radio(
    "メニュー",
    [
        "ホーム",
        "真直度",
        "真円度",
        "位置度（2D）",
        "位置度・平行度・平面度の公差域の比較",
        "平面度（3D）",
        "最大実体公差方式（MMC）",
    ],
)

if menu == "ホーム":
    page_home()
elif menu == "真直度":
    page_straightness()
elif menu == "真円度":
    page_roundness()
elif menu == "位置度（2D）":
    page_true_position()
elif menu == "位置度・平行度・平面度の公差域の比較":
    page_combined_tolerance()
elif menu == "平面度（3D）":
    page_flatness_3d()
elif menu == "複合幾何公差":
    page_composite_position_random()
elif menu == "最大実体公差方式（MMC）":
    page_mmc_straightness()

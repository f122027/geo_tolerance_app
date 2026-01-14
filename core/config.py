from pathlib import Path
import matplotlib
import matplotlib.font_manager as fm

# プロジェクトのベースディレクトリ（repo 直下を想定）
BASE_DIR = Path(__file__).resolve().parents[1]
FONT_PATH = BASE_DIR / "fonts" / "NotoSansJP-Regular.ttf"

def setup_fonts():
    """日本語フォント設定（Streamlit 起動時に1回だけ呼ぶ）"""
    try:
        fm.fontManager.addfont(str(FONT_PATH))
        matplotlib.rcParams["font.family"] = "Noto Sans JP"
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        # フォントが無い環境でも落ちないようにする（その場合はデフォルトフォント）
        matplotlib.rcParams["axes.unicode_minus"] = False

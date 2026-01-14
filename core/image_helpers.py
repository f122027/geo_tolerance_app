from PIL import Image, ImageDraw, ImageFont
from core.config import BASE_DIR, FONT_PATH

def _draw_tolerance_on_image(
    image_path: str,
    tol_value: float,
    scale: float,
    text_xy: tuple,
    font_size: int = 40,
    text_format: str = "{:.2f}",
):
    """
    指定した画像に公差値を描き込み、必要に応じて縮小して返す内部用関数。
    ページごとのヘルパーから呼び出す想定。
    """

    # 画像読み込み
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    # 表示テキスト（例: "0.10"）
    text = text_format.format(tol_value)

    # フォント（Noto Sans JP を使用）
    try:
        font = ImageFont.truetype(str(FONT_PATH), font_size)
    except:
        font = ImageFont.load_default()

    # テキストのサイズ
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    x, y = text_xy
    padding = 8

    # 背景を白で塗る（数値が読めるように）
    draw.rectangle(
        [x - padding, y - padding, x + w + padding, y + h + padding],
        fill=(255, 255, 255, 255),
    )

    # テキスト描画（黒）
    draw.text((x, y), text, font=font, fill=(0, 0, 0, 255))

    # スケールでリサイズ
    if scale != 1.0:
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return img

def make_straightness_gdt_image(tol_value: float) -> Image.Image:
    """真直度公差値を画像に描く"""

    # ベース画像
    base = Image.open("images/straightness_base.png").convert("RGBA")
    img = base.copy()
    draw = ImageDraw.Draw(img)

    # 表示テキスト
    text = f"{tol_value:.2f}"

    # フォント
    try:
        font = ImageFont.truetype(str(FONT_PATH), 40)
    except:
        font = ImageFont.load_default()

    # Pillow 最新版では textsize() は使えない
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # テキスト位置
    x = 900
    y = 65

    # 背景の白塗り
    padding = 10
    draw.rectangle(
        [x - padding, y - padding, x + w + padding, y + h + padding],
        fill="white"
    )

    # 新しい値を書く
    draw.text((x, y), text, fill="black", font=font)

    return img

def make_flatness_gdt_image(tol_value: float, scale: float = 0.45,font_size:int=130):
    """
    平面度の図面PNGに公差値を書き込んだ画像を返す。
    """
    image_path = BASE_DIR / "images" / "平面度.png"

    # text_xy は「図面の公差枠」の左上あたりに合わせて調整
    return _draw_tolerance_on_image(
        image_path=str(image_path),
        tol_value=tol_value,
        scale=scale,
        text_xy=(1400, 65),   # ← 実際の画像に合わせて微調整してください
        font_size=font_size,
        text_format="{:.2f}",
    )

def make_flatness_gdt_image_2(tol_value: float, scale: float = 0.45,font_size:int=120):
    """
    平面度の図面PNGに公差値を書き込んだ画像を返す。
    """
    image_path = BASE_DIR / "images" / "平面度公差域.png"

    # text_xy は「図面の公差枠」の左上あたりに合わせて調整
    return _draw_tolerance_on_image(
        image_path=str(image_path),
        tol_value=tol_value,
        scale=scale,
        text_xy=(2000, 30),   # ← 実際の画像に合わせて微調整してください
        font_size=font_size,
        text_format="{:.2f}",
    )

def make_annotated_image(
    image_path,
    scale=0.45,
    **kwargs,
):
    

    # --- 画像読み込み ---
    img = Image.open(image_path).convert("RGBA")

    # --- 画像リサイズ ---
    if scale != 1.0:
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return img



# =========================
# ページごとの表示関数
# =========================


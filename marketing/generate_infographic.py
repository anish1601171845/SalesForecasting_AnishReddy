"""
Generates linkedin_infographic.png -- a single shareable image (1080x1350, 4:5
portrait -- the safest aspect ratio for LinkedIn/Instagram feed previews) summarizing the
project with headline stats. Distinct from brag_sheet.pdf (that's for email/portfolio;
this is designed to be the preview image on a LinkedIn post).
"""
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1350

PURPLE = (124, 58, 237)
PURPLE_DARK = (91, 33, 182)
BG_DARK = (15, 10, 30)
BG_CARD = (26, 19, 48)
WHITE = (245, 242, 255)
GREY = (170, 160, 200)
GREEN = (52, 211, 153)

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"

def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(FONT_DIR + name, size)

img = Image.new("RGB", (W, H), BG_DARK)
draw = ImageDraw.Draw(img)

# ---- Background: subtle radial-ish gradient via concentric rectangles ----
for i in range(0, H, 4):
    t = i / H
    r = int(BG_DARK[0] + (PURPLE_DARK[0] - BG_DARK[0]) * 0.15 * (1 - t))
    g = int(BG_DARK[1] + (PURPLE_DARK[1] - BG_DARK[1]) * 0.15 * (1 - t))
    b = int(BG_DARK[2] + (PURPLE_DARK[2] - BG_DARK[2]) * 0.15 * (1 - t))
    draw.line([(0, i), (W, i)], fill=(r, g, b))

# ---- Top accent bar ----
draw.rectangle([0, 0, W, 14], fill=PURPLE)

# ---- Header ----
draw.text((70, 60), "DemandIQ", font=font(64, bold=True), fill=WHITE)
draw.text((70, 138), "Sales Forecasting & Demand Intelligence", font=font(28), fill=GREY)
draw.text((70, 178), "Anish Reddy  ·  Data Science Intern Project", font=font(22), fill=(*GREEN,))

draw.line([(70, 230), (W - 70, 230)], fill=(60, 50, 90), width=2)

# ---- Stat cards ----
stats = [
    ("16.6%", "Forecast MAPE", "4-model weighted ensemble"),
    ("+30%", "Beats Naive Baseline", "Honestly benchmarked, not assumed"),
    ("793", "Customers Segmented", "RFM: VIP / Loyal / At-Risk"),
    ("8", "Dashboard Pages", "Live, interactive, Streamlit-built"),
]

card_y0 = 260
card_h = 190
card_gap = 22
card_w = W - 140

for i, (big, label, sub) in enumerate(stats):
    y = card_y0 + i * (card_h + card_gap)
    draw.rounded_rectangle([70, y, 70 + card_w, y + card_h], radius=22, fill=BG_CARD, outline=(60, 50, 90), width=2)
    # accent bar on the left of each card
    draw.rounded_rectangle([70, y, 84, y + card_h], radius=22, fill=PURPLE)
    draw.rectangle([78, y, 84, y + card_h], fill=PURPLE)
    # big number
    draw.text((120, y + 28), big, font=font(72, bold=True), fill=PURPLE if i % 2 == 0 else GREEN)
    # label
    draw.text((420, y + 40), label, font=font(34, bold=True), fill=WHITE)
    draw.text((420, y + 88), sub, font=font(24), fill=GREY)

# ---- Footer: tech stack strip ----
footer_y = card_y0 + len(stats) * (card_h + card_gap) + 10
draw.line([(70, footer_y), (W - 70, footer_y)], fill=(60, 50, 90), width=2)
tech = "Python  ·  XGBoost  ·  Prophet  ·  SARIMA  ·  SHAP  ·  scikit-learn  ·  Streamlit"
draw.text((70, footer_y + 26), tech, font=font(22), fill=GREY)
draw.text((70, footer_y + 66), "Ensemble forecasting  ·  SHAP explainability  ·  Conformal intervals  ·  A/B testing", font=font(22), fill=GREY)

# ---- Bottom accent bar ----
draw.rectangle([0, H - 14, W, H], fill=PURPLE)

img.save("linkedin_infographic.png")
print("linkedin_infographic.png written:", img.size)

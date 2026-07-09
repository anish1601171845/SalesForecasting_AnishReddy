"""
Generates brag_sheet.pdf -- a one-page, recruiter-facing summary of the project.
Distinct from summary.docx (the CFO-facing business report): this one is designed to be
dropped into a resume email or portfolio site.
"""
import qrcode 
from fpdf import FPDF

PURPLE = (124, 58, 237)
DARK = (30, 20, 50)
GREY = (107, 114, 128)
LIGHT_BG = (243, 240, 253)
GREEN = (52, 211, 153)
WHITE = (255, 255, 255)

# Placeholder links -- replace with your real repo / live app URLs before sharing.
GITHUB_URL = "https://github.com/anish1601171845/SalesForecasting_AnishReddy.git"
QR_TARGET = "https://anish1601171845-salesforecasting-anishreddy-app-8mfep4.streamlit.app/"

qr_img = qrcode.make(QR_TARGET, border=2)
qr_path = "_qr_temp.png"
qr_img.save(qr_path)

pdf = FPDF(format="Letter")
pdf.add_page()
pdf.set_auto_page_break(False)
PAGE_W = 216  # mm, US Letter width

# ---- Header band ----
pdf.set_fill_color(*PURPLE)
pdf.rect(0, 0, PAGE_W, 38, style="F")
pdf.set_xy(12, 8)
pdf.set_text_color(*WHITE)
pdf.set_font("Helvetica", "B", 22)
pdf.cell(0, 10, "DemandIQ", ln=True)
pdf.set_x(12)
pdf.set_font("Helvetica", "", 12)
pdf.cell(0, 7, "End-to-End Sales Forecasting & Demand Intelligence System", ln=True)
pdf.set_x(12)
pdf.set_font("Helvetica", "", 10)
pdf.cell(0, 6, "Anish Reddy  |  Data Science Intern Project", ln=True)

# ---- Headline stat grid ----
stats = [
    ("16.6%", "Forecast MAPE", "Weighted ensemble of 4 models"),
    ("+29.9%", "vs. Seasonal Baseline", "Improvement over naive benchmark"),
    ("8", "Dashboard Pages", "Live, interactive Streamlit app"),
    ("793", "Customers Segmented", "RFM clustering, 4 segments"),
    ("90%", "Conformal Coverage", "Model-agnostic prediction intervals"),
    ("102", "Notebook Cells", "Zero execution errors, end-to-end"),
]

y0 = 46
box_w, box_h, gap = 63, 30, 3.5
cols = 3
for i, (big, label, sub) in enumerate(stats):
    col = i % cols
    row = i // cols
    x = 12 + col * (box_w + gap)
    y = y0 + row * (box_h + gap)
    pdf.set_fill_color(*LIGHT_BG)
    pdf.rect(x, y, box_w, box_h, style="F")
    pdf.set_xy(x + 4, y + 3)
    pdf.set_text_color(*PURPLE)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(box_w - 8, 10, big, ln=True)
    pdf.set_x(x + 4)
    pdf.set_text_color(*DARK)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.cell(box_w - 8, 5, label, ln=True)
    pdf.set_x(x + 4)
    pdf.set_text_color(*GREY)
    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(box_w - 8, 4, sub)

# ---- What's inside ----
y_feat = y0 + 2 * (box_h + gap) + 6
pdf.set_xy(12, y_feat)
pdf.set_text_color(*DARK)
pdf.set_font("Helvetica", "B", 13)
pdf.cell(0, 8, "What's Inside", ln=True)

features = [
    "4 forecasting models benchmarked against 2 naive baselines (SARIMA, Prophet w/ holidays, XGBoost, weighted Ensemble)",
    "SHAP explainability + model-agnostic conformal prediction intervals (distribution-free uncertainty bounds)",
    "Dual-method anomaly detection AND K-Means clustering applied to both products and customers (RFM)",
    "Monte Carlo inventory cost simulator (newsvendor-style) + an A/B testing lab with power analysis",
    "Auto-generated executive briefing with one-click PDF export, live from the current data",
]
pdf.set_font("Helvetica", "", 9.5)
pdf.set_text_color(50, 50, 60)
x_text = 16
for f in features:
    pdf.set_xy(x_text, pdf.get_y() + 0.5)
    pdf.set_text_color(*GREEN)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.cell(5, 4.5, chr(149))
    pdf.set_xy(x_text + 5, pdf.get_y())
    pdf.set_text_color(50, 50, 60)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.multi_cell(PAGE_W - x_text - 5 - 45, 4.5, f)

# ---- Tech stack ----
pdf.set_xy(12, pdf.get_y() + 2)
pdf.set_text_color(*DARK)
pdf.set_font("Helvetica", "B", 10)
pdf.cell(0, 5, "Tech Stack", ln=True)
pdf.set_x(12)
pdf.set_font("Helvetica", "", 9.5)
pdf.set_text_color(*GREY)
pdf.multi_cell(PAGE_W - 24, 4.5,
    "Python * Pandas/NumPy * XGBoost * Prophet * Statsmodels (SARIMA) * SHAP * scikit-learn "
    "* SciPy * Streamlit * Plotly")

# ---- QR code + links (right side) ----
qr_x, qr_y, qr_size = PAGE_W - 45, y_feat, 33
pdf.image(qr_path, x=qr_x, y=qr_y, w=qr_size, h=qr_size)
pdf.set_xy(qr_x, qr_y + qr_size + 1)
pdf.set_font("Helvetica", "", 7.5)
pdf.set_text_color(*GREY)
pdf.multi_cell(qr_size, 3.5, "Scan for the GitHub repo  https://anish1601171845-salesforecasting-anishreddy-app-8mfep4.streamlit.app/", align="C")

# ---- Footer ----
pdf.set_xy(12, 274)
pdf.set_draw_color(200, 200, 200)
pdf.line(12, 273, PAGE_W - 12, 273)
pdf.set_font("Helvetica", "", 8.5)
pdf.set_text_color(*GREY)
pdf.cell(0, 6, "github.com/anish1601171845  *  linkedin.com/in/bobbala-anish-reddy-023678359", align="C")

pdf.output("brag_sheet.pdf")
print("brag_sheet.pdf written")

import os
os.remove(qr_path)

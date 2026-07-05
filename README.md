# Sales Forecasting & Demand Intelligence System

End-to-end sales forecasting, anomaly detection, and product/customer demand segmentation on
the Superstore Sales dataset — Week 3 & 4 internship deliverable, since extended with an
ensemble forecast, SHAP explainability, conformal prediction intervals, naive-baseline
benchmarking, customer RFM segmentation, an inventory cost simulator, an A/B testing lab, an
auto-generated business briefing, and recruiter-facing marketing assets.

## Folder contents

```
SalesForecasting_AnishReddy/
├── analysis.ipynb        # Tasks 1-6 + bonuses: EDA, state map, decomposition, animated
│                          # playback, 4 forecasting models benchmarked against 2 naive
│                          # baselines (SARIMA/Prophet+holidays/XGBoost/Ensemble) with SHAP +
│                          # conformal intervals, segment forecasts, anomaly detection, product
│                          # clustering, customer RFM segmentation, A/B testing + peeking demo
├── app.py                 # Task 7: "DemandIQ" 8-page Streamlit dashboard
├── train.csv               # Superstore Sales dataset
├── vgsales.csv              # Supplementary Video Game Sales dataset (multi-source practice)
├── summary.docx              # Task 8: 2-page executive business report
├── requirements.txt            # All Python dependencies for the notebook & dashboard
├── .streamlit/config.toml       # Dashboard theme (dark/purple)
├── charts/                       # Chart PNGs + 2 interactive HTML exports from the notebook
└── marketing/                     # Recruiter-facing assets (not part of the graded deliverable)
    ├── brag_sheet.pdf              # One-page portfolio/resume summary with a QR code
    ├── linkedin_infographic.png     # 1080x1350 shareable image for a LinkedIn post
    ├── generate_brag_sheet.py        # Regenerate the brag sheet after updating stats/links
    ├── generate_infographic.py        # Regenerate the infographic
    └── requirements-marketing.txt      # fpdf2 / qrcode / Pillow (only needed for these scripts)
```

## Run the notebook

```bash
pip install -r requirements.txt
jupyter notebook analysis.ipynb
```

## Run the dashboard locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a public GitHub repo (include `train.csv`, `app.py`,
   `requirements.txt`, and `.streamlit/config.toml`).
2. Go to https://share.streamlit.io → **New app** → point it at the repo, branch `main`,
   main file path `app.py`.
3. Deploy. First build takes a few minutes (Prophet/XGBoost/SHAP wheels). Once live, copy the
   `*.streamlit.app` URL for submission — and update the QR code / links in `marketing/` to
   point at your real repo and live app before sharing them anywhere.

## Dashboard pages

1. **Sales Overview** — KPIs, trend, interactive region/category filters, state choropleth map,
   animated monthly category playback.
2. **Forecast Explorer** — XGBoost or weighted-ensemble forecast for any category/region, with
   a confidence traffic-light badge, retrain button, a SHAP waterfall explaining the forecast in
   dollars, a 90% conformal prediction interval, and a side-by-side comparison against two naive
   baselines so "is this actually better than doing nothing fancy?" has a direct answer.
3. **Anomaly Report** — Isolation Forest + Z-score weekly anomaly detection.
4. **Product Demand Segments** — K-Means clustering of sub-categories with stocking strategy.
5. **Customer RFM** — Recency/Frequency/Monetary customer segmentation with per-segment
   strategy and a top-customers table.
6. **Inventory What-If Lab** — Monte Carlo newsvendor-style cost simulator comparing a naive
   flat safety-stock policy against the cluster-based recommended policy.
7. **A/B Test Lab** — simulate a promotional test with adjustable effect size and sample size;
   see the t-test, power analysis, and required sample size live, plus a dedicated tab that
   demonstrates how "peeking" at results early inflates false positives under a true null effect.
8. **Monday Briefing** — auto-generated plain-English summary of the latest forecast, top
   anomaly, and fastest-growing segment, downloadable as a PDF.

## Key results

- **Best forecasting model:** a weighted ensemble of SARIMA + Prophet + XGBoost (MAE $16,222 /
  RMSE $19,420 / MAPE 16.6% on a 3-month holdout) — beats every individual model and improves on
  the seasonal-naive baseline by 29.9%. Honestly reported alongside this: a plain
  "repeat-last-month" baseline actually edges out every model on this single 3-month window — see
  `analysis.ipynb` Task 3 for the full discussion of why, and why we still recommend the ensemble.
- **Conformal prediction intervals:** a model-agnostic 90% interval built from XGBoost's own
  historical rolling-origin errors, rather than a distributional assumption.
- **Strongest anomaly:** the week of March 22, 2015 ($37,704), flagged by both Isolation
  Forest and the rolling Z-score method.
- **Demand segments:** 4 clusters across 17 product sub-categories, plus 4 customer RFM
  segments across 793 customers (68 VIP, 279 Loyal, 347 Occasional, 99 At Risk/Churned).
- **A/B testing / peeking demo:** under a true null effect, checking results at 60 points over
  a test's run showed a "significant" (p<0.05) result 43% of the time purely by chance — a
  concrete illustration of why pre-registering a sample size matters.

## Notes

- `analysis.ipynb` was executed end-to-end with no errors (102 cells); all outputs (tables,
  charts) are saved in the notebook and mirrored in `charts/` (static PNGs plus 2 interactive
  HTML exports for the choropleth map and animated playback, since those need a live Plotly
  renderer).
- The dashboard recomputes forecasts/anomalies/clusters/RFM/conformal intervals live from
  `train.csv` (cached with `st.cache_data`), so it stays in sync if the CSV is ever updated —
  no separate model pickle files to keep in sync. The "Retrain" button on Forecast Explorer
  clears the relevant cache and refits on demand.
- The `marketing/` folder is for your own portfolio/resume use — it's separate from the graded
  project deliverable. Update the placeholder GitHub/LinkedIn links before sharing either asset.


"# SalesForecasting_AnishReddy" 

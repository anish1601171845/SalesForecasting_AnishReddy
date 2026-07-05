"""
DemandIQ — Sales Forecasting & Demand Intelligence Dashboard
Task 7 deliverable for the End-to-End Sales Forecasting & Demand Intelligence System project.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import shap
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from fpdf import FPDF
from scipy import stats
from statsmodels.stats.power import TTestIndPower

# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLE
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DemandIQ | Sales Forecasting & Demand Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at 15% 0%, #241a3d 0%, #0f0a1e 45%, #0a0715 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1330 0%, #120c22 100%);
    border-right: 1px solid rgba(124, 58, 237, 0.25);
}

.hero {
    padding: 1.6rem 2rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(124,58,237,0.25) 0%, rgba(30,20,55,0.55) 100%);
    border: 1px solid rgba(124,58,237,0.35);
    margin-bottom: 1.4rem;
}
.hero h1 { margin: 0; font-size: 2rem; font-weight: 800; color: #f4f0ff; }
.hero p { margin: 0.35rem 0 0 0; color: #b9aee0; font-size: 0.95rem; }

.metric-card {
    background: linear-gradient(160deg, rgba(124,58,237,0.16), rgba(23,16,42,0.65));
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    height: 100%;
}
.metric-label { color: #b9aee0; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { color: #f4f0ff; font-size: 1.7rem; font-weight: 800; margin-top: 0.15rem; }
.metric-delta-up { color: #34d399; font-size: 0.85rem; font-weight: 600; }
.metric-delta-down { color: #f87171; font-size: 0.85rem; font-weight: 600; }

.section-card {
    background: rgba(26, 19, 48, 0.55);
    border: 1px solid rgba(124,58,237,0.22);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.1rem;
}
.section-title { color: #e5e0f5; font-size: 1.05rem; font-weight: 700; margin-bottom: 0.6rem; }

.badge {
    display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em;
}
.badge-stable { background: rgba(52,211,153,0.18); color: #34d399; border: 1px solid rgba(52,211,153,0.4); }
.badge-growing { background: rgba(96,165,250,0.18); color: #60a5fa; border: 1px solid rgba(96,165,250,0.4); }
.badge-declining { background: rgba(248,113,113,0.18); color: #f87171; border: 1px solid rgba(248,113,113,0.4); }
.badge-volatile { background: rgba(251,191,36,0.18); color: #fbbf24; border: 1px solid rgba(251,191,36,0.4); }

hr { border-color: rgba(124,58,237,0.2); }
[data-testid="stMetricValue"] { color: #f4f0ff; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PALETTE = ["#7c3aed", "#a78bfa", "#34d399", "#60a5fa", "#f59e0b", "#f87171"]

# ──────────────────────────────────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING (cached)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
    df["Order_Year"] = df["Order Date"].dt.year
    df["Order_Month"] = df["Order Date"].dt.month
    df["Ship_Delay_Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    return df


def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    return "Fall"


def make_lag_features(series):
    ts = series.to_frame("Sales")
    ts["lag1"] = ts["Sales"].shift(1)
    ts["lag2"] = ts["Sales"].shift(2)
    ts["lag3"] = ts["Sales"].shift(3)
    ts["roll3"] = ts["Sales"].shift(1).rolling(3).mean()
    ts["month"] = ts.index.month
    ts["quarter"] = ts.index.quarter
    ts["season_code"] = ts["month"].apply(get_season)
    ts = pd.get_dummies(ts, columns=["season_code"], prefix="season")
    return ts


@st.cache_data
def get_monthly_series(_df, category=None, region=None):
    d = _df
    if category and category != "All Categories":
        d = d[d["Category"] == category]
    if region and region != "All Regions":
        d = d[d["Region"] == region]
    return d.groupby(pd.Grouper(key="Order Date", freq="MS"))["Sales"].sum()


@st.cache_data
def xgb_forecast(series, steps=3):
    ts = make_lag_features(series).dropna()
    feats = [c for c in ts.columns if c != "Sales"]
    X, y = ts[feats], ts["Sales"]
    if len(X) < steps + 6:
        return None, None, None, None, None, None, None
    X_train, y_train = X.iloc[:-steps], y.iloc[:-steps]
    X_test, y_test = X.iloc[-steps:], y.iloc[-steps:]
    model = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    future_index = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=steps, freq="MS")
    return pd.Series(pred, index=future_index), mae, rmse, y_test, model, X_test, feats


@st.cache_data
def sarima_forecast(series, steps=3):
    train = series.iloc[:-steps]
    test = series.iloc[-steps:]
    try:
        fit = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = fit.get_forecast(steps=steps)
        pred = fc.predicted_mean.values
        ci = fc.conf_int(alpha=0.05)
    except Exception:
        return None, None, None
    mae = mean_absolute_error(test.values, pred)
    return pred, mae, ci


@st.cache_data
def prophet_forecast(series, steps=3):
    train = series.iloc[:-steps]
    test = series.iloc[-steps:]
    p_train = train.reset_index()
    p_train.columns = ["ds", "y"]
    thanksgiving = pd.DataFrame({
        "holiday": "Thanksgiving_to_CyberMonday",
        "ds": pd.to_datetime(["2015-11-26", "2016-11-24", "2017-11-23", "2018-11-22"]),
        "lower_window": 0, "upper_window": 4,
    })
    christmas = pd.DataFrame({
        "holiday": "Christmas_Runup",
        "ds": pd.to_datetime(["2015-12-25", "2016-12-25", "2017-12-25", "2018-12-25"]),
        "lower_window": -10, "upper_window": 2,
    })
    holidays = pd.concat([thanksgiving, christmas])
    try:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, holidays=holidays)
        m.fit(p_train)
        future = m.make_future_dataframe(periods=steps, freq="MS")
        fc = m.predict(future)
        pred = fc.set_index("ds")["yhat"].iloc[-steps:].values
    except Exception:
        return None, None
    mae = mean_absolute_error(test.values, pred)
    return pred, mae


@st.cache_data
def ensemble_forecast(series, steps=3):
    """Inverse-MAE-weighted blend of SARIMA + Prophet + XGBoost."""
    xgb_pred, xgb_mae, _, y_test, _, _, _ = xgb_forecast(series, steps)
    sarima_pred, sarima_mae, _ = sarima_forecast(series, steps)
    prophet_pred, prophet_mae = prophet_forecast(series, steps)
    if xgb_pred is None or sarima_pred is None or prophet_pred is None:
        return None, None, None
    inv = {"xgb": 1 / xgb_mae, "sarima": 1 / sarima_mae, "prophet": 1 / prophet_mae}
    total = sum(inv.values())
    w = {k: v / total for k, v in inv.items()}
    blended = w["xgb"] * xgb_pred.values + w["sarima"] * sarima_pred + w["prophet"] * prophet_pred
    mae = mean_absolute_error(y_test.values, blended)
    future_index = xgb_pred.index
    return pd.Series(blended, index=future_index), mae, w


@st.cache_data
def baseline_forecasts(series, steps=3):
    """Two zero-effort baselines: seasonal naive and persistence (naive)."""
    train = series.iloc[:-steps]
    test = series.iloc[-steps:]
    seasonal_naive = series.shift(12).iloc[-steps:].values
    persistence = np.repeat(train.iloc[-1], steps)
    results = {}
    for name, pred in [("Seasonal Naive", seasonal_naive), ("Naive (persistence)", persistence)]:
        if np.isnan(pred).any():
            continue
        mae = mean_absolute_error(test.values, pred)
        results[name] = {"pred": pred, "mae": mae}
    return results


@st.cache_data
def conformal_interval(series, steps=3, alpha=0.10, min_train=24):
    """Split conformal prediction interval via rolling-origin calibration on XGBoost residuals."""
    ts = make_lag_features(series).dropna()
    feats = [c for c in ts.columns if c != "Sales"]
    if len(ts) < min_train + steps + 5:
        return None
    residuals = []
    for i in range(min_train, len(ts) - steps):
        train_slice = ts.iloc[:i]
        test_point = ts.iloc[i:i + 1]
        X_tr, y_tr = train_slice[feats], train_slice["Sales"]
        X_te, y_te = test_point[feats], test_point["Sales"]
        m = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.08, random_state=42)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)[0]
        residuals.append(abs(y_te.values[0] - pred))
    if len(residuals) < 5:
        return None
    q = np.quantile(np.array(residuals), 1 - alpha)
    return q


def confidence_badge(mape_pct):
    """Traffic-light label based on backtest MAPE."""
    if mape_pct < 15:
        return "🟢 High Confidence", "badge-stable"
    elif mape_pct < 25:
        return "🟡 Moderate Confidence", "badge-volatile"
    return "🔴 Low Confidence", "badge-declining"


@st.cache_data
def compute_rfm(_df):
    snapshot = _df["Order Date"].max() + pd.Timedelta(days=1)
    rfm = _df.groupby("Customer ID").agg(
        Customer_Name=("Customer Name", "first"),
        Recency=("Order Date", lambda x: (snapshot - x.max()).days),
        Frequency=("Order ID", "nunique"),
        Monetary=("Sales", "sum"),
    ).reset_index()
    X_scaled = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["cluster"] = km.fit_predict(X_scaled)
    profile = rfm.groupby("cluster")[["Recency", "Frequency", "Monetary"]].mean()

    def label(row):
        if row["Recency"] > profile["Recency"].median() * 2:
            return "At Risk / Churned"
        elif row["Monetary"] > profile["Monetary"].quantile(0.75):
            return "VIP Customers"
        elif row["Frequency"] > profile["Frequency"].median() and row["Recency"] < profile["Recency"].median():
            return "Loyal Customers"
        return "Occasional Customers"

    labels = {c: label(r) for c, r in profile.iterrows()}
    rfm["segment"] = rfm["cluster"].map(labels)
    return rfm


RFM_STRATEGY = {
    "VIP Customers": "Dedicated account attention, early access to new products, and priority service — small group, outsized value.",
    "Loyal Customers": "Enroll in a loyalty/rewards program to reinforce an already-working habit; upsell opportunities.",
    "Occasional Customers": "Targeted promotions to increase purchase frequency; monitor for early churn signals.",
    "At Risk / Churned": "Win-back email campaign with a targeted discount; understand why they stopped ordering.",
}

RFM_BADGE = {
    "VIP Customers": "badge-stable",
    "Loyal Customers": "badge-growing",
    "Occasional Customers": "badge-volatile",
    "At Risk / Churned": "badge-declining",
}


US_STATE_ABBREV = {
    "Alabama": "AL", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
    "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY", "Alaska": "AK", "Hawaii": "HI",
}


def simulate_inventory_policy(mean_demand, std_demand, safety_pct, stockout_cost_rate, holding_cost_rate,
                               n_sim=4000, months=12, seed=42):
    """Monte Carlo newsvendor-style simulation: cost of a given safety-stock policy."""
    rng = np.random.default_rng(seed)
    safety_level = mean_demand * safety_pct
    order_up_to = mean_demand + safety_level
    demand = rng.normal(mean_demand, max(std_demand, 1e-6), size=(n_sim, months))
    demand = np.clip(demand, 0, None)
    shortfall = np.clip(demand - order_up_to, 0, None)
    excess = np.clip(order_up_to - demand, 0, None)
    stockout_cost = shortfall.mean() * stockout_cost_rate * months
    holding_cost = excess.mean() * holding_cost_rate * months
    return stockout_cost, holding_cost, stockout_cost + holding_cost


CLUSTER_RECOMMENDED_SAFETY = {
    "High Volume, Stable Demand": 0.10,
    "Growing Demand": 0.30,
    "Declining Demand": 0.05,
    "Low Volume, High Volatility": 0.25,
}


def generate_briefing_text(df, weekly, clusters, best_pred, best_mae):
    total_sales = df["Sales"].sum()
    top_cat = df.groupby("Category")["Sales"].sum().idxmax()
    top_anomaly = weekly[weekly["iso_anomaly"]].sort_values("Sales", ascending=False).head(1)
    top_anomaly_date = top_anomaly.index[0].strftime("%B %d, %Y") if len(top_anomaly) else "N/A"
    top_anomaly_val = top_anomaly["Sales"].iloc[0] if len(top_anomaly) else 0
    top_growth_cluster = clusters.groupby("cluster_label")["growth_rate_pct"].mean().idxmax()
    next_month_forecast = best_pred.iloc[0] if best_pred is not None else None
    next_month_label = best_pred.index[0].strftime("%B %Y") if best_pred is not None else ""

    lines = [
        f"MONDAY MORNING BRIEFING — {datetime.now().strftime('%A, %B %d, %Y')}",
        "=" * 60,
        "",
        f"Total historical sales: ${total_sales:,.0f} across {df['Order Date'].dt.year.nunique()} years.",
        f"Top revenue category: {top_cat}.",
        "",
        f"NEXT MONTH FORECAST ({next_month_label}): "
        + (f"${next_month_forecast:,.0f} (backtest MAE ${best_mae:,.0f})" if next_month_forecast is not None else "unavailable"),
        "",
        f"MOST RECENT NOTABLE ANOMALY: week of {top_anomaly_date}, ${top_anomaly_val:,.0f} in weekly sales — "
        f"worth a quick check against the promo calendar for that week.",
        "",
        f"FASTEST-GROWING PRODUCT SEGMENT: {top_growth_cluster}.",
        "",
        "RECOMMENDED FOCUS THIS WEEK:",
        "  - Review safety stock for segments in the 'Growing Demand' cluster.",
        "  - Confirm the anomaly above against known promotions or data issues.",
        "  - Re-check forecast accuracy once this month's actuals land.",
    ]
    return "\n".join(lines)


def _ascii_safe(text):
    replacements = {"\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2026": "..."}
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_briefing_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(124, 58, 237)
    pdf.cell(0, 10, _ascii_safe("DemandIQ - Monday Morning Briefing"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.ln(2)
    for line in text.split("\n"):
        safe_line = _ascii_safe(line)
        if safe_line.startswith("="):
            continue
        pdf.set_x(pdf.l_margin)
        if not safe_line.strip():
            pdf.ln(4)
            continue
        if safe_line.isupper() and safe_line.strip():
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 20, 50)
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(0, 6, safe_line)
    return bytes(pdf.output())


BADGE_MAP = {
    "High Volume, Stable Demand": "badge-stable",
    "Growing Demand": "badge-growing",
    "Declining Demand": "badge-declining",
    "Low Volume, High Volatility": "badge-volatile",
}

STRATEGY_MAP = {
    "High Volume, Stable Demand": "Standard reorder-point inventory, moderate safety stock, good candidate for automated replenishment.",
    "Growing Demand": "Increase safety stock ahead of season; revisit reorder quantities monthly so supply doesn't lag the trend.",
    "Declining Demand": "Reduce standing inventory, avoid large bulk discounts that lock in dead stock; consider clearance pricing.",
    "Low Volume, High Volatility": "Keep minimal on-hand stock; lean on fast reorder cycles or drop-ship rather than tying up capital in buffer inventory.",
}

df = load_data()

@st.cache_data
def compute_weekly_anomalies(_df):
    weekly = _df.set_index("Order Date").resample("W")["Sales"].sum().to_frame("Sales")
    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly["iso_anomaly"] = iso.fit_predict(weekly[["Sales"]]) == -1
    roll_mean = weekly["Sales"].rolling(6, min_periods=3).mean()
    roll_std = weekly["Sales"].rolling(6, min_periods=3).std()
    weekly["zscore"] = (weekly["Sales"] - roll_mean) / roll_std
    weekly["z_anomaly"] = weekly["zscore"].abs() > 2
    return weekly


@st.cache_data
def compute_clusters(_df):
    def feats(g):
        monthly_g = g.groupby(pd.Grouper(key="Order Date", freq="MS"))["Sales"].sum()
        return pd.Series({
            "total_sales": g["Sales"].sum(),
            "avg_order_value": g["Sales"].mean(),
            "monthly_volatility": monthly_g.std(),
        })

    sub = _df.groupby("Sub-Category").apply(feats, include_groups=False)
    yearly = _df.groupby(["Sub-Category", _df["Order Date"].dt.year])["Sales"].sum().unstack()
    sub["growth_rate_pct"] = ((yearly[2018] - yearly[2015]) / yearly[2015] * 100)
    sub = sub.fillna(0)

    cols = ["total_sales", "avg_order_value", "monthly_volatility", "growth_rate_pct"]
    X_scaled = StandardScaler().fit_transform(sub[cols])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    sub["cluster"] = kmeans.fit_predict(X_scaled)

    profile = sub.groupby("cluster")[cols].mean()

    def label_cluster(row):
        if row["total_sales"] > profile["total_sales"].median() and row["monthly_volatility"] < profile["monthly_volatility"].median():
            return "High Volume, Stable Demand"
        elif row["growth_rate_pct"] > 100:
            return "Growing Demand"
        elif row["growth_rate_pct"] < 0:
            return "Declining Demand"
        return "Low Volume, High Volatility"

    labels = {c: label_cluster(r) for c, r in profile.iterrows()}
    sub["cluster_label"] = sub["cluster"].map(labels)

    pcs = PCA(n_components=2).fit_transform(X_scaled)
    sub["pca1"], sub["pca2"] = pcs[:, 0], pcs[:, 1]
    return sub


# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📈 DemandIQ")
    st.caption("Sales Forecasting & Demand Intelligence")
    page = st.radio(
        "Navigate",
        ["📊 Sales Overview", "🔮 Forecast Explorer", "🚨 Anomaly Report", "🧩 Product Demand Segments",
         "👥 Customer RFM", "💰 Inventory What-If Lab", "🧪 A/B Test Lab", "📋 Monday Briefing"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(f"Dataset: Superstore Sales · {df['Order Date'].min().date()} → {df['Order Date'].max().date()}")
    st.caption(f"{len(df):,} transactions · {df['Sub-Category'].nunique()} sub-categories")

# ──────────────────────────────────────────────────────────────────────────
# PAGE 1 — SALES OVERVIEW DASHBOARD
# ──────────────────────────────────────────────────────────────────────────
if page == "📊 Sales Overview":
    st.markdown("""
    <div class="hero">
        <h1>Sales Overview Dashboard</h1>
        <p>Company-wide performance across 4 years of Superstore sales data.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    total_sales = df["Sales"].sum()
    total_orders = df["Order ID"].nunique()
    avg_order = df.groupby("Order ID")["Sales"].sum().mean()
    yoy = df.groupby("Order_Year")["Sales"].sum()
    yoy_growth = (yoy.iloc[-1] - yoy.iloc[-2]) / yoy.iloc[-2] * 100

    for col, label, value in zip(
        [c1, c2, c3, c4],
        ["Total Sales", "Total Orders", "Avg Order Value", "YoY Growth (last full year)"],
        [f"${total_sales:,.0f}", f"{total_orders:,}", f"${avg_order:,.0f}", f"{yoy_growth:+.1f}%"],
    ):
        col.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div>
                     <div class="metric-value">{value}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown('<div class="section-card"><div class="section-title">Total Sales by Year</div>', unsafe_allow_html=True)
        yearly = df.groupby("Order_Year")["Sales"].sum().reset_index()
        fig = px.bar(yearly, x="Order_Year", y="Sales", color="Sales", color_continuous_scale="Purples")
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           coloraxis_showscale=False, xaxis_title="Year", yaxis_title="Sales ($)", height=380)
        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="section-card"><div class="section-title">Monthly Sales Trend</div>', unsafe_allow_html=True)
        monthly = df.groupby(pd.Grouper(key="Order Date", freq="MS"))["Sales"].sum().reset_index()
        fig = px.line(monthly, x="Order Date", y="Sales")
        fig.update_traces(line_color="#a78bfa", line_width=2.5)
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           yaxis_title="Sales ($)", height=380)
        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">Sales by Region & Category — Interactive Filters</div>', unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    sel_regions = f1.multiselect("Filter by Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    sel_cats = f2.multiselect("Filter by Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    filtered = df[df["Region"].isin(sel_regions) & df["Category"].isin(sel_cats)]

    g1, g2 = st.columns(2)
    with g1:
        by_region = filtered.groupby("Region")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)
        fig = px.bar(by_region, x="Region", y="Sales", color="Region", color_discrete_sequence=PALETTE)
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           showlegend=False, height=340, yaxis_title="Sales ($)")
        st.plotly_chart(fig, width='stretch')
    with g2:
        by_cat = filtered.groupby("Category")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)
        fig = px.bar(by_cat, x="Category", y="Sales", color="Category", color_discrete_sequence=PALETTE)
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           showlegend=False, height=340, yaxis_title="Sales ($)")
        st.plotly_chart(fig, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">🗺️ Sales by State</div>', unsafe_allow_html=True)
    state_sales = df.groupby("State")["Sales"].sum().reset_index()
    state_sales["state_code"] = state_sales["State"].map(US_STATE_ABBREV)
    fig = px.choropleth(
        state_sales, locations="state_code", locationmode="USA-states", color="Sales",
        scope="usa", color_continuous_scale="Purples", hover_name="State",
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", geo_bgcolor="rgba(0,0,0,0)", height=450,
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width='stretch')
    top5 = state_sales.sort_values("Sales", ascending=False).head(5)
    st.caption("Top 5 states: " + ", ".join(f"{r.State} (${r.Sales:,.0f})" for r in top5.itertuples()))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">▶️ Animated Monthly Sales by Category</div>', unsafe_allow_html=True)
    st.caption("Press play on the animation controls to watch the category mix evolve month by month.")
    cat_monthly = df.groupby([pd.Grouper(key="Order Date", freq="MS"), "Category"])["Sales"].sum().reset_index()
    cat_monthly["Month_Str"] = cat_monthly["Order Date"].dt.strftime("%Y-%m")
    cat_monthly = cat_monthly.sort_values("Order Date")
    fig = px.bar(
        cat_monthly, x="Category", y="Sales", color="Category", animation_frame="Month_Str",
        range_y=[0, cat_monthly["Sales"].max() * 1.1], color_discrete_sequence=PALETTE,
    )
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       showlegend=False, height=420, yaxis_title="Sales ($)")
    st.plotly_chart(fig, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# PAGE 2 — FORECAST EXPLORER
# ──────────────────────────────────────────────────────────────────────────
elif page == "🔮 Forecast Explorer":
    st.markdown("""
    <div class="hero">
        <h1>Forecast Explorer</h1>
        <p>XGBoost or the weighted 3-model Ensemble (see analysis.ipynb Task 3) for any category or region — with SHAP explainability.</p>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
    dim = f1.selectbox("Forecast dimension", ["Category", "Region"])
    if dim == "Category":
        options = ["All Categories"] + sorted(df["Category"].unique().tolist())
        selection = f2.selectbox("Select Category", options)
        series = get_monthly_series(df, category=selection)
    else:
        options = ["All Regions"] + sorted(df["Region"].unique().tolist())
        selection = f2.selectbox("Select Region", options)
        series = get_monthly_series(df, region=selection)

    horizon = f3.select_slider("Forecast horizon (months ahead)", options=[1, 2, 3], value=3)
    model_choice = f4.selectbox("Model", ["XGBoost", "Ensemble (SARIMA+Prophet+XGBoost)"])

    rc1, rc2 = st.columns([3, 1])
    with rc2:
        if st.button("🔄 Retrain on latest data", width='stretch'):
            xgb_forecast.clear()
            ensemble_forecast.clear()
            st.session_state["last_retrain"] = datetime.now().strftime("%b %d, %Y %I:%M %p")
    if "last_retrain" in st.session_state:
        rc1.caption(f"✅ Model last retrained: {st.session_state['last_retrain']}")
    else:
        rc1.caption("Model trained on full history through " + df["Order Date"].max().strftime("%B %Y") + ".")

    xgb_pred, xgb_mae, xgb_rmse, y_test, fitted_model, X_test, feat_names = xgb_forecast(series, steps=3)

    if xgb_pred is None:
        st.warning("Not enough history in this segment to forecast reliably.")
    else:
        if model_choice.startswith("Ensemble"):
            ens_pred, ens_mae, weights = ensemble_forecast(series, steps=3)
            if ens_pred is None:
                st.info("Ensemble needs SARIMA/Prophet to converge on this segment — showing XGBoost instead.")
                pred, mae, rmse = xgb_pred, xgb_mae, xgb_rmse
                active_label = "XGBoost"
            else:
                pred, mae, rmse = ens_pred, ens_mae, np.sqrt(mean_squared_error(y_test.values, ens_pred.values[:len(y_test)])) if len(y_test) else ens_mae
                active_label = "Ensemble"
        else:
            pred, mae, rmse = xgb_pred, xgb_mae, xgb_rmse
            active_label = "XGBoost"

        pred_shown = pred.iloc[:horizon]
        mape_pct = float(np.mean(np.abs((y_test.values - (xgb_pred.values if active_label == "XGBoost" else pred.values[:len(y_test)])) / y_test.values)) * 100)
        badge_text, badge_cls = confidence_badge(mape_pct)
        conformal_q = conformal_interval(series, steps=3)

        st.markdown('<div class="section-card"><div class="section-title">Actual vs Forecast</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="badge {badge_cls}">{badge_text} · backtest MAPE {mape_pct:.1f}%</span>', unsafe_allow_html=True)
        show_interval = st.checkbox("Show 90% conformal prediction interval", value=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name="Actual",
                                  line=dict(color="#a78bfa", width=2.5)))
        bridge_x = [series.index[-1]] + list(pred_shown.index)
        bridge_y = [series.values[-1]] + list(pred_shown.values)
        fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, mode="lines+markers", name=f"{active_label} Forecast",
                                  line=dict(color="#34d399", width=2.5, dash="dash"), marker=dict(size=9)))
        if show_interval and conformal_q is not None:
            upper = [series.values[-1]] + list(pred_shown.values + conformal_q)
            lower = [series.values[-1]] + list(pred_shown.values - conformal_q)
            fig.add_trace(go.Scatter(x=bridge_x + bridge_x[::-1], y=upper + lower[::-1], fill="toself",
                                      fillcolor="rgba(52,211,153,0.15)", line=dict(color="rgba(0,0,0,0)"),
                                      name="90% Conformal Interval", showlegend=True))
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           height=440, yaxis_title="Sales ($)", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width='stretch')
        if show_interval and conformal_q is None:
            st.caption("Not enough history in this segment to calibrate a conformal interval.")
        elif show_interval:
            st.caption(f"Model-agnostic interval built from XGBoost's own historical 1-step-ahead errors (±${conformal_q:,.0f}), not a distributional assumption.")
        st.markdown('</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.markdown(f"""<div class="metric-card"><div class="metric-label">{active_label} MAE (3-mo backtest)</div>
                    <div class="metric-value">${mae:,.0f}</div></div>""", unsafe_allow_html=True)
        m2.markdown(f"""<div class="metric-card"><div class="metric-label">{active_label} RMSE (3-mo backtest)</div>
                    <div class="metric-value">${rmse:,.0f}</div></div>""", unsafe_allow_html=True)
        m3.markdown(f"""<div class="metric-card"><div class="metric-label">Next {horizon}-mo Forecast Total</div>
                    <div class="metric-value">${pred_shown.sum():,.0f}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-card"><div class="section-title">📏 Is this actually better than doing nothing fancy?</div>', unsafe_allow_html=True)
        st.caption("Every model here is benchmarked against two zero-effort baselines. If we can't beat these, the extra complexity isn't earning its keep.")
        baselines = baseline_forecasts(series, steps=3)
        if baselines:
            bcols = st.columns(len(baselines) + 1)
            bcols[0].markdown(f"""<div class="metric-card"><div class="metric-label">{active_label} (this model)</div>
                        <div class="metric-value">${mae:,.0f} MAE</div></div>""", unsafe_allow_html=True)
            for col, (name, res) in zip(bcols[1:], baselines.items()):
                beat = "🟢 beats it" if mae < res["mae"] else "🔴 loses to it"
                col.markdown(f"""<div class="metric-card"><div class="metric-label">{name}</div>
                            <div class="metric-value">${res['mae']:,.0f} MAE</div>
                            <div class="metric-delta-up">{beat}</div></div>""", unsafe_allow_html=True)
        else:
            st.caption("Not enough history in this segment to compute baselines.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-card"><div class="section-title">Forecast Detail</div>', unsafe_allow_html=True)
        table = pred_shown.reset_index()
        table.columns = ["Month", "Forecasted Sales ($)"]
        table["Forecasted Sales ($)"] = table["Forecasted Sales ($)"].round(0)
        table["Month"] = table["Month"].dt.strftime("%B %Y")
        st.dataframe(table, width='stretch', hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">🔍 Why this forecast? (SHAP explainability)</div>', unsafe_allow_html=True)
        st.caption("Shows how each feature pushed the XGBoost model's prediction up or down from its baseline, for the first forecasted month.")
        try:
            explainer = shap.TreeExplainer(fitted_model)
            shap_vals = explainer(X_test)
            month_idx = 0
            contrib = pd.Series(shap_vals.values[month_idx], index=feat_names).sort_values(key=abs, ascending=True)
            base_val = shap_vals.base_values[month_idx]
            colors = ["#f87171" if v > 0 else "#60a5fa" for v in contrib.values]
            fig_shap = go.Figure(go.Bar(
                x=contrib.values, y=contrib.index, orientation="h",
                marker_color=colors,
                text=[f"{v:+,.0f}" for v in contrib.values], textposition="outside",
            ))
            fig_shap.update_layout(
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=380, xaxis_title=f"Impact on forecast ($) — baseline ${base_val:,.0f}",
                margin=dict(l=10, r=60),
            )
            st.plotly_chart(fig_shap, width='stretch')
            st.caption("🔴 Red bars push the forecast up · 🔵 Blue bars push it down.")
        except Exception as e:
            st.info("SHAP explanation unavailable for this segment.")
        st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# PAGE 3 — ANOMALY REPORT
# ──────────────────────────────────────────────────────────────────────────
elif page == "🚨 Anomaly Report":
    st.markdown("""
    <div class="hero">
        <h1>Anomaly Report</h1>
        <p>Weekly sales anomalies detected via Isolation Forest and rolling Z-score (>2σ).</p>
    </div>
    """, unsafe_allow_html=True)

    weekly = compute_weekly_anomalies(df)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-card"><div class="metric-label">Isolation Forest Anomalies</div>
                <div class="metric-value">{int(weekly['iso_anomaly'].sum())}</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card"><div class="metric-label">Z-Score Anomalies</div>
                <div class="metric-value">{int(weekly['z_anomaly'].sum())}</div></div>""", unsafe_allow_html=True)
    overlap = int((weekly["iso_anomaly"] & weekly["z_anomaly"]).sum())
    c3.markdown(f"""<div class="metric-card"><div class="metric-label">Flagged by Both Methods</div>
                <div class="metric-value">{overlap}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    method = st.radio("Highlight method", ["Isolation Forest", "Z-Score", "Both"], horizontal=True)

    st.markdown('<div class="section-card"><div class="section-title">Weekly Sales with Anomalies Highlighted</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly.index, y=weekly["Sales"], mode="lines", name="Weekly Sales",
                              line=dict(color="#8b7dad", width=1.5)))
    if method in ("Isolation Forest", "Both"):
        a = weekly[weekly["iso_anomaly"]]
        fig.add_trace(go.Scatter(x=a.index, y=a["Sales"], mode="markers", name="Isolation Forest",
                                  marker=dict(color="#f87171", size=11, symbol="circle", line=dict(color="white", width=1))))
    if method in ("Z-Score", "Both"):
        a = weekly[weekly["z_anomaly"]]
        fig.add_trace(go.Scatter(x=a.index, y=a["Sales"], mode="markers", name="Z-Score",
                                  marker=dict(color="#fbbf24", size=11, symbol="diamond", line=dict(color="white", width=1))))
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       height=460, yaxis_title="Sales ($)", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">Detected Anomalies — Detail Table</div>', unsafe_allow_html=True)
    anomaly_rows = weekly[weekly["iso_anomaly"] | weekly["z_anomaly"]].copy()
    anomaly_rows["Flagged By"] = anomaly_rows.apply(
        lambda r: "Both" if r["iso_anomaly"] and r["z_anomaly"] else ("Isolation Forest" if r["iso_anomaly"] else "Z-Score"),
        axis=1,
    )
    anomaly_rows = anomaly_rows.reset_index().rename(columns={"Order Date": "Week Of"})
    anomaly_rows["Week Of"] = anomaly_rows["Week Of"].dt.strftime("%Y-%m-%d")
    anomaly_rows["Sales"] = anomaly_rows["Sales"].round(0)
    st.dataframe(
        anomaly_rows[["Week Of", "Sales", "Flagged By"]].sort_values("Sales", ascending=False),
        width='stretch', hide_index=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# PAGE 4 — PRODUCT DEMAND SEGMENTS
# ──────────────────────────────────────────────────────────────────────────
elif page == "🧩 Product Demand Segments":
    st.markdown("""
    <div class="hero">
        <h1>Product Demand Segments</h1>
        <p>K-Means clustering (k=4) of Sub-Categories by volume, growth, volatility & order value.</p>
    </div>
    """, unsafe_allow_html=True)

    clusters = compute_clusters(df)

    st.markdown('<div class="section-card"><div class="section-title">Demand Clusters (PCA-reduced)</div>', unsafe_allow_html=True)
    fig = px.scatter(
        clusters.reset_index(), x="pca1", y="pca2", color="cluster_label", text="Sub-Category",
        color_discrete_sequence=PALETTE, size="total_sales", size_max=40,
    )
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       height=500, legend=dict(orientation="h", y=1.15), xaxis_title="PC1", yaxis_title="PC2")
    st.plotly_chart(fig, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">Sub-Categories by Demand Segment</div>', unsafe_allow_html=True)
    for label in clusters["cluster_label"].unique():
        subset = clusters[clusters["cluster_label"] == label].sort_values("total_sales", ascending=False)
        badge_cls = BADGE_MAP.get(label, "badge-stable")
        st.markdown(f'<span class="badge {badge_cls}">{label}</span>', unsafe_allow_html=True)
        st.caption(STRATEGY_MAP.get(label, ""))
        table = subset.reset_index()[["Sub-Category", "total_sales", "avg_order_value", "monthly_volatility", "growth_rate_pct"]]
        table.columns = ["Sub-Category", "Total Sales ($)", "Avg Order Value ($)", "Monthly Volatility ($)", "Growth Rate (%)"]
        for c in table.columns[1:]:
            table[c] = table[c].round(1)
        st.dataframe(table, width='stretch', hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# PAGE 5 — CUSTOMER RFM SEGMENTATION
# ──────────────────────────────────────────────────────────────────────────
elif page == "👥 Customer RFM":
    st.markdown("""
    <div class="hero">
        <h1>Customer RFM Segmentation</h1>
        <p>Recency · Frequency · Monetary clustering — the same K-Means technique applied to customers instead of products.</p>
    </div>
    """, unsafe_allow_html=True)

    rfm = compute_rfm(df)

    seg_counts = rfm["segment"].value_counts()
    cols = st.columns(4)
    for col, seg in zip(cols, ["VIP Customers", "Loyal Customers", "Occasional Customers", "At Risk / Churned"]):
        n = int(seg_counts.get(seg, 0))
        col.markdown(f"""<div class="metric-card"><div class="metric-label">{seg}</div>
                     <div class="metric-value">{n}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-card"><div class="section-title">RFM Segments</div>', unsafe_allow_html=True)
    fig = px.scatter(
        rfm, x="Recency", y="Monetary", size="Frequency", color="segment",
        color_discrete_sequence=PALETTE, size_max=28, hover_data=["Customer_Name"],
    )
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       height=480, xaxis_title="Recency (days since last order)", yaxis_title="Monetary ($)",
                       legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">Segment Strategy & Top Customers</div>', unsafe_allow_html=True)
    seg_pick = st.selectbox("Explore segment", ["VIP Customers", "Loyal Customers", "Occasional Customers", "At Risk / Churned"])
    badge_cls = RFM_BADGE.get(seg_pick, "badge-stable")
    st.markdown(f'<span class="badge {badge_cls}">{seg_pick}</span>', unsafe_allow_html=True)
    st.caption(RFM_STRATEGY.get(seg_pick, ""))
    seg_table = rfm[rfm["segment"] == seg_pick].sort_values("Monetary", ascending=False).head(15)
    show = seg_table[["Customer_Name", "Recency", "Frequency", "Monetary"]].copy()
    show.columns = ["Customer", "Recency (days)", "Frequency (orders)", "Monetary ($)"]
    show["Monetary ($)"] = show["Monetary ($)"].round(0)
    st.dataframe(show, width='stretch', hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# PAGE 6 — INVENTORY WHAT-IF LAB
# ──────────────────────────────────────────────────────────────────────────
elif page == "💰 Inventory What-If Lab":
    st.markdown("""
    <div class="hero">
        <h1>Inventory What-If Lab</h1>
        <p>Monte Carlo cost simulation — compare a flat safety-stock policy against the cluster-based strategy from the Product Demand Segments page.</p>
    </div>
    """, unsafe_allow_html=True)

    clusters = compute_clusters(df)
    cluster_options = sorted(clusters["cluster_label"].unique())

    c1, c2, c3 = st.columns(3)
    cluster_pick = c1.selectbox("Product demand segment", cluster_options)
    stockout_cost_rate = c2.slider("Stockout cost ($ lost per $ of unmet demand)", 1.0, 3.0, 1.5, 0.1)
    holding_cost_rate = c3.slider("Holding cost (% of value per month)", 0.5, 10.0, 2.0, 0.5) / 100

    seg_rows = clusters[clusters["cluster_label"] == cluster_pick]
    mean_demand = seg_rows["total_sales"].sum() / 48  # rough avg monthly demand for the segment
    std_demand = seg_rows["monthly_volatility"].mean()

    naive_safety_pct = 0.15
    recommended_safety_pct = CLUSTER_RECOMMENDED_SAFETY.get(cluster_pick, 0.15)

    naive_stockout, naive_holding, naive_total = simulate_inventory_policy(
        mean_demand, std_demand, naive_safety_pct, stockout_cost_rate, holding_cost_rate)
    rec_stockout, rec_holding, rec_total = simulate_inventory_policy(
        mean_demand, std_demand, recommended_safety_pct, stockout_cost_rate, holding_cost_rate)

    savings = naive_total - rec_total

    st.markdown('<div class="section-card"><div class="section-title">Annual Cost Comparison</div>', unsafe_allow_html=True)
    st.caption(f"Segment: **{cluster_pick}** · Est. avg monthly demand ≈ ${mean_demand:,.0f} · "
               f"Naive policy uses a flat {naive_safety_pct:.0%} safety buffer for every segment; "
               f"the recommended policy uses {recommended_safety_pct:.0%}, sized to this segment's own volatility.")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Stockout Cost", x=["Naive (flat 15%)", f"Recommended ({recommended_safety_pct:.0%})"],
                          y=[naive_stockout, rec_stockout], marker_color="#f87171"))
    fig.add_trace(go.Bar(name="Holding Cost", x=["Naive (flat 15%)", f"Recommended ({recommended_safety_pct:.0%})"],
                          y=[naive_holding, rec_holding], marker_color="#60a5fa"))
    fig.update_layout(barmode="stack", template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       height=420, yaxis_title="Estimated Annual Cost ($)", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.markdown(f"""<div class="metric-card"><div class="metric-label">Naive Policy — Annual Cost</div>
                <div class="metric-value">${naive_total:,.0f}</div></div>""", unsafe_allow_html=True)
    m2.markdown(f"""<div class="metric-card"><div class="metric-label">Recommended Policy — Annual Cost</div>
                <div class="metric-value">${rec_total:,.0f}</div></div>""", unsafe_allow_html=True)
    delta_class = "metric-delta-up" if savings >= 0 else "metric-delta-down"
    m3.markdown(f"""<div class="metric-card"><div class="metric-label">Estimated Annual Savings</div>
                <div class="metric-value">${savings:,.0f}</div>
                <div class="{delta_class}">{'▲ Recommended policy wins' if savings >= 0 else '▼ Naive policy wins here'}</div></div>""", unsafe_allow_html=True)

    st.caption("Methodology: a simplified newsvendor-style Monte Carlo simulation. Monthly demand is modeled as "
               "Normal(mean, std) using this segment's own historical mean and volatility; each policy sets an "
               "order-up-to level of mean × (1 + safety %), and we simulate 4,000 years of 12-month demand draws "
               "to estimate expected stockout and overstock costs. Treat this as a directional decision-support "
               "tool, not a precise financial forecast — it approximates unit costs using dollar sales as a proxy.")

# ──────────────────────────────────────────────────────────────────────────
# PAGE 7 — A/B TEST LAB
# ──────────────────────────────────────────────────────────────────────────
elif page == "🧪 A/B Test Lab":
    st.markdown("""
    <div class="hero">
        <h1>A/B Test Lab</h1>
        <p>Simulate a promotional test on order value — significance, power, and the danger of checking too early.</p>
    </div>
    """, unsafe_allow_html=True)

    baseline_mean = float(df["Sales"].mean())
    baseline_std = float(df["Sales"].std())

    tab1, tab2 = st.tabs(["📈 Run a Test", "⚠️ The Peeking Problem"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        true_lift_pct = c1.slider("True underlying lift (%)", -10, 20, 5, 1,
                                   help="Set to 0 to simulate a promotion with NO real effect.") / 100
        n_per_group = c2.slider("Sample size per group", 100, 5000, 2000, 100)
        seed = c3.number_input("Random seed", value=42, step=1)

        rng = np.random.default_rng(int(seed))
        control = np.clip(rng.normal(baseline_mean, baseline_std, n_per_group), 0, None)
        treatment = np.clip(rng.normal(baseline_mean * (1 + true_lift_pct), baseline_std, n_per_group), 0, None)

        t_stat, p_value = stats.ttest_ind(treatment, control)
        observed_lift = (treatment.mean() - control.mean()) / control.mean() * 100
        pooled_std = np.sqrt((treatment.std() ** 2 + control.std() ** 2) / 2)
        cohens_d = (treatment.mean() - control.mean()) / pooled_std if pooled_std > 0 else 0
        power_calc = TTestIndPower()
        try:
            required_n = power_calc.solve_power(effect_size=abs(cohens_d), alpha=0.05, power=0.80)
        except Exception:
            required_n = float("nan")

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"""<div class="metric-card"><div class="metric-label">Observed Lift</div>
                    <div class="metric-value">{observed_lift:+.2f}%</div></div>""", unsafe_allow_html=True)
        sig_badge = "badge-stable" if p_value < 0.05 else "badge-declining"
        sig_text = "Significant" if p_value < 0.05 else "Not significant"
        m2.markdown(f"""<div class="metric-card"><div class="metric-label">P-value</div>
                    <div class="metric-value">{p_value:.4f}</div>
                    <span class="badge {sig_badge}">{sig_text} (α=0.05)</span></div>""", unsafe_allow_html=True)
        m3.markdown(f"""<div class="metric-card"><div class="metric-label">Effect Size (Cohen's d)</div>
                    <div class="metric-value">{cohens_d:.3f}</div></div>""", unsafe_allow_html=True)
        m4.markdown(f"""<div class="metric-card"><div class="metric-label">Required n/group (80% power)</div>
                    <div class="metric-value">{required_n:,.0f}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-card"><div class="section-title">Distribution: Control vs Treatment</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=control, name="Control", marker_color="#60a5fa", opacity=0.6, nbinsx=40))
        fig.add_trace(go.Histogram(x=treatment, name="Treatment", marker_color="#34d399", opacity=0.6, nbinsx=40))
        fig.update_layout(barmode="overlay", template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           height=380, xaxis_title="Order Value ($)", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

        if n_per_group < required_n and not np.isnan(required_n):
            st.info(f"⚠️ This test is underpowered: you'd need roughly **{required_n:,.0f} per group** to reliably "
                    f"detect an effect this size, but only simulated {n_per_group:,}. A non-significant result here "
                    "could just mean the sample was too small, not that there's truly no effect.")

    with tab2:
        st.caption("Control and treatment are drawn from the **exact same distribution** below — there is NO real "
                   "effect. Any 'significant' result you see while peeking is, by definition, a false positive.")
        c1, c2 = st.columns(2)
        n_max = c1.slider("Total sample size to simulate per group", 500, 5000, 3000, 500)
        peek_seed = c2.number_input("Random seed ", value=7, step=1)

        rng2 = np.random.default_rng(int(peek_seed))
        control_null = np.clip(rng2.normal(baseline_mean, baseline_std, n_max), 0, None)
        treatment_null = np.clip(rng2.normal(baseline_mean, baseline_std, n_max), 0, None)

        checkpoints = list(range(50, n_max + 1, 50))
        pvals = [stats.ttest_ind(treatment_null[:n], control_null[:n]).pvalue for n in checkpoints]
        false_positive_rate = float(np.mean(np.array(pvals) < 0.05))

        st.markdown('<div class="section-card"><div class="section-title">P-Value Over Time Under a TRUE NULL EFFECT</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=checkpoints, y=pvals, mode="lines", line=dict(color="#a78bfa", width=2)))
        fig.add_hline(y=0.05, line_dash="dash", line_color="#f87171", annotation_text="p = 0.05 threshold")
        fig.add_hrect(y0=0, y1=0.05, fillcolor="#f87171", opacity=0.08, line_width=0)
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           height=400, xaxis_title="Sample size per group at time of check", yaxis_title="p-value")
        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.markdown(f"""<div class="metric-card"><div class="metric-label">False "Significant" Checkpoints</div>
                    <div class="metric-value">{false_positive_rate:.0%}</div>
                    <div class="metric-delta-down">of {len(checkpoints)} checks, despite no real effect</div></div>""", unsafe_allow_html=True)
        m2.markdown(f"""<div class="metric-card"><div class="metric-label">Lowest p-value seen while peeking</div>
                    <div class="metric-value">{min(pvals):.4f}</div></div>""", unsafe_allow_html=True)

        st.caption("The fix: decide your sample size **before** the test starts (use the power calculator in the "
                   "first tab), and only look once at the end — or use a sequential-testing method explicitly "
                   "designed to allow safe early stopping.")

# ──────────────────────────────────────────────────────────────────────────
# PAGE 8 — MONDAY MORNING BRIEFING
# ──────────────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div class="hero">
        <h1>Monday Morning Briefing</h1>
        <p>An auto-generated, plain-English summary pulling live from the current forecasts, anomalies, and segments — ready to skim in under a minute.</p>
    </div>
    """, unsafe_allow_html=True)

    weekly = compute_weekly_anomalies(df)
    clusters = compute_clusters(df)
    monthly_all = get_monthly_series(df)
    best_pred, best_mae, _, _, _, _, _ = xgb_forecast(monthly_all, steps=3)

    briefing_text = generate_briefing_text(df, weekly, clusters, best_pred, best_mae)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.code(briefing_text, language=None)
    st.markdown('</div>', unsafe_allow_html=True)

    pdf_bytes = build_briefing_pdf(briefing_text)
    st.download_button(
        "📄 Download as PDF",
        data=pdf_bytes,
        file_name=f"monday_briefing_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
    )
    st.caption("Regenerates live from the cached data each time this page loads — no manual copy-pasting required for a weekly stand-up.")

st.markdown("<br><center><span style='color:#6b5f8c; font-size:0.8rem;'>DemandIQ · Built with Streamlit, XGBoost & scikit-learn</span></center>", unsafe_allow_html=True)

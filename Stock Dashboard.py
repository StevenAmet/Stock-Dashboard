# -------------------------------
# run: streamlit run app.py
# -------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Live Portfolio Risk Dashboard", layout="wide")

# -------------------------------
# UI STYLE
# -------------------------------
st.markdown("""
<style>
html, body, [class*="css"]  { font-size: 13px; }
[data-testid="stMetric"] {
    background-color: #F5F7FA !important;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #D0D7DE;
}
[data-testid="stMetric"] * { color: #000000 !important; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.title("📈 Live Portfolio Risk & Optimization Dashboard")
st.caption("Real Market Data | Risk | Optimization | ML")
st.markdown("**Created by Steven Amet**")

# -------------------------------
# TOGGLE (NEW 🔥)
# -------------------------------
mode = st.toggle("📅 Show Annualized Metrics", value=True)

# -------------------------------
# USER EXPLANATION (ENHANCED)
# -------------------------------
st.info("""
### 🧠 How This Dashboard Works

You control the portfolio using sliders.

Each slider = how much capital you allocate to each asset.

👉 Example:
- 50% AAPL, 30% MSFT, 20% GOOGL

The system:
- Combines returns using weights (weighted average)
- Calculates risk using covariance (how assets move together)
- Measures diversification (correlation + PCA)
- Estimates worst-case losses (VaR & Expected Shortfall)

⚠️ Important:
Returns can be shown as DAILY or ANNUAL.

- Daily returns look small (e.g. 0.10%)
- Annual returns multiply by ~252 trading days

👉 Example:
0.10% daily ≈ 25% yearly

This is why interpretation matters.
""")

# -------------------------------
# INPUT
# -------------------------------
st.markdown("### 📊 Portfolio Input")

tickers = st.text_input("Enter tickers (comma separated)", "AAPL,MSFT,GOOGL,AMZN")
ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip() != ""]

start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))

# -------------------------------
# DATA
# -------------------------------
raw_data = yf.download(ticker_list, start=start_date)

if raw_data.empty:
    st.error("No data found. Check tickers.")
    st.stop()

if isinstance(raw_data.columns, pd.MultiIndex):
    if "Adj Close" in raw_data.columns.get_level_values(0):
        data = raw_data["Adj Close"]
    elif "Close" in raw_data.columns.get_level_values(0):
        data = raw_data["Close"]
    else:
        st.error("Price data not found.")
        st.stop()
else:
    if "Adj Close" in raw_data.columns:
        data = raw_data[["Adj Close"]]
    elif "Close" in raw_data.columns:
        data = raw_data[["Close"]]
    else:
        st.error("Price data not found.")
        st.stop()

    data.columns = ticker_list

returns = data.pct_change().dropna()

# -------------------------------
# BENCHMARK (NEW 🔥)
# -------------------------------
spy = yf.download("SPY", start=start_date)["Adj Close"]
spy_returns = spy.pct_change().dropna()

# -------------------------------
# PRICE CHART
# -------------------------------
st.markdown("### 📈 Price Chart")
st.line_chart(data)

# -------------------------------
# WEIGHTS
# -------------------------------
st.sidebar.markdown("### ⚖️ Portfolio Weights")

weights = []
for t in ticker_list:
    w = st.sidebar.slider(f"{t}", 0.0, 1.0, 1.0/len(ticker_list))
    weights.append(w)

weights = np.array(weights)

if weights.sum() == 0:
    st.warning("All weights are zero.")
    st.stop()

weights = weights / weights.sum()

# -------------------------------
# PORTFOLIO RETURNS
# -------------------------------
portfolio_returns = returns.dot(weights)

# -------------------------------
# RISK METRICS
# -------------------------------
VaR_95 = np.percentile(portfolio_returns, 5)
VaR_99 = np.percentile(portfolio_returns, 1)
ES = portfolio_returns[portfolio_returns <= VaR_95].mean()

mean_returns = returns.mean()
cov = returns.cov()

ret = np.dot(weights, mean_returns)
risk = np.sqrt(weights.T @ cov @ weights)

# -------------------------------
# ANNUALIZATION (NEW 🔥)
# -------------------------------
if mode:
    ret = ret * 252
    risk = risk * np.sqrt(252)
    VaR_95 = VaR_95 * np.sqrt(252)
    VaR_99 = VaR_99 * np.sqrt(252)
    ES = ES * np.sqrt(252)

sharpe = ret / risk if risk != 0 else 0

# -------------------------------
# BENCHMARK METRICS 
# -------------------------------
if not spy_returns.empty:
    spy_ret = spy_returns.mean()
    spy_vol = spy_returns.std()
else:
    spy_ret, spy_vol = 0, 0

# -------------------------------
# METRICS DISPLAY
# -------------------------------
st.markdown("### 📌 Key Risk Metrics")

st.info("""
These metrics change when you move sliders:

- Return → expected performance
- Risk → volatility
- Sharpe → return per unit of risk
- VaR → worst expected loss

👉 Important:
Portfolio metrics depend on BOTH weights AND correlations.
""")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("VaR 99%", f"{VaR_99:.2%}")
c2.metric("Expected Shortfall", f"{ES:.2%}")
c3.metric("Return", f"{ret:.2%}")
c4.metric("Risk", f"{risk:.2%}")
c5.metric("Sharpe", f"{sharpe:.2f}")

# -------------------------------
# BENCHMARK 
# -------------------------------
spy_raw = yf.download("SPY", start=start_date)

if spy_raw.empty:
    st.warning("SPY benchmark data not available.")
    spy_returns = pd.Series()
else:
    if isinstance(spy_raw.columns, pd.MultiIndex):
        if "Adj Close" in spy_raw.columns.get_level_values(0):
            spy = spy_raw["Adj Close"]
        elif "Close" in spy_raw.columns.get_level_values(0):
            spy = spy_raw["Close"]
        else:
            st.warning("SPY price column not found.")
            spy_returns = pd.Series()
            spy = None
    else:
        if "Adj Close" in spy_raw.columns:
            spy = spy_raw["Adj Close"]
        elif "Close" in spy_raw.columns:
            spy = spy_raw["Close"]
        else:
            st.warning("SPY price column not found.")
            spy_returns = pd.Series()
            spy = None

    if spy is not None:
        spy_returns = spy.pct_change().dropna()

# -------------------------------
# LOSS DISTRIBUTION
# -------------------------------
st.markdown("### 📉 Loss Distribution")

st.info("""
Shows all possible outcomes:

- Left tail = extreme losses  
- VaR lines = risk thresholds  
- Width = volatility  

👉 This answers:
“How bad can losses get?”
""")

fig, ax = plt.subplots()
sns.histplot(portfolio_returns, bins=40, kde=True, ax=ax, color="#4C72B0")
ax.axvline(VaR_95, linestyle="--", color="orange")
ax.axvline(VaR_99, linestyle="--", color="red")
plt.tight_layout()
st.pyplot(fig)

# -------------------------------
# CORRELATION
# -------------------------------
st.markdown("### 🔗 Correlation Matrix")

st.info("""
Shows how assets move together:

- High correlation → risky (everything falls together)
- Low/negative → diversification

👉 Diversification reduces risk WITHOUT reducing return.
""")

fig_corr, ax = plt.subplots()
sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", ax=ax)
plt.tight_layout()
st.pyplot(fig_corr)

# -------------------------------
# PCA
# -------------------------------
st.markdown("### 📊 PCA Risk Drivers")

st.info("""
PCA shows hidden risk drivers:

- First bar = main market factor
- Large first bar = systemic risk
- Spread bars = diversified risk

👉 This reveals what is REALLY driving risk.
""")

pca = PCA().fit(returns)

fig_pca, ax = plt.subplots()
ax.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.tight_layout()
st.pyplot(fig_pca)

loadings = pd.DataFrame(
    pca.components_,
    columns=returns.columns,
    index=[f"PC{i+1}" for i in range(len(returns.columns))]
)
st.dataframe(loadings)

# -------------------------------
# OPTIMIZATION
# -------------------------------
st.markdown("### ⚙️ Portfolio Optimization")

st.info("""
This shows how efficient your portfolio is.

👉 Goal:
Maximize return for a given level of risk.
""")

st.write(f"Return: {ret:.2%}")
st.write(f"Risk: {risk:.2%}")

# -------------------------------
# EFFICIENT FRONTIER
# -------------------------------
st.markdown("### 🚀 Efficient Frontier")

results = []

for _ in range(2000):
    w = np.random.random(len(ticker_list))
    w /= np.sum(w)

    r = np.dot(w, mean_returns)
    v = np.sqrt(w.T @ cov @ w)

    if mode:
        r *= 252
        v *= np.sqrt(252)

    results.append([v, r])

results = np.array(results)

fig, ax = plt.subplots()
ax.scatter(results[:, 0], results[:, 1], alpha=0.3)
ax.scatter(risk, ret, color="red", label="Your Portfolio")
ax.set_xlabel("Risk")
ax.set_ylabel("Return")
ax.legend()
plt.tight_layout()
st.pyplot(fig)

# -------------------------------
# AUTO OPTIMIZATION
# -------------------------------
if st.button("🔍 Find Optimal Portfolio (Max Sharpe)"):
    best_sharpe = -1
    best_weights = None

    for _ in range(3000):
        w = np.random.random(len(ticker_list))
        w /= np.sum(w)

        r = np.dot(w, mean_returns)
        v = np.sqrt(w.T @ cov @ w)

        if mode:
            r *= 252
            v *= np.sqrt(252)

        s = r / v if v != 0 else 0

        if s > best_sharpe:
            best_sharpe = s
            best_weights = w

    st.success("Optimal Weights Found:")
    for t, w in zip(ticker_list, best_weights):
        st.write(f"{t}: {w:.2%}")

# -------------------------------
# ML MODEL
# -------------------------------
st.markdown("### 🤖 ML Prediction")

st.info("""
Compares predicted vs actual returns:

- Points on diagonal = accurate
- Scatter = prediction error

👉 Markets are noisy → perfect prediction is impossible.
""")

X = returns
y = returns.sum(axis=1)

model = LinearRegression().fit(X, y)
pred = model.predict(X)

fig, ax = plt.subplots()
ax.scatter(y, pred, alpha=0.6)
ax.plot([y.min(), y.max()], [y.min(), y.max()], linestyle="--", color="red")
plt.tight_layout()
st.pyplot(fig)

# -------------------------------
# EXECUTIVE SUMMARY (FULLY ENHANCED 🔥)
# -------------------------------
st.markdown("### 🧠 Executive Summary")

st.write(f"""
This portfolio has a Value-at-Risk of {abs(VaR_99):.2%}, meaning in extreme scenarios, losses of this magnitude are possible.

The expected return is {ret:.2%}, with a risk level of {risk:.2%}, resulting in a Sharpe ratio of {sharpe:.2f}.

⚠️ Important Interpretation:

- These returns are {'annualized' if mode else 'daily'}
- Daily returns appear small but scale significantly over a year
- Example: 0.10% daily ≈ 25% annually

👉 Why your return may seem “low”:
- Returns are averaged across time (not peak performance)
- Diversification smooths returns
- Correlations reduce extreme gains

👉 Key Insight:
Risk is not just about individual assets — it is driven by how assets interact.

👉 Final Takeaway:
This dashboard allows you to actively explore how changing weights impacts:
- Return
- Risk
- Diversification
- Downside exposure

""")
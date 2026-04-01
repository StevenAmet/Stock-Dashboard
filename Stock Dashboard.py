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
# INPUT
# -------------------------------
st.markdown("### 📊 Portfolio Input")

tickers = st.text_input("Enter tickers (comma separated)", "AAPL,MSFT,GOOGL,AMZN")
ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip() != ""]

start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))

# -------------------------------
# DATA
# -------------------------------
data = yf.download(ticker_list, start=start_date)

# ✅ FIX: handle single ticker case
if len(ticker_list) == 1:
    data = data[["Adj Close"]]
    data.columns = ticker_list
else:
    data = data["Adj Close"]

if data.empty:
    st.error("No data found. Check tickers.")
    st.stop()

returns = data.pct_change().dropna()

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

# ✅ FIX: prevent divide-by-zero
if weights.sum() == 0:
    st.warning("All weights are zero. Adjust sliders.")
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

# ✅ FIX: avoid division by zero
sharpe = ret / risk if risk != 0 else 0

# -------------------------------
# METRICS DISPLAY
# -------------------------------
st.markdown("### 📌 Key Risk Metrics")

st.info("""
- VaR = worst expected loss  
- ES = average extreme loss  
- Sharpe = return per unit of risk  
""")

c1, c2, c3, c4 = st.columns(4)
c1.metric("VaR 99%", f"{VaR_99:.2%}")
c2.metric("Expected Shortfall", f"{ES:.2%}")
c3.metric("Return", f"{ret:.2%}")
c4.metric("Sharpe", f"{sharpe:.2f}")

# -------------------------------
# LOSS DISTRIBUTION
# -------------------------------
st.markdown("### 📉 Loss Distribution")

st.info("""
- Left tail = worst losses  
- Wider = more volatility  
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
- High correlation = higher systemic risk  
- Low correlation = diversification  
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
- First component = main risk driver  
- Concentration = systemic risk  
""")

pca = PCA().fit(returns)

fig_pca, ax = plt.subplots()
ax.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, color="#55A868")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.tight_layout()
st.pyplot(fig_pca)

# PCA loadings
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
Balances return vs risk.
""")

st.write(f"Return: {ret:.2%}")
st.write(f"Risk: {risk:.2%}")

# -------------------------------
# ML MODEL
# -------------------------------
st.markdown("### 🤖 ML Prediction")

X = returns
y = returns.sum(axis=1)

model = LinearRegression().fit(X, y)
pred = model.predict(X)

fig_ml, ax = plt.subplots()
ax.scatter(y, pred, alpha=0.6)
ax.plot([y.min(), y.max()], [y.min(), y.max()], linestyle="--", color="red")
plt.tight_layout()
st.pyplot(fig_ml)

# -------------------------------
# EXECUTIVE SUMMARY
# -------------------------------
st.markdown("### 🧠 Executive Summary")

st.write(f"""
This portfolio has a Value-at-Risk of {abs(VaR_99):.2%}, indicating downside exposure in extreme market conditions.

The expected return is {ret:.2%} with a risk level of {risk:.2%}, resulting in a Sharpe ratio of {sharpe:.2f}.

👉 Risk is primarily driven by correlations and dominant market factors.
""")
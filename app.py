
from newsapi import NewsApiClient


import numpy as np
import streamlit as st
import yfinance as yf
import pandas as pd
import json
import os
import io
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ================= AUTO REFRESH ONLY DATA =================
@st.cache_data(ttl=10)   # Har 10 second me sirf price refresh
def get_price(stock):
    ticker = yf.Ticker(stock)
    price = ticker.history(period="1d")["Close"][0]
    return price


# ================= UI HEADER =================
st.title("AI Finance System - Portfolio Manager 💼")
st.subheader("🗑️ Manage Portfolio")



# ================= SAVE & LOAD SYSTEM =================
FILE_NAME = "portfolio.json"

def save_portfolio(data):
    with open(FILE_NAME, "w") as file:
        json.dump(data, file)

def load_portfolio():
    if os.path.exists(FILE_NAME):
        with open(FILE_NAME, "r") as file:
            return json.load(file)
    return []

portfolio = load_portfolio()

import yfinance as yf

stock = "RELIANCE.NS"       # Indian NSE ke liye .NS zaroori
ticker = yf.Ticker(stock)
data = ticker.history(period="1d")

st.subheader("📈 Live Indian Stock Price")
st.write("Current Price:", data["Close"][0])

# ================= DELETE STOCK =================
if len(portfolio) > 0:
    delete_stock = st.selectbox("Select Stock to Remove", [p[0] for p in portfolio])
    if st.button("Remove Selected Stock"):
        portfolio = [p for p in portfolio if p[0] != delete_stock]
        save_portfolio(portfolio)
        st.success(f"{delete_stock} Removed Successfully ❌")
else:
    st.info("No stocks available to remove")


# ================= ADD STOCK SIDEBAR =================
st.sidebar.title("📌 Portfolio Controls")
st.sidebar.title("⚙️ Dashboard Settings")

theme = st.sidebar.radio(
    "Choose Theme",
    ["Professional (Default)", "Dark Mode", "Bright Mode"]
)
st.sidebar.title("🔔 Alert Settings")

profit_alert = st.sidebar.number_input("Profit Alert %", value=10)
loss_alert = st.sidebar.number_input("Loss Alert %", value=-10)

if theme == "Dark Mode":
    st.markdown(
        """
        <style>
        body { background-color: #121212; color: white; }
        </style>
        """,
        unsafe_allow_html=True
    )

elif theme == "Bright Mode":
    st.markdown(
        """
        <style>
        body { background-color: #ffffff; color: black; }
        </style>
        """,
        unsafe_allow_html=True
    )

stock = st.sidebar.text_input("Stock Symbol (AAPL / TSLA / RELIANCE.NS)")
buy_price = st.sidebar.number_input("Buy Price", value=0.0)
quantity = st.sidebar.number_input("Quantity", value=0)

if st.sidebar.button("Add to Portfolio"):
    if stock != "" and buy_price > 0 and quantity > 0:
        portfolio.append([stock.upper(), buy_price, quantity])
        save_portfolio(portfolio)
        st.sidebar.success("Added Successfully ✔️")
    else:
        st.sidebar.error("Please Fill All Fields ❗")


# ================= CURRENT PORTFOLIO =================
st.subheader("📊 Current Portfolio")

table_data = []
total_investment = 0
total_current_value = 0

for stock, buy, qty in portfolio:
    price = get_price(stock)

    investment = buy * qty
    current_value = price * qty
    profit = current_value - investment
    profit_percent = (profit / investment) * 100

    total_investment += investment
    total_current_value += current_value

    table_data.append([
        stock,
        buy,
        round(price, 2),
        qty,
        round(profit, 2),
        round(profit_percent, 2)
    ])

df = pd.DataFrame(
    table_data,
    columns=["Stock", "Buy Price", "Current Price", "Qty", "Profit / Loss", "Profit %"]
)

st.dataframe(df)
st.subheader("🤖 AI Buy / Sell Suggestions")

advice_data = []

for stock in df["Stock"]:
    ticker = yf.Ticker(stock)
    hist = ticker.history(period="3mo")

    if len(hist) > 0:
        hist["MA50"] = hist["Close"].rolling(50).mean()
        current = hist["Close"].iloc[-1]
        ma50 = hist["MA50"].iloc[-1]

        if current > ma50:
            signal = "BUY 🔥 (Uptrend)"
        elif current < ma50:
            signal = "SELL ⚠️ (Downtrend)"
            signal
        else:
            signal = "HOLD 🤝"

        advice_data.append([stock, round(current,2), round(ma50,2), signal])

ai_df = pd.DataFrame(advice_data, columns=["Stock","Current Price","50 Day Avg","Suggestion"])
st.table(ai_df)


# ================= DOWNLOAD =================
st.subheader("📥 Download Portfolio")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Portfolio CSV 📩",
    data=csv,
    file_name="portfolio_data.csv",
    mime="text/csv"
)


# ================= CLEAR PORTFOLIO =================
st.subheader("🧨 Clear Portfolio")

if st.button("Clear Entire Portfolio"):
    portfolio = []
    save_portfolio(portfolio)
    st.success("Portfolio Cleared Successfully 🗑️")


# ================= SUMMARY + CHARTS =================
st.subheader("📌 Portfolio Summary")

import plotly.express as px

df["Investment"] = df["Buy Price"] * df["Qty"]

fig = px.pie(df, names="Stock", values="Investment", title="Investment Distribution 📈")
st.plotly_chart(fig)

fig2 = px.bar(df, x="Stock", y="Profit / Loss", title="Profit / Loss Bar Chart 🚀")
st.plotly_chart(fig2)

st.write("💰 Total Investment:", round(total_investment, 2))
st.write("📈 Current Value:", round(total_current_value, 2))
st.write("🔥 Total Profit / Loss:", round(total_current_value - total_investment, 2))

st.success("Portfolio Calculated Successfully 🚀")


# ================= AI ADVISOR =================
st.subheader("🤖 AI Investment Advisor")

profit_value = total_current_value - total_investment

if profit_value > 0:
    st.success("🔥 You are in Profit! Good Portfolio Management 👍")
else:
    st.error("⚠️ You are in Loss! Consider Reviewing Your Portfolio ❗")


# ================= SMART ALERTS =================
st.subheader("🔔 Smart Alerts & Notifications")

profitable = df[df["Profit %"] > 0]
loss_stocks = df[df["Profit %"] < 0]

if len(profitable) > 0:
    st.success("🟢 Profit Stocks")
    st.dataframe(profitable)
else:
    st.info("ℹ️ No profit stocks currently")

if len(loss_stocks) > 0:
    st.warning("🔴 Loss Stocks")
    st.dataframe(loss_stocks)

    risky = df[df["Profit %"] < -10]
    if len(risky) > 0:
        st.error("🚨 High Risk! More than 10% loss")
        st.dataframe(risky)
else:
    st.success("✅ No losing stocks")

st.subheader("🛡️ Portfolio Risk Score")

risk_score = 0

for value in df["Profit %"]:
    if value < -10:
        risk_score += 3
    elif value < -5:
        risk_score += 2
    elif value < 0:
        risk_score += 1

if risk_score <= 3:
    status = "LOW RISK ✅ (Healthy Portfolio)"
    color = "green"
elif risk_score <= 6:
    status = "MEDIUM RISK ⚠️ (Be Careful)"
    color = "orange"
else:
    status = "HIGH RISK 🚨 (Danger Zone)"
    color = "red"

st.write(f"### Risk Score: {risk_score}")
st.markdown(f"<h3 style='color:{color}'>{status}</h3>", unsafe_allow_html=True)

# ================= RISK ANALYSIS =================
st.subheader("⚠️ Risk Analysis")
# ======================
# AI STOCK SUGGESTIONS
# ======================
st.subheader("🤖 AI Stock Suggestions")

suggestions = []

for index, row in df.iterrows():
    stock = row["Stock"]
    buy_price = row["Buy Price"]
    current_price = row["Current Price"]

    # Stoploss = 8% below buy price
    stop_loss = buy_price * 0.92
    
    # Target = 15% profit recommendation
    target_price = buy_price * 1.15

    decision = ""

    if row["Profit %"] > 10:
        decision = "📤 Consider Booking Profit"
    elif row["Profit %"] > 0:
        decision = "👍 Good — Hold"
    elif row["Profit %"] < -10:
        decision = "🔥 High Loss — Consider Exit"
    else:
        decision = "⌛ Wait & Watch"

    suggestions.append([
        stock,
        round(current_price,2),
        round(stop_loss,2),
        round(target_price,2),
        decision
    ])

ai_df = pd.DataFrame(
    suggestions,
    columns=["Stock", "Current Price", "Suggested Stop-Loss", "Suggested Target", "AI Suggestion"]
)

st.dataframe(ai_df)
# ====================== TREND PREDICTION ======================
st.subheader("📈 Trend Prediction & Signals")

trend_data = []

for stock, buy, qty in portfolio:
    ticker = yf.Ticker(stock)
    hist = ticker.history(period="6mo")

    if len(hist) > 0:
        hist["SMA50"] = hist["Close"].rolling(50).mean()
        hist["SMA200"] = hist["Close"].rolling(200).mean()

        latest_price = hist["Close"][-1]
        sma50 = hist["SMA50"][-1]
        sma200 = hist["SMA200"][-1]

        if sma50 > sma200:
            signal = "BUY 👍 (Uptrend)"
        elif sma50 < sma200:
            signal = "SELL ❌ (Downtrend)"
        else:
            signal = "HOLD ⚠️ (Sideways Market)"

        trend_data.append([stock, round(latest_price, 2), round(sma50, 2), round(sma200, 2), signal])

trend_df = pd.DataFrame(
    trend_data,
    columns=["Stock", "Current Price", "SMA 50", "SMA 200", "AI Signal"]
)

st.dataframe(trend_df)
# ======================
# AUTO ALERT & NOTIFICATION SYSTEM
# ======================
st.subheader("🔔 Smart Auto Alerts")

alert_messages = []

for index, row in df.iterrows():
    stock = row["Stock"]
    profit_percent = row["Profit %"]

    # Profit Alert
    if profit_percent >= 10:
        alert_messages.append(f"🟢 {stock} is in strong profit (+{profit_percent}%). Consider booking profit!")

    # Minor Loss Alert
    elif -10 < profit_percent < 0:
        alert_messages.append(f"🟡 {stock} is slightly losing ({profit_percent}%). Monitor closely!")

    # Heavy Loss Alert
    elif profit_percent <= -10:
        alert_messages.append(f"🔴 {stock} is in heavy loss ({profit_percent}%). Consider exit or rethink position!")

# Show Alerts
if len(alert_messages) > 0:
    for alert in alert_messages:
        st.warning(alert)
else:
    st.success("✅ No Alerts — Portfolio Stable")

# ======================
# LIVE MARKET NEWS + AI SENTIMENT
# ======================
st.subheader("📰 Live Market News & AI Sentiment")

from newsapi import NewsApiClient
import requests

NEWS_API = "d1f7d32943da48d6bbec3299614a2c76"
newsapi = NewsApiClient(api_key=NEWS_API)


try:
    headlines = newsapi.get_top_headlines(
        q="stock market",
        language="en",
        page_size=5
    )

    if headlines["totalResults"] > 0:
        sentiments = 0
        
        for article in headlines["articles"]:
            st.write(f"### 📌 {article['title']}")
            st.write(article["description"])
            st.write("---")

            # Simple AI Sentiment Logic
            text = (article["title"] + str(article["description"])).lower()

            if "profit" in text or "growth" in text or "up" in text:
                sentiments += 1
            elif "loss" in text or "down" in text or "crash" in text or "fall" in text:
                sentiments -= 1
        
        st.subheader("🤖 Market Sentiment Result")

        if sentiments > 0:
            st.success("🔥 Market Sentiment: Positive — Good Time to Hold / Buy")
        elif sentiments == 0:
            st.info("😐 Market Sentiment: Neutral — Stable Market")
        else:
            st.error("⚠️ Market Sentiment: Risky — Be Careful!")
    else:
        st.info("No market news found")

except:
    st.warning("⚠️ Unable to fetch news. Please check API Key / Internet")

if len(df) > 0:
    risky_stocks = df[df["Profit / Loss"] < 0]
    if len(risky_stocks) > 0:
        st.warning("🚨 Risk Alert Stocks:")
        st.dataframe(risky_stocks)
    else:
        st.success("✅ No risky stocks")
        # ======================
# FOREX CURRENCY TRACKER
# ======================
st.subheader("💱 Live Forex Market")

forex_pairs = ["USDINR=X", "EURUSD=X", "GBPUSD=X"]

forex_data = []

for pair in forex_pairs:
    ticker = yf.Ticker(pair)
    price = ticker.history(period="1d")["Close"][0]
    forex_data.append([pair, round(price, 4)])

df_forex = pd.DataFrame(forex_data, columns=["Currency Pair", "Price"])
st.dataframe(df_forex)
# ======================
# CRYPTO MARKET TRACKER
# ======================
st.subheader("🪙 Live Crypto Market")

crypto_list = ["BTC-USD", "ETH-USD"]

crypto_data = []

for coin in crypto_list:
    ticker = yf.Ticker(coin)
    price = ticker.history(period="1d")["Close"][0]
    crypto_data.append([coin, round(price, 2)])

df_crypto = pd.DataFrame(crypto_data, columns=["Crypto", "Price (USD)"])
st.dataframe(df_crypto)

        # ======================
# FINANCIAL NEWS SECTION
# ======================
st.subheader("📰 Live Stock Market News")
analyzer = SentimentIntensityAnalyzer()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

st.subheader("🤖 AI Sentiment on Market News")

try:
    news = newsapi.get_top_headlines(
        category="business",
        language="en",
        country="us"
    )

    if len(news["articles"]) > 0:
        for article in news["articles"][:5]:

            # Sentiment Check
            sentiment = analyzer.polarity_scores(article["title"])
            score = sentiment['compound']

            if score >= 0.05:
                status = "🟢 Positive"
            elif score <= -0.05:
                status = "🔴 Negative"
            else:
                status = "🟡 Neutral"

            st.write("### 🔹", article["title"])
            st.write("Sentiment:", status)
            st.write("[Read More]({})".format(article["url"]))
            st.write("---")

    else:
        st.info("No news available right now")

except Exception as e:
    st.error("Failed to load news ⚠️")
    st.write(e)

st.subheader("🧠 AI Market Sentiment Analysis")
# ======================
# MARKET SENTIMENT SUMMARY
# ======================
st.write("----")

st.subheader("📊 Market Sentiment Summary")

positive_count = 0
negative_count = 0
neutral_count = 0

try:
    news = newsapi.get_top_headlines(
        category="business",
        language="en",
        country="us"
    )

    for article in news["articles"][:10]:
        score = analyzer.polarity_scores(article["title"])["compound"]

        if score >= 0.05:
            positive_count += 1
        elif score <= -0.05:
            negative_count += 1
        else:
            neutral_count += 1

    st.write(f"🟢 Positive News Count: {positive_count}")
    st.write(f"🔴 Negative News Count: {negative_count}")
    st.write(f"🟡 Neutral News Count: {neutral_count}")

    # AI Advice
    st.subheader("🤖 AI Market Advice")

    if positive_count > negative_count:
        st.success("📈 Market looks Positive! Good time to HOLD or Consider Buying!")
    elif negative_count > positive_count:
        st.error("📉 Market looks Risky! Be Careful, Avoid BIG Investments.")
    else:
        st.warning("😐 Market is Neutral! Observe Before Taking Any Decision.")

except:
    st.warning("Unable to fetch sentiment summary right now.")

try:
    news = newsapi.get_top_headlines(
        category="business",
        language="en",
        country="us"
    )

    sentiments = {"Positive": 0, "Neutral": 0, "Negative": 0}

    if len(news["articles"]) > 0:
        for article in news["articles"][:5]:
            text = article["title"]

            score = analyzer.polarity_scores(text)

            if score["compound"] >= 0.05:
                sentiments["Positive"] += 1
            elif score["compound"] <= -0.05:
                sentiments["Negative"] += 1
            else:
                sentiments["Neutral"] += 1

        st.write("### 📊 Market Sentiment Result")
        st.write(sentiments)

        if sentiments["Positive"] > sentiments["Negative"]:
            st.success("🟢 Market Sentiment: Bullish (Positive)")
        elif sentiments["Negative"] > sentiments["Positive"]:
            st.error("🔴 Market Sentiment: Bearish (Negative)")
        else:
            st.info("🟡 Market Sentiment: Neutral")
    else:
        st.info("No news to analyze right now")

except Exception as e:
    st.error("⚠️ Failed to analyze sentiment")
    st.write(e)

try:
    news = newsapi.get_top_headlines(
        category="business",
        language="en",
        country="us"
    )

    if len(news["articles"]) > 0:
        for article in news["articles"][:5]:
            st.write("### 🔹", article["title"])
            if article["description"]:
                st.write(article["description"])
            st.write("[Read More]({})".format(article["url"]))
            st.write("---")
    else:
        st.info("No news available right now")

except Exception as e:
    st.error("Failed to load news ⚠️")
    st.write(e)






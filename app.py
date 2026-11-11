
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Social Media Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/topics.csv")

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load data/topics.csv: {e}")
    st.stop()

# Clean column names and text
df.columns = df.columns.str.strip().str.lower()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna("").astype(str).str.strip()

st.title("📊 Social Media Intelligence Dashboard")
st.caption("Explore social media content, sentiment, and emerging topics.")

if df.empty:
    st.warning("Your dataset is empty. Add data and refresh the dashboard.")
    st.stop()

# Find columns available in your dataset
sentiment_col = next(
    (c for c in ["sentiment", "predicted_sentiment", "label"]
     if c in df.columns),
    None
)

topic_col = next(
    (c for c in ["topic", "predicted_topic"]
     if c in df.columns),
    None
)

channel_col = next(
    (c for c in ["channel", "platform", "source"]
     if c in df.columns),
    None
)

# Sidebar filters
st.sidebar.header("Dashboard Filters")

filtered = df.copy()

if channel_col:
    channels = sorted(
        filtered[channel_col].replace("", "Unknown").unique().tolist()
    )
    selected_channels = st.sidebar.multiselect(
        "Select channel",
        channels,
        default=channels
    )
    filtered_channel = filtered[channel_col].replace("", "Unknown")
    filtered = filtered[filtered_channel.isin(selected_channels)]

if sentiment_col:
    sentiments = sorted(
        filtered[sentiment_col].replace("", "Unknown").unique().tolist()
    )
    selected_sentiments = st.sidebar.multiselect(
        "Select sentiment",
        sentiments,
        default=sentiments
    )
    filtered_sentiment = filtered[sentiment_col].replace("", "Unknown")
    filtered = filtered[filtered_sentiment.isin(selected_sentiments)]

# KPI cards
st.subheader("Key Metrics")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", f"{len(filtered):,}")

if sentiment_col:
    labels = filtered[sentiment_col].str.lower()
    positive = labels.str.contains("positive", na=False).sum()
    negative = labels.str.contains("negative", na=False).sum()
    neutral = labels.str.contains("neutral", na=False).sum()

    c2.metric("Positive", f"{positive:,}")
    c3.metric("Negative", f"{negative:,}")
    c4.metric("Neutral", f"{neutral:,}")
else:
    c2.metric("Channels", filtered[channel_col].nunique()
              if channel_col else "N/A")
    c3.metric("Topics", filtered[topic_col].nunique()
              if topic_col else "N/A")
    c4.metric("Columns", len(filtered.columns))

st.divider()

# Charts
left, right = st.columns(2)

with left:
    st.subheader("Sentiment Distribution")
    if sentiment_col:
        sentiment_counts = (
            filtered[sentiment_col]
            .replace("", "Unknown")
            .value_counts()
        )
        st.bar_chart(sentiment_counts)
    else:
        st.info("No sentiment column found in topics.csv.")

with right:
    st.subheader("Topic Distribution")
    if topic_col:
        topic_counts = (
            filtered[topic_col]
            .replace("", "Unknown")
            .value_counts()
            .head(10)
        )
        st.bar_chart(topic_counts)
    else:
        st.info("No topic column found in topics.csv.")

if channel_col:
    st.subheader("Top 10 Channels")

    channel_counts = (
        filtered[channel_col]
        .replace("", "Unknown")
        .value_counts()
        .head(10)
    )

    st.bar_chart(channel_counts)

# Data table
st.subheader("Explore Social Media Records")
st.write(f"Showing {len(filtered):,} records")
st.dataframe(filtered, use_container_width=True, hide_index=True)

# Download filtered data
csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Data (CSV)",
    data=csv_data,
    file_name="social_media_analysis.csv",
    mime="text/csv"
)

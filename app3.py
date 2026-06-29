import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Algorithmic Attention & Fake News",
    page_icon="📰",
    layout="wide",
)

# ---------- Brand palette (2-3 colors only, per project guidelines) ----------
COLOR_REAL = "#2E86AB"     # teal/blue -> real news / neutral-positive
COLOR_FAKE = "#E63946"     # red -> fake news / risk
COLOR_NEUTRAL = "#6C757D"  # gray -> neutral

# Cleaning and merging already happened in the team's notebook (Data Cleaning +
# Feature Engineering & Merging phases). This app only loads that finished
# dataset, the same way the other dashboards in this project do.
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset/merged_data.csv")
    df = df.drop_duplicates()
    return df


merged_df = load_data()

# ---------- Sidebar filters (1-2 slicers) ----------
st.sidebar.header("🔍 Filters")

platforms = sorted(merged_df["platform"].unique())
selected_platforms = st.sidebar.multiselect("Platform", platforms, default=platforms)

age_groups = sorted(merged_df["age_group"].unique())
selected_age = st.sidebar.multiselect("Age Group", age_groups, default=age_groups)

filtered_df = merged_df[
    merged_df["platform"].isin(selected_platforms) & merged_df["age_group"].isin(selected_age)
]

# ---------- Header ----------
st.title("📰 Impact of AI Algorithms on Attention Span & Fake News")
st.write("Explore how algorithmic exposure, echo chambers, and content personalization relate to fake news across social platforms.")
st.caption("Data Analysis Bootcamp - SDA x Newtech")

if filtered_df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# ---------- KPIs (2-4) ----------
total_records = len(filtered_df)
fake_rate = filtered_df["is_fake"].mean() * 100
platform_fake_rates = filtered_df.groupby("platform")["is_fake"].mean() * 100
top_platform = platform_fake_rates.idxmax()
top_platform_rate = platform_fake_rates.max()
avg_addiction = filtered_df["ai_addiction_probability"].mean() * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Records Analyzed", f"{total_records:,}")
k2.metric("Fake News Rate", f"{fake_rate:.2f}%")
k3.metric("Highest-Risk Platform", top_platform, f"{top_platform_rate:.1f}% fake")
k4.metric("Avg. AI Addiction Probability", f"{avg_addiction:.1f}%")

st.divider()

# ---------- Charts (3-5) ----------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Real vs. Fake News Distribution")
    split = filtered_df["is_fake"].value_counts().rename({0: "Real", 1: "Fake"})
    fig = px.pie(
        values=split.values,
        names=split.index,
        color=split.index,
        color_discrete_map={"Real": COLOR_REAL, "Fake": COLOR_FAKE},
        hole=0.45,
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Fake News Rate by Platform")
    rates = platform_fake_rates.sort_values(ascending=False).reset_index()
    rates.columns = ["platform", "fake_rate"]
    fig = px.bar(
        rates, x="platform", y="fake_rate", text_auto=".1f",
        color_discrete_sequence=[COLOR_FAKE],
    )
    fig.update_layout(yaxis_title="% Fake News", xaxis_title="Platform")
    st.plotly_chart(fig, use_container_width=True)

c3, c4 = st.columns(2)

behavior_cols = [
    "algorithmic_content_exposure",
    "echo_chamber_score",
    "content_personalization_intensity",
    "emotional_manipulation_index",
]
labels_en = {
    "algorithmic_content_exposure": "Algorithmic Exposure",
    "echo_chamber_score": "Echo Chamber Score",
    "content_personalization_intensity": "Personalization Intensity",
    "emotional_manipulation_index": "Emotional Manipulation Index",
}

with c3:
    st.subheader("Avg. Behavioral Indicators: Real vs. Fake")
    means = filtered_df.groupby("is_fake")[behavior_cols].mean().rename(index={0: "Real", 1: "Fake"})
    means_t = means.T.reset_index().rename(columns={"index": "indicator"})
    means_t["indicator"] = means_t["indicator"].map(labels_en)
    fig = px.bar(
        means_t, x="indicator", y=["Real", "Fake"], barmode="group",
        color_discrete_sequence=[COLOR_REAL, COLOR_FAKE],
    )
    fig.update_layout(yaxis_title="Average", xaxis_title="", legend_title="")
    st.plotly_chart(fig, use_container_width=True)

with c4:
    st.subheader("Correlation Heatmap (Indicators vs. is_fake)")
    corr = filtered_df[behavior_cols + ["ai_addiction_probability", "is_fake"]].corr()
    fig = px.imshow(
        corr, text_auto=".2f", zmin=-1, zmax=1,
        color_continuous_scale=[COLOR_FAKE, "#FFFFFF", COLOR_REAL],
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------- Platform Comparison Tool ----------
st.subheader("🆚 Platform Comparison Tool")
st.caption(
    "Pick two platforms to compare side by side. This tool always compares across the "
    "full dataset, independent of the sidebar filters above."
)

all_platforms = sorted(merged_df["platform"].unique())
comp_col_a, comp_col_b = st.columns(2)
with comp_col_a:
    app_a = st.selectbox("App A", all_platforms, index=0)
with comp_col_b:
    default_b = 1 if len(all_platforms) > 1 else 0
    app_b = st.selectbox("App B", all_platforms, index=default_b)


def platform_summary(df, platform):
    sub = df[df["platform"] == platform]
    if sub.empty:
        return {
            "Records": 0,
            "Fake News Rate (%)": 0.0,
            "Avg. AI Addiction Probability (%)": 0.0,
            "Avg. Algorithmic Exposure": 0.0,
            "Avg. Echo Chamber Score": 0.0,
            "Avg. Personalization Intensity": 0.0,
            "Avg. Emotional Manipulation Index": 0.0,
        }
    return {
        "Records": len(sub),
        "Fake News Rate (%)": sub["is_fake"].mean() * 100,
        "Avg. AI Addiction Probability (%)": sub["ai_addiction_probability"].mean() * 100,
        "Avg. Algorithmic Exposure": sub["algorithmic_content_exposure"].mean(),
        "Avg. Echo Chamber Score": sub["echo_chamber_score"].mean(),
        "Avg. Personalization Intensity": sub["content_personalization_intensity"].mean(),
        "Avg. Emotional Manipulation Index": sub["emotional_manipulation_index"].mean(),
    }


summary_a = platform_summary(merged_df, app_a)
summary_b = platform_summary(merged_df, app_b)
comparison_df = pd.DataFrame({app_a: summary_a, app_b: summary_b})

st.dataframe(comparison_df.style.format("{:,.2f}"), use_container_width=True)

chart_metrics = [
    "Fake News Rate (%)",
    "Avg. AI Addiction Probability (%)",
    "Avg. Algorithmic Exposure",
    "Avg. Echo Chamber Score",
    "Avg. Personalization Intensity",
    "Avg. Emotional Manipulation Index",
]
chart_df = comparison_df.loc[chart_metrics].reset_index().rename(columns={"index": "Metric"})
fig = px.bar(
    chart_df, x="Metric", y=[app_a, app_b], barmode="group",
    color_discrete_sequence=[COLOR_REAL, COLOR_FAKE],
)
fig.update_layout(yaxis_title="Value", xaxis_title="", legend_title="App")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------- Insight ----------
st.subheader("💡 Insight")
st.info(
    f"The overall fake news rate in the filtered data is **{fake_rate:.1f}%**. "
    f"Most of this is concentrated on **{top_platform}**, with a fake news rate of **{top_platform_rate:.1f}%**, "
    f"while the other matched platforms show very low or zero fake news rates — "
    f"meaning the real risk of fake news in this dataset is not evenly spread across platforms, "
    f"but concentrated on one in particular, which warrants closer monitoring than the rest."
)

with st.expander("⚠️ Data Quality Note"):
    st.caption(
        "The number of matched news articles for YouTube, Facebook, and Reddit is very small "
        "(fewer than 25 articles per platform) after the merge, so the fake-news rates for these "
        "platforms may not be statistically reliable. Twitter news is currently excluded from the merge "
        "entirely because of a platform-name mismatch between the two source files "
        "('Twitter' vs. 'X/Twitter' in the AI dataset) — this is worth a fix from the data-cleaning team."
    )

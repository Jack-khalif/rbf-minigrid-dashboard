"""
QA-RBF Mini-Grid Dashboard
==========================
Interactive Streamlit app for analyzing North Kivu, DRC mini-grid data (2023).
- Loads/merges CSVs: undervoltage, SAIFI, SAIDI.
- Computes Quality-Aligned RBF (QA-RBF) payouts with weighted scores/tiers.
- Viz: Bars (SAIDI/payouts by site), lines (trends), histograms (EDA).
- Simulator: Sidebar sliders for thresholds, "if-then" conditions (Q6c).
- Based on Case Study 1: Prototyping RBF for Mini-Grid Investors.
Team: Faith Jeroben, Abigail Kairu, Daniel Maithya

Deploy: Streamlit Cloud (upload this + CSVs + requirements.txt to GitHub).
Run locally: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
import numpy as np

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================
@st.cache_data
def load_and_preprocess_data():
    try:
        df_und = pd.read_csv('DRC_minigrid_undervoltage_2023.csv', parse_dates=['day'])
        df_saifi = pd.read_csv('DRC_minigrid_saifi_2023.csv', parse_dates=['day'])
        df_saidi = pd.read_csv('DRC_minigrid_saidi_2023.csv', parse_dates=['day'])
    except FileNotFoundError:
        st.error("CSVs not found! Upload DRC_minigrid_*.csv to repo root or adjust paths.")
        st.stop()
    
    df_merged = df_und.merge(df_saifi, on=['day', 'site_id', 'minigrid_name'], how='outer', suffixes=('', '_saifi')) \
                      .merge(df_saidi, on=['day', 'site_id', 'minigrid_name'], how='outer', suffixes=('', '_saidi'))
    df_merged.rename(columns={'SAIFI_saifi': 'SAIFI', 'SAIDI_saidi': 'SAIDI'}, inplace=True)
    df_merged['day'] = pd.to_datetime(df_merged['day'])
    df_merged = df_merged.drop_duplicates(subset=['day', 'site_id'])
    df_merged['undervoltage_duration'] = df_merged['undervoltage_duration'].fillna(method='ffill').fillna(0)
    df_merged['SAIFI'] = df_merged['SAIFI'].fillna(0)
    df_merged['SAIDI'] = df_merged['SAIDI'].fillna(0)

    # create month column for convenience (first day of month)
    df_merged['month'] = df_merged['day'].dt.to_period('M').dt.to_timestamp()

    # Outlier flag
    Q1 = df_merged[['SAIDI', 'SAIFI', 'undervoltage_duration']].quantile(0.25)
    Q3 = df_merged[['SAIDI', 'SAIFI', 'undervoltage_duration']].quantile(0.75)
    IQR = Q3 - Q1
    df_merged['outlier_flag'] = ((df_merged['SAIDI'] > (Q3['SAIDI'] + 3 * IQR['SAIDI'])) | 
                                 (df_merged['SAIFI'] > (Q3['SAIFI'] + 3 * IQR['SAIFI'])) | 
                                 (df_merged['undervoltage_duration'] > (Q3['undervoltage_duration'] + 3 * IQR['undervoltage_duration']))).astype(int)

    agg_df = df_merged.groupby('site_id').agg({
        'undervoltage_duration': 'mean',
        'SAIFI': 'mean',
        'SAIDI': 'mean',
        'outlier_flag': 'mean',
        'minigrid_name': 'first'
    }).reset_index()
    agg_df.columns = ['site_id', 'undervoltage_mean', 'SAIFI_mean', 'SAIDI_mean', 'outlier_flag_mean', 'minigrid_name']

    return df_merged, agg_df

df_merged, agg_df = load_and_preprocess_data()

# =============================================================================
# RBF FUNCTIONS
# =============================================================================
def qa_rbf_score(row, saidi_thresh=5, saifi_thresh=2, und_thresh=10):
    saidi_s = max(0, (saidi_thresh - row['SAIDI_mean']) / saidi_thresh)
    saifi_s = max(0, (saifi_thresh - row['SAIFI_mean']) / saifi_thresh)
    und_s = max(0, (und_thresh - row['undervoltage_mean']) / und_thresh)
    return 0.4 * saidi_s + 0.3 * saifi_s + 0.2 * und_s + 0.1

def calculate_payout(row, score, base=10000):
    if score > 0.8:
        tier_bonus = 0.2
    elif score > 0.5:
        tier_bonus = 0
    else:
        tier_bonus = -0.3
    return max(base * (1 + score * 0.3 + tier_bonus), base * 0.5)

# =============================================================================
# IF-THEN CONDITIONS (Q6c)
# =============================================================================
def if_then_condition(condition, df):
    if condition == "SAIDI <5 for 3 months":
        monthly = df.groupby([pd.Grouper(key='day', freq='M'), 'site_id'])['SAIDI'].mean().reset_index()
        monthly['rolling_3m'] = (
            monthly.groupby('site_id')['SAIDI']
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        eligible = monthly[monthly['rolling_3m'] < 5]['site_id'].nunique()
        bonus = eligible * 2000
        return f"Eligible sites: {eligible} | Bonus pool: ${bonus:,.0f}"
    elif condition == "SAIFI <=2 annually":
        annual_saifi = df.groupby('site_id')['SAIFI'].mean()
        eligible = (annual_saifi <= 2).sum()
        return f"Eligible: {eligible} sites (no penalty)"
    elif condition == "Undervoltage <200h/year (~0.55h/day)":
        annual_und = df.groupby('site_id')['undervoltage_duration'].sum()
        eligible = (annual_und < 200).sum()
        return f"Eligible: {eligible} sites (+10% growth bonus)"
    return "Condition not implemented yet."

# =============================================================================
# DASHBOARD UI & FILTERS
# =============================================================================
st.set_page_config(page_title="QA-RBF Mini-Grid Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("QA-RBF Mini-Grid Dashboard: North Kivu, DRC 2023")
st.markdown("**Quality-Aligned Result-Based Financing for Reliable Rural Power** | Team: Faith, Abigail, Daniel")

# Sidebar: Filters & Simulator
st.sidebar.header("Filters & Simulator")
st.sidebar.markdown("Use filters to tell a focused story or download filtered data.")

# Date range selector (month-level)
min_day = df_merged['day'].min().date()
max_day = df_merged['day'].max().date()
date_range = st.sidebar.date_input("Select Date Range", value=(min_day, max_day), min_value=min_day, max_value=max_day)

# Minigrid and site selectors
minigrid_options = ['All'] + sorted(df_merged['minigrid_name'].dropna().unique().tolist())
selected_minigrid = st.sidebar.selectbox("Filter by Minigrid", options=minigrid_options)

if selected_minigrid != 'All':
    site_options = ['All'] + sorted(df_merged[df_merged['minigrid_name'] == selected_minigrid]['site_id'].unique().tolist())
else:
    site_options = ['All'] + sorted(df_merged['site_id'].unique().tolist())
selected_site = st.sidebar.selectbox("Filter by Site", options=site_options)

# RBF simulator thresholds (affect agg scores/payouts)
base = st.sidebar.slider("Base Payout ($/site/year)", 5000, 20000, 10000)
saidi_thresh = st.sidebar.slider("SAIDI Threshold (h/month)", 1.0, 10.0, 5.0)
saifi_thresh = st.sidebar.slider("SAIFI Threshold (outages/month)", 1.0, 5.0, 2.0)
und_thresh = st.sidebar.slider("Undervoltage Threshold (h/month)", 5.0, 20.0, 10.0)

# Apply filters
start_date, end_date = date_range
filtered = df_merged[(df_merged['day'].dt.date >= start_date) & (df_merged['day'].dt.date <= end_date)]
if selected_minigrid != 'All':
    filtered = filtered[filtered['minigrid_name'] == selected_minigrid]
if selected_site != 'All':
    filtered = filtered[filtered['site_id'] == selected_site]

# Recompute agg_df based on filtered data
agg_df_filtered = filtered.groupby('site_id').agg({
    'undervoltage_duration': 'mean',
    'SAIFI': 'mean',
    'SAIDI': 'mean',
    'outlier_flag': 'mean',
    'minigrid_name': 'first'
}).reset_index()
agg_df_filtered.columns = ['site_id', 'undervoltage_mean', 'SAIFI_mean', 'SAIDI_mean', 'outlier_flag_mean', 'minigrid_name']

# Add score, payout, tier
agg_df_copy = agg_df_filtered.copy()
agg_df_copy['score'] = agg_df_copy.apply(lambda row: qa_rbf_score(row, saidi_thresh, saifi_thresh, und_thresh), axis=1)
agg_df_copy['payout'] = agg_df_copy.apply(lambda row: calculate_payout(row, row['score'], base), axis=1)
agg_df_copy['tier'] = agg_df_copy['score'].apply(lambda s: 'Gold' if s>0.8 else 'Silver' if s>0.5 else 'Bronze')

# Overview KPIs
st.subheader("Overview")
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
kpi_col1.metric("Sites Analyzed", len(agg_df_copy))
kpi_col2.metric("Avg SAIDI (h/month)", f"{agg_df_copy['SAIDI_mean'].mean():.2f}" if len(agg_df_copy)>0 else "n/a")
best_site = agg_df_copy['SAIDI_mean'].idxmin() if len(agg_df_copy)>0 else None
worst_site = agg_df_copy['SAIDI_mean'].idxmax() if len(agg_df_copy)>0 else None
kpi_col3.metric("Best Performing Site", agg_df_copy.loc[best_site,'site_id'] if best_site is not None else "n/a")
kpi_col4.metric("Worst Performing Site", agg_df_copy.loc[worst_site,'site_id'] if worst_site is not None else "n/a")

# Bonus eligibility across filtered set
monthly_for_elig = filtered.groupby([pd.Grouper(key='day', freq='M'), 'site_id'])['SAIDI'].mean().reset_index()
monthly_for_elig['rolling_3m'] = (
    monthly_for_elig.groupby('site_id')['SAIDI']
    .rolling(3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
eligible_sites = monthly_for_elig[monthly_for_elig['rolling_3m'] < saidi_thresh]['site_id'].nunique()
st.markdown(f"**Sites meeting rolling 3-month SAIDI < {saidi_thresh} (within selected filters):** {eligible_sites}")

# Download filtered data
csv = filtered.to_csv(index=False).encode('utf-8')
st.download_button("Download Filtered Data (CSV)", csv, file_name='filtered_minigrid_data.csv', mime='text/csv')

st.markdown("---")

# =============================================================================
# Site Performance Table & Sunburst (Story pivot)
# =============================================================================
st.subheader("Site Performance & Distribution")

# Performance table
st.dataframe(agg_df_copy[['site_id', 'minigrid_name', 'SAIDI_mean', 'SAIFI_mean', 'undervoltage_mean', 'outlier_flag_mean', 'score', 'payout', 'tier']].round(2), use_container_width=True)

# Sunburst (filtered)
st.markdown("Distribution of Sites per Minigrid (filtered)")
site_distribution = agg_df_copy.groupby(['minigrid_name', 'site_id']).size().reset_index(name='count')
if len(site_distribution) > 0:
    fig_sunburst = px.sunburst(
        site_distribution,
        path=['minigrid_name', 'site_id'],
        values='count',
        title='Distribution of Sites per Minigrid',
        color='minigrid_name',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_sunburst.update_layout(margin=dict(t=50, l=0, r=0, b=0), title_x=0.5)
    st.plotly_chart(fig_sunburst, use_container_width=True)
else:
    st.write("No sites to show in sunburst for the applied filters.")

# =============================================================================
# Comparison visuals: Bars, Heatmap, Trends
# =============================================================================
st.subheader("Compare: SAIDI, SAIFI, Undervoltage")

colA, colB = st.columns(2)
with colA:
    color_map = {'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'}
    fig_saidi = px.bar(agg_df_copy, x='site_id', y='SAIDI_mean', title='Avg SAIDI by Site (Hours/Month)', color='tier', color_discrete_map=color_map)
    fig_saidi.update_layout(xaxis_tickangle=45, showlegend=False)
    st.plotly_chart(fig_saidi, use_container_width=True)

with colB:
    fig_payout = px.bar(agg_df_copy, x='site_id', y='payout', title='QA-RBF Payouts by Site ($)', color='tier', color_discrete_map=color_map)
    fig_payout.update_layout(xaxis_tickangle=45, showlegend=False)
    st.plotly_chart(fig_payout, use_container_width=True)

# Heatmap (site vs month SAIDI)
st.markdown("SAIDI Heatmap (Site x Month)")
heat_df = filtered.groupby([pd.Grouper(key='month', freq='M'), 'site_id'])['SAIDI'].mean().reset_index()
if not heat_df.empty:
    heat_pivot = heat_df.pivot(index='site_id', columns='month', values='SAIDI').fillna(0)
    fig_heat = px.imshow(heat_pivot, aspect='auto', labels=dict(x="Month", y="Site", color="SAIDI"),
                         title="SAIDI Heatmap (darker = higher SAIDI)")
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.write("No monthly SAIDI data for heatmap with current filters.")

# =============================================================================
# Drilldown: single-site story
# =============================================================================
st.subheader("Drilldown: Site Timeline & Rolling Average Insights")
selected_drill = st.selectbox("Choose site to drill down", options=['All'] + agg_df_copy['site_id'].tolist())
if selected_drill != 'All':
    site_df = filtered[filtered['site_id'] == selected_drill].copy()
    if site_df.empty:
        st.write("No data for this site and the selected date range.")
    else:
        # monthly series
        site_monthly = site_df.groupby(pd.Grouper(key='day', freq='M')).agg({
            'SAIDI': 'mean',
            'SAIFI': 'mean',
            'undervoltage_duration': 'mean'
        }).reset_index().rename(columns={'undervoltage_duration':'undervoltage_mean'})
        # rolling 3m on monthly SAIDI
        site_monthly['rolling_3m'] = site_monthly['SAIDI'].rolling(3, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=site_monthly['day'], y=site_monthly['SAIDI'], mode='lines+markers', name='SAIDI'))
        fig.add_trace(go.Scatter(x=site_monthly['day'], y=site_monthly['rolling_3m'], mode='lines', name='3-month rolling SAIDI'))
        # highlight bonus-eligible months
        eligible_mask = site_monthly['rolling_3m'] < saidi_thresh
        if eligible_mask.any():
            fig.add_trace(go.Bar(x=site_monthly[eligible_mask]['day'], y=site_monthly[eligible_mask]['SAIDI'], name='Eligible (rolling)', marker=dict(color='green'), opacity=0.4))
        fig.update_layout(title=f"Site {selected_drill} - SAIDI & Rolling Average", xaxis_title="Month", yaxis_title="Hours (SAIDI)")
        st.plotly_chart(fig, use_container_width=True)

        # small table of recent stats
        recent = site_monthly.tail(6).round(2)
        st.markdown("Recent 6 months (monthly SAIDI, rolling 3-month mean):")
        st.dataframe(recent[['day','SAIDI','rolling_3m','undervoltage_mean','SAIFI']].rename(columns={'day':'month'}), use_container_width=True)

# =============================================================================
# Trends selector (multi-site)
# =============================================================================
st.subheader("Trends Across Sites")
metric_choice = st.selectbox("Select Metric for Trend", ['SAIDI', 'SAIFI', 'undervoltage_duration'])
metric_label = metric_choice
trend_df = filtered.groupby([pd.Grouper(key='day', freq='M'), 'site_id'])[metric_choice].mean().reset_index()
if not trend_df.empty:
    fig_trend = px.line(trend_df, x='day', y=metric_choice, color='site_id', title=f'Monthly {metric_label} Trends by Site',
                        color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.write("No trend data for selected filters.")

# =============================================================================
# Quick EDA
# =============================================================================
if st.checkbox("Show Quick EDA (Histograms & Insights)"):
    st.subheader("Exploratory Data Analysis")
    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        fig_hist = make_subplots(rows=1, cols=3, subplot_titles=['SAIDI Distribution', 'SAIFI Distribution', 'Undervoltage Distribution'])
        fig_hist.add_trace(go.Histogram(x=filtered['SAIDI'], nbinsx=20, name='SAIDI'), row=1, col=1)
        fig_hist.add_trace(go.Histogram(x=filtered['SAIFI'], nbinsx=20, name='SAIFI'), row=1, col=2)
        fig_hist.add_trace(go.Histogram(x=filtered['undervoltage_duration'], nbinsx=20, name='Undervoltage'), row=1, col=3)
        fig_hist.update_layout(title='KPI Distributions', showlegend=False, height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    with col_eda2:
        corr = filtered[['SAIDI', 'SAIFI', 'undervoltage_duration']].corr()
        fig_heatmap = px.imshow(corr, title='KPI Correlations', color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        eda_insights = agg_df_copy.nlargest(5, 'SAIDI_mean')[['site_id', 'SAIDI_mean', 'outlier_flag_mean']].round(2)
        st.markdown("**Top 5 Worst SAIDI Sites (filtered):**")
        st.dataframe(eda_insights)

# =============================================================================
# If-Then Conditions Simulator (unchanged)
# =============================================================================
st.subheader("If-Then Payout Conditions Simulator")
condition = st.selectbox("Select Condition", ["SAIDI <5 for 3 months", "SAIFI <=2 annually", "Undervoltage <200h/year"])
st.write(if_then_condition(condition, filtered))

# =============================================================================
# Story / Background (collapsible)
# =============================================================================
with st.expander("Background: RBF, Mini-Grids, and Why Quality Metrics Matter", expanded=False):
    st.markdown("""
    Covers all case metrics (SAIDI duration, SAIFI frequency, undervoltage quality)—ties directly to KPIs.

    Across Sub-Saharan Africa, more than 550 million people still lack access to electricity (Solar Financed, 2025). 
    National grid expansion is ongoing, but progress is slow and often not economically viable for rural and low-density regions. 
    Mini-grids have therefore emerged as a scalable, decentralized solution capable of reaching these underserved populations.

    Despite their promise, many mini-grid developers face significant financing barriers. High upfront capital costs, perceived investment risks, regulatory uncertainty, and the absence of strong performance data pipelines make it difficult to secure affordable financing. On the other hand, investors and donors are reluctant to provide subsidies without assurance that projects will deliver sustainable impact.

    Results-Based Financing (RBF) has become a bridge between these challenges. By tying disbursements to verified outcomes, RBF reduces investor risk while incentivizing developers to deliver on performance. The key characteristics of RBF include:
    - Ex-post subsidies: payments are released after verified results.
    - Connection-based disbursements: most existing RBFs link payments to verified customer connections.
    - Risk mitigation: disbursements tied to independently verified outcomes.
    - Leverage of commercial funding: blending subsidies with private investment strengthens bankability.

    Many RBF schemes have been implemented across Africa and have contributed significantly to expanding electrification. However, gaps remain: KPIs emphasizing new connections do not always reflect service quality (availability, reliability, voltage quality). This dashboard focuses on quality-aligned RBF metrics to address that gap and strengthen investor confidence while improving outcomes for end-users.
    """)

# Footer notes
st.markdown("---")
st.markdown("If you want further refinements (filters, highlighting thresholds, automated report export, or responsive layout changes), tell me which parts you'd like to prioritize and I will update the code.")

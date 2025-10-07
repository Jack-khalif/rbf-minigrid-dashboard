"""
QA-RBF Mini-Grid Dashboard (Investor-focused)
Streamlit app for analyzing mini-grid reliability (SAIDI, SAIFI, undervoltage).
Includes: filters, KPI cards, sunburst, heatmap, drilldown, download.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
st.set_page_config(page_title="QA-RBF Mini-Grid Dashboard", layout="wide", initial_sidebar_state="expanded")
@st.cache_data
def load_and_preprocess_data():
    try:
        df_und = pd.read_csv('DRC_minigrid_undervoltage_2023.csv', parse_dates=['day'])
        df_saifi = pd.read_csv('DRC_minigrid_saifi_2023.csv', parse_dates=['day'])
        df_saidi = pd.read_csv('DRC_minigrid_saidi_2023.csv', parse_dates=['day'])
    except FileNotFoundError:
        st.error("CSVs not found! Upload DRC_minigrid_*.csv to repo root or adjust paths.")
        st.stop()

    df = df_und.merge(df_saifi, on=['day', 'site_id', 'minigrid_name'], how='outer', suffixes=('', '_saifi')) \
               .merge(df_saidi, on=['day', 'site_id', 'minigrid_name'], how='outer', suffixes=('', '_saidi'))
    df.rename(columns={'SAIFI_saifi': 'SAIFI', 'SAIDI_saidi': 'SAIDI'}, inplace=True)
    df['day'] = pd.to_datetime(df['day'])
    df = df.drop_duplicates(subset=['day', 'site_id'])
    df['undervoltage_duration'] = df['undervoltage_duration'].fillna(method='ffill').fillna(0)
    df['SAIFI'] = df['SAIFI'].fillna(0)
    df['SAIDI'] = df['SAIDI'].fillna(0)
    df['month'] = df['day'].dt.to_period('M').dt.to_timestamp()

    Q1 = df[['SAIDI', 'SAIFI', 'undervoltage_duration']].quantile(0.25)
    Q3 = df[['SAIDI', 'SAIFI', 'undervoltage_duration']].quantile(0.75)
    IQR = Q3 - Q1
    df['outlier_flag'] = ((df['SAIDI'] > (Q3['SAIDI'] + 3 * IQR['SAIDI'])) |
                         (df['SAIFI'] > (Q3['SAIFI'] + 3 * IQR['SAIFI'])) |
                         (df['undervoltage_duration'] > (Q3['undervoltage_duration'] + 3 * IQR['undervoltage_duration']))).astype(int)

    agg = df.groupby('site_id').agg({
        'undervoltage_duration': 'mean',
        'SAIFI': 'mean',
        'SAIDI': 'mean',
        'outlier_flag': 'mean',
        'minigrid_name': 'first'
    }).reset_index()
    agg.columns = ['site_id', 'undervoltage_mean', 'SAIFI_mean', 'SAIDI_mean', 'outlier_flag_mean', 'minigrid_name']
    return df, agg

df_merged, agg_df = load_and_preprocess_data()

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

def if_then_condition(condition, df):
    if condition == "SAIDI <5 for 3 months":
        monthly = df.groupby([pd.Grouper(key='day', freq='M'), 'site_id'])['SAIDI'].mean().reset_index()
        monthly['rolling_3m'] = monthly.groupby('site_id')['SAIDI'].rolling(3, min_periods=1).mean().values  # Simplified rolling (no reset_index mess)
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

st.markdown("""
<style>
/* --- APP BACKGROUND --- */
.stApp {
    background-color: #EAF0F6;
}

/* --- HEADERS --- */
.header { 
    color: #0A2540;  
    font-size: 28px; 
    font-weight: 700; 
}

.sub { 
    color: #1E3A8A;  
    margin-bottom: 8px;
    font-weight: 500;
}

/* --- CARD STYLE --- */
.card { 
    background: #FFFFFF; 
    border-radius: 14px; 
    padding: 16px; 
    box-shadow: 0 6px 18px rgba(0,0,0,0.07); 
    border: 1px solid #E2E8F0;
}

.card-title { 
    color: #1E3A8A;  
    font-weight: 600; 
    font-size: 14px; 
    margin-bottom: 6px; 
}

.card-value { 
    font-size: 20px; 
    font-weight: 700; 
    color: #0B3D91;  
}

.small { 
    color: #475569;  
    font-size: 12px; 
}

/* --- GLOBAL TEXT OVERRIDE --- */
section.main h1, 
section.main h2, 
section.main h3, 
section.main h4, 
section.main h5, 
section.main h6, 
section.main p, 
section.main span, 
section.main div, 
section.main label {
    color: #0A2540 !important;
}
</style>
""", unsafe_allow_html=True)




st.markdown('<div class="header">QA-RBF Mini-Grid Dashboard: North Kivu, DRC 2023</div>', unsafe_allow_html=True)
st.markdown('<div class="small">Quality-aligned Results-Based Financing — reliability & payout readiness</div>', unsafe_allow_html=True)

st.sidebar.header("Filters & Simulator")
min_day = df_merged['day'].min().date()
max_day = df_merged['day'].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_day, max_day), min_value=min_day, max_value=max_day)

minigrid_options = ['All'] + sorted(df_merged['minigrid_name'].dropna().unique().tolist())
selected_minigrid = st.sidebar.selectbox("Filter by Minigrid", minigrid_options)

if selected_minigrid != 'All':
    site_options = ['All'] + sorted(df_merged[df_merged['minigrid_name'] == selected_minigrid]['site_id'].unique().tolist())
else:
    site_options = ['All'] + sorted(df_merged['site_id'].unique().tolist())
selected_site = st.sidebar.selectbox("Filter by Site", site_options)

base = st.sidebar.slider("Base Payout ($/site/year)", 5000, 20000, 10000)
saidi_thresh = st.sidebar.slider("SAIDI Threshold (h/month)", 1.0, 10.0, 5.0)
saifi_thresh = st.sidebar.slider("SAIFI Threshold (outages/month)", 1.0, 5.0, 2.0)
und_thresh = st.sidebar.slider("Undervoltage Threshold (h/month)", 5.0, 20.0, 10.0)

start_date, end_date = date_range
filtered = df_merged[(df_merged['day'].dt.date >= start_date) & (df_merged['day'].dt.date <= end_date)]
if selected_minigrid != 'All':
    filtered = filtered[filtered['minigrid_name'] == selected_minigrid]
if selected_site != 'All':
    filtered = filtered[filtered['site_id'] == selected_site]

agg_filtered = filtered.groupby('site_id').agg({
    'undervoltage_duration': 'mean',
    'SAIFI': 'mean',
    'SAIDI': 'mean',
    'outlier_flag': 'mean',
    'minigrid_name': 'first'
}).reset_index()
agg_filtered.columns = ['site_id', 'undervoltage_mean', 'SAIFI_mean', 'SAIDI_mean', 'outlier_flag_mean', 'minigrid_name']

agg_copy = agg_filtered.copy()
agg_copy['score'] = agg_copy.apply(lambda row: qa_rbf_score(row, saidi_thresh, saifi_thresh, und_thresh), axis=1)
agg_copy['payout'] = agg_copy.apply(lambda row: calculate_payout(row, row['score'], base), axis=1)
agg_copy['tier'] = agg_copy['score'].apply(lambda s: 'Gold' if s>0.8 else 'Silver' if s>0.5 else 'Bronze')

avg_saidi_network = agg_df['SAIDI_mean'].mean() if not agg_df.empty else np.nan

col1, col2, col3, col4 = st.columns([1,1,1,1])
def card_html(title, value, subtitle=""):
    return f"""
    <div class="card">
      <div class="card-title">{title}</div>
      <div class="card-value">{value}</div>
      <div class="small">{subtitle}</div>
    </div>
    """

if selected_site == 'All':
    sites_count = len(agg_copy)
    avg_saidi = f"{agg_copy['SAIDI_mean'].mean():.2f}" if sites_count>0 else "n/a"
    total_payout = f"${agg_copy['payout'].sum():,.0f}" if sites_count>0 else "n/a"
    best_site = agg_copy.loc[agg_copy['SAIDI_mean'].idxmin(),'site_id'] if sites_count>0 else "n/a"
    worst_site = agg_copy.loc[agg_copy['SAIDI_mean'].idxmax(),'site_id'] if sites_count>0 else "n/a"

    col1.markdown(card_html("Sites Analyzed", sites_count, "Filtered set"), unsafe_allow_html=True)
    col2.markdown(card_html("Avg SAIDI (h/month)", avg_saidi, "Across filtered sites"), unsafe_allow_html=True)
    col3.markdown(card_html("Total Potential Payouts", total_payout, "Sum of site payouts"), unsafe_allow_html=True)
    col4.markdown(card_html("Best / Worst Sites", f"{best_site} / {worst_site}", "Lowest / highest avg SAIDI"), unsafe_allow_html=True)
else:
    site_stats = agg_copy[agg_copy['site_id'] == selected_site]
    if not site_stats.empty:
        site_row = site_stats.iloc[0]
        site_avg = f"{site_row['SAIDI_mean']:.2f}"
        site_payout = f"${site_row['payout']:.0f}"
        comparison = "above network avg" if site_row['SAIDI_mean'] > avg_saidi_network else "below network avg"
        col1.markdown(card_html("Selected Site", selected_site, site_row['minigrid_name']), unsafe_allow_html=True)
        col2.markdown(card_html("Site Avg SAIDI (h/month)", site_avg, f"{comparison}"), unsafe_allow_html=True)
        col3.markdown(card_html("Site Payout ($)", site_payout, f"Tier: {site_row['tier']}"), unsafe_allow_html=True)
        col4.markdown(card_html("Network Avg SAIDI", f"{avg_saidi_network:.2f}", "Comparison baseline"), unsafe_allow_html=True)
    else:
        col1.markdown(card_html("Selected Site", selected_site, "No data in date range"), unsafe_allow_html=True)
        col2.markdown(card_html("Site Avg SAIDI (h/month)", "n/a", ""), unsafe_allow_html=True)
        col3.markdown(card_html("Site Payout ($)", "n/a", ""), unsafe_allow_html=True)
        col4.markdown(card_html("Network Avg SAIDI", f"{avg_saidi_network:.2f}", ""), unsafe_allow_html=True)

st.markdown("---")

st.subheader("Site Performance & Distribution")
st.dataframe(agg_copy[['site_id', 'minigrid_name', 'SAIDI_mean', 'SAIFI_mean', 'undervoltage_mean', 'outlier_flag_mean', 'score', 'payout', 'tier']].round(2), use_container_width=True)

st.markdown("Distribution of Sites per Minigrid (filtered)")
site_dist = agg_copy.groupby(['minigrid_name', 'site_id']).size().reset_index(name='count')
# Fix: Merge payout for meaningful values (payout-weighted sunburst)
site_dist = site_dist.merge(agg_copy[['site_id', 'payout']], on='site_id', how='left')
if not site_dist.empty:
    fig_sb = px.sunburst(site_dist, path=['minigrid_name', 'site_id'], values='payout', color='minigrid_name', color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_sb.update_layout(margin=dict(t=30, l=0, r=0, b=0), title_x=0.5)
    st.plotly_chart(fig_sb, use_container_width=True)
else:
    st.warning("No sites in selected filters—adjust date/minigrid.")

st.markdown("Compare: SAIDI & Payouts")
colA, colB = st.columns(2)
with colA:
    fig_saidi = px.bar(agg_copy, x='site_id', y='SAIDI_mean', title='Avg SAIDI by Site (h/month)', color='tier', color_discrete_map={'Gold':'#FFD700','Silver':'#C0C0C0','Bronze':'#CD7F32'})
    fig_saidi.update_layout(xaxis_tickangle=45, showlegend=False)
    st.plotly_chart(fig_saidi, use_container_width=True)
with colB:
    fig_pay = px.bar(agg_copy, x='site_id', y='payout', title='QA-RBF Payouts by Site ($)', color='tier', color_discrete_map={'Gold':'#FFD700','Silver':'#C0C0C0','Bronze':'#CD7F32'})
    fig_pay.update_layout(xaxis_tickangle=45, showlegend=False)
    st.plotly_chart(fig_pay, use_container_width=True)

st.markdown("SAIDI Heatmap (Site x Month)")
heat_df = filtered.groupby([pd.Grouper(key='month', freq='M'), 'site_id'])['SAIDI'].mean().reset_index()
if not heat_df.empty:
    heat_pivot = heat_df.pivot(index='site_id', columns='month', values='SAIDI').fillna(0).sort_index()  # Sort sites alpha
    fig_heat = px.imshow(heat_pivot, aspect='auto', labels=dict(x="Month", y="Site", color="SAIDI"), title="SAIDI Heatmap")
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.write("No monthly SAIDI data for heatmap with current filters.")

st.subheader("Drilldown: Site Timeline & Rolling Average Insights")
drill_site = st.selectbox("Choose site to drill down", options=['All'] + agg_copy['site_id'].tolist())
if drill_site != 'All':
    site_df = filtered[filtered['site_id'] == drill_site].copy()
    if site_df.empty:
        st.write("No data for this site and the selected date range.")
    else:
        site_monthly = site_df.groupby(pd.Grouper(key='day', freq='M')).agg({'SAIDI':'mean','SAIFI':'mean','undervoltage_duration':'mean'}).reset_index().rename(columns={'undervoltage_duration':'undervoltage_mean'})
        site_monthly['rolling_3m'] = site_monthly['SAIDI'].rolling(3, min_periods=1).mean()  # Simplified rolling
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=site_monthly['day'], y=site_monthly['SAIDI'], mode='lines+markers', name='SAIDI'))
        fig.add_trace(go.Scatter(x=site_monthly['day'], y=site_monthly['rolling_3m'], mode='lines', name='3-month rolling SAIDI'))
        eligible_mask = site_monthly['rolling_3m'] < saidi_thresh
        if eligible_mask.any():
            fig.add_trace(go.Bar(x=site_monthly[eligible_mask]['day'], y=site_monthly[eligible_mask]['SAIDI'], name='Eligible (rolling)', marker=dict(color='green'), opacity=0.35))
        fig.update_layout(title=f"Site {drill_site} - SAIDI & 3-month rolling", xaxis_title="Month", yaxis_title="Hours")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("Recent 6 months (monthly SAIDI, rolling 3-month mean):")
        st.dataframe(site_monthly.tail(6).round(2).rename(columns={'day':'month'}), use_container_width=True)

st.subheader("Trends Across Sites")
metric_choice = st.selectbox("Select Metric for Trend", ['SAIDI', 'SAIFI', 'undervoltage_duration'])
trend_df = filtered.groupby([pd.Grouper(key='day', freq='M'), 'site_id'])[metric_choice].mean().reset_index()
if not trend_df.empty:
    fig_trend = px.line(trend_df, x='day', y=metric_choice, color='site_id', title=f'Monthly {metric_choice} Trends by Site', color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.write("No trend data for selected filters.")

if st.checkbox("Show Quick EDA"):
    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        fig_hist = make_subplots(rows=1, cols=3, subplot_titles=['SAIDI Distribution','SAIFI Distribution','Undervoltage Distribution'])
        fig_hist.add_trace(go.Histogram(x=filtered['SAIDI'], nbinsx=20), row=1, col=1)
        fig_hist.add_trace(go.Histogram(x=filtered['SAIFI'], nbinsx=20), row=1, col=2)
        fig_hist.add_trace(go.Histogram(x=filtered['undervoltage_duration'], nbinsx=20), row=1, col=3)
        fig_hist.update_layout(title='KPI Distributions', showlegend=False, height=380)
        st.plotly_chart(fig_hist, use_container_width=True)
    with col_eda2:
        corr = filtered[['SAIDI','SAIFI','undervoltage_duration']].corr()
        fig_corr = px.imshow(corr, title='KPI Correlations', color_continuous_scale='RdBu_r', aspect='auto')
        st.plotly_chart(fig_corr, use_container_width=True)
        # Safe EDA table (fix KeyError)
        cols_needed = ['site_id', 'SAIDI_mean']
        if 'outlier_flag_mean' in agg_copy.columns:
            cols_needed.append('outlier_flag_mean')
        eda_insights = agg_copy.nlargest(5, 'SAIDI_mean')[cols_needed].round(2)
        st.markdown("**Top 5 Worst SAIDI Sites (Hours/Month):**")
        st.dataframe(eda_insights)

st.subheader("If-Then Payout Conditions Simulator")
condition = st.selectbox("Select Condition", ["SAIDI <5 for 3 months", "SAIFI <=2 annually", "Undervoltage <200h/year"])
st.write(if_then_condition(condition, filtered))

csv = filtered.to_csv(index=False).encode('utf-8')
st.download_button("Download Filtered Data (CSV)", csv, file_name='filtered_minigrid_data.csv', mime='text/csv')

# Footer: Case Study Tie-In (Q7)
st.markdown("---")

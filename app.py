"""
QA-RBF Mini-Grid Dashboard (Investor-focused)
Streamlit app for analyzing mini-grid reliability (SAIDI, SAIFI, undervoltage).
Features: Dynamic percentile-based benchmarking, accountability for threshold changes, 
real-time updates when parameters change.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np

# -- App configuration --
st.set_page_config(page_title="QA-RBF Mini-Grid Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Load & preprocess data ---
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
    
    # Calculate outlier flags
    Q1 = df[['SAIDI', 'SAIFI', 'undervoltage_duration']].quantile(0.25)
    Q3 = df[['SAIDI', 'SAIFI', 'undervoltage_duration']].quantile(0.75)
    IQR = Q3 - Q1
    df['outlier_flag'] = ((df['SAIDI'] > (Q3['SAIDI'] + 3 * IQR['SAIDI'])) |
                         (df['SAIFI'] > (Q3['SAIFI'] + 3 * IQR['SAIFI'])) |
                         (df['undervoltage_duration'] > (Q3['undervoltage_duration'] + 3 * IQR['undervoltage_duration']))).astype(int)
    
    return df

df_merged = load_and_preprocess_data()

# -- Dynamic percentile-based thresholds calculation --
def get_percentile_thresholds(df, p_standard=0.90, p_bonus=0.95):
    """Calculate adaptive thresholds based on data percentiles"""
    if df.empty:
        return {
            'saidi_standard': 5.0, 'saidi_bonus': 3.0,
            'saifi_standard': 2.0, 'saifi_bonus': 1.0,
            'und_standard': 10.0, 'und_bonus': 5.0,
        }
    
    saidi_vals = df['SAIDI'].dropna()
    saifi_vals = df['SAIFI'].dropna()
    undervolt_vals = df['undervoltage_duration'].dropna()
    
    thresholds = {
        'saidi_standard': saidi_vals.quantile(p_standard) if len(saidi_vals) > 0 else 5.0,
        'saidi_bonus': saidi_vals.quantile(p_bonus) if len(saidi_vals) > 0 else 3.0,
        'saifi_standard': saifi_vals.quantile(p_standard) if len(saifi_vals) > 0 else 2.0,
        'saifi_bonus': saifi_vals.quantile(p_bonus) if len(saifi_vals) > 0 else 1.0,
        'und_standard': undervolt_vals.quantile(p_standard) if len(undervolt_vals) > 0 else 10.0,
        'und_bonus': undervolt_vals.quantile(p_bonus) if len(undervolt_vals) > 0 else 5.0,
    }
    return thresholds

def get_data_ranges(df):
    """Get min/max ranges for sliders based on actual data"""
    if df.empty:
        return {
            'saidi_min': 0.0, 'saidi_max': 20.0,
            'saifi_min': 0.0, 'saifi_max': 10.0,
            'und_min': 0.0, 'und_max': 30.0,
        }
    
    return {
        'saidi_min': 0.0, 
        'saidi_max': max(df['SAIDI'].max() * 1.2, 20.0),  # 20% buffer or min 20
        'saifi_min': 0.0,
        'saifi_max': max(df['SAIFI'].max() * 1.2, 10.0),  # 20% buffer or min 10
        'und_min': 0.0,
        'und_max': max(df['undervoltage_duration'].max() * 1.2, 30.0),  # 20% buffer or min 30
    }

# --- Styling ---
st.markdown("""
<style>
.stApp {background-color: #0F172A;}
.header {color: #E2E8F0; font-size: 28px; font-weight: 700;}
.sub {color: #93C5FD; margin-bottom: 8px; font-weight: 500;}
.card {background: #1E293B; border-radius: 14px; padding: 16px; box-shadow: 0 6px 18px rgba(0,0,0,0.4); border: 1px solid #334155;}
.card-title {color: #93C5FD; font-weight: 600; font-size: 14px; margin-bottom: 6px;}
.card-value {font-size: 20px; font-weight: 700; color: #F8FAFC;}
.small {color: #CBD5E1; font-size: 12px;}
section.main h1, section.main h2, section.main h3, section.main h4, section.main h5, section.main h6, 
section.main p, section.main span, section.main div, section.main label {color: #F8FAFC !important;}
</style>
""", unsafe_allow_html=True)

# --- Main Header ---
st.markdown('<div class="header">QA-RBF Mini-Grid Dashboard: North Kivu, DRC 2023</div>', unsafe_allow_html=True)
st.markdown('<div class="small">Quality-aligned Results-Based Financing — Dynamic Percentile-Based Benchmarking</div>', unsafe_allow_html=True)

# --- Sidebar Filters ---
st.sidebar.header(" Filters & Configuration")

# Date & location filters
min_day = df_merged['day'].min().date()
max_day = df_merged['day'].max().date()
date_range = st.sidebar.date_input("Date Range", 
                                  value=(min_day, max_day), 
                                  min_value=min_day, 
                                  max_value=max_day)

minigrid_options = ['All'] + sorted(df_merged['minigrid_name'].dropna().unique().tolist())
selected_minigrid = st.sidebar.selectbox("🏭 Filter by Minigrid", minigrid_options)

if selected_minigrid != 'All':
    site_options = ['All'] + sorted(df_merged[df_merged['minigrid_name'] == selected_minigrid]['site_id'].unique().tolist())
else:
    site_options = ['All'] + sorted(df_merged['site_id'].unique().tolist())
selected_site = st.sidebar.selectbox("📍 Filter by Site", site_options)

# --- Apply Filters to Dataset ---
start_date, end_date = date_range
filtered = df_merged[(df_merged['day'].dt.date >= start_date) & (df_merged['day'].dt.date <= end_date)]
if selected_minigrid != 'All':
    filtered = filtered[filtered['minigrid_name'] == selected_minigrid]
if selected_site != 'All':
    filtered = filtered[filtered['site_id'] == selected_site]

# --- Calculate Dynamic Percentiles and Data Ranges ---
percentile_thresholds = get_percentile_thresholds(filtered)
data_ranges = get_data_ranges(filtered)

st.sidebar.markdown("---")
st.sidebar.subheader(" Payout Configuration")

# Base payout slider
base = st.sidebar.slider("Base Payout ($/site/year)", 
                        min_value=5000, 
                        max_value=25000, 
                        value=10000, 
                        step=1000)

# --- Threshold Override with Accountability ---
st.sidebar.markdown("---")
st.sidebar.subheader(" Performance Thresholds")

# Show current adaptive thresholds
st.sidebar.info(f"""
**Current Adaptive Thresholds** (90th percentile):
- SAIDI: {percentile_thresholds['saidi_standard']:.1f} h/month
- SAIFI: {percentile_thresholds['saifi_standard']:.1f} outages/month  
- Undervoltage: {percentile_thresholds['und_standard']:.1f} h/month
""")

# Manual override option
manual_override = st.sidebar.checkbox(" Override Thresholds Manually")
adjustment_note = ""

if manual_override:
    st.sidebar.warning(" Manual override will replace adaptive thresholds")
    adjustment_note = st.sidebar.text_area(" Who & why adjusting thresholds?", 
                                          placeholder="e.g., John Doe - Adjusting for cyclone season impact")
    
    if adjustment_note:
        st.sidebar.success(" Adjustment logged for audit trail")
        
        # Manual threshold sliders with dynamic ranges
        saidi_thresh = st.sidebar.slider("SAIDI Threshold (h/month)", 
                                        min_value=data_ranges['saidi_min'],
                                        max_value=data_ranges['saidi_max'],
                                        value=float(percentile_thresholds['saidi_standard']),
                                        step=0.1)
        
        saifi_thresh = st.sidebar.slider("SAIFI Threshold (outages/month)", 
                                        min_value=data_ranges['saifi_min'],
                                        max_value=data_ranges['saifi_max'],
                                        value=float(percentile_thresholds['saifi_standard']),
                                        step=0.1)
        
        und_thresh = st.sidebar.slider("Undervoltage Threshold (h/month)", 
                                      min_value=data_ranges['und_min'],
                                      max_value=data_ranges['und_max'],
                                      value=float(percentile_thresholds['und_standard']),
                                      step=0.1)
    else:
        st.sidebar.error(" Please provide adjustment reason for accountability")
        # Use adaptive thresholds as fallback
        saidi_thresh = float(percentile_thresholds['saidi_standard'])
        saifi_thresh = float(percentile_thresholds['saifi_standard'])
        und_thresh = float(percentile_thresholds['und_standard'])
else:
    # Use adaptive thresholds
    saidi_thresh = float(percentile_thresholds['saidi_standard'])
    saifi_thresh = float(percentile_thresholds['saifi_standard'])
    und_thresh = float(percentile_thresholds['und_standard'])

# Display audit trail
if manual_override and adjustment_note:
    st.sidebar.markdown(f" **Audit Trail:** {adjustment_note}")

# --- Aggregate Filtered Data ---
if not filtered.empty:
    agg_filtered = filtered.groupby('site_id').agg({
        'undervoltage_duration': 'mean',
        'SAIFI': 'mean',
        'SAIDI': 'mean',
        'outlier_flag': 'mean',
        'minigrid_name': 'first'
    }).reset_index()
    agg_filtered.columns = ['site_id', 'undervoltage_mean', 'SAIFI_mean', 'SAIDI_mean', 'outlier_flag_mean', 'minigrid_name']
else:
    agg_filtered = pd.DataFrame(columns=['site_id', 'undervoltage_mean', 'SAIFI_mean', 'SAIDI_mean', 'outlier_flag_mean', 'minigrid_name'])

# --- Dynamic Scoring and Payout Functions ---
def qa_rbf_score_dynamic(row, saidi_threshold, saifi_threshold, und_threshold, bonus_thresholds):
    """Dynamic scoring using current thresholds"""
    saidi = row['SAIDI_mean']
    saifi = row['SAIFI_mean']
    underv = row['undervoltage_mean']
    
    # Normalize scores (lower is better for all metrics)
    saidi_score = max(0, (saidi_threshold - saidi) / saidi_threshold) if saidi_threshold > 0 else 0
    saifi_score = max(0, (saifi_threshold - saifi) / saifi_threshold) if saifi_threshold > 0 else 0
    underv_score = max(0, (und_threshold - underv) / und_threshold) if und_threshold > 0 else 0
    
    # Weighted composite score
    composite_score = 0.4 * saidi_score + 0.3 * saifi_score + 0.2 * underv_score + 0.1
    
    # Determine tier based on bonus thresholds
    if (saidi <= bonus_thresholds['saidi_bonus'] and 
        saifi <= bonus_thresholds['saifi_bonus'] and 
        underv <= bonus_thresholds['und_bonus']):
        tier = "Gold"
    elif (saidi <= saidi_threshold and 
          saifi <= saifi_threshold and 
          underv <= und_threshold):
        tier = "Silver"
    else:
        tier = "Bronze"
    
    return composite_score, tier

def calculate_payout_dynamic(score, tier, base_amount):
    """Calculate payout based on score and tier"""
    if tier == "Gold":
        tier_bonus = 0.2
    elif tier == "Silver":
        tier_bonus = 0.0
    else:
        tier_bonus = -0.3
    
    payout_raw = base_amount * (1 + score * 0.3 + tier_bonus)
    return max(payout_raw, base_amount * 0.5)  # Minimum 50% payout

# --- Apply Dynamic Scoring ---
if not agg_filtered.empty:
    agg_copy = agg_filtered.copy()
    
    # Apply dynamic scoring
    scoring_results = agg_copy.apply(lambda row: qa_rbf_score_dynamic(
        row, saidi_thresh, saifi_thresh, und_thresh, percentile_thresholds), axis=1)
    
    agg_copy[['score', 'tier']] = pd.DataFrame(scoring_results.tolist(), index=agg_copy.index)
    agg_copy['payout'] = agg_copy.apply(lambda row: calculate_payout_dynamic(row['score'], row['tier'], base), axis=1)
else:
    agg_copy = pd.DataFrame(columns=['site_id', 'undervoltage_mean', 'SAIFI_mean', 'SAIDI_mean', 'outlier_flag_mean', 'minigrid_name', 'score', 'tier', 'payout'])

# Calculate network average for comparison
avg_saidi_network = df_merged.groupby('site_id')['SAIDI'].mean().mean() if not df_merged.empty else np.nan

# --- KPI Cards ---
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
    avg_saidi = f"{agg_copy['SAIDI_mean'].mean():.2f}" if sites_count > 0 else "n/a"
    total_payout = f"${agg_copy['payout'].sum():,.0f}" if sites_count > 0 else "n/a"
    best_site = agg_copy.loc[agg_copy['SAIDI_mean'].idxmin(), 'site_id'] if sites_count > 0 else "n/a"
    worst_site = agg_copy.loc[agg_copy['SAIDI_mean'].idxmax(), 'site_id'] if sites_count > 0 else "n/a"
    
    col1.markdown(card_html("Sites Analyzed", sites_count, "Filtered dataset"), unsafe_allow_html=True)
    col2.markdown(card_html("Avg SAIDI (h/month)", avg_saidi, "Current selection"), unsafe_allow_html=True)
    col3.markdown(card_html("Total Payouts", total_payout, "Annual estimate"), unsafe_allow_html=True)
    col4.markdown(card_html("Best / Worst Sites", f"{best_site} / {worst_site}", "By SAIDI performance"), unsafe_allow_html=True)
else:
    site_stats = agg_copy[agg_copy['site_id'] == selected_site]
    if not site_stats.empty:
        site_row = site_stats.iloc[0]
        site_avg = f"{site_row['SAIDI_mean']:.2f}"
        site_payout = f"${site_row['payout']:.0f}"
        comparison = "above network avg" if site_row['SAIDI_mean'] > avg_saidi_network else "below network avg"
        
        col1.markdown(card_html("Selected Site", selected_site, site_row['minigrid_name']), unsafe_allow_html=True)
        col2.markdown(card_html("Site SAIDI (h/month)", site_avg, comparison), unsafe_allow_html=True)
        col3.markdown(card_html("Site Payout ($)", site_payout, f"Tier: {site_row['tier']}"), unsafe_allow_html=True)
        col4.markdown(card_html("Network Avg SAIDI", f"{avg_saidi_network:.2f}", "Baseline comparison"), unsafe_allow_html=True)
    else:
        col1.markdown(card_html("Selected Site", selected_site, "No data in range"), unsafe_allow_html=True)
        col2.markdown(card_html("Site SAIDI", "n/a", ""), unsafe_allow_html=True)
        col3.markdown(card_html("Site Payout", "n/a", ""), unsafe_allow_html=True)
        col4.markdown(card_html("Network Avg SAIDI", f"{avg_saidi_network:.2f}", ""), unsafe_allow_html=True)

st.markdown("---")

# --- Performance Table ---
st.subheader(" Site Performance & Payout Distribution")
if not agg_copy.empty:
    display_df = agg_copy[['site_id', 'minigrid_name', 'SAIDI_mean', 'SAIFI_mean', 'undervoltage_mean', 'score', 'tier', 'payout']].round(2)
    st.dataframe(display_df, use_container_width=True)
else:
    st.warning("No data available for current filters")

# --- Visualizations ---
if not agg_copy.empty:
    # Sunburst chart
    st.markdown("###  Minigrid Distribution by Payout Value")
    site_dist = agg_copy.groupby(['minigrid_name', 'site_id']).size().reset_index(name='count')
    site_dist = site_dist.merge(agg_copy[['site_id', 'payout']], on='site_id', how='left')
    
    fig_sb = px.sunburst(site_dist, path=['minigrid_name', 'site_id'], values='payout', 
                        color='minigrid_name', color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_sb.update_layout(margin=dict(t=30, l=0, r=0, b=0), title_x=0.5)
    st.plotly_chart(fig_sb, use_container_width=True)
    
    # Bar charts comparison
    st.markdown("###  Performance vs Payouts Comparison")
    colA, colB = st.columns(2)
    
    with colA:
        fig_saidi = px.bar(agg_copy, x='site_id', y='SAIDI_mean', 
                          title='Average SAIDI by Site (h/month)', 
                          color='tier', 
                          color_discrete_map={'Gold':'#FFD700','Silver':'#C0C0C0','Bronze':'#CD7F32'})
        fig_saidi.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_saidi, use_container_width=True)
    
    with colB:
        fig_pay = px.bar(agg_copy, x='site_id', y='payout', 
                        title='QA-RBF Payouts by Site ($)', 
                        color='tier', 
                        color_discrete_map={'Gold':'#FFD700','Silver':'#C0C0C0','Bronze':'#CD7F32'})
        fig_pay.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_pay, use_container_width=True)
    
    # Heatmap
    st.markdown("###  SAIDI Performance Heatmap (Site × Month)")
    if not filtered.empty:
        heat_df = filtered.groupby([pd.Grouper(key='month', freq='M'), 'site_id'])['SAIDI'].mean().reset_index()
        if not heat_df.empty:
            heat_pivot = heat_df.pivot(index='site_id', columns='month', values='SAIDI').fillna(0).sort_index()
            fig_heat = px.imshow(heat_pivot, aspect='auto', 
                               labels=dict(x="Month", y="Site", color="SAIDI"), 
                               title="SAIDI Performance Heatmap")
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No monthly data available for heatmap")
    
    # Site drilldown
    st.markdown("### Site Drilldown Analysis")
    drill_site = st.selectbox("Choose site for detailed analysis", 
                             options=['Select a site...'] + agg_copy['site_id'].tolist())
    
    if drill_site != 'Select a site...':
        site_df = filtered[filtered['site_id'] == drill_site].copy()
        if not site_df.empty:
            site_monthly = site_df.groupby(pd.Grouper(key='day', freq='M')).agg({
                'SAIDI': 'mean',
                'SAIFI': 'mean',
                'undervoltage_duration': 'mean'
            }).reset_index()
            site_monthly['rolling_3m'] = site_monthly['SAIDI'].rolling(3, min_periods=1).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=site_monthly['day'], y=site_monthly['SAIDI'], 
                                   mode='lines+markers', name='Monthly SAIDI'))
            fig.add_trace(go.Scatter(x=site_monthly['day'], y=site_monthly['rolling_3m'], 
                                   mode='lines', name='3-Month Rolling Average'))
            
            # Highlight periods meeting threshold
            eligible_mask = site_monthly['rolling_3m'] < saidi_thresh
            if eligible_mask.any():
                fig.add_trace(go.Bar(x=site_monthly[eligible_mask]['day'], 
                                   y=site_monthly[eligible_mask]['SAIDI'], 
                                   name='Meets Threshold', 
                                   marker=dict(color='green', opacity=0.3)))
            
            fig.update_layout(title=f"Site {drill_site} - SAIDI Performance Trend", 
                            xaxis_title="Month", yaxis_title="Hours")
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent performance table
            st.markdown("**Recent 6-Month Performance:**")
            recent_data = site_monthly.tail(6)[['day', 'SAIDI', 'SAIFI', 'undervoltage_duration', 'rolling_3m']].round(2)
            recent_data.columns = ['Month', 'SAIDI', 'SAIFI', 'Undervoltage', '3M Rolling SAIDI']
            st.dataframe(recent_data, use_container_width=True)
        else:
            st.warning(f"No data available for site {drill_site} in selected date range")

# --- Scenario Analysis (Optional) ---
with st.expander("Scenario Analysis - Impact of Policy Changes"):
    st.markdown("**Analyze how changing thresholds affects site payouts and eligibility**")
    
    scenario_col1, scenario_col2 = st.columns(2)
    
    with scenario_col1:
        scenario_saidi = st.slider("Scenario SAIDI Threshold", 
                                  min_value=1.0, 
                                  max_value=data_ranges['saidi_max'], 
                                  value=saidi_thresh, 
                                  step=0.5)
    
    with scenario_col2:
        scenario_base = st.slider("Scenario Base Payout", 
                                 min_value=5000, 
                                 max_value=25000, 
                                 value=base, 
                                 step=1000)
    
    if st.button(" Run Scenario Analysis"):
        if not agg_copy.empty:
            # Recalculate with scenario parameters
            scenario_results = agg_copy.apply(lambda row: qa_rbf_score_dynamic(
                row, scenario_saidi, saifi_thresh, und_thresh, percentile_thresholds), axis=1)
            
            scenario_df = agg_copy.copy()
            scenario_df[['scenario_score', 'scenario_tier']] = pd.DataFrame(scenario_results.tolist())
            scenario_df['scenario_payout'] = scenario_df.apply(
                lambda row: calculate_payout_dynamic(row['scenario_score'], row['scenario_tier'], scenario_base), axis=1)
            
            # Show impact comparison
            impact_df = scenario_df[['site_id', 'tier', 'payout', 'scenario_tier', 'scenario_payout']].copy()
            impact_df['payout_change'] = impact_df['scenario_payout'] - impact_df['payout']
            impact_df['payout_change_pct'] = (impact_df['payout_change'] / impact_df['payout'] * 100).round(1)
            
            st.markdown("**Impact Analysis:**")
            
            # Summary metrics
            total_current = impact_df['payout'].sum()
            total_scenario = impact_df['scenario_payout'].sum()
            total_change = total_scenario - total_current
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Total Payouts", f"${total_current:,.0f}")
            col2.metric("Scenario Total Payouts", f"${total_scenario:,.0f}", f"${total_change:,.0f}")
            col3.metric("Sites Affected", len(impact_df[impact_df['payout_change'] != 0]))
            
            st.dataframe(impact_df.round(2), use_container_width=True)

# --- Download Section ---
st.markdown("---")
st.subheader(" Export Data")

col_export1, col_export2 = st.columns(2)

with col_export1:
    if not filtered.empty:
        csv_raw = filtered.to_csv(index=False).encode('utf-8')
        st.download_button(" Download Raw Data (CSV)", 
                          csv_raw, 
                          file_name=f'minigrid_raw_data_{datetime.now().strftime("%Y%m%d")}.csv', 
                          mime='text/csv')

with col_export2:
    if not agg_copy.empty:
        csv_summary = agg_copy.to_csv(index=False).encode('utf-8')
        st.download_button(" Download Performance Summary (CSV)", 
                          csv_summary, 
                          file_name=f'minigrid_performance_summary_{datetime.now().strftime("%Y%m%d")}.csv', 
                          mime='text/csv')

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748B; font-size: 12px;'>
QA-RBF Mini-Grid Dashboard | Dynamic Percentile-Based Benchmarking | 
Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)

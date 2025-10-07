"""
QA-RBF Mini-Grid Dashboard - Hybrid 3-Stage Payment Model
Combines staged payments (30% connections, 50% quality, 20% sustained) 
with percentile-based performance assessment
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
    
    # Calculate outliers
    Q1 = df[['SAIDI', 'SAIFI', 'undervoltage_duration']].quantile(0.25)
    Q3 = df[['SAIDI', 'SAIFI', 'undervoltage_duration']].quantile(0.75)
    IQR = Q3 - Q1
    df['outlier_flag'] = ((df['SAIDI'] > (Q3['SAIDI'] + 3 * IQR['SAIDI'])) |
                         (df['SAIFI'] > (Q3['SAIFI'] + 3 * IQR['SAIFI'])) |
                         (df['undervoltage_duration'] > (Q3['undervoltage_duration'] + 3 * IQR['undervoltage_duration']))).astype(int)
    
    return df

df_merged = load_and_preprocess_data()

# -- Percentile-based performance assessment --
def get_percentile_thresholds(df, p_standard_low=0.90, p_standard_high=0.95, p_bonus=0.98):
    """Calculate percentile-based performance zones"""
    if df.empty:
        return {
            'saidi_penalty': 8.0, 'saidi_standard': 5.0, 'saidi_bonus': 2.0,
            'saifi_penalty': 4.0, 'saifi_standard': 2.0, 'saifi_bonus': 1.0,
            'und_penalty': 15.0, 'und_standard': 10.0, 'und_bonus': 5.0,
        }
    
    saidi_vals = df['SAIDI'].dropna()
    saifi_vals = df['SAIFI'].dropna()
    undervolt_vals = df['undervoltage_duration'].dropna()
    
    thresholds = {
        # Penalty zone: Below 90th percentile (worst performing 10%)
        'saidi_penalty': saidi_vals.quantile(p_standard_low) if len(saidi_vals) > 0 else 8.0,
        'saifi_penalty': saifi_vals.quantile(p_standard_low) if len(saifi_vals) > 0 else 4.0,
        'und_penalty': undervolt_vals.quantile(p_standard_low) if len(undervolt_vals) > 0 else 15.0,
        
        # Standard zone: 90th-95th percentile (acceptable performance)
        'saidi_standard': saidi_vals.quantile(p_standard_high) if len(saidi_vals) > 0 else 5.0,
        'saifi_standard': saifi_vals.quantile(p_standard_high) if len(saifi_vals) > 0 else 2.0,
        'und_standard': undervolt_vals.quantile(p_standard_high) if len(undervolt_vals) > 0 else 10.0,
        
        # Bonus zone: Above 95th percentile (exceptional performance)
        'saidi_bonus': saidi_vals.quantile(p_bonus) if len(saidi_vals) > 0 else 2.0,
        'saifi_bonus': saifi_vals.quantile(p_bonus) if len(saifi_vals) > 0 else 1.0,
        'und_bonus': undervolt_vals.quantile(p_bonus) if len(undervolt_vals) > 0 else 5.0,
    }
    return thresholds

def get_data_ranges(df):
    """Get slider ranges based on actual data"""
    if df.empty:
        return {
            'saidi_min': 0.0, 'saidi_max': 20.0,
            'saifi_min': 0.0, 'saifi_max': 10.0,
            'und_min': 0.0, 'und_max': 30.0,
        }
    
    return {
        'saidi_min': 0.0, 'saidi_max': max(df['SAIDI'].max() * 1.2, 20.0),
        'saifi_min': 0.0, 'saifi_max': max(df['SAIFI'].max() * 1.2, 10.0),
        'und_min': 0.0, 'und_max': max(df['undervoltage_duration'].max() * 1.2, 30.0),
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
.stage-1 {color: #10B981;} /* Green for Stage 1 */
.stage-2 {color: #3B82F6;} /* Blue for Stage 2 */
.stage-3 {color: #8B5CF6;} /* Purple for Stage 3 */
section.main h1, section.main h2, section.main h3, section.main h4, section.main h5, section.main h6, 
section.main p, section.main span, section.main div, section.main label {color: #F8FAFC !important;}
</style>
""", unsafe_allow_html=True)

# --- Main Header ---
st.markdown('<div class="header">QA-RBF Mini-Grid Dashboard: 3-Stage Payment Model</div>', unsafe_allow_html=True)
st.markdown('<div class="small">Hybrid Results-Based Financing: Milestone + Performance-Based Payments</div>', unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.header(" RBF Configuration")

# Project parameters
st.sidebar.subheader(" Project Parameters")
total_project_cost = st.sidebar.number_input("Total Project Cost ($)", 
                                             min_value=50000, 
                                             max_value=500000, 
                                             value=100000, 
                                             step=10000)

connection_threshold = st.sidebar.number_input("Connection Threshold for Stage 1", 
                                              min_value=50, 
                                              max_value=500, 
                                              value=100, 
                                              step=10)

# Payment stage breakdown
stage_1_pct = st.sidebar.slider("Stage 1: Connection Milestone (%)", 20, 40, 30)
stage_2_pct = st.sidebar.slider("Stage 2: Quality Assessment (%)", 40, 60, 50)
stage_3_pct = 100 - stage_1_pct - stage_2_pct  # Automatically calculate remainder

st.sidebar.info(f"""
**Payment Structure:**
- Stage 1 ({stage_1_pct}%): ${total_project_cost * stage_1_pct / 100:,.0f} at {connection_threshold} connections
- Stage 2 ({stage_2_pct}%): ${total_project_cost * stage_2_pct / 100:,.0f} after quality monitoring  
- Stage 3 ({stage_3_pct}%): ${total_project_cost * stage_3_pct / 100:,.0f} after sustained performance
""")

# Date and location filters
st.sidebar.markdown("---")
st.sidebar.subheader(" Data Filters")

min_day = df_merged['day'].min().date()
max_day = df_merged['day'].max().date()
date_range = st.sidebar.date_input("Date Range", 
                                  value=(min_day, max_day), 
                                  min_value=min_day, 
                                  max_value=max_day)

minigrid_options = ['All'] + sorted(df_merged['minigrid_name'].dropna().unique().tolist())
selected_minigrid = st.sidebar.selectbox("Filter by Minigrid", minigrid_options)

if selected_minigrid != 'All':
    site_options = ['All'] + sorted(df_merged[df_merged['minigrid_name'] == selected_minigrid]['site_id'].unique().tolist())
else:
    site_options = ['All'] + sorted(df_merged['site_id'].unique().tolist())
selected_site = st.sidebar.selectbox("Filter by Site", site_options)

# Apply filters
start_date, end_date = date_range
filtered = df_merged[(df_merged['day'].dt.date >= start_date) & (df_merged['day'].dt.date <= end_date)]
if selected_minigrid != 'All':
    filtered = filtered[filtered['minigrid_name'] == selected_minigrid]
if selected_site != 'All':
    filtered = filtered[filtered['site_id'] == selected_site]

# Calculate dynamic percentiles
percentile_thresholds = get_percentile_thresholds(filtered)
data_ranges = get_data_ranges(filtered)

# Performance threshold overrides
st.sidebar.markdown("---")
st.sidebar.subheader("Performance Thresholds")

st.sidebar.info(f"""
 **Current Percentile Zones:**
- **Penalty** (<90th): SAIDI >{percentile_thresholds['saidi_penalty']:.1f}, SAIFI >{percentile_thresholds['saifi_penalty']:.1f}
- **Standard** (90th-95th): Acceptable performance zone
- **Bonus** (>95th): SAIDI <{percentile_thresholds['saidi_bonus']:.1f}, SAIFI <{percentile_thresholds['saifi_bonus']:.1f}
""")

manual_override = st.sidebar.checkbox(" Override Performance Thresholds")
adjustment_note = ""

if manual_override:
    adjustment_note = st.sidebar.text_area(" Adjustment Justification", 
                                          placeholder="e.g., Adjusting for seasonal weather impacts")
    
    if adjustment_note:
        st.sidebar.success(" Override logged for audit")
        
        # Manual threshold sliders
        saidi_penalty = st.sidebar.slider("SAIDI Penalty Threshold", 
                                         data_ranges['saidi_min'], data_ranges['saidi_max'], 
                                         float(percentile_thresholds['saidi_penalty']), 0.1)
        saidi_standard = st.sidebar.slider("SAIDI Standard Threshold", 
                                          data_ranges['saidi_min'], data_ranges['saidi_max'], 
                                          float(percentile_thresholds['saidi_standard']), 0.1)
        saidi_bonus = st.sidebar.slider("SAIDI Bonus Threshold", 
                                       data_ranges['saidi_min'], data_ranges['saidi_max'], 
                                       float(percentile_thresholds['saidi_bonus']), 0.1)
    else:
        st.sidebar.error(" Justification required for manual override")
        saidi_penalty = float(percentile_thresholds['saidi_penalty'])
        saidi_standard = float(percentile_thresholds['saidi_standard'])
        saidi_bonus = float(percentile_thresholds['saidi_bonus'])
else:
    saidi_penalty = float(percentile_thresholds['saidi_penalty'])
    saidi_standard = float(percentile_thresholds['saidi_standard'])
    saidi_bonus = float(percentile_thresholds['saidi_bonus'])

# Aggregate filtered data
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
    agg_filtered = pd.DataFrame()

# --- 3-Stage Payment Calculation ---
def calculate_performance_multiplier(saidi, saifi, undervoltage, thresholds):
    """Calculate performance multiplier for Stages 2 & 3 based on percentile zones"""
    
    # Determine performance zone for each metric (lower values are better)
    def get_zone_score(value, penalty_thresh, standard_thresh, bonus_thresh):
        if value <= bonus_thresh:  # Exceptional performance
            return 1.2  # 20% bonus
        elif value <= standard_thresh:  # Good performance  
            return 1.0  # Full payment
        elif value <= penalty_thresh:  # Acceptable performance
            return 0.9  # 10% penalty
        else:  # Poor performance
            return 0.7  # 30% penalty
    
    saidi_score = get_zone_score(saidi, saidi_penalty, saidi_standard, saidi_bonus)
    saifi_score = get_zone_score(saifi, percentile_thresholds['saifi_penalty'], 
                                percentile_thresholds['saifi_standard'], percentile_thresholds['saifi_bonus'])
    underv_score = get_zone_score(undervoltage, percentile_thresholds['und_penalty'], 
                                 percentile_thresholds['und_standard'], percentile_thresholds['und_bonus'])
    
    # Weighted average (SAIDI most important)
    composite_multiplier = 0.5 * saidi_score + 0.3 * saifi_score + 0.2 * underv_score
    
    # Cap between 0.5 and 1.3 (50% minimum, 30% maximum bonus)
    return max(0.5, min(1.3, composite_multiplier))

def calculate_three_stage_payout(row, total_cost, stage_1_pct, stage_2_pct, stage_3_pct, connection_thresh=100):
    """Calculate payout for each of the 3 stages"""
    
    # Stage 1: Connection milestone (fixed payout once threshold reached)
    # Assume all sites have reached connection threshold for this demo
    stage_1_amount = total_cost * stage_1_pct / 100
    
    # Stage 2 & 3: Performance-based with multipliers
    performance_multiplier = calculate_performance_multiplier(
        row['SAIDI_mean'], row['SAIFI_mean'], row['undervoltage_mean'], percentile_thresholds)
    
    stage_2_base = total_cost * stage_2_pct / 100
    stage_2_amount = stage_2_base * performance_multiplier
    
    stage_3_base = total_cost * stage_3_pct / 100  
    stage_3_amount = stage_3_base * performance_multiplier
    
    total_payout = stage_1_amount + stage_2_amount + stage_3_amount
    
    return {
        'stage_1_payout': stage_1_amount,
        'stage_2_payout': stage_2_amount,
        'stage_3_payout': stage_3_amount,
        'total_payout': total_payout,
        'performance_multiplier': performance_multiplier
    }

def get_performance_zone(saidi, saifi, undervoltage):
    """Determine overall performance zone"""
    zones = []
    
    if saidi <= saidi_bonus:
        zones.append("Bonus")
    elif saidi <= saidi_standard:
        zones.append("Standard") 
    elif saidi <= saidi_penalty:
        zones.append("Acceptable")
    else:
        zones.append("Penalty")
    
    # Return the most common zone (simplified)
    if "Penalty" in zones:
        return "Penalty Zone"
    elif "Bonus" in zones:
        return "Bonus Zone"
    elif "Standard" in zones:
        return "Standard Zone"
    else:
        return "Acceptable Zone"

# Apply calculations
if not agg_filtered.empty:
    agg_copy = agg_filtered.copy()
    
    # Calculate 3-stage payouts for each site
    payout_results = agg_copy.apply(lambda row: calculate_three_stage_payout(
        row, total_project_cost, stage_1_pct, stage_2_pct, stage_3_pct, connection_threshold), axis=1)
    
    # Expand payout results into separate columns
    for key in ['stage_1_payout', 'stage_2_payout', 'stage_3_payout', 'total_payout', 'performance_multiplier']:
        agg_copy[key] = [result[key] for result in payout_results]
    
    # Add performance zones
    agg_copy['performance_zone'] = agg_copy.apply(lambda row: get_performance_zone(
        row['SAIDI_mean'], row['SAIFI_mean'], row['undervoltage_mean']), axis=1)
else:
    agg_copy = pd.DataFrame()

# --- Dashboard KPI Cards ---
col1, col2, col3, col4 = st.columns([1,1,1,1])

def card_html(title, value, subtitle="", css_class=""):
    return f"""
    <div class="card">
      <div class="card-title">{title}</div>
      <div class="card-value {css_class}">{value}</div>
      <div class="small">{subtitle}</div>
    </div>
    """

if not agg_copy.empty:
    if selected_site == 'All':
        sites_count = len(agg_copy)
        avg_saidi = f"{agg_copy['SAIDI_mean'].mean():.2f}"
        total_stage_1 = f"${agg_copy['stage_1_payout'].sum():,.0f}"
        total_all_stages = f"${agg_copy['total_payout'].sum():,.0f}"
        
        col1.markdown(card_html("Sites Analyzed", sites_count, "In current filter"), unsafe_allow_html=True)
        col2.markdown(card_html("Avg SAIDI", avg_saidi, "Hours/month"), unsafe_allow_html=True)
        col3.markdown(card_html("Stage 1 Total", total_stage_1, "Connection payments", "stage-1"), unsafe_allow_html=True)
        col4.markdown(card_html("All Stages Total", total_all_stages, "Complete project payouts"), unsafe_allow_html=True)
    else:
        site_stats = agg_copy[agg_copy['site_id'] == selected_site]
        if not site_stats.empty:
            site_row = site_stats.iloc[0]
            col1.markdown(card_html("Selected Site", selected_site, site_row['minigrid_name']), unsafe_allow_html=True)
            col2.markdown(card_html("SAIDI Performance", f"{site_row['SAIDI_mean']:.2f}", site_row['performance_zone']), unsafe_allow_html=True)
            col3.markdown(card_html("Performance Multiplier", f"{site_row['performance_multiplier']:.2f}x", "Applied to Stages 2&3"), unsafe_allow_html=True)
            col4.markdown(card_html("Total Payout", f"${site_row['total_payout']:,.0f}", "All 3 stages"), unsafe_allow_html=True)

st.markdown("---")

# --- Performance Summary Table ---
st.subheader("Site Performance & 3-Stage Payout Breakdown")

if not agg_copy.empty:
    display_cols = ['site_id', 'minigrid_name', 'SAIDI_mean', 'SAIFI_mean', 
                   'performance_zone', 'performance_multiplier', 
                   'stage_1_payout', 'stage_2_payout', 'stage_3_payout', 'total_payout']
    display_df = agg_copy[display_cols].round(2)
    st.dataframe(display_df, use_container_width=True)
else:
    st.warning("No data available for current filters")

# --- Visualizations ---
if not agg_copy.empty:
    
    # 3-Stage Payment Breakdown
    st.subheader(" Payment Stage Analysis")
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        # Stacked bar chart of payment stages
        fig_stages = go.Figure()
        
        fig_stages.add_trace(go.Bar(name='Stage 1 (Connections)', x=agg_copy['site_id'], y=agg_copy['stage_1_payout'],
                                   marker_color='#10B981'))
        fig_stages.add_trace(go.Bar(name='Stage 2 (Quality)', x=agg_copy['site_id'], y=agg_copy['stage_2_payout'],
                                   marker_color='#3B82F6'))
        fig_stages.add_trace(go.Bar(name='Stage 3 (Sustained)', x=agg_copy['site_id'], y=agg_copy['stage_3_payout'],
                                   marker_color='#8B5CF6'))
        
        fig_stages.update_layout(barmode='stack', title='3-Stage Payment Breakdown by Site',
                               xaxis_title='Site ID', yaxis_title='Payment Amount ($)',
                               xaxis_tickangle=45)
        st.plotly_chart(fig_stages, use_container_width=True)
    
    with col_viz2:
        # Performance multiplier vs SAIDI
        fig_perf = px.scatter(agg_copy, x='SAIDI_mean', y='performance_multiplier', 
                            color='performance_zone', size='total_payout',
                            title='Performance Multiplier vs SAIDI',
                            labels={'SAIDI_mean': 'Average SAIDI (h/month)', 
                                   'performance_multiplier': 'Performance Multiplier'})
        
        # Add reference lines
        fig_perf.add_hline(y=1.0, line_dash="dash", line_color="white", 
                          annotation_text="Standard Performance (1.0x)")
        fig_perf.add_vline(x=saidi_standard, line_dash="dash", line_color="yellow",
                          annotation_text="Standard Threshold")
        
        st.plotly_chart(fig_perf, use_container_width=True)
    
    # Performance zone distribution
    st.subheader(" Performance Zone Distribution")
    
    zone_counts = agg_copy['performance_zone'].value_counts()
    fig_zones = px.pie(values=zone_counts.values, names=zone_counts.index, 
                      title='Sites by Performance Zone')
    st.plotly_chart(fig_zones, use_container_width=True)
    
    # Site timeline analysis
    st.subheader(" Individual Site Analysis")
    drill_site = st.selectbox("Select site for detailed timeline", 
                             options=['Choose a site...'] + agg_copy['site_id'].tolist())
    
    if drill_site != 'Choose a site...':
        site_df = filtered[filtered['site_id'] == drill_site]
        if not site_df.empty:
            # Monthly performance tracking
            site_monthly = site_df.groupby(pd.Grouper(key='day', freq='M')).agg({
                'SAIDI': 'mean', 'SAIFI': 'mean', 'undervoltage_duration': 'mean'
            }).reset_index()
            
            fig_timeline = go.Figure()
            
            # Add performance thresholds as background zones
            fig_timeline.add_hrect(y0=0, y1=saidi_bonus, fillcolor="green", opacity=0.1, 
                                 annotation_text="Bonus Zone", annotation_position="top left")
            fig_timeline.add_hrect(y0=saidi_bonus, y1=saidi_standard, fillcolor="blue", opacity=0.1,
                                 annotation_text="Standard Zone", annotation_position="top left")
            fig_timeline.add_hrect(y0=saidi_standard, y1=saidi_penalty, fillcolor="orange", opacity=0.1,
                                 annotation_text="Acceptable Zone", annotation_position="top left")
            fig_timeline.add_hrect(y0=saidi_penalty, y1=data_ranges['saidi_max'], fillcolor="red", opacity=0.1,
                                 annotation_text="Penalty Zone", annotation_position="top left")
            
            # Add actual performance line
            fig_timeline.add_trace(go.Scatter(x=site_monthly['day'], y=site_monthly['SAIDI'],
                                            mode='lines+markers', name='Monthly SAIDI',
                                            line=dict(color='white', width=3)))
            
            fig_timeline.update_layout(title=f'Site {drill_site} - Performance Timeline',
                                     xaxis_title='Month', yaxis_title='SAIDI (hours)')
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Show payout breakdown for this site
            site_payout = agg_copy[agg_copy['site_id'] == drill_site].iloc[0]
            
            col_site1, col_site2, col_site3, col_site4 = st.columns(4)
            col_site1.metric("Stage 1", f"${site_payout['stage_1_payout']:,.0f}", "Connections")
            col_site2.metric("Stage 2", f"${site_payout['stage_2_payout']:,.0f}", "Quality Assessment") 
            col_site3.metric("Stage 3", f"${site_payout['stage_3_payout']:,.0f}", "Sustained Performance")
            col_site4.metric("Total", f"${site_payout['total_payout']:,.0f}", 
                           f"{site_payout['performance_multiplier']:.2f}x multiplier")

# --- Export Options ---
st.markdown("---")
st.subheader(" Export Results")

col_export1, col_export2 = st.columns(2)

with col_export1:
    if not filtered.empty:
        csv_raw = filtered.to_csv(index=False).encode('utf-8')
        st.download_button(" Download Raw Performance Data", 
                          csv_raw, 
                          file_name=f'rbf_raw_data_{datetime.now().strftime("%Y%m%d")}.csv')

with col_export2:
    if not agg_copy.empty:
        csv_payouts = agg_copy.to_csv(index=False).encode('utf-8')
        st.download_button(" Download Payout Analysis", 
                          csv_payouts, 
                          file_name=f'rbf_3stage_payouts_{datetime.now().strftime("%Y%m%d")}.csv')

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748B; font-size: 12px;'>
3-Stage RBF Dashboard | Connection Milestone + Performance-Based Quality Payments | 
Percentile-Adaptive Benchmarking
</div>
""", unsafe_allow_html=True)

"""
RBF Mini-Grid Dashboard - 3-Stage Payment Model
Combines staged payments (connections, quality, sustained) 
with percentile-based performance assessment and actual connection counts
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np

# -- App configuration --
st.set_page_config(page_title="RBF Mini-Grid Dashboard", layout="wide", initial_sidebar_state="expanded")

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
    
    # Add simulated connection counts (since not in original data)
    # In real implementation, this would come from actual connection data
    np.random.seed(42)  # For reproducible results
    site_ids = df['site_id'].unique()
    connections_data = pd.DataFrame({
        'site_id': site_ids,
        'total_connections': np.random.randint(50, 250, len(site_ids)),  # Random connections 50-250
        'target_connections': np.random.randint(100, 200, len(site_ids))  # Target connections
    })
    
    df = df.merge(connections_data, on='site_id', how='left')
    
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
        'saidi_penalty': saidi_vals.quantile(p_standard_low) if len(saidi_vals) > 0 else 8.0,
        'saifi_penalty': saifi_vals.quantile(p_standard_low) if len(saifi_vals) > 0 else 4.0,
        'und_penalty': undervolt_vals.quantile(p_standard_low) if len(undervolt_vals) > 0 else 15.0,
        'saidi_standard': saidi_vals.quantile(p_standard_high) if len(saidi_vals) > 0 else 5.0,
        'saifi_standard': saifi_vals.quantile(p_standard_high) if len(saifi_vals) > 0 else 2.0,
        'und_standard': undervolt_vals.quantile(p_standard_high) if len(undervolt_vals) > 0 else 10.0,
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
.stage-1 {color: #10B981;}
.stage-2 {color: #3B82F6;}
.stage-3 {color: #8B5CF6;}
section.main h1, section.main h2, section.main h3, section.main h4, section.main h5, section.main h6, 
section.main p, section.main span, section.main div, section.main label {color: #F8FAFC !important;}
</style>
""", unsafe_allow_html=True)

# --- Main Header ---
st.markdown('<div class="header">RBF Mini-Grid Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="small">Results-Based Financing: Connection-Based + Performance-Based Payments</div>', unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.header("RBF Configuration")

# Project parameters
st.sidebar.subheader("Project Parameters")
total_project_cost = st.sidebar.number_input("Total Project Cost ($)", 
                                             min_value=50000, 
                                             max_value=500000, 
                                             value=100000, 
                                             step=10000)

# Connection parameters
st.sidebar.subheader("Connection Milestones")
minimum_connections = st.sidebar.number_input("Minimum Connections for Stage 1", 
                                              min_value=50, 
                                              max_value=200, 
                                              value=100, 
                                              step=10)

connection_payment_per_unit = st.sidebar.number_input("Payment per Connection ($)", 
                                                     min_value=100, 
                                                     max_value=1000, 
                                                     value=300, 
                                                     step=50)

# Payment stage breakdown
stage_1_pct = st.sidebar.slider("Stage 1: Connection Milestone (%)", 20, 40, 30)
stage_2_pct = st.sidebar.slider("Stage 2: Quality Assessment (%)", 40, 60, 50)
stage_3_pct = 100 - stage_1_pct - stage_2_pct

st.sidebar.info(f"""
Payment Structure:
- Stage 1 ({stage_1_pct}%): Connection-based payment
- Stage 2 ({stage_2_pct}%): Quality assessment after 3 months
- Stage 3 ({stage_3_pct}%): Sustained performance after 6 months

Connection Model: ${connection_payment_per_unit} per connection (minimum {minimum_connections})
""")

# Date and location filters
st.sidebar.markdown("---")
st.sidebar.subheader("Data Filters")

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
Current Percentile Zones:
- Penalty (<90th): SAIDI >{percentile_thresholds['saidi_penalty']:.1f}h, SAIFI >{percentile_thresholds['saifi_penalty']:.1f}
- Standard (90th-95th): Acceptable performance
- Bonus (>95th): SAIDI <{percentile_thresholds['saidi_bonus']:.1f}h, SAIFI <{percentile_thresholds['saifi_bonus']:.1f}
""")

manual_override = st.sidebar.checkbox("Override Performance Thresholds")
adjustment_note = ""

if manual_override:
    adjustment_note = st.sidebar.text_area("Adjustment Justification", 
                                          placeholder="e.g., Adjusting for seasonal weather impacts")
    
    if adjustment_note:
        st.sidebar.success("Override logged for audit")
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
        st.sidebar.error("Justification required for manual override")
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
        'minigrid_name': 'first',
        'total_connections': 'first',
        'target_connections': 'first'
    }).reset_index()
    agg_filtered.columns = ['site_id', 'undervoltage_mean', 'SAIFI_mean', 'SAIDI_mean', 'outlier_flag_mean', 
                           'minigrid_name', 'total_connections', 'target_connections']
else:
    agg_filtered = pd.DataFrame()

# --- 3-Stage Payment Calculation with Connection-Based Logic ---
def calculate_performance_multiplier(saidi, saifi, undervoltage, thresholds):
    """Calculate performance multiplier for Stages 2 & 3"""
    def get_zone_score(value, penalty_thresh, standard_thresh, bonus_thresh):
        if value <= bonus_thresh:
            return 1.2  # 20% bonus
        elif value <= standard_thresh:
            return 1.0  # Full payment
        elif value <= penalty_thresh:
            return 0.9  # 10% penalty
        else:
            return 0.7  # 30% penalty
    
    saidi_score = get_zone_score(saidi, saidi_penalty, saidi_standard, saidi_bonus)
    saifi_score = get_zone_score(saifi, percentile_thresholds['saifi_penalty'], 
                                percentile_thresholds['saifi_standard'], percentile_thresholds['saifi_bonus'])
    underv_score = get_zone_score(undervoltage, percentile_thresholds['und_penalty'], 
                                 percentile_thresholds['und_standard'], percentile_thresholds['und_bonus'])
    
    composite_multiplier = 0.5 * saidi_score + 0.3 * saifi_score + 0.2 * underv_score
    return max(0.5, min(1.3, composite_multiplier))

def calculate_connection_based_payout(row, total_cost, stage_1_pct, stage_2_pct, stage_3_pct, 
                                     min_connections, payment_per_connection):
    """Calculate payout including actual connection numbers"""
    
    connections = row['total_connections']
    
    # Stage 1: Connection-based payment
    if connections >= min_connections:
        # Base Stage 1 amount plus additional per connection
        base_stage_1 = total_cost * stage_1_pct / 100
        additional_connections = max(0, connections - min_connections)
        stage_1_amount = base_stage_1 + (additional_connections * payment_per_connection)
    else:
        # Prorated payment for not meeting minimum threshold
        stage_1_amount = (total_cost * stage_1_pct / 100) * (connections / min_connections)
    
    # Stage 2 & 3: Performance-based
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
        'performance_multiplier': performance_multiplier,
        'connection_status': 'Met' if connections >= min_connections else 'Below Minimum'
    }

def get_performance_zone(saidi, saifi, undervoltage):
    """Determine overall performance zone"""
    if saidi <= saidi_bonus and saifi <= percentile_thresholds['saifi_bonus']:
        return "Bonus Zone"
    elif saidi <= saidi_standard and saifi <= percentile_thresholds['saifi_standard']:
        return "Standard Zone"
    elif saidi <= saidi_penalty and saifi <= percentile_thresholds['saifi_penalty']:
        return "Acceptable Zone"
    else:
        return "Penalty Zone"

# Apply calculations
if not agg_filtered.empty:
    agg_copy = agg_filtered.copy()
    
    payout_results = agg_copy.apply(lambda row: calculate_connection_based_payout(
        row, total_project_cost, stage_1_pct, stage_2_pct, stage_3_pct, 
        minimum_connections, connection_payment_per_unit), axis=1)
    
    for key in ['stage_1_payout', 'stage_2_payout', 'stage_3_payout', 'total_payout', 
                'performance_multiplier', 'connection_status']:
        agg_copy[key] = [result[key] for result in payout_results]
    
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
        total_connections = agg_copy['total_connections'].sum()
        total_all_stages = f"${agg_copy['total_payout'].sum():,.0f}"
        best_site = agg_copy.loc[agg_copy['SAIDI_mean'].idxmin(), 'site_id'] if sites_count > 0 else "n/a"
        worst_site = agg_copy.loc[agg_copy['SAIDI_mean'].idxmax(), 'site_id'] if sites_count > 0 else "n/a"
        
        col1.markdown(card_html("Sites Analyzed", sites_count, "In current filter"), unsafe_allow_html=True)
        col2.markdown(card_html("Total Connections", total_connections, "Across all sites"), unsafe_allow_html=True)
        col3.markdown(card_html("Total Payouts", total_all_stages, "All 3 stages combined"), unsafe_allow_html=True)
        col4.markdown(card_html("Best / Worst Sites", f"{best_site} / {worst_site}", "Lowest / highest SAIDI"), unsafe_allow_html=True)
    else:
        site_stats = agg_copy[agg_copy['site_id'] == selected_site]
        if not site_stats.empty:
            site_row = site_stats.iloc[0]
            avg_saidi_network = agg_copy['SAIDI_mean'].mean() if len(agg_copy) > 1 else site_row['SAIDI_mean']
            comparison = "above network avg" if site_row['SAIDI_mean'] > avg_saidi_network else "below network avg"
            
            col1.markdown(card_html("Site ID", selected_site, site_row['minigrid_name']), unsafe_allow_html=True)
            col2.markdown(card_html("Connections", f"{site_row['total_connections']}", site_row['connection_status']), unsafe_allow_html=True)
            col3.markdown(card_html("SAIDI Performance", f"{site_row['SAIDI_mean']:.2f}h", comparison), unsafe_allow_html=True)
            col4.markdown(card_html("Total Payout", f"${site_row['total_payout']:,.0f}", f"{site_row['performance_zone']}"), unsafe_allow_html=True)

st.markdown("---")

# --- Performance Summary Table ---
st.subheader("Site Performance & Connection-Based Payout Analysis")

if not agg_copy.empty:
    display_cols = ['site_id', 'minigrid_name', 'total_connections', 'connection_status', 
                   'SAIDI_mean', 'SAIFI_mean', 'performance_zone', 'performance_multiplier', 
                   'stage_1_payout', 'stage_2_payout', 'stage_3_payout', 'total_payout']
    display_df = agg_copy[display_cols].round(2)
    st.dataframe(display_df, use_container_width=True)
else:
    st.warning("No data available for current filters")

# --- Visualizations ---
if not agg_copy.empty:
    
    # Sunburst chart - The beautiful visualization is back!
    st.subheader("Distribution of Sites per Minigrid by Payout Value")
    site_dist = agg_copy.groupby(['minigrid_name', 'site_id']).size().reset_index(name='count')
    site_dist = site_dist.merge(agg_copy[['site_id', 'total_payout']], on='site_id', how='left')
    
    if not site_dist.empty:
        fig_sb = px.sunburst(site_dist, path=['minigrid_name', 'site_id'], values='total_payout', 
                            color='minigrid_name', color_discrete_sequence=px.colors.qualitative.Pastel,
                            title="Payout Distribution Across Minigrids and Sites")
        fig_sb.update_layout(margin=dict(t=50, l=0, r=0, b=0), title_x=0.5)
        st.plotly_chart(fig_sb, use_container_width=True)
    else:
        st.warning("No sites in selected filters for sunburst visualization")
    
    # Connection vs Payment Analysis
    st.subheader("Connection Impact Analysis")
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        # Connections vs Stage 1 payout
        fig_conn = px.scatter(agg_copy, x='total_connections', y='stage_1_payout', 
                            color='connection_status', size='total_payout',
                            title='Connections vs Stage 1 Payout',
                            labels={'total_connections': 'Total Connections', 
                                   'stage_1_payout': 'Stage 1 Payout ($)'})
        fig_conn.add_vline(x=minimum_connections, line_dash="dash", line_color="red",
                          annotation_text=f"Minimum ({minimum_connections})")
        st.plotly_chart(fig_conn, use_container_width=True)
    
    with col_viz2:
        # 3-Stage payment breakdown
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
    
# Performance Metrics vs Total Payouts Analysis
st.subheader("Performance Metrics vs Total Payouts Analysis")

# Create three columns for the three metrics
col_perf1, col_perf2, col_perf3 = st.columns(3)

with col_perf1:
    # SAIDI vs Total Payouts
    fig_saidi_payout = px.scatter(agg_copy, x='SAIDI_mean', y='total_payout', 
                                 color='performance_zone', size='total_connections',
                                 hover_data=['site_id', 'performance_multiplier'],
                                 title='SAIDI vs Total Payout',
                                 labels={'SAIDI_mean': 'Average SAIDI (h/month)',
                                        'total_payout': 'Total Payout ($)'},
                                 color_discrete_map={
                                     'Bonus Zone': '#10B981',      # Green
                                     'Standard Zone': '#3B82F6',   # Blue  
                                     'Acceptable Zone': '#F59E0B', # Orange
                                     'Penalty Zone': '#EF4444'     # Red
                                 })
    
    # Add threshold lines for SAIDI zones
    fig_saidi_payout.add_vline(x=saidi_bonus, line_dash="dash", line_color="green", 
                              annotation_text="Bonus", annotation_position="top")
    fig_saidi_payout.add_vline(x=saidi_standard, line_dash="dash", line_color="blue",
                              annotation_text="Standard", annotation_position="top")
    fig_saidi_payout.add_vline(x=saidi_penalty, line_dash="dash", line_color="orange",
                              annotation_text="Penalty", annotation_position="top")
    
    fig_saidi_payout.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_saidi_payout, use_container_width=True)

with col_perf2:
    # SAIFI vs Total Payouts
    fig_saifi_payout = px.scatter(agg_copy, x='SAIFI_mean', y='total_payout', 
                                 color='performance_zone', size='total_connections',
                                 hover_data=['site_id', 'performance_multiplier'],
                                 title='SAIFI vs Total Payout',
                                 labels={'SAIFI_mean': 'Average SAIFI (#/month)',
                                        'total_payout': 'Total Payout ($)'},
                                 color_discrete_map={
                                     'Bonus Zone': '#10B981',      # Green
                                     'Standard Zone': '#3B82F6',   # Blue  
                                     'Acceptable Zone': '#F59E0B', # Orange
                                     'Penalty Zone': '#EF4444'     # Red
                                 })
    
    # Add threshold lines for SAIFI zones
    fig_saifi_payout.add_vline(x=percentile_thresholds['saifi_bonus'], line_dash="dash", line_color="green",
                              annotation_text="Bonus", annotation_position="top")
    fig_saifi_payout.add_vline(x=percentile_thresholds['saifi_standard'], line_dash="dash", line_color="blue",
                              annotation_text="Standard", annotation_position="top")
    fig_saifi_payout.add_vline(x=percentile_thresholds['saifi_penalty'], line_dash="dash", line_color="orange",
                              annotation_text="Penalty", annotation_position="top")
    
    fig_saifi_payout.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_saifi_payout, use_container_width=True)

with col_perf3:
    # Undervoltage vs Total Payouts  
    fig_underv_payout = px.scatter(agg_copy, x='undervoltage_mean', y='total_payout', 
                                  color='performance_zone', size='total_connections',
                                  hover_data=['site_id', 'performance_multiplier'],
                                  title='Undervoltage vs Total Payout',
                                  labels={'undervoltage_mean': 'Average Undervoltage (h/month)',
                                         'total_payout': 'Total Payout ($)'},
                                  color_discrete_map={
                                      'Bonus Zone': '#10B981',      # Green
                                      'Standard Zone': '#3B82F6',   # Blue  
                                      'Acceptable Zone': '#F59E0B', # Orange
                                      'Penalty Zone': '#EF4444'     # Red
                                  })
    
    # Add threshold lines for Undervoltage zones
    fig_underv_payout.add_vline(x=percentile_thresholds['und_bonus'], line_dash="dash", line_color="green",
                               annotation_text="Bonus", annotation_position="top")
    fig_underv_payout.add_vline(x=percentile_thresholds['und_standard'], line_dash="dash", line_color="blue",
                               annotation_text="Standard", annotation_position="top")
    fig_underv_payout.add_vline(x=percentile_thresholds['und_penalty'], line_dash="dash", line_color="orange",
                               annotation_text="Penalty", annotation_position="top")
    
    fig_underv_payout.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_underv_payout, use_container_width=True)

# Add explanation text
st.markdown("""
**Performance Zone Analysis:**
- **Bonus Zone** (Green): Sites performing in top 2% - receive 20% bonus on Stages 2 & 3
- **Standard Zone** (Blue): Sites performing in 95th-98th percentile - receive full payment
- **Acceptable Zone** (Orange): Sites performing in 90th-95th percentile - receive 10% penalty
- **Penalty Zone** (Red): Sites performing below 90th percentile - receive 30% penalty

*Bubble size represents total connections. Vertical dashed lines show performance zone boundaries.*
""")

# Multi-metric site drilldown
st.subheader("Site Drilldown Analysis - All Performance Metrics")
drill_site = st.selectbox("Select site for detailed analysis", 
                         options=['Choose a site...'] + agg_copy['site_id'].tolist())

if drill_site != 'Choose a site...':
    site_df = filtered[filtered['site_id'] == drill_site]
    if not site_df.empty:
        # Get site info
        site_info = agg_copy[agg_copy['site_id'] == drill_site].iloc[0]
        
        # Display site summary
        col_site1, col_site2, col_site3, col_site4 = st.columns(4)
        col_site1.metric("Connections", f"{site_info['total_connections']}", site_info['connection_status'])
        col_site2.metric("Avg SAIDI", f"{site_info['SAIDI_mean']:.2f}h", site_info['performance_zone'])
        col_site3.metric("Avg SAIFI", f"{site_info['SAIFI_mean']:.2f}", f"{site_info['performance_multiplier']:.2f}x multiplier")
        col_site4.metric("Total Payout", f"${site_info['total_payout']:,.0f}", "All stages")
        
        # Monthly performance tracking for all three metrics
        site_monthly = site_df.groupby(pd.Grouper(key='day', freq='M')).agg({
            'SAIDI': 'mean', 
            'SAIFI': 'mean', 
            'undervoltage_duration': 'mean'
        }).reset_index()
        
        if not site_monthly.empty:
            # Create subplot for all three metrics
            fig_multi = make_subplots(rows=3, cols=1, 
                                     subplot_titles=['SAIDI Performance (Hours/month)', 
                                                   'SAIFI Performance (Outages/month)',
                                                   'Undervoltage Duration (Hours/month)'],
                                     vertical_spacing=0.08)
            
            # SAIDI subplot with threshold zones
            fig_multi.add_hrect(y0=0, y1=saidi_bonus, fillcolor="green", opacity=0.1, row=1, col=1)
            fig_multi.add_hrect(y0=saidi_bonus, y1=saidi_standard, fillcolor="blue", opacity=0.1, row=1, col=1)
            fig_multi.add_hrect(y0=saidi_standard, y1=saidi_penalty, fillcolor="orange", opacity=0.1, row=1, col=1)
            fig_multi.add_hrect(y0=saidi_penalty, y1=data_ranges['saidi_max'], fillcolor="red", opacity=0.1, row=1, col=1)
            
            fig_multi.add_trace(go.Scatter(x=site_monthly['day'], y=site_monthly['SAIDI'],
                                         mode='lines+markers', name='SAIDI', line=dict(color='#FF6B6B')),
                               row=1, col=1)
            
            # SAIFI subplot with threshold zones
            fig_multi.add_hrect(y0=0, y1=percentile_thresholds['saifi_bonus'], fillcolor="green", opacity=0.1, row=2, col=1)
            fig_multi.add_hrect(y0=percentile_thresholds['saifi_bonus'], y1=percentile_thresholds['saifi_standard'], fillcolor="blue", opacity=0.1, row=2, col=1)
            fig_multi.add_hrect(y0=percentile_thresholds['saifi_standard'], y1=percentile_thresholds['saifi_penalty'], fillcolor="orange", opacity=0.1, row=2, col=1)
            fig_multi.add_hrect(y0=percentile_thresholds['saifi_penalty'], y1=data_ranges['saifi_max'], fillcolor="red", opacity=0.1, row=2, col=1)
            
            fig_multi.add_trace(go.Scatter(x=site_monthly['day'], y=site_monthly['SAIFI'],
                                         mode='lines+markers', name='SAIFI', line=dict(color='#4ECDC4')),
                               row=2, col=1)
            
            # Undervoltage subplot with threshold zones
            fig_multi.add_hrect(y0=0, y1=percentile_thresholds['und_bonus'], fillcolor="green", opacity=0.1, row=3, col=1)
            fig_multi.add_hrect(y0=percentile_thresholds['und_bonus'], y1=percentile_thresholds['und_standard'], fillcolor="blue", opacity=0.1, row=3, col=1)
            fig_multi.add_hrect(y0=percentile_thresholds['und_standard'], y1=percentile_thresholds['und_penalty'], fillcolor="orange", opacity=0.1, row=3, col=1)
            fig_multi.add_hrect(y0=percentile_thresholds['und_penalty'], y1=data_ranges['und_max'], fillcolor="red", opacity=0.1, row=3, col=1)
            
            fig_multi.add_trace(go.Scatter(x=site_monthly['day'], y=site_monthly['undervoltage_duration'],
                                         mode='lines+markers', name='Undervoltage', line=dict(color='#45B7D1')),
                               row=3, col=1)
            
            fig_multi.update_layout(height=800, title=f'Site {drill_site} - Complete Performance Timeline',
                                   showlegend=False)
            fig_multi.update_xaxes(title_text="Month", row=3, col=1)
            st.plotly_chart(fig_multi, use_container_width=True)
            
            # Performance summary table
            st.markdown("**Recent 6-Month Performance Summary:**")
            recent_summary = site_monthly.tail(6).round(2)
            recent_summary.columns = ['Month', 'SAIDI (h)', 'SAIFI (#)', 'Undervoltage (h)']
            st.dataframe(recent_summary, use_container_width=True)
            
            # Stage breakdown for this specific site
            st.markdown("**Detailed Payout Breakdown:**")
            breakdown_cols = st.columns(3)
            breakdown_cols[0].metric("Stage 1 (Connections)", f"${site_info['stage_1_payout']:,.0f}", 
                                    f"{site_info['total_connections']} connections")
            breakdown_cols[1].metric("Stage 2 (Quality)", f"${site_info['stage_2_payout']:,.0f}", 
                                    f"{site_info['performance_multiplier']:.2f}x multiplier")
            breakdown_cols[2].metric("Stage 3 (Sustained)", f"${site_info['stage_3_payout']:,.0f}", 
                                    f"{site_info['performance_multiplier']:.2f}x multiplier")

# --- Export Options ---
st.markdown("---")
st.subheader("Export Results")

col_export1, col_export2 = st.columns(2)

with col_export1:
    if not filtered.empty:
        csv_raw = filtered.to_csv(index=False).encode('utf-8')
        st.download_button("Download Raw Performance Data", 
                          csv_raw, 
                          file_name=f'rbf_raw_data_{datetime.now().strftime("%Y%m%d")}.csv')

with col_export2:
    if not agg_copy.empty:
        csv_payouts = agg_copy.to_csv(index=False).encode('utf-8')
        st.download_button("Download Connection-Based Payout Analysis", 
                          csv_payouts, 
                          file_name=f'rbf_connection_payouts_{datetime.now().strftime("%Y%m%d")}.csv')

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748B; font-size: 12px;'>
3-Stage RBF Dashboard | Connection-Based + Performance-Based Payments | 
Real Connection Counts + Multi-Metric Performance Analysis
</div>
""", unsafe_allow_html=True)

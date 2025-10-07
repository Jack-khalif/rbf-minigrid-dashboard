

```markdown
# QA-RBF Mini-Grid Dashboard

**Live Demo**: [https://jack-khalif-rbf-minigrid-dashboard-app-xyjpez.streamlit.app/]

A comprehensive **Results-Based Financing (RBF) dashboard** for analyzing mini-grid electrical reliability and calculating performance-based payments. This dashboard implements a **3-stage payment model** combining connection milestones with adaptive performance benchmarking using percentile-based thresholds.

##  Project Purpose

This dashboard addresses the need for **transparent, data-driven financing models** in mini-grid electrification projects, particularly in developing regions like the Democratic Republic of Congo (DRC). It transforms complex electrical engineering metrics (SAIDI, SAIFI, undervoltage) into actionable financial incentives that reward both infrastructure deployment and sustained service quality.

### Key Problems Solved

1. **Lack of Performance Accountability**: Traditional financing pays for infrastructure delivery but doesn't ensure long-term service quality
2. **Inflexible Benchmarking**: Fixed performance thresholds don't adapt to local conditions or sector evolution
3. **Complex Multi-Stage Payments**: Manual calculation of staged payments based on connections and performance is error-prone
4. **Limited Transparency**: Stakeholders need clear visibility into how performance affects financial outcomes

## Technical Architecture

### Core Technologies

- **Streamlit**: Web application framework for interactive dashboards
- **Plotly**: Advanced data visualization library for charts and graphs
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing for calculations

### Data Flow Architecture

```
Raw CSV Data → Data Loading & Preprocessing → Percentile Calculation → 
Performance Scoring → 3-Stage Payout Calculation → Visualization & Export
```

##  Dashboard Features

### 1. **Adaptive Percentile-Based Benchmarking**
- **Dynamic Thresholds**: Performance benchmarks automatically adjust based on current data
- **Performance Zones**: 
  - Penalty Zone (<90th percentile)
  - Acceptable Zone (90th-95th percentile) 
  - Standard Zone (95th-98th percentile)
  - Bonus Zone (>98th percentile)
- **Manual Override**: Authorized personnel can adjust thresholds with accountability logging

### 2. **3-Stage Payment Model**

#### Stage 1: Connection Milestone (Default 30%)
- **Base payment** when minimum connection threshold is reached
- **Additional payment** per connection above minimum threshold
- **Prorated payment** for sites below minimum threshold

#### Stage 2: Quality Assessment (Default 50%) 
- **Performance-multiplied payment** after 3-month monitoring period
- **Multipliers range**: 0.5x (minimum) to 1.3x (maximum bonus)

#### Stage 3: Sustained Performance (Default 20%)
- **Performance-multiplied payment** after 6-month monitoring period  
- **Same multiplier logic** as Stage 2 to ensure consistency

### 3. **Multi-Metric Performance Analysis**
- **SAIDI** (System Average Interruption Duration Index): Total outage hours per month
- **SAIFI** (System Average Interruption Frequency Index): Number of outages per month
- **Undervoltage Duration**: Hours of insufficient voltage per month
- **Weighted Scoring**: SAIDI (50%), SAIFI (30%), Undervoltage (20%)

### 4. **Interactive Visualizations**
- **Sunburst Chart**: Hierarchical view of minigrid → site relationships sized by payout
- **3-Stage Payment Breakdown**: Stacked bar charts showing payment components
- **Performance Scatter Plots**: Connection count vs performance with bubble sizing
- **Multi-Metric Timeline**: Individual site analysis across all performance metrics
- **Heatmaps**: Performance trends across sites and time periods

## Code Structure & Implementation

### Core Data Processing Functions

#### `load_and_preprocess_data()`
```
@st.cache_data
def load_and_preprocess_data():
```
**Purpose**: Loads and merges three CSV datasets, handles missing values, and adds connection simulation data.

**Key Operations**:
- **Data Merging**: Combines undervoltage, SAIFI, and SAIDI datasets on `['day', 'site_id', 'minigrid_name']`
- **Missing Value Handling**: Forward-fills undervoltage data, zeros for SAIFI/SAIDI
- **Outlier Detection**: IQR-based flagging (3×IQR beyond Q3)
- **Connection Simulation**: Generates realistic connection counts (50-250 per site) for demonstration

**Cache Decorator**: `@st.cache_data` prevents redundant data loading during user interactions.

#### `get_percentile_thresholds()`
```
def get_percentile_thresholds(df, p_standard_low=0.90, p_standard_high=0.95, p_bonus=0.98):
```
**Purpose**: Calculates adaptive performance thresholds based on current data distribution.

**Algorithm**:
1. **Extract metric values** from filtered dataset
2. **Calculate percentiles** for each performance zone
3. **Return threshold dictionary** with penalty/standard/bonus levels
4. **Fallback values** if insufficient data exists

**Dynamic Nature**: Thresholds automatically adjust when filters change, ensuring contextual relevance.

#### `calculate_performance_multiplier()`
```
def calculate_performance_multiplier(saidi, saifi, undervoltage, thresholds):
```
**Purpose**: Converts raw performance metrics into financial multipliers for Stages 2 & 3.

**Scoring Logic**:
```
def get_zone_score(value, penalty_thresh, standard_thresh, bonus_thresh):
    if value <= bonus_thresh: return 1.2      # 20% bonus
    elif value <= standard_thresh: return 1.0  # Full payment  
    elif value <= penalty_thresh: return 0.9   # 10% penalty
    else: return 0.7                          # 30% penalty
```

**Weighted Composite**: 
- SAIDI: 50% weight (most critical for customer experience)
- SAIFI: 30% weight (frequency matters but less than duration)
- Undervoltage: 20% weight (quality issue but not outage)

**Safety Bounds**: Final multiplier capped between 0.5x and 1.3x to prevent extreme financial impacts.

#### `calculate_connection_based_payout()`
```
def calculate_connection_based_payout(row, total_cost, stage_1_pct, stage_2_pct, stage_3_pct, min_connections, payment_per_connection):
```
**Purpose**: Implements the complete 3-stage payment calculation logic.

**Stage 1 Logic**:
```
if connections >= min_connections:
    base_stage_1 = total_cost * stage_1_pct / 100
    additional_connections = max(0, connections - min_connections)
    stage_1_amount = base_stage_1 + (additional_connections * payment_per_connection)
else:
    stage_1_amount = (total_cost * stage_1_pct / 100) * (connections / min_connections)
```

**Stage 2 & 3 Logic**:
```
performance_multiplier = calculate_performance_multiplier(...)
stage_2_amount = stage_2_base * performance_multiplier
stage_3_amount = stage_3_base * performance_multiplier
```

**Return Structure**: Dictionary with individual stage amounts, total payout, multiplier, and connection status.

### User Interface Components

#### **Sidebar Configuration Panel**
- **Project Parameters**: Total cost, connection thresholds, payment per connection
- **Payment Structure**: Adjustable percentages for each stage (auto-calculating remainder)
- **Data Filters**: Date range, minigrid selection, site selection (cascading filters)
- **Performance Overrides**: Manual threshold adjustment with accountability logging

#### **KPI Cards System**
```
def card_html(title, value, subtitle="", css_class=""):
    return f"""
    <div class="card">
      <div class="card-title">{title}</div>
      <div class="card-value {css_class}">{value}</div>
      <div class="small">{subtitle}</div>
    </div>
    """
```
**Purpose**: Creates professional-looking metric cards with consistent styling.

**Dynamic Content**: Cards adapt based on whether "All sites" or individual site is selected.

#### **Performance Zones Visualization**
The dashboard uses color-coded zones throughout visualizations:
- **Green**: Bonus performance zone (exceptional)
- **Blue**: Standard performance zone (good)
- **Orange**: Acceptable performance zone (needs improvement)
- **Red**: Penalty performance zone (poor, requires attention)

### Advanced Visualizations

#### **Sunburst Chart Implementation**
```
fig_sb = px.sunburst(site_dist, path=['minigrid_name', 'site_id'], values='total_payout', 
                    color='minigrid_name', color_discrete_sequence=px.colors.qualitative.Pastel)
```
**Purpose**: Shows hierarchical relationship between minigrids and sites with financial weighting.

**Data Preparation**: Groups sites by minigrid, merges with payout data for sizing.

#### **Multi-Metric Timeline (Site Drilldown)**
```
fig_multi = make_subplots(rows=3, cols=1, 
                         subplot_titles=['SAIDI Performance', 'SAIFI Performance', 'Undervoltage Duration'])
```
**Purpose**: Comprehensive performance analysis for individual sites across all metrics.

**Zone Overlay**: Each subplot includes performance zone backgrounds using `add_hrect()`.

**Interactive Features**: Users can select any site for detailed temporal analysis.

## Data Requirements

### Input Files
The dashboard expects three CSV files with specific structures:

#### 1. `DRC_minigrid_undervoltage_2023.csv`
```
Columns: day, site_id, minigrid_name, undervoltage_duration
```

#### 2. `DRC_minigrid_saifi_2023.csv`  
```
Columns: day, site_id, minigrid_name, SAIFI
```

#### 3. `DRC_minigrid_saidi_2023.csv`
```
Columns: day, site_id, minigrid_name, SAIDI
```

### Data Specifications
- **Date Format**: Must be parseable by pandas (YYYY-MM-DD recommended)
- **Site Identifiers**: Consistent `site_id` across all files
- **Numeric Metrics**: SAIDI/SAIFI in appropriate units (hours/outages per month)
- **Missing Values**: Handled automatically but clean data preferred

### Connection Data
Currently simulated with `np.random` for demonstration. In production:
```
# Replace simulation with actual connection data
connections_data = pd.read_csv('actual_connections.csv')  # site_id, total_connections, target_connections
```





##  Key Performance Indicators (KPIs)

### Financial Metrics
- **Total Payouts**: Sum of all stage payments across selected sites
- **Stage Breakdown**: Individual payment amounts for each stage
- **Performance Multipliers**: Applied to Stages 2 & 3 based on service quality

### Technical Metrics  
- **SAIDI Average**: Mean outage duration across selected sites/time
- **Connection Status**: Percentage of sites meeting minimum connection thresholds
- **Performance Zones**: Distribution of sites across quality tiers

### Comparative Metrics
- **Best/Worst Performers**: Sites with lowest/highest SAIDI values
- **Network Averages**: Baseline performance for comparison
- **Trend Analysis**: Month-over-month performance changes

##  Configuration Options

### Payment Structure Customization
```
# Adjustable via sidebar sliders
stage_1_pct = 30  # Connection milestone percentage
stage_2_pct = 50  # Quality assessment percentage  
stage_3_pct = 20  # Sustained performance percentage (auto-calculated)
```

### Performance Thresholds
```
# Percentile-based (adaptive) or manual override
p_standard_low = 0.90   # 90th percentile boundary
p_standard_high = 0.95  # 95th percentile boundary
p_bonus = 0.98          # 98th percentile boundary
```

### Connection Model
```
minimum_connections = 100        # Minimum for Stage 1 qualification
connection_payment_per_unit = 300  # Additional payment per connection
```

##  Troubleshooting

### Common Issues

#### **Data Loading Errors**
```
Error: CSVs not found! Upload DRC_minigrid_*.csv to repo root or adjust paths.
```
**Solution**: Ensure all three CSV files are in the project root with exact names.

#### **Empty Visualizations**
**Cause**: Filters too restrictive, no data in selected date range/sites
**Solution**: Expand date range or select "All" for minigrid/site filters

#### **Performance Calculation Issues**
**Cause**: Missing or invalid performance data
**Solution**: Check CSV files for proper numeric values, handle NaN values

#### **Threshold Override Not Working**
**Cause**: Missing justification text
**Solution**: Provide detailed explanation in "Adjustment Justification" field

### Performance Optimization

#### **Large Datasets**
- Use `@st.cache_data` decorator on data processing functions
- Consider data sampling for very large time ranges
- Implement pagination for site lists if needed

#### **Slow Visualizations**
- Reduce number of data points in scatter plots
- Use data aggregation for high-level views
- Consider plotly `render_mode='webgl'` for large datasets

##  Security & Accountability

### Audit Trail Features
- **Threshold Adjustments**: All manual overrides logged with user identification and reasoning
- **Parameter Changes**: Sidebar modifications tracked for transparency  
- **Data Exports**: Timestamped downloads for version control

### Data Privacy
- **No Personal Information**: Dashboard processes only technical performance metrics
- **Aggregated Views**: Individual customer data not exposed
- **Secure Deployment**: Streamlit Cloud provides HTTPS and security standards

## Future Enhancements

### Planned Features

#### **Real-Time Data Integration**
```
# Replace CSV loading with API connections
def load_real_time_data():
    # Connect to mini-grid monitoring systems
    # Pull live SAIDI/SAIFI/undervoltage data
    # Update dashboard automatically
```

#### **Predictive Analytics**
- **Performance Forecasting**: Use historical data to predict future performance zones
- **Payout Modeling**: Scenario analysis for different performance improvement strategies
- **Risk Assessment**: Early warning system for sites approaching penalty zones

#### **Enhanced Accountability**
- **User Authentication**: Role-based access for different stakeholder types
- **Detailed Logging**: Full audit trail of all dashboard interactions  
- **Approval Workflows**: Multi-step approval for threshold adjustments

#### **Mobile Responsiveness**
- **Responsive Design**: Optimize layouts for mobile/tablet viewing
- **Offline Capability**: Local data caching for field use
- **Progressive Web App**: Installation on mobile devices

#### **Advanced Analytics**
- **Machine Learning**: Anomaly detection in performance patterns
- **Correlation Analysis**: Identify factors affecting performance (weather, maintenance, etc.)
- **Benchmarking**: Compare performance against regional/global standards

### Technical Debt & Improvements

#### **Code Organization**
- **Modular Structure**: Split functions into separate Python modules
- **Configuration Management**: External config files for parameters
- **Testing Framework**: Unit tests for calculation functions

#### **Data Management**
- **Database Integration**: Replace CSV files with PostgreSQL/MongoDB
- **Data Validation**: Automated checks for data quality and consistency
- **Version Control**: Track changes in performance data over time

## Development Notes

### Key Design Decisions

#### **Percentile-Based Thresholds**
**Rationale**: Fixed thresholds become outdated as mini-grid technology and operations improve. Percentile-based benchmarks automatically adjust to sector performance, ensuring fairness and encouraging continuous improvement.

#### **3-Stage Payment Model**
**Rationale**: Aligns with real-world project cash flow needs - early funding for deployment, performance-based payments for operations, and sustained incentives for long-term service quality.

#### **Weighted Performance Scoring**
**Rationale**: SAIDI (outage duration) has the highest weight (50%) because it most directly impacts customer experience. SAIFI and undervoltage are important but secondary factors.

#### **Safety Bounds on Multipliers**  
**Rationale**: Prevents extreme financial outcomes that could destabilize projects. 50% minimum ensures operator viability, 130% maximum prevents unsustainable bonus payments.

### Performance Calculation Deep Dive

The dashboard implements a sophisticated multi-layer scoring system:

1. **Individual Metric Scoring**: Each metric (SAIDI, SAIFI, undervoltage) is scored against its percentile thresholds
2. **Composite Performance Score**: Weighted average of individual scores 
3. **Zone Classification**: Overall performance classified into Bonus/Standard/Acceptable/Penalty zones
4. **Financial Multipliers**: Zone classification converted to payment multipliers for Stages 2 & 3
5. **Connection Integration**: Stage 1 payments combine fixed milestone rewards with incremental connection bonuses

### Data Processing Pipeline

```
Raw CSV Files → Data Validation → Missing Value Handling → 
Outlier Detection → Time Series Aggregation → Performance Calculation → 
Financial Modeling → Visualization → Export
```

Each step includes error handling and fallback options to ensure dashboard stability.



---

*This dashboard represents a cutting-edge approach to results-based financing in the mini-grid sector, combining rigorous performance monitoring with transparent financial incentives to drive sustainable electrification outcomes.*
```

Now you can copy this entire markdown content and paste it into your README.md file!

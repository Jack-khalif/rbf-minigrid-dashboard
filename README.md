##   RBF Mini-Grid Dashboard  
### Results-Based Financing for Mini-Grid Excellence  

Live Demo: https://jack-khalif-rbf-minigrid-dashboard-app-xyjpez.streamlit.app  
---
##  Project Overview

The **RBF Mini-Grid Dashboard** revolutionizes how mini-grid projects are financed and monitored in developing regions. By implementing a sophisticated **3-Stage Payment Model** combined with **adaptive percentile-based benchmarking**, this dashboard ensures that funding flows are directly tied to both infrastructure deployment and sustained service quality.

### Key Problems Solved

| Challenge | Solution |
|-----------|----------|
|  **Lack of Performance Accountability** | 3-stage payments tied to connections + service quality |
|  **Inflexible Benchmarking** | Adaptive percentile thresholds that evolve with data |
| **Complex Payment Calculations** | Automated multi-stage payout computation |
|  **Limited Stakeholder Transparency** | Interactive visualizations for all stakeholders |
|  **One-size-fits-all Approaches** | Context-sensitive performance zones |

---
### **Proposed model**
The RBF Mini-Grid model is a data-driven platform that connects funding to real mini-grid performance. It ensures that financial disbursements reflect service quality and reliability through a structured three-stage payment model.  

Stage 1 (30%): Based on connection milestones  
Stage 2 (50%): Evaluated after a three-month quality assessment  
Stage 3 (20%): Verified after six months of sustained performance  

The system uses adaptive percentile-based benchmarking to rank sites dynamically:  
Bonus Zone – Above 95th percentile  
Standard Zone – Between 90th and 95th percentile  
Penalty Zone – Below 90th percentile  

Performance is measured using three key metrics:  
SAIDI – System Average Interruption Duration Index (50%)  
SAIFI – System Average Interruption Frequency Index (30%)  
Undervoltage – Power quality monitoring (20%)  

The dashboard automatically computes payments, visualizes results, and allows users to explore site performance, payout breakdowns, and historical trends. Data is processed from CSV files using Pandas and NumPy, and visualized through Plotly inside a Streamlit web interface.  

## **Dashboard Screenshots**

###  **Main Dashboard Overview**

![Main Dashboard](images/maindashboard.png)

### **Performance Analytics**

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px;">
  <img src="images/saifi.png" alt="SAIFI Performance" width="45%">
  <img src="images/saidi.png" alt="SAIDI Performance" width="45%">
  <img src="images/undervoltage.png" alt="Undervoltage Performance" width="45%">


</div>

![Connection Completion](images/connection_completion.png)

### **Sunburst Visualization**



![Sunburst Chart](images/sunburst_chart.png)

### **Site Drilldown Analysis**


![Site Analysis](images/site-drilldown.png)

###  **Performance vs Payouts**


![Performance Payouts](images/perfomance_payouts.png)

---

Technology Stack  

Frontend: Streamlit – Interactive web app interface  
Visualization: Plotly – Data visualization and charts  
Backend: Python – Logic and computation  
Data Processing: Pandas – Data manipulation  
Math Engine: NumPy – Numerical operations  

---

Performance Metrics  

Financial  
- Total project payouts across all stages  
- Stage-wise payment breakdown  
- Comparison between connection-based and performance-based allocations  

Technical  
- SAIDI averages and reliability trends  
- Distribution across performance zones  
- Connection completion rates  

Comparative  
- Best and worst performing sites  
- Network-wide performance trends  
- Site-to-site benchmarking  

---

Data Requirements  

Required Files:  
DRC_minigrid_undervoltage_2023.csv – day, site_id, minigrid_name, undervoltage_duration  
DRC_minigrid_saifi_2023.csv – day, site_id, minigrid_name, SAIFI  
DRC_minigrid_saidi_2023.csv – day, site_id, minigrid_name, SAIDI  

File Specifications:  
- Date format: YYYY-MM-DD  
- Consistent site identifiers across files  
- Numeric values in standard units  
- Complete time coverage preferred  

---

Configuration Examples  

Payment Structure  
stage_1_pct = 30  
stage_2_pct = 50  
stage_3_pct = 20  

Performance Thresholds  
p_standard_low = 0.90  
p_standard_high = 0.95  

Connection Model  
connection_completion_pct = 70  
stage_1_cap = 0.30  

---

Future Roadmap  

Real-Time Integration  
- Live data streams from monitoring systems  
- Automated updates via API  
- Continuous data ingestion  

Advanced Analytics  
- Performance forecasting using machine learning  
- Anomaly detection in reliability patterns  
- Predictive payout modeling  

Accessibility  
- Mobile-responsive design  
- Offline field-use support   



````markdown
# **QA-RBF Mini-Grid Dashboard**
### *Results-Based Financing for Mini-Grid Excellence*

[![Live Demo](https://img.shields.io/badge/Live_Demo-Available-success?style=for-the-badge)](https://jack-khalif-rbf-minigrid-dashboard-app-xyjpez.streamlit.app/)
[![Streamlit](https://img.shields.io/badge/Built_with-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**[Live Dashboard](https://jack-khalif-rbf-minigrid-dashboard-app-xyjpez.streamlit.app/)** •
**[Documentation](#features)** •
**[Quick Start](#installation--setup)** •
**[Contributing](#contributing)**

---

### *Transforming Mini-Grid Performance Data into Smart Financial Incentives*

---

## **Overview**

The **QA-RBF Mini-Grid Dashboard** redefines how mini-grid projects are financed and evaluated in developing regions.  
It introduces a **3-Stage Payment Model** linked to measurable reliability and quality metrics, ensuring that **funding aligns with actual performance** and not just infrastructure rollout.

### **Key Problems Solved**

| Challenge | Solution |
|------------|-----------|
| Lack of performance accountability | 3-stage payments linked to connections and quality |
| Inflexible benchmarking | Adaptive percentile thresholds that evolve with data |
| Complex calculations | Automated payout computation across all stages |
| Limited transparency | Interactive visual insights for all stakeholders |
| One-size-fits-all models | Context-aware performance zones |

---

## **Features**

### **Three-Stage Payment Model**
- **Stage 1 (30%)** – Payments based on connection milestones  
- **Stage 2 (50%)** – Quality assessment after 3 months  
- **Stage 3 (20%)** – Sustained performance verification after 6 months  

### **Adaptive Performance Benchmarking**
- Dynamic thresholds updated with sector-wide data  
- Performance zones: Bonus (>95th percentile), Standard (90-95th), Penalty (<90th)  
- Manual overrides for justified exceptions  

### **Multi-Metric Performance Analysis**
- **SAIDI** – System Average Interruption Duration Index (50%)  
- **SAIFI** – System Average Interruption Frequency Index (30%)  
- **Undervoltage** – Power quality metric (20%)  

---

## **Dashboard Previews**

**Main Dashboard Overview**  
![Main Dashboard](images/main-dashboard.png)

**Performance Analytics**  
![Performance Analytics](images/performance-analytics.png)

**Sunburst Visualization**  
![Sunburst Chart](images/sunburst-chart.png)

**Site Drilldown Analysis**  
![Site Analysis](images/site-drilldown.png)

**Performance vs Payouts**  
![Performance Payouts](images/performance-payouts.png)

---

## **System Architecture**

```mermaid
graph TD
    A[CSV Data Files] --> B[Data Preprocessing]
    B --> C[Percentile Calculation]
    C --> D[Performance Scoring]
    D --> E[3-Stage Payout Calculation]
    E --> F[Interactive Visualizations]
    F --> G[Export & Reporting]
    H[User Controls] --> C
    H --> D
    H --> E
````

---

## **Technology Stack**

| Layer           | Tool      | Purpose                       |
| --------------- | --------- | ----------------------------- |
| Frontend        | Streamlit | Interactive web app interface |
| Visualization   | Plotly    | Data visualization & charts   |
| Backend         | Python    | Logic & computation           |
| Data Processing | Pandas    | Data manipulation             |
| Math Engine     | NumPy     | Numerical operations          |

---

## **Performance Metrics**

### **Financial**

* Total project payouts (all stages)
* Stage-wise payment breakdown
* Performance vs connection-based allocations

### **Technical**

* SAIDI averages and reliability trends
* Performance zone distributions
* Connection completion rates

### **Comparative**

* Best/worst sites by reliability
* Network-wide performance trends
* Site-to-site benchmarking

---

## **Data Requirements**

| File                                 | Required Columns                                   | Description           |
| ------------------------------------ | -------------------------------------------------- | --------------------- |
| `DRC_minigrid_undervoltage_2023.csv` | day, site_id, minigrid_name, undervoltage_duration | Power quality data    |
| `DRC_minigrid_saifi_2023.csv`        | day, site_id, minigrid_name, SAIFI                 | Outage frequency data |
| `DRC_minigrid_saidi_2023.csv`        | day, site_id, minigrid_name, SAIDI                 | Outage duration data  |

**Specifications**

* Date format – `YYYY-MM-DD`
* Consistent site identifiers
* Numeric metrics in standard units
* Complete temporal coverage preferred

---

## **Configuration Examples**

**Payment Structure**

```python
stage_1_pct = 30
stage_2_pct = 50
stage_3_pct = 20
```

**Performance Thresholds**

```python
p_standard_low = 0.90
p_standard_high = 0.95
```

**Connection Model**

```python
connection_completion_pct = 70
stage_1_cap = 0.30
```

---

## **Future Roadmap**

### **Real-Time Integration**

* [ ] Live data streams from monitoring systems
* [ ] Automated updates via API
* [ ] Continuous data ingestion

### **Advanced Analytics**

* [ ] Performance forecasting using ML
* [ ] Anomaly detection
* [ ] Predictive payout modeling

### **Accessibility**

* [ ] Mobile-responsive UI
* [ ] Offline field-use mode
* [ ] Progressive Web App (PWA) features

### **Enterprise**

* [ ] Role-based access control
* [ ] Approval workflows
* [ ] Enhanced audit trail

---

### **Quick Links**

* [Live Dashboard](https://jack-khalif-rbf-minigrid-dashboard-app-xyjpez.streamlit.app/)
* [License (MIT)](LICENSE)
* [Contribute](#contributing)
* [Report Issues](../../issues)

---



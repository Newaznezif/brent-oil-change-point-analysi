# Project Plan and Analysis Workflow
# Brent Oil Price Change Point Analysis - Project Plan

## Analysis Workflow

### Phase 1: Data Preparation & Exploration (Feb 4-5)
1. **Data Loading & Cleaning**
   - Load historical Brent oil prices (May 20, 1987 - Sep 30, 2022)
   - Convert 'Date' column to datetime format
   - Handle missing values and outliers
   - Validate data consistency

2. **Event Data Compilation**
   - Research and compile 10-15 key geopolitical/economic events
   - Create structured CSV with dates, descriptions, and impact levels
   - Align event dates with price data

3. **Exploratory Data Analysis**
   - Plot raw price series over time
   - Calculate and visualize log returns
   - Analyze trends, seasonality, and volatility
   - Test for stationarity (ADF test)

### Phase 2: Modeling & Analysis (Feb 6-7)
4. **Bayesian Change Point Model**
   - Define prior distributions for change points
   - Implement switch-point model using PyMC
   - Run MCMC sampling (NUTS/Hamiltonian Monte Carlo)
   - Check convergence (R-hat statistics, trace plots)

5. **Change Point Detection**
   - Identify structural breaks in price series
   - Quantify before/after parameter changes
   - Calculate confidence intervals for change points

6. **Event Association**
   - Match detected change points with historical events
   - Quantify impact sizes (% change, absolute change)
   - Formulate causal hypotheses

### Phase 3: Dashboard Development (Feb 8-9)
7. **Backend Development**
   - Create Flask API endpoints
   - Serve processed data and model results
   - Implement data querying functionality

8. **Frontend Development**
   - Build React dashboard with interactive charts
   - Implement event highlighting
   - Add filtering and date selection
   - Ensure responsive design

### Phase 4: Reporting & Communication (Feb 10)
9. **Insight Generation**
   - Summarize key findings
   - Create visualizations for stakeholders
   - Prepare impact quantification statements

10. **Documentation**
    - Write comprehensive report
    - Document assumptions and limitations
    - Prepare presentation materials

## Success Metrics
- Identify at least 5-7 significant change points with high confidence
- Quantify price impacts with 95% credible intervals
- Successfully associate 70%+ of major change points with known events
- Create functional dashboard with 3+ interactive features

## Dependencies & Tools
- Python: PyMC, ArviZ, pandas, numpy, matplotlib
- Statistics: Bayesian inference, MCMC methods
- Dashboard: Flask, React, Recharts
- Version Control: Git/GitHub
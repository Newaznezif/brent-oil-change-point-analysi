# Assumptions and Limitations

# Assumptions and Limitations

## Key Assumptions

### Statistical Assumptions
1. **Change Point Model Assumptions:**
   - Structural breaks manifest as shifts in distribution parameters
   - Changes are instantaneous (not gradual transitions)
   - Time between significant changes is reasonably spaced

2. **Data Assumptions:**
   - Reported prices are accurate and consistent
   - Daily frequency captures relevant price movements
   - No systematic measurement errors in the dataset

3. **Modeling Assumptions:**
   - Normal distribution adequately describes price variations
   - Log returns are stationary for modeling purposes
   - Single change point model captures major structural breaks

### Causal Inference Assumptions
1. **Temporal Proximity:**
   - Change points occurring near events are potentially caused by them
   - Maximum lag of 30 days between event and detectable change

2. **Exclusion of Confounders:**
   - Major events dominate other market factors during crisis periods
   - Geopolitical events have immediate market impacts

## Important Limitations

### Statistical Limitations
1. **Correlation ≠ Causation:**
   - Detecting a change point near an event demonstrates statistical association
   - Does NOT prove causal relationship without controlled experiments
   - Multiple simultaneous events may confound attribution

2. **Model Limitations:**
   - Bayesian change point models detect parameter shifts, not event impacts directly
   - Model assumes constant volatility within regimes (may not hold)
   - Single change point per analysis window limitation

### Data Limitations
1. **Data Scope:**
   - Only Brent crude prices analyzed (not WTI, Dubai, etc.)
   - Daily frequency may miss intraday shocks
   - No volume or open interest data included

2. **Event Data:**
   - Event selection based on historical significance (subjective)
   - Event dates may not match exact market reaction timing
   - Impact quantification is approximate

### Methodological Limitations
1. **Simplified Model:**
   - Focuses only on mean shifts, ignoring variance changes
   - Does not model gradual transitions or anticipatory movements
   - Assumes independence between observations within regimes

2. **Missing Factors:**
   - Does not account for:
     - USD exchange rate fluctuations
     - Global economic growth rates
     - Technological changes in extraction
     - Renewable energy adoption
     - Storage capacity and inventory levels

## Critical Distinction: Statistical vs. Causal

### Statistical Correlation
- We detect: "Price change occurred near Event X"
- Statistical evidence: Posterior probability of change point
- Quantification: Magnitude of parameter shift

### Causal Impact
- We infer: "Event X caused price change"
- Requires: Counterfactual scenario (what would have happened without event)
- Evidence needed: Exclusion of alternative explanations

## Mitigation Strategies

1. **Multiple Model Validation:**
   - Compare results across different change point detection methods
   - Use rolling window analysis to check robustness

2. **Sensitivity Analysis:**
   - Test different prior distributions
   - Vary model hyperparameters
   - Analyze subsets of data

3. **Transparent Reporting:**
   - Clearly state all assumptions
   - Report confidence/credible intervals
   - Acknowledge alternative explanations

## Recommendations for Stakeholders

1. **Use findings as:**
   - Evidence for further investigation
   - Input for risk assessment models
   - Historical context for decision-making

2. **Avoid using as:**
   - Definitive causal proof
   - Sole basis for major investments
   - Predictor of future events
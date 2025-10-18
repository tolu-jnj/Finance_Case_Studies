---
applyTo: '**'
---

# Financial Forecasting Interview Assistant

I'll help you systematically approach two forecasting case studies. This document is a structured system prompt and a suggested project plan aligned with the Data Theory methodology.

## 🎯 Project overview & approach

You're tackling two time-series forecasting problems that will showcase end-to-end data science capabilities. The plan below walks through governance, data generation, standardization, feature engineering, modeling, and deployment.

## 📋 Phase 1 — Governance & problem scoping

### Case study 1 — Country-product monthly forecasting

Key questions to define
- Business context: what drives country–product combinations? Seasonality? Economic indicators?
- Success metrics: RMSE, MAPE, or a business-relevant metric (for example: forecast accuracy within X%).

Constraints
- How many country–product combinations exist?
- Are there missing combinations or sparse data?
- Should we forecast all combinations or prioritize high-value ones?

Objective
- Forecast the target variable for the year 2020 on Monthly basis for all country-product combinations

### Case study 2 — Multi-country quarterly forecasting (36 time series)

Key questions
- Y-variable nature: are Y1–Y36 related (e.g., different products in the same country)?
- Cross-series dependencies: can we leverage correlations between series?
- Hierarchy: is there a natural grouping (Country A: Y1–Y12, Country B: Y13–Y24, etc.)?

Objective
- Forecast all time series (CountryA_Y1 to CountryC_Y36) for the year 2020 on Quarterly basis

Governance tasks
- Define data quality checks (missingness, outliers, frequency consistency).
- Establish model performance thresholds.
- Document assumptions and limitations.
- Create a project requirements document.

## 📊 Phase 2 — Data generation & collection

Follow the 2-Generation principles.

For both case studies — key activities
```python
# Key activities
1. Read functions: ingest CSV data
2. Identifier functions: validate data integrity
	- Confirm date ranges (2011–2019 vs 2010–2019)
	- Check for missing months/quarters
	- Identify data types and anomalies
3. Metadata documentation:
	- Data source details
	- Feature descriptions
	- Known data quality issues
```

Suggested code structure (follow repo conventions)

project_name/
- 1_Governance/
  - README.md (project requirements)
  - data_quality_checks.py
- 2_Generation/
  - README.md (data sources documentation)
  - read_data.py
  - validate_data.py
- 3_Collection_Standardization/
- 4_Aggregation/
- 5_Analysis/
- 6_Application/

## 🔧 Phase 3 — Collection & standardization

Key tasks
- Refactor: convert dates to proper datetime objects; ensure consistent frequency.
- Standardize:
  - Harmonize country/product naming conventions.
  - Handle quarterly vs monthly frequencies.
  - Create a uniform feature schema.
- Filter/select: identify relevant country–product combinations.

Technical decisions to document
- Should you aggregate monthly to quarterly for Case 1 comparison?
- How to handle series with different start dates?
- Outlier treatment strategy?

## 📈 Phase 4 — Aggregation (feature engineering)

Demonstrate advanced capabilities

Time-series features
- Lag features (1, 3, 6, 12 months).
- Rolling statistics (mean, std, min, max).
- Seasonal decomposition (trend, seasonality, residuals).
- Fourier features for cyclical patterns.
- Date features (month, quarter, year, is_holiday).

Domain-specific features
- Country-level aggregations (total across products).
- Product-level aggregations (total across countries).
- Hierarchical features (country → product hierarchy).

For Case 2 (multivariate)
- Cross-series features: correlations between Y variables.
- Dimensionality reduction: PCA on related series.
- External regressors (if justifiable or available).

## 🤖 Phase 5 — Analysis (modeling strategy)

Show breadth and depth

Baseline models
- Naive forecasts (last value, seasonal naive).
- Moving averages.
- Exponential smoothing (Holt–Winters).

Statistical models
- ARIMA/SARIMA — include parameter tuning.
- Prophet — capture seasonality and holidays.
- ETS — error, trend, seasonality models.

Machine learning
- XGBoost / LightGBM — with engineered features.
- Random Forest — for feature importance.
- Linear models with regularization (Ridge/Lasso).

Deep learning (optional)
- LSTM / GRU — for long-term dependencies.
- Temporal Fusion Transformer (TFT) — state-of-the-art.
- N-BEATS — pure time-series architecture.

Ensemble approach
- Combine multiple models (weighted average, stacking).
- Use different models for different series based on characteristics.

Case 2 specific
- Global models (train on all 36 series together).
- Hierarchical forecasting (reconciliation techniques).
- VAR / VARMAX if cross-series dependencies exist.

## 🚀 Phase 6 — Application (deployment & presentation)

### Streamlit application structure
```python
# app.py structure
1. Homepage
	- Project overview
	- Methodology summary
	- Key insights
2. Data Explorer tab
	- Interactive EDA visualizations
	- Data quality reports
	- Time-series plots by country/product
3. Forecasting tab
	- Model selection dropdown
	- Country/product filter
	- Generate forecast button
	- Visualization of historical + forecast
	- Confidence intervals
4. Model Performance tab
	- Metrics comparison table
	- Error distribution plots
	- Feature importance (if applicable)
5. AI Assistance documentation
	- How you used AI at each phase
	- Code generation examples
	- Problem-solving approaches
```

#### Docker setup (example Dockerfile)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 📊 Presentation slide structure

Suggested flow (15–20 slides)
- Title & executive summary
- Problem statement (Cases 1 & 2 overview)
- Data understanding (EDA highlights, challenges identified)
- Methodology overview (Data Theory 6-phase framework)
- Feature engineering (key innovations)
- Modeling approach (justification for chosen methods)
- Model performance (Case 1 results)
- Model performance (Case 2 results)
- Key insights & findings
- Technical architecture (repo structure, Streamlit, Docker)
- AI assistance strategy (how you leveraged GenAI)
- Challenges & solutions
- Production considerations (monitoring, retraining)
- Future enhancements
- Demo transition (live application walkthrough)

## 🤖 AI assistance documentation strategy

Document at each phase
- Problem decomposition: "Asked AI to help structure multi-series forecasting problem."
- Code generation: "Used AI for boilerplate SARIMA implementation, then customized."
- Debugging: "AI helped identify pandas datetime indexing issue."
- Optimization: "Requested AI suggestions for hyperparameter tuning approach."
- Documentation: "AI assisted in generating docstrings and README content."

Create a separate markdown file for the AI collaboration log, for example `ai_collaboration_log.md`:

```markdown
# AI Collaboration Log

## Phase: Data Exploration
- **Prompt**: "Generate EDA code for 36 time series with missing values"
- **AI Output**: [code snippet]
- **My Modifications**: [your enhancements]
- **Reasoning**: [why you kept/changed suggestions]
```

## 🎯 Immediate next steps

I'll help you with:
- First: load and explore your CSV data.
- Share the data structure/sample and I'll generate EDA code.
- Define business questions and success metrics (what accuracy is "good enough").
- Identify critical countries/products to prioritize.
- Choose model families to prioritize based on data characteristics, computational constraints, and interpretability requirements.
- Create the repository structure using the Data Theory conventions.

How would you like to proceed? Share your data (columns, row count, sample) and I will build the first phase: data validation and EDA following these best practices.

	- Model selection dropdown
	- Country/product filter
	- Generate forecast button
	- Visualization of historical + forecast
	- Confidence intervals
4. Model Performance tab
	- Metrics comparison table
	- Error distribution plots
	- Feature importance (if applicable)
5. AI Assistance documentation
	- How you used AI at each phase
	- Code generation examples
	- Problem-solving approaches
```

#### Docker setup (example Dockerfile)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 📊 Presentation slide structure

Suggested flow (15–20 slides)
- Title & executive summary
- Problem statement (Cases 1 & 2 overview)
- Data understanding (EDA highlights, challenges identified)
- Methodology overview (Data Theory 6-phase framework)
- Feature engineering (key innovations)
- Modeling approach (justification for chosen methods)
- Model performance (Case 1 results)
- Model performance (Case 2 results)
- Key insights & findings
- Technical architecture (repo structure, Streamlit, Docker)
- AI assistance strategy (how you leveraged GenAI)
- Challenges & solutions
- Production considerations (monitoring, retraining)
- Future enhancements
- Demo transition (live application walkthrough)

## 🤖 AI assistance documentation strategy

Document at each phase
- Problem decomposition: "Asked AI to help structure multi-series forecasting problem."
- Code generation: "Used AI for boilerplate SARIMA implementation, then customized."
- Debugging: "AI helped identify pandas datetime indexing issue."
- Optimization: "Requested AI suggestions for hyperparameter tuning approach."
- Documentation: "AI assisted in generating docstrings and README content."

Create a separate markdown file for the AI collaboration log, for example `ai_collaboration_log.md`:

```markdown
# AI Collaboration Log

## Phase: Data Exploration
- **Prompt**: "Generate EDA code for 36 time series with missing values"
- **AI Output**: [code snippet]
- **My Modifications**: [your enhancements]
- **Reasoning**: [why you kept/changed suggestions]
```

## 🎯 Immediate next steps

I'll help you with:
- First: load and explore your CSV data.
- Share the data structure/sample and I'll generate EDA code.
- Define business questions and success metrics (what accuracy is "good enough"?).
- Identify critical countries/products to prioritize.
- Choose model families to prioritize based on data characteristics, computational constraints, and interpretability requirements.
- Create the repository structure using the Data Theory conventions.

How would you like to proceed? Share your data (columns, row count, sample) and I will build the first phase: data validation and EDA following these best practices.

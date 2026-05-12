# Early-Life Production Forecasting for Unconventional Wells

**Tennessee Technological University — CSC-4615**

Logan McDavid · Matt Hazelwood · Alex Lujan · Kyle Monday · Taylor Turner

*Clients: James Cassanelli · Nikolaos Mitsakos*

*Faculty Advisors: Dr. Jesse Roberts · Dr. William Eberle · Dr. Chester T. Little*

---

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Objective](#objective)
4. [Pipeline Overview](#pipeline-overview)
5. [Phase 01 — Preprocessing & Cleaning](#phase-01--preprocessing--cleaning)
6. [Phase 02 — Decline Curve Analysis Baseline](#phase-02--decline-curve-analysis-baseline)
7. [Phase 03 — LSTM Sequence Model Training](#phase-03--lstm-sequence-model-training)
8. [Phase 04 — LSTM Evaluation](#phase-04--lstm-evaluation)
9. [Phase 05 — XGBoost Training & Evaluation](#phase-05--xgboost-training--evaluation)
10. [Phase 06 — DCA Performance Analysis](#phase-06--dca-performance-analysis)
11. [Results](#results)
12. [Discussion](#discussion)
13. [Tech Stack](#tech-stack)
14. [References](#references)

---

## Introduction

Decline Curve Analysis (DCA) is the industry standard for oil production forecasting and performs well once a well has 12–18 months of production history and enters stable decline. For unconventional horizontal wells, however, operators must allocate capital for new wells far earlier — often with just an early time threshold of production data — before the decline trend is established.

DCA's early time fits are unreliable in these conditions. The Arps Hyperbolic model is defined from the point of peak production, and when it is calibrated to only an early time threshold, the resulting parameter estimates for initial rate, curve shape, and decline coefficient are highly sensitive to noise. This produces large forecast errors that propagate directly into investment decisions: a well that is projected to produce twice what it actually delivers can drive operators to drill adjacent wells, commit capital to infrastructure, and take on debt — all against a forecast that was structurally unreliable from the start.

---

## Problem Statement

Develop a machine learning model that forecasts monthly oil production out five years using only an early time threshold of a well's production history, and demonstrate that it materially outperforms the DCA early time baseline on independent, unseen wells.

---

## Objective

This project addresses that gap by building a full forecasting pipeline on the Wyoming Oil & Gas Commission's public horizontal wells dataset. The pipeline:

- Cleans and standardizes raw government production records into analysis-ready time series
- Fits Arps Hyperbolic Decline Curves to every well to establish a physics-based baseline
- Trains both an LSTM sequence model and an XGBoost gradient-boosted tree model to forecast five years of monthly production from the same early time window given to DCA
- Evaluates all models on a strictly held-out 20% test cohort of wells never seen during training
- Produces a complete audit trail of per-well metrics, visualizations, and diagnostic plots

---

## Pipeline Overview

The project is organized as a six-phase sequential pipeline. Each phase produces artifact files consumed by the next, and every phase is independently executable.

```
Raw Wyoming Production CSV
          │
          ▼
┌─────────────────────────────────────┐
│  Phase 01: Preprocessing & Cleaning │  Wide → Long format, filter, clean
└─────────────────────────────────────┘
          │
          ▼  phase1_cleaned.csv
┌─────────────────────────────────────┐
│  Phase 02: DCA Baseline             │  Arps curve fitting, feature engineering,
│                                     │  train/test split
└─────────────────────────────────────┘
          │
          ├──► phase2_train_baseline.csv
          └──► phase2_test_baseline.csv
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
┌──────────────────┐  ┌──────────────────────┐
│ Phase 03: LSTM   │  │ Phase 05: XGBoost     │
│ Training         │  │ Training & Evaluation │
└──────────────────┘  └──────────────────────┘
          │
          ▼
┌──────────────────┐
│ Phase 04: LSTM   │
│ Evaluation       │
└──────────────────┘
          
┌──────────────────────────────────────┐
│ Phase 06: DCA Performance Analysis   │  Standalone DCA diagnostic
└──────────────────────────────────────┘
```

---

## Phase 01 — Preprocessing & Cleaning

**Script:** `step_01_reshape_filter_clean.py`

**Input:** Raw Wyoming state production CSV (wide format, one row per well-year)
**Output:** `data/artifacts/world_drafts/phase1_cleaned.csv`

The Wyoming Oil & Gas Commission publishes production records in a wide format where each row represents a single well in a single calendar year, and monthly oil, gas, water, and days-on figures are stored as separate columns. Before any modeling can take place, this structure must be resolved into a chronological time series. Phase 01 is the pipeline's entry point and addresses three structural problems in the raw data.

**Reshaping** — The wide format is melted into long format by iterating over all 12 calendar months, extracting the four production columns for each, and concatenating the result. Every row in the output represents one well's production for one calendar month. An `Index_Feature` column is synthesized from the year and month fields and cast to `datetime64`.

**Filtering** — Only active horizontal wells whose first production month falls on or after January 1, 2015 are retained. This cutoff enforces a minimum level of data quality, modern multi-stage fracture completion technology, and sufficient temporal depth for the 60-month training windows required downstream. Months where `Active_Duration = 0` (shut-in periods) are excluded throughout, including when computing first-production start dates.

**Ramp-Up Trimming** — Horizontal wells exhibit a brief production ramp-up before reaching peak flush production. Including pre-peak data would force the decline curve optimizer to reconcile incompatible behaviors, corrupting the `Param_A` parameter. For each well, the peak production month is identified, all prior months are removed, and the cumulative metric produced during the ramp-up is preserved as `Pre_Peak_Metric` for accurate cumulative accounting in Phase 02.

**Outlier Removal** — A rolling Median Absolute Deviation (MAD) filter is applied to each well's production time series. Data points whose absolute deviation from the rolling median exceeds a configurable multiple of the MAD are replaced with `NaN` and linearly interpolated. MAD is used in place of Z-score because production distributions are heavily right-skewed and contain legitimate step-changes that would inflate standard deviation, making Z-score thresholds unreliable. The early time threshold after peak is protected from flagging to prevent the natural early decline from being classified as anomalous.

**Metadata Enrichment** — External static well characteristics (AllComps) are joined to the cleaned time series on the `Entity_ID` well identifier.

---

## Phase 02 — Decline Curve Analysis Baseline

**Script:** `step_01_fit_dca.py`

**Input:** `phase1_cleaned.csv`
**Outputs:** `dca_params.csv`, `phase2_dca_baseline.csv`, `phase2_train_baseline.csv`, `phase2_test_baseline.csv`

Phase 02 is the analytical backbone of the pipeline. It fits the Arps Hyperbolic Decline Curve model independently to every well using two strategies, generates full-timeline predictions and residuals, augments the dataset with engineered features, and produces the train/test split consumed by all downstream ML phases.

**The Arps Hyperbolic Model** characterizes production rate over time using three parameters:

- `Param_A` — Initial operational metric at peak
- `Param_B` — Curve shape exponent controlling curve shape 
- `Param_C` — Nominal decline rate parameter

```
q(t) = Param_A / (1 + Param_B · Param_C · t)^(1/Param_B)
```

**Two Fitting Strategies** are applied to every well:

- **Full Fit** — Uses the complete production history. Represents the best achievable DCA accuracy given full information.
- **Early Time Fit** — Uses only the early time threshold of production. Simulates the real-world scenario where a forecaster must commit to a five-year EUR prediction before the decline trend is established. This is the baseline against which all ML models are benchmarked.

The gap in error between these two fits quantifies the information value of additional production history and sets an upper bound on how much the ML models can improve over early-life DCA.

**Fitting bounds** are enforced to prevent physically implausible parameter estimates. `Param_B` is constrained to `[0.7, 1.3]` reflecting industry norms for Wyoming horizontal tight-oil wells. `Param_C` is bounded at `0.5` per month to prevent step-decline fits, and given a stricter lower floor for wells with fewer than 12 months to prevent the optimizer from collapsing toward a flat-line (near-zero decline) forecast on short, noisy series.

**Feature Augmentation** computes early time production statistics (`Series_Feature_1`–`Series_Feature_N`, `Rolling_Stat_1`, `Rolling_Stat_2`, `Rolling_Stat_3`) and time polynomial features (`Time_Trans_1`, `Time_Trans_2`, `Time_Trans_3`, `Time_Trans_4`) that are embedded into the output CSVs. Computing these here, rather than in each downstream phase, guarantees consistency and prevents implementation divergence.

**Train/Test Split** is performed at the well level — never at the row level — to prevent data leakage. An 80/20 random split assigns entire wells to either the training cohort or the test cohort. Placing early and late production months from the same well on both sides of the split would constitute direct leakage, producing optimistically biased metrics.

---

## Phase 03 — LSTM Sequence Model Training

**Script:** `step_01_train_lstm.py`

**Input:** `phase2_train_baseline.csv`
**Outputs:** `lstm_error_corrector.keras`, `seq_scaler.pkl`, `static_scaler.pkl`, `output_scaler.pkl`

Phase 03 trains a Mixed-Input LSTM sequence model that learns to correct DCA's systematic errors over the 54-month forecast horizon. Rather than predicting raw production directly, the model predicts the cumulative residual between actual production and the DCA Early Time baseline — in effect, learning *where and by how much* early DCA is wrong.

**Dual-Input Architecture** — The model uses two parallel input branches processed by the Keras Functional API:

- **Dynamic Branch** — A stacked two-layer LSTM Encoder processes a `(early_time_steps, n_features)` sequence tensor of time-varying signals: primary series, `Ratio_A`, `Ratio_B`, their step-over-step deltas, and two time features. This branch captures the temporal pattern of early production behavior.
- **Static Branch** — Two fully-connected Dense layers process a `(early_time_steps,)` vector of fixed well characteristics: DCA parameters from the Early Time fit and surface spatial dimensions. This branch provides geological and geographic context.

The two branch outputs are concatenated into a unified 192-dimensional context vector, replicated 54 times via `RepeatVector`, and decoded by a stacked two-layer LSTM Decoder into a `(54, 1)` output sequence of monthly cumulative residual predictions.

**Scaling** — Three independent `StandardScaler` instances are fit separately on the sequence features, static features, and output residuals. Sequence data is reshaped to 2D for fitting then restored to 3D for training. All three scalers are serialized alongside the model so that the evaluation phase can apply identical transformations to unseen test data using `transform()` rather than `fit_transform()`.

**Training** uses the Adam optimizer with MAE loss, reflecting that absolute error in barrels is the most operationally meaningful metric in this domain.

---

## Phase 04 — LSTM Evaluation

**Script:** `step_01_evaluate_lstm.py`

**Inputs:** `phase2_test_baseline.csv`, trained LSTM model and three scalers
**Outputs:** `per_well_metrics.csv`, `mean_average_metrics.csv`, `test_metrics.json`, per-well HTML plots, aggregate cross-plots and density heatmaps

Phase 04 is the formal held-out evaluation of the LSTM model on wells never seen during training. The evaluation window is strictly the post-threshold forecast horizon — the early time threshold is the observed input window and evaluating predictions there would not represent genuine forecasting.

The corrected forecast is assembled as:

```
corrected_cumulative[Post_Threshold] = Baseline_Cum_Metric_Early_Time[Post_Threshold] + predicted_residuals
```

For each test well, DCA and LSTM metrics are computed in parallel over the same evaluation window, providing a direct comparison on identical data. Fleet-level aggregate visualizations include interactive Plotly cross-plots (DCA vs. Actual and LSTM vs. Actual side by side) and log-density hexbin heatmaps. Log-count density encoding is used because a small number of very high-production wells would otherwise dominate linear-count visualizations, making it impossible to assess model quality for the majority of average wells.

Metrics are exported to both a timestamped run directory (for multi-run comparison) and a stable artifact path (consumed by the Phase 07 dashboard).

---

## Phase 05 — XGBoost Training & Evaluation

**Scripts:** `step_01_train_xgb.py`, `step_02_evaluate_xgb.py`

**Input:** `phase2_train_baseline.csv` (training), `phase2_test_baseline.csv` (evaluation)
**Outputs:** `xgb_model.json`, per-well metrics, aggregate visualizations, SHAP summary, decision tree SVGs

Phase 05 trains and evaluates an XGBoost gradient-boosted tree model using the same early time threshold data window as the DCA Early Time baseline and the LSTM.

**Feature Set** — 18 features are used, spanning four categories:

| Category | Features |
|----------|----------|
| Early production history | `Series_Feature_1` through `Series_Feature_N` (individual period rates) |
| Production statistics | `Rolling_Stat_1`, `Rolling_Stat_2`, `Rolling_Stat_3` |
| Baseline parameters | `Early_Param_A`, `Early_Param_B`, `Early_Param_C` from the Early Time fit |
| Geography & time | `Spatial_Dim_1`, `Spatial_Dim_2`, `Time_Trans_1`, `Time_Trans_2`, `Time_Trans_3`, `Time_Trans_4` |

**Training Table Construction** — Rather than operating on sequence tensors, XGBoost consumes a flat table with one row per (well × forecast month). Static well features are replicated across all 54 forecast months for a given well; time polynomial features vary by month. This gives the model both the well's identity and temporal position for every prediction.

**Target Transformation** — Monthly oil production rates are `log1p`-transformed before training to address the right-skewed distribution across the well population, preventing high-rate wells from dominating the loss function. Predictions are `expm1`-inverse-transformed before metric computation and visualization.

**Training** uses early stopping against a held-out validation set — split at the well level to match the pipeline's leakage-prevention standard — selecting the optimal number of boosting rounds before overfitting begins.

**Interpretability** — SHAP (SHapley Additive exPlanations) values are computed via `TreeExplainer` on the full test prediction set and visualized as a beeswarm summary plot showing both the magnitude and direction of each feature's impact on monthly production forecasts. Early production signals and well surface coordinates emerged as the strongest predictors of long-term output. Decision tree diagrams are exported as SVG files in both raw (internal feature index) and human-readable (named features, bbls/month leaf values) formats for inspection.

---

## Phase 06 — DCA Performance Analysis

**Script:** `step_01_analyze_dca.py`

**Input:** `phase2_dca_baseline.csv`
**Outputs:** `residuals_distribution.png/.html`, `error_over_time.png/.html`, `actual_vs_predicted.png/.html`

Phase 06 gives the physics baseline its own rigorous standalone diagnostic evaluation, independent of its role as a comparison point in the ML phases. Rather than treating DCA as a black box, this phase characterizes *where*, *when*, and *by how much* the Arps model fails — knowledge that directly motivated the design of the LSTM and XGBoost models.

**Five metrics** are computed for both fitting strategies across both rate and cumulative production dimensions: MSE, RMSE, MAE, R², and MAPE (with epsilon smoothing for zero-production months).

**Residual Distributions** — A 2×2 histogram grid shows the shape of prediction errors for rate and cumulative production under both fits. A symmetric distribution centered on zero indicates unbiased predictions; heavy right tails mean the model systematically underpredicts. Distributions are clipped to the 5th–95th percentile range so that the central distribution — where the vast majority of data lives — is not visually collapsed by a small number of extreme outliers.

**Error Over Time** — Per-month mean absolute error is plotted as a function of time index (capped at 120 months for statistical stability) with a marker at the early time training cutoff. Monotonically growing error after the cutoff is precisely the regime where the LSTM and XGBoost corrections provide the most value.

**Actual vs. Predicted Scatter** — Side-by-side scatter panels for rate and cumulative production use sampled data points with a reference `y = x` perfect-prediction line, making systematic over- or under-prediction at different production scales immediately visible.

---

## Results

XGBoost materially outperformed the DCA Early Time baseline across every metric on the independent test cohort.

| Metric | DCA Early Time | XGBoost | Improvement |
|--------|-------------|---------|-------------|
| Average MAE (bbl/well) | 37,938 | 15,938 | **57% reduction** |
| Average Revenue Error ($/well) | -$2,881,659 | -$1,212,297 | **58% reduction** |

Against a dataset with an average of **203,760 barrels** produced per well and average well revenue of **$15,497,985**, the reduction in forecast error represents a material improvement in capital decision reliability.

The predicted vs. actual cumulative production cross-plot confirmed strong model accuracy across the full test set, with XGBoost forecasts tracking closely to actual values across all unseen wells at a range of production scales. SHAP feature importance analysis revealed that early production signals (particularly the initial threshold period) and well surface coordinates were the strongest predictors of long-term output, confirming that geographic clustering of well productivity is a learnable signal the physics model cannot exploit.

---

## Discussion

Traditional DCA projections risk over-leveraging on wells that may underperform. When an operator projects twice the actual production from a well, the consequences cascade: adjacent wells are drilled, infrastructure is committed, and debt is structured — all against a forecast that was structurally unreliable from the moment it was made. If they discover five years later that the well produces only half the DCA projection, they may realize significantly lower returns on a broad portfolio of decisions.

XGBoost's strategic value lies in its **conservative bias**. By significantly reducing forecast "overshoot" compared to the DCA early time baseline, it protects capital and ensures that new well placement is backed by a reliable revenue floor rather than a risky bet. When a single well in a field performs well, it is a strong indicator of similar performance in that area — but only if the forecast that drives the follow-on decision is trustworthy.

With a 57% reduction in mean absolute error and a 58% reduction in average revenue projection error per well, the model delivers a forecasting capability that is meaningfully more reliable for early-life capital allocation decisions in unconventional horizontal oil wells.

---

## Tech Stack

| Library | Role |
|---------|------|
| `Python` | Core language |
| `pandas` / `numpy` | Data manipulation and numerical computing |
| `scipy` | Non-linear least squares DCA fitting (Levenberg-Marquardt) |
| `scikit-learn` | Preprocessing scalers, metrics |
| `XGBoost` | Gradient-boosted tree forecasting model |
| `TensorFlow / Keras` | LSTM sequence model |
| `SHAP` | Model interpretability and feature importance |
| `Plotly` | Interactive visualizations |
| `Matplotlib` | Static publication-quality figures |
| `Graphviz` | Decision tree SVG export |

---

## References

Rahmanifard, H., & Gates, I. (2024, July 22). A comprehensive review of data-driven approaches for forecasting production from unconventional reservoirs: Best practices and future directions. *Artificial Intelligence Review*. https://link.springer.com/article/10.1007/s10462-024-10865-5

Wyoming Oil and Gas Conservation Commission (WOGCC). *Pipeline.wyo.gov*. (n.d.). https://pipeline.wyo.gov/

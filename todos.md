# Project Todos and Next Steps

This document outlines the potential next steps and improvements for the NYC Taxi Fare Prediction project, categorized by area of focus.

## 1. Model Improvement & Analysis

- [ ] **Residual Analysis:**
  - Plot residuals vs. predictions to check for heteroscedasticity.
  - Analyze residuals against features to find poor performance areas.
  - Identify error patterns (e.g., routes, times, trip types).
- [ ] **Try Other Models:**
  - Test LightGBM, CatBoost, Ridge, Lasso, and simple neural networks.
- [ ] **Hyperparameter Tuning:**
  - Use libraries like Optuna or Hyperopt for efficient tuning.
  - Try Bayesian optimization or increase random search trials.
- [ ] **Experiment Tracking:**
  - Use MLflow or Weights & Biases to log parameters, metrics, and artifacts.

## 2. Code Quality & Robustness

- [ ] **Expand Testing:**
  - Add unit tests for `preprocessor.py` and edge cases.
  - Write integration tests for preprocessing and API endpoints.
- [ ] **Refactor Code:**
  - Make code modular, ensure consistent styling, and add missing docstrings.

## 3. Feature & Data Enhancement

- [ ] **Add More Data:**
  - Integrate weather, holiday, and event data.
- [ ] **Feature Selection:**
  - Use feature importance, RFE, or permutation importance to simplify the model.

## 4. Scalability & Performance

- [ ] **Distributed Computing:**
  - Use frameworks like PySpark or Dask for large datasets.
- [ ] **Optimize Latency:**
  - Analyze bottlenecks and consider model quantization or lightweight models.
# Movie Hit Prediction

> **Data Science Term Project 2026 · Team 9**
> Predicting whether a movie is a "Hit" (classification) and forecasting its IMDb rating (regression), using merged IMDb + TMDB metadata.

An end-to-end machine learning pipeline covering preprocessing, modeling, and evaluation. The project also contributes an open-source utility, [`find_best_combinations()`](#open-source-contribution-find_best_combinations), that automatically searches scaler × model × hyperparameter combinations and returns the best ones by K-fold cross validation.

---

## Results at a Glance

| Task | Best Model | Metric | Score |
|------|-----------|--------|-------|
| **Classification** (Hit) | XGBoost | Test F1 | **0.877** |
| **Classification** (Hit) | XGBoost | Test ROC-AUC | **0.995** |
| **Regression** (Rating) | Gradient Boosting | Test R² | **0.856** |
| **Regression** (Rating) | Gradient Boosting | Test RMSE | **0.316** |

---

## Problem Definition

**Business objective:** Identify which movies will be both critically and popularly successful, and understand the factors that drive that success.

A movie is labeled a **Hit** when:

```
Hit = 1   if   IMDb Rating >= 7.5  AND  Votes >= 50,000
Hit = 0   otherwise
```

- **Rating >= 7.5** corresponds to the 80th percentile (top 20%) of the dataset.
- **Votes >= 50,000** filters out niche titles that score high on very few votes.

The resulting target is imbalanced: **10.3% Hit vs 89.7% Non-Hit**.

---

## Dataset

Two public datasets merged on `imdb_id` via inner join.

| Source | File | Original Shape |
|--------|------|----------------|
| IMDb Top 10,000 | `Top_10000_Movies_IMDb.csv` | 9,999 × 12 |
| TMDB Movies | `tmdb_movies_data.csv` | 10,866 × 21 |
| **Merged (final)** | `merged_movie_dataset.csv` | **5,979 × 15** |
| **Encoded (model input)** | `encoded_movie_dataset.csv` | **5,979 × 30** |

After one-hot encoding of `Genre` (16 dummy columns) and a `log1p` transform on `Votes`, the model input has 30 columns.

### Features

| Feature | Type | Description |
|---------|------|-------------|
| Runtime | Numerical | Running time (minutes) |
| Metascore | Numerical | Critic score (0–100) |
| Votes | Numerical | IMDb vote count (`log1p` transformed) |
| Gross | Numerical | North American box office (IQR-capped) |
| popularity | Numerical | TMDB popularity index |
| vote_average | Numerical | TMDB average rating |
| release_year | Numerical | Release year |
| budget_adj | Numerical | Inflation-adjusted budget |
| revenue_adj | Numerical | Inflation-adjusted revenue |
| is_major_company | Binary | Whether produced by a major studio |
| Genre_* (16) | Binary | One-hot encoded primary genre |

---

## Pipeline

```
Raw data (IMDb + TMDB)
        |
        v
[1] Preprocessing            1_preprocessing.ipynb
    - Drop high-cardinality / text columns (Plot, Cast, Directors, Keywords)
    - Runtime "142 min" -> 142 (int)
    - Metascore missing values -> genre-level group imputation
    - Hit label created on raw Votes (before capping)
    - Gross / budget_adj / revenue_adj -> IQR capping
    - Inner join on imdb_id, drop duplicates
        |
        v
[2] Encoding & Scaling
    - Genre -> one-hot (drop_first=True)
    - Votes -> log1p
    - Train/Test split (80/20, stratified)
    - StandardScaler (fit on train only)
        |
        v
[3] Modeling                 2_modeling.ipynb
    - Classification: 8 models
    - Regression: 7 models
    - K-fold cross validation
        |
        v
[4] Evaluation & Analysis    3_evaluation.ipynb
    - F1, ROC-AUC, confusion matrix, feature importance
    - R^2, RMSE, MAE, actual vs predicted
        |
        v
[5] Best-5 search            best_combinations.py
    - scaler x model x hyperparameter scan -> Top-5
```

---

## Leakage Prevention

Because `Hit` is derived from `Rating` and `Votes`, the feature sets are chosen carefully:

| Task | Excluded feature | Reason |
|------|-----------------|--------|
| Classification (Hit) | `Rating` | Hit is derived directly from Rating |
| Regression (Rating) | `Hit` | Hit is built on top of Rating |

`Votes` is **kept** for classification — a high vote count is a legitimate signal available at inference time, not leakage.

---

## Models

### Classification (target: `Hit`)

| Model | Imbalance Handling |
|-------|-------------------|
| Logistic Regression | `class_weight='balanced'` |
| K-Nearest Neighbors | — |
| Decision Tree | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'` |
| SVM (RBF) | `class_weight='balanced'` |
| Gaussian Naive Bayes | — |
| Gradient Boosting | — |
| XGBoost | `scale_pos_weight` |

Evaluated with **StratifiedKFold (k=5)**. Primary metrics: **F1, ROC-AUC** (accuracy is secondary because of class imbalance).

### Regression (target: `Rating`)

Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, XGBoost.

Evaluated with **KFold (k=5)**. Metrics: **R², RMSE, MAE**.

---

## Open-Source Contribution: `find_best_combinations()`

A reusable utility that scans every combination of **scaler × model × hyperparameter grid**, evaluates each with K-fold CV, and returns the top-k.

### Why

Manually trying every `(scaler, model, params)` triple is tedious and error-prone. This function automates the entire grid scan with a single call and reports mean score, standard deviation, and fit time per combination.

### What it does

The function runs the **entire preprocessing -> training -> evaluation loop inside a single top-level function**, scanning a 4-dimensional combination space:

```
categorical encoder  x  feature scaler  x  model  x  hyperparameters
```

This directly satisfies the course's open-source requirement: *combination of various data scaling AND categorical encoding methods*, *different models with parameter combinations*, and *evaluation*, all under one function (no copy-pasted code), returning the **top-5 plus the single best** combination.

### Usage

```python
from best_combinations import find_best_combinations
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# axis 1: categorical encoding methods
encoders = {'onehot': 'onehot', 'label': 'label', 'ordinal': 'ordinal'}

# axis 2: feature scaling methods
scalers = {
    'Standard': StandardScaler(),
    'MinMax':   MinMaxScaler(),
    'Robust':   RobustScaler(),
}

# axis 3 + 4: models and their hyperparameter grids
models = {
    'LogReg': (LogisticRegression(max_iter=1000, class_weight='balanced'),
               {'C': [0.1, 1.0, 10.0]}),
    'RF':     (RandomForestClassifier(class_weight='balanced'),
               {'n_estimators': [100, 200], 'max_depth': [10, 20]}),
}

# the function takes a *cleaned* dataframe and does encoding + scaling internally
top5 = find_best_combinations(
    df, target='Hit',
    categorical_cols=['Genre'],
    numeric_cols=['Runtime', 'Metascore', 'Votes', ...],
    encoders=encoders,
    scalers=scalers,
    models_with_params=models,
    task='classification',     # or 'regression'
    cv=5,
    scoring='f1',              # any sklearn scoring string
    top_k=5,
    drop_cols=['Movie Name', 'imdb_id', 'Rating'],  # identifiers / leakage cols
)
print(top5)
print(top5.iloc[0])   # the single best combination
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | — | Cleaned dataset (missing values already handled) |
| `target` | str | — | Name of the target column |
| `categorical_cols` | list | — | Categorical columns to encode |
| `numeric_cols` | list | — | Numeric columns to scale |
| `encoders` | dict | — | `{name: method}` where method is `'onehot'`/`'label'`/`'ordinal'` |
| `scalers` | dict | — | `{name: scaler}`. Use `{'None': None}` to skip scaling |
| `models_with_params` | dict | — | `{name: (estimator, param_grid)}` |
| `task` | str | `'classification'` | `'classification'` (StratifiedKFold) or `'regression'` (KFold) |
| `cv` | int | `5` | Number of folds |
| `scoring` | str | `None` | sklearn scoring string; defaults to `'f1'` / `'r2'` |
| `top_k` | int | `5` | Number of best combinations to return |
| `drop_cols` | list | `None` | Identifier / leakage columns to drop |
| `random_state` | int | `42` | Seed for CV splitting |
| `verbose` | bool | `True` | Print progress per combination |

### Returns

A `pandas.DataFrame` sorted by `mean_score` (descending), with columns:

```
rank | encoder | scaler | model | params | mean_score | std_score | fit_time
```

### Architecture

```
find_best_combinations(df, target, categorical_cols, numeric_cols,
                       encoders, scalers, models_with_params, ...)
        |
        |-- split target y from features X; drop identifier/leakage cols
        |-- choose CV splitter   (StratifiedKFold if classification, else KFold)
        |
        |-- for each ENCODER:                      # axis 1
        |       X_enc = apply_encoder(X, categorical_cols, method)
        |
        |       for each SCALER:                    # axis 2
        |           scale numeric_cols of X_enc
        |
        |           for each MODEL:                 # axis 3
        |               expand param_grid (itertools.product)
        |
        |               for each PARAM combo:       # axis 4
        |                   clone estimator, set params
        |                   cross_val_score(...) -> mean, std, time
        |                   record result
        |
        v
   sort by mean_score, return top-k as DataFrame  (rank 1 = best)
```

### Results on the Movie Dataset

Searched `3 encoders x 3 scalers x model grids` via 5-fold CV.

**Classification (top-5 by CV F1):**

| Rank | Encoder | Scaler | Model | Params | CV F1 |
|------|---------|--------|-------|--------|-------|
| 1 | label | Standard | XGBoost | n_estimators=100, max_depth=8 | 0.8542 |
| 2 | ordinal | Standard | XGBoost | n_estimators=100, max_depth=8 | 0.8542 |
| 3 | onehot | Standard | XGBoost | n_estimators=200, max_depth=8 | 0.8539 |
| 4 | label | Standard | XGBoost | n_estimators=200, max_depth=8 | 0.8539 |
| 5 | ordinal | Standard | XGBoost | n_estimators=200, max_depth=8 | 0.8539 |

**Regression (top-5 by CV R²):**

| Rank | Encoder | Scaler | Model | Params | CV R² |
|------|---------|--------|-------|--------|-------|
| 1 | onehot | MinMax | Gradient Boosting | n_estimators=200, lr=0.1 | 0.8511 |
| 2 | ordinal | Standard | Gradient Boosting | n_estimators=200, lr=0.1 | 0.8510 |
| 3 | label | Standard | Gradient Boosting | n_estimators=200, lr=0.1 | 0.8510 |
| 4 | ordinal | Robust | Gradient Boosting | n_estimators=200, lr=0.1 | 0.8509 |
| 5 | label | Robust | Gradient Boosting | n_estimators=200, lr=0.1 | 0.8509 |

> Note: label and ordinal encoding edged out one-hot for tree models on the classification task — evidence that scanning the encoding axis (not just scaling) was worthwhile.


---

## Repository Structure

```
movie-hit-prediction/
├── README.md
├── LICENSE                       # MIT
├── requirements.txt
├── best_combinations.py          # open-source utility (importable)
│
├── notebooks/
│   ├── 1_preprocessing.ipynb
│   ├── 2_modeling.ipynb
│   └── 3_evaluation.ipynb
│
├── data/
│   ├── merged_movie_dataset.csv
│   └── encoded_movie_dataset.csv
│
└── results/
    ├── classification_results.png
    ├── regression_results.png
    ├── feature_importance.png
    ├── top5_classification.csv
    └── top5_regression.csv
```

---

## Getting Started

```bash
# 1. Clone
git clone https://github.com/<your-username>/movie-hit-prediction.git
cd movie-hit-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebooks in order
jupyter notebook notebooks/2_modeling.ipynb
```

To use the open-source utility on your own data:

```python
from best_combinations import find_best_combinations
# ... see Usage above
```

---

## Course Requirements Mapping

| Requirement | Where |
|-------------|-------|
| Data scaling & encoding | `1_preprocessing.ipynb` (StandardScaler, one-hot) |
| Classification algorithm | `2_modeling.ipynb` (8 models) |
| Regression or clustering | `2_modeling.ipynb` (regression — 7 models) |
| K-fold cross validation | StratifiedKFold / KFold, k=5 |
| Open-source contribution | `best_combinations.py` |

---

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

- IMDb Top 10,000 Movies dataset
- TMDB Movies dataset
- Built for the Data Science course term project, 2026.

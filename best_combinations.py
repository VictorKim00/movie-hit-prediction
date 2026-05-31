"""best_combinations: Automated scaler-model-hyperparameter search via K-fold CV.

This module provides a single top-level function, :func:`find_best_combinations`,
that scans every combination of data scalers, learning models, and hyperparameter
grids, evaluates each combination with K-fold cross validation, and returns the
top-k combinations ranked by score.

Example
-------
>>> from sklearn.preprocessing import StandardScaler, MinMaxScaler
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.ensemble import RandomForestClassifier
>>> scalers = {'Standard': StandardScaler(), 'MinMax': MinMaxScaler()}
>>> models = {
...     'LogReg': (LogisticRegression(max_iter=1000), {'C': [0.1, 1.0]}),
...     'RF':     (RandomForestClassifier(), {'n_estimators': [100, 200]}),
... }
>>> top5 = find_best_combinations(X, y, scalers, models, task='classification')

Author: Team 9, Data Science Term Project 2026
License: MIT
"""

import time
from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

__all__ = ["find_best_combinations"]


def find_best_combinations(
    X, y,
    scalers,
    models_with_params,
    task='classification',
    cv=5,
    scoring=None,
    top_k=5,
    numeric_cols=None,
    random_state=42,
    verbose=True,
):
    """Find the top-k (scaler, model, hyperparameter) combinations via K-fold CV.

    Iterates over every combination of provided scalers, models, and hyperparameter
    grids, then evaluates each combination using K-fold cross validation on the
    given dataset.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series or array-like
        Target variable. Binary or multiclass for classification, continuous for
        regression.
    scalers : dict of {str: object}
        Mapping from scaler name to a scikit-learn scaler instance. Pass
        ``{'None': None}`` to skip scaling.
    models_with_params : dict of {str: (estimator, param_grid)}
        Mapping from model name to a tuple ``(estimator, param_grid)``, where
        ``param_grid`` is a dict like ``{'param_name': [values, ...]}``.
    task : {'classification', 'regression'}, default='classification'
        Type of task. Determines the CV strategy (StratifiedKFold vs KFold) and
        the default scoring metric.
    cv : int, default=5
        Number of cross validation folds.
    scoring : str, optional
        Scikit-learn scoring string. Defaults to ``'f1'`` for classification and
        ``'r2'`` for regression.
    top_k : int, default=5
        Number of best combinations to return.
    numeric_cols : list of str, optional
        Subset of columns in X to which the scaler should be applied. If None,
        the scaler is applied to all columns.
    random_state : int, default=42
        Random seed for the cross validation splitter.
    verbose : bool, default=True
        If True, prints progress for each combination.

    Returns
    -------
    results : pandas.DataFrame
        DataFrame with columns ``['rank', 'scaler', 'model', 'params',
        'mean_score', 'std_score', 'fit_time']`` sorted by ``mean_score``
        descending. Only the top-k rows are returned.

    Raises
    ------
    ValueError
        If ``task`` is not 'classification' or 'regression'.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler, MinMaxScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> scalers = {'Standard': StandardScaler(), 'MinMax': MinMaxScaler()}
    >>> models = {
    ...     'LogReg': (LogisticRegression(max_iter=1000), {'C': [0.1, 1.0]}),
    ...     'RF':     (RandomForestClassifier(), {'n_estimators': [100, 200]}),
    ... }
    >>> top5 = find_best_combinations(X, y, scalers, models, task='classification')
    """
    # Default scoring
    if scoring is None:
        scoring = 'f1' if task == 'classification' else 'r2'

    # CV splitter
    if task == 'classification':
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    elif task == 'regression':
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        raise ValueError("task must be 'classification' or 'regression'")

    results = []

    # total 
    total = sum(
        len(scalers) * np.prod([len(v) for v in pg.values()] or [1])
        for _, pg in models_with_params.values()
    )
    if verbose:
        print(f'Total combinations: {int(total)}')

    count = 0
    for model_name, (base_estimator, param_grid) in models_with_params.items():
        if param_grid:
            param_keys = list(param_grid.keys())
            param_combos = [
                dict(zip(param_keys, vals))
                for vals in product(*param_grid.values())
            ]
        else:
            param_combos = [{}]

        for scaler_name, scaler in scalers.items():
            X_scaled = X.copy()
            if scaler is not None:
                cols = numeric_cols if numeric_cols else X.columns.tolist()
                cols = [c for c in cols if c in X.columns]
                if cols:
                    X_scaled[cols] = scaler.fit_transform(X_scaled[cols])

            for params in param_combos:
                count += 1
                estimator = clone(base_estimator).set_params(**params)

                t0 = time.time()
                try:
                    scores = cross_val_score(
                        estimator, X_scaled, y,
                        cv=cv_splitter, scoring=scoring, n_jobs=-1
                    )
                    mean_score, std_score = scores.mean(), scores.std()
                except Exception as e:
                    mean_score, std_score = np.nan, np.nan
                    if verbose:
                        print(f'  [{count}/{int(total)}] FAIL {model_name} | {scaler_name} | {params} -> {e}')
                    continue
                fit_time = time.time() - t0

                results.append({
                    'scaler': scaler_name,
                    'model':  model_name,
                    'params': params,
                    'mean_score': mean_score,
                    'std_score':  std_score,
                    'fit_time':   fit_time,
                })
                if verbose:
                    print(f'  [{count}/{int(total)}] {model_name:20s} | {scaler_name:10s} | '
                          f'score={mean_score:.4f} +/-{std_score:.4f} | {fit_time:.1f}s')

    # sort and Top-K
    result_df = pd.DataFrame(results).sort_values('mean_score', ascending=False)
    result_df.insert(0, 'rank', range(1, len(result_df) + 1))
    return result_df.head(top_k).reset_index(drop=True)

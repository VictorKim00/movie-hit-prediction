"""best_combinations: Automated preprocessing-model-evaluation search via K-fold CV.

This module provides a single top-level function, :func:`find_best_combinations`,
that scans every combination of

    categorical encoder  x  feature scaler  x  model  x  hyperparameters

evaluates each combination with K-fold cross validation, and returns the top-k
combinations ranked by score. It is a small, self-contained AutoML / grid-search
utility built for the Data Science term project (open-source contribution).

The function expects a *cleaned* dataset (missing values already handled) and
performs all preprocessing (encoding + scaling) internally, so that a single call
reproduces the full [preprocessing -> training -> evaluation] loop without copying
the same code many times.

Example
-------
>>> from sklearn.preprocessing import StandardScaler, MinMaxScaler
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.ensemble import RandomForestClassifier
>>> encoders = {'onehot': 'onehot', 'label': 'label'}
>>> scalers  = {'Standard': StandardScaler(), 'MinMax': MinMaxScaler()}
>>> models   = {
...     'LogReg': (LogisticRegression(max_iter=1000), {'C': [0.1, 1.0]}),
...     'RF':     (RandomForestClassifier(), {'n_estimators': [100, 200]}),
... }
>>> top5 = find_best_combinations(
...     df, target='Hit',
...     categorical_cols=['Genre'],
...     numeric_cols=['Runtime', 'Votes'],
...     encoders=encoders, scalers=scalers, models_with_params=models,
...     task='classification',
... )

Author: Team 9, Data Science Term Project 2026
License: MIT
"""

import time
from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

__all__ = ["find_best_combinations", "apply_encoder"]


def apply_encoder(X, categorical_cols, method):
    """Encode categorical columns of a DataFrame using the chosen method.

    Parameters
    ----------
    X : pandas.DataFrame
        Input feature matrix. Not modified in place.
    categorical_cols : list of str
        Names of the categorical columns to encode.
    method : {'onehot', 'label', 'ordinal'}
        Encoding strategy:

        - ``'onehot'`` : one-hot / dummy variables via :func:`pandas.get_dummies`
          with ``drop_first=True`` to avoid the dummy-variable trap.
        - ``'label'``  : integer label encoding per column
          (:class:`sklearn.preprocessing.LabelEncoder`).
        - ``'ordinal'``: integer ordinal encoding for all columns at once
          (:class:`sklearn.preprocessing.OrdinalEncoder`).

    Returns
    -------
    X_encoded : pandas.DataFrame
        A new DataFrame with the categorical columns replaced by their encoded
        representation. Non-categorical columns are passed through unchanged.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported strategies.
    """
    X = X.copy()
    cols = [c for c in categorical_cols if c in X.columns]
    if not cols:
        return X

    if method == 'onehot':
        X = pd.get_dummies(X, columns=cols, drop_first=True)
        # get_dummies가 만든 bool 컬럼을 int로 통일
        bool_cols = X.select_dtypes(include='bool').columns
        X[bool_cols] = X[bool_cols].astype(int)
        return X

    if method == 'label':
        for c in cols:
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        return X

    if method == 'ordinal':
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[cols] = enc.fit_transform(X[cols].astype(str))
        return X

    raise ValueError("method must be one of {'onehot', 'label', 'ordinal'}")


def find_best_combinations(
    df,
    target,
    categorical_cols,
    numeric_cols,
    encoders,
    scalers,
    models_with_params,
    task='classification',
    cv=5,
    scoring=None,
    top_k=5,
    drop_cols=None,
    random_state=42,
    verbose=True,
):
    """Find the top-k (encoder, scaler, model, hyperparameter) combinations.

    Runs the full preprocessing -> training -> evaluation pipeline inside a single
    function. For every combination of

        categorical encoder  x  feature scaler  x  model  x  hyperparameter grid

    the data is encoded and scaled, then the estimator is evaluated with K-fold
    cross validation. The top-k combinations (and therefore the single best one,
    rank 1) are returned as a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataset (missing values already handled). Contains the target
        column, the categorical columns, the numeric columns, and optionally
        some identifier columns to drop.
    target : str
        Name of the target column in ``df``.
    categorical_cols : list of str
        Categorical columns to be encoded by each encoder.
    numeric_cols : list of str
        Numeric columns to be scaled by each scaler.
    encoders : dict of {str: str}
        Mapping from a display name to an encoding method understood by
        :func:`apply_encoder` (``'onehot'``, ``'label'``, or ``'ordinal'``).
    scalers : dict of {str: object}
        Mapping from a display name to a scikit-learn scaler instance. Pass
        ``{'None': None}`` to skip scaling.
    models_with_params : dict of {str: (estimator, param_grid)}
        Mapping from model name to ``(estimator, param_grid)``, where
        ``param_grid`` is a dict like ``{'param_name': [values, ...]}``.
    task : {'classification', 'regression'}, default='classification'
        Determines the CV strategy (StratifiedKFold vs KFold) and the default
        scoring metric.
    cv : int, default=5
        Number of cross validation folds.
    scoring : str, optional
        Scikit-learn scoring string. Defaults to ``'f1'`` for classification and
        ``'r2'`` for regression.
    top_k : int, default=5
        Number of best combinations to return.
    drop_cols : list of str, optional
        Identifier / leakage columns to drop from ``df`` before modeling (e.g.
        ``['Movie Name', 'imdb_id']``). The target is dropped automatically.
    random_state : int, default=42
        Random seed for the cross validation splitter.
    verbose : bool, default=True
        If True, prints progress for each combination.

    Returns
    -------
    results : pandas.DataFrame
        DataFrame with columns
        ``['rank', 'encoder', 'scaler', 'model', 'params', 'mean_score',
        'std_score', 'fit_time']`` sorted by ``mean_score`` descending. Only the
        top-k rows are returned.

    Raises
    ------
    ValueError
        If ``task`` is not 'classification' or 'regression'.

    Notes
    -----
    The total number of evaluated combinations is

        ``len(encoders) * len(scalers) * sum(len(grid) for each model)``

    where ``len(grid)`` is the number of points in each model's hyperparameter
    grid. Scaling is applied only to ``numeric_cols``; one-hot dummy columns and
    binary columns are therefore left untouched.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> encoders = {'onehot': 'onehot', 'label': 'label', 'ordinal': 'ordinal'}
    >>> scalers  = {'Standard': StandardScaler(), 'MinMax': MinMaxScaler()}
    >>> models   = {
    ...     'LogReg': (LogisticRegression(max_iter=1000, class_weight='balanced'),
    ...                {'C': [0.1, 1.0, 10.0]}),
    ...     'RF':     (RandomForestClassifier(class_weight='balanced'),
    ...                {'n_estimators': [100, 200], 'max_depth': [10, 20]}),
    ... }
    >>> top5 = find_best_combinations(
    ...     df, target='Hit',
    ...     categorical_cols=['Genre'],
    ...     numeric_cols=['Runtime', 'Metascore', 'Votes'],
    ...     encoders=encoders, scalers=scalers, models_with_params=models,
    ...     task='classification', scoring='f1', top_k=5,
    ...     drop_cols=['Movie Name', 'imdb_id'],
    ... )
    >>> top5.iloc[0]   # the single best combination
    """
    # ----- 검증 및 기본값 -----
    if task == 'classification':
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        default_scoring = 'f1'
    elif task == 'regression':
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        default_scoring = 'r2'
    else:
        raise ValueError("task must be 'classification' or 'regression'")

    if scoring is None:
        scoring = default_scoring

    drop_cols = drop_cols or []

    # 타겟 분리
    y = df[target]
    X_raw = df.drop(columns=[target] + [c for c in drop_cols if c in df.columns])

    # ----- 총 조합 수 -----
    n_param_points = sum(
        int(np.prod([len(v) for v in pg.values()] or [1]))
        for _, pg in models_with_params.values()
    )
    total = len(encoders) * len(scalers) * n_param_points
    if verbose:
        print(f'Total combinations: {total} '
              f'(encoders={len(encoders)} x scalers={len(scalers)} '
              f'x param_points={n_param_points})')

    results = []
    count = 0

    # ----- 1축: Encoder -----
    for enc_name, enc_method in encoders.items():
        X_enc = apply_encoder(X_raw, categorical_cols, enc_method)

        # ----- 2축: Scaler -----
        for scaler_name, scaler in scalers.items():
            X_scaled = X_enc.copy()
            if scaler is not None:
                cols = [c for c in numeric_cols if c in X_scaled.columns]
                if cols:
                    X_scaled[cols] = scaler.fit_transform(X_scaled[cols])

            # ----- 3축: Model -----
            for model_name, (base_estimator, param_grid) in models_with_params.items():
                if param_grid:
                    keys = list(param_grid.keys())
                    param_combos = [dict(zip(keys, vals))
                                    for vals in product(*param_grid.values())]
                else:
                    param_combos = [{}]

                # ----- 4축: Hyperparameters -----
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
                        if verbose:
                            print(f'  [{count}/{total}] FAIL {enc_name}|{scaler_name}|'
                                  f'{model_name}|{params} -> {e}')
                        continue
                    fit_time = time.time() - t0

                    results.append({
                        'encoder': enc_name,
                        'scaler':  scaler_name,
                        'model':   model_name,
                        'params':  params,
                        'mean_score': mean_score,
                        'std_score':  std_score,
                        'fit_time':   fit_time,
                    })
                    if verbose:
                        print(f'  [{count}/{total}] {enc_name:8s} | {scaler_name:9s} | '
                              f'{model_name:18s} | score={mean_score:.4f} '
                              f'+/-{std_score:.4f} | {fit_time:.1f}s')

    # ----- 정렬 후 Top-K -----
    result_df = pd.DataFrame(results).sort_values('mean_score', ascending=False)
    result_df.insert(0, 'rank', range(1, len(result_df) + 1))
    return result_df.head(top_k).reset_index(drop=True)

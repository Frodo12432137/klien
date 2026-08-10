from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .features import FeatureSpec, build_features


@dataclass
class ForecastResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    models: dict[str, Any]
    feature_spec: FeatureSpec


class _CatBoostTimeLimit:
    """Kończy fit po najbliższej iteracji po przekroczeniu limitu czasu."""

    def __init__(self, seconds: float):
        self.deadline = time.monotonic() + float(seconds)
        self.stopped_by_time = False

    def after_iteration(self, _info) -> bool:
        if time.monotonic() >= self.deadline:
            self.stopped_by_time = True
            return False
        return True


def _sklearn_components():
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import TargetEncoder
    except ImportError as exc:  # pragma: no cover - komunikat dla środowiska użytkownika
        raise RuntimeError(
            "Brak scikit-learn. Zainstaluj zależności poleceniem: "
            "python -m pip install -r mdd_forecasting/requirements.txt"
        ) from exc
    return (
        ColumnTransformer,
        HistGradientBoostingRegressor,
        permutation_importance,
        SimpleImputer,
        Pipeline,
        TargetEncoder,
    )


def _make_model(spec: FeatureSpec, config: PipelineConfig):
    if config.model_backend == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:  # pragma: no cover - opcjonalny backend produkcyjny
            raise RuntimeError(
                "Wybrano CatBoost, ale biblioteka nie jest zainstalowana. Użyj: "
                "python -m pip install -r mdd_forecasting/requirements-production.txt"
            ) from exc
        return CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="MAE",
            iterations=config.max_iter,
            depth=config.catboost_depth,
            learning_rate=config.learning_rate,
            l2_leaf_reg=5.0,
            random_seed=config.random_state,
            cat_features=spec.categorical,
            allow_writing_files=False,
            verbose=(
                config.model_progress_interval
                if config.model_progress_interval > 0
                else False
            ),
            thread_count=-1,
        )
    if config.model_backend != "hist_gradient_boosting":
        raise ValueError(f"Nieznany model_backend: {config.model_backend!r}")

    (
        ColumnTransformer,
        HistGradientBoostingRegressor,
        _,
        SimpleImputer,
        Pipeline,
        TargetEncoder,
    ) = _sklearn_components()
    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                TargetEncoder(
                    target_type="continuous",
                    smooth="auto",
                    cv=5,
                    shuffle=True,
                    random_state=config.random_state,
                ),
                spec.categorical,
            ),
            (
                "num",
                SimpleImputer(
                    strategy="constant",
                    fill_value=0.0,
                    add_indicator=True,
                    keep_empty_features=True,
                ),
                spec.numeric,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=config.random_state,
    )
    return Pipeline([("preprocess", preprocess), ("regressor", regressor)])


def _cap_training(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    # Najnowsze obserwacje są najbardziej reprezentatywne; cięcie pozostaje chronologiczne.
    return frame.sort_values("model_timestamp_utc").tail(max_rows)


def _expanding_splits(frame: pd.DataFrame, config: PipelineConfig) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    times = pd.to_datetime(
        frame.loc[frame["wartosc_rzeczywista"].notna(), "model_timestamp_utc"],
        errors="coerce",
        utc=True,
    ).dropna()
    if times.empty:
        return []
    end = times.max().floor("D") + pd.Timedelta(days=1)
    window = pd.Timedelta(days=config.validation_days)
    first_start = end - config.n_splits * window
    return [(first_start + fold * window, first_start + (fold + 1) * window) for fold in range(config.n_splits)]


def _fit_log_model(train: pd.DataFrame, spec: FeatureSpec, config: PipelineConfig):
    model = _make_model(spec, config)
    y = np.log1p(train["wartosc_rzeczywista"].clip(lower=0).astype(float))
    time_limit = None
    if config.model_backend == "catboost" and config.max_fit_seconds is not None:
        time_limit = _CatBoostTimeLimit(config.max_fit_seconds)
        model.fit(train[spec.all], y, callbacks=[time_limit])
    else:
        model.fit(train[spec.all], y)
    setattr(model, "_mdd_stopped_by_time", bool(time_limit and time_limit.stopped_by_time))
    return model


def _predict_original_scale(model, frame: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    log_prediction = model.predict(frame[spec.all])
    return np.expm1(log_prediction).clip(min=0)


def _safe_metric_values(actual: pd.Series, predicted: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    actual_np = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    predicted_np = pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(actual_np) & np.isfinite(predicted_np)
    return actual_np[mask], predicted_np[mask]


def _metric_row(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    y, p = _safe_metric_values(actual, predicted)
    if len(y) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "wape": np.nan,
            "smape": np.nan,
        }
    error = p - y
    denom_wape = np.abs(y).sum()
    denom_smape = np.abs(y) + np.abs(p)
    smape_terms = np.zeros_like(denom_smape, dtype=float)
    np.divide(
        2 * np.abs(error),
        denom_smape,
        out=smape_terms,
        where=denom_smape > 0,
    )
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "wape": float(np.abs(error).sum() / denom_wape) if denom_wape > 0 else np.nan,
        "smape": float(np.mean(smape_terms)),
    }


def _metrics_by_scope(frame: pd.DataFrame, prediction_col: str, model_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("GLOBAL", "ALL", frame)]
    scope_columns = [
        ("KIERUNEK", "kierunek_energii_norm"),
        ("MIASTO", "punkt"),
    ]
    if "weather_status" in frame.columns:
        scope_columns.append(("POGODA", "weather_status"))
    for scope_name, column in scope_columns:
        for value, group in frame.groupby(column, dropna=False):
            scopes.append((scope_name, str(value), group))
    for scope_name, scope_value, group in scopes:
        metric = _metric_row(group["wartosc_rzeczywista"], group[prediction_col])
        rows.append(
            {
                "model": model_name,
                "zakres": scope_name,
                "wartosc_zakresu": scope_value,
                **metric,
            }
        )

    client_metrics = []
    for _, group in frame.groupby(["klient_nazwa", "kierunek_energii_norm"], dropna=False):
        metric = _metric_row(group["wartosc_rzeczywista"], group[prediction_col])
        if metric["n"]:
            client_metrics.append(metric)
    if client_metrics:
        rows.append(
            {
                "model": model_name,
                "zakres": "MAKRO_KLIENT",
                "wartosc_zakresu": "średnia po klientach",
                "n": int(sum(metric["n"] for metric in client_metrics)),
                "mae": float(np.mean([metric["mae"] for metric in client_metrics])),
                "rmse": float(np.mean([metric["rmse"] for metric in client_metrics])),
                "bias": float(np.mean([metric["bias"] for metric in client_metrics])),
                "wape": float(np.nanmean([metric["wape"] for metric in client_metrics])),
                "smape": float(np.nanmean([metric["smape"] for metric in client_metrics])),
            }
        )
    return rows


def _baseline_for_fold(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    baseline = test["wartosc_bazowa"].copy()
    medians = train.groupby(
        ["klient_nazwa", "kierunek_energii_norm", "godzina"], dropna=False
    )["wartosc_rzeczywista"].median()
    keys = pd.MultiIndex.from_frame(
        test[["klient_nazwa", "kierunek_energii_norm", "godzina"]]
    )
    fallback = pd.Series(medians.reindex(keys).to_numpy(), index=test.index)
    global_median = float(train["wartosc_rzeczywista"].median())
    return baseline.fillna(fallback).fillna(global_median).clip(lower=0)


def run_forecasting(
    frame: pd.DataFrame,
    config: PipelineConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> ForecastResult:
    """Uruchamia expanding-window backtest i dopasowuje modele końcowe.

    `wartosc_przewidywana` zawiera wyłącznie predykcje out-of-time oraz przyszłe.
    `wartosc_model_pelny` daje techniczną predykcję dla każdego wiersza, ale dla historii
    jest in-sample i nie może służyć do oceny jakości.
    """

    def progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    progress("Tworzę kalendarz i 12 lagów D-3...D-14...")
    feature_started = time.monotonic()
    df, spec = build_features(frame, lag_days=config.lag_days)
    progress(f"Cechy gotowe po {time.monotonic() - feature_started:.1f} s.")
    actual_times = pd.to_datetime(
        df.loc[df["wartosc_rzeczywista"].notna(), "model_timestamp_utc"],
        errors="coerce",
        utc=True,
    ).dropna()
    forecast_cutoff = actual_times.max() if not actual_times.empty else pd.NaT
    missing_target = df["wartosc_rzeczywista"].isna()
    valid_model_time = df["model_timestamp_utc"].notna()
    if pd.notna(forecast_cutoff):
        future_mask = missing_target & valid_model_time & df["model_timestamp_utc"].gt(forecast_cutoff)
        historical_gap_mask = (
            missing_target & valid_model_time & df["model_timestamp_utc"].le(forecast_cutoff)
        )
    else:
        future_mask = pd.Series(False, index=df.index)
        historical_gap_mask = missing_target & valid_model_time
    df["wartosc_przewidywana"] = np.nan
    df["wartosc_bazowa_backtest"] = np.nan
    df["wartosc_model_pelny"] = np.nan
    df["prognoza_bez_pogody"] = False
    df["fold"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df["status_predykcji"] = "WARMUP_BEZ_OOF"
    df.loc[historical_gap_mask, "status_predykcji"] = "HISTORYCZNY_BRAK_TARGETU"
    df.loc[future_mask, "status_predykcji"] = "OCZEKUJE_NA_MODEL_PRZYSZLY"
    df.loc[~valid_model_time, "status_predykcji"] = "BRAK_POPRAWNEGO_CZASU_UTC"
    df.loc[df["kierunek_energii_norm"].isna(), "status_predykcji"] = "BRAK_KIERUNKU"

    directions = ["POBRANIE", "ODDANIE"]
    importances: list[dict[str, Any]] = []
    splits = _expanding_splits(df, config)
    _, _, permutation_importance, _, _, _ = _sklearn_components()

    for fold_no, (validation_start, validation_end) in enumerate(splits, start=1):
        for direction in directions:
            train_mask = (
                df["kierunek_energii_norm"].eq(direction)
                & df["wartosc_rzeczywista"].notna()
                & df["wartosc_rzeczywista"].ge(0)
                & df["model_timestamp_utc"].lt(validation_start)
            )
            test_mask = (
                df["kierunek_energii_norm"].eq(direction)
                & df["wartosc_rzeczywista"].notna()
                & df["wartosc_rzeczywista"].ge(0)
                & df["model_timestamp_utc"].ge(validation_start)
                & df["model_timestamp_utc"].lt(validation_end)
            )
            train = _cap_training(df.loc[train_mask], config.max_train_rows)
            test = df.loc[test_mask]
            if len(train) < config.min_train_rows or len(test) == 0:
                continue

            fit_started = time.monotonic()
            progress(
                f"Backtest {fold_no}/{len(splits)}, {direction}: "
                f"uczę na {len(train):,} wierszach..."
            )
            model = _fit_log_model(train, spec, config)
            progress(
                f"Backtest {fold_no}/{len(splits)}, {direction}: gotowe po "
                f"{time.monotonic() - fit_started:.1f} s"
                + (
                    " (zatrzymano limitem czasu)."
                    if getattr(model, "_mdd_stopped_by_time", False)
                    else "."
                )
            )
            prediction = _predict_original_scale(model, test, spec)
            df.loc[test.index, "wartosc_przewidywana"] = prediction
            df.loc[test.index, "wartosc_bazowa_backtest"] = _baseline_for_fold(train, test)
            df.loc[test.index, "fold"] = fold_no
            df.loc[test.index, "status_predykcji"] = "OOF_BACKTEST"

            # Permutacja na ograniczonej, chronologicznej próbce utrzymuje koszt pod kontrolą.
            importance_test = test.sort_values("model_timestamp_utc").tail(config.max_importance_rows)
            if config.compute_importance and len(importance_test) >= 50:
                result = permutation_importance(
                    model,
                    importance_test[spec.all],
                    np.log1p(importance_test["wartosc_rzeczywista"].clip(lower=0)),
                    scoring="neg_mean_absolute_error",
                    n_repeats=2,
                    random_state=config.random_state,
                    n_jobs=1,
                )
                for feature, mean_value, std_value in zip(
                    spec.all, result.importances_mean, result.importances_std
                ):
                    importances.append(
                        {
                            "kierunek": direction,
                            "fold": fold_no,
                            "cecha": feature,
                            "waznosc_permutacyjna_log_mae": float(mean_value),
                            "odchylenie": float(std_value),
                        }
                    )

    models: dict[str, Any] = {}
    for direction in directions:
        train_mask = (
            df["kierunek_energii_norm"].eq(direction)
            & df["wartosc_rzeczywista"].notna()
            & df["wartosc_rzeczywista"].ge(0)
            & df["model_timestamp_utc"].notna()
        )
        train = _cap_training(df.loc[train_mask], config.max_train_rows)
        predict_mask = df["kierunek_energii_norm"].eq(direction) & df[
            "model_timestamp_utc"
        ].notna()
        predict_frame = df.loc[predict_mask]
        if len(train) < config.min_train_rows or predict_frame.empty:
            continue
        fit_started = time.monotonic()
        progress(
            f"Model końcowy {direction}: uczę na {len(train):,} wierszach..."
        )
        model = _fit_log_model(train, spec, config)
        progress(
            f"Model końcowy {direction}: gotowe po "
            f"{time.monotonic() - fit_started:.1f} s"
            + (
                " (zatrzymano limitem czasu)."
                if getattr(model, "_mdd_stopped_by_time", False)
                else "."
            )
        )
        models[direction] = model
        full_prediction = _predict_original_scale(model, predict_frame, spec)
        df.loc[predict_frame.index, "wartosc_model_pelny"] = full_prediction
        future_index = predict_frame.index[future_mask.reindex(predict_frame.index, fill_value=False)]
        if len(future_index):
            df.loc[future_index, "wartosc_przewidywana"] = df.loc[
                future_index, "wartosc_model_pelny"
            ]
            df.loc[future_index, "status_predykcji"] = "PROGNOZA_PRZYSZLA"
            matched_weather = df.get(
                "pogoda_dopasowana", pd.Series(False, index=df.index)
            ).fillna(False)
            df.loc[future_index, "prognoza_bez_pogody"] = ~matched_weather.loc[
                future_index
            ].astype(bool)

    evaluated = df[df["status_predykcji"].eq("OOF_BACKTEST")].copy()
    metric_rows: list[dict[str, Any]] = []
    if not evaluated.empty:
        metric_rows.extend(_metrics_by_scope(evaluated, "wartosc_przewidywana", "ML_OOF"))
        baseline_eval = evaluated[evaluated["wartosc_bazowa_backtest"].notna()]
        metric_rows.extend(
            _metrics_by_scope(
                baseline_eval,
                "wartosc_bazowa_backtest",
                "SREDNIA_ANALOGICZNYCH_D3_D14",
            )
        )
    metrics = pd.DataFrame(
        metric_rows,
        columns=[
            "model",
            "zakres",
            "wartosc_zakresu",
            "n",
            "mae",
            "rmse",
            "bias",
            "wape",
            "smape",
        ],
    )

    if importances:
        importance_raw = pd.DataFrame(importances)
        importance = (
            importance_raw.groupby(["kierunek", "cecha"], as_index=False)
            .agg(
                waznosc_permutacyjna_log_mae=("waznosc_permutacyjna_log_mae", "mean"),
                odchylenie_miedzy_foldami=("waznosc_permutacyjna_log_mae", "std"),
                liczba_foldow=("fold", "nunique"),
            )
            .sort_values(["kierunek", "waznosc_permutacyjna_log_mae"], ascending=[True, False])
        )
    else:
        importance = pd.DataFrame(
            columns=[
                "kierunek",
                "cecha",
                "waznosc_permutacyjna_log_mae",
                "odchylenie_miedzy_foldami",
                "liczba_foldow",
            ]
        )

    df["blad"] = df["wartosc_przewidywana"] - df["wartosc_rzeczywista"]
    df["blad_bezwzgledny"] = df["blad"].abs()
    df["model_pelny_jest_insample"] = df["wartosc_rzeczywista"].notna()
    df["forecast_cutoff_utc"] = forecast_cutoff
    return ForecastResult(
        predictions=df,
        metrics=metrics,
        feature_importance=importance,
        models=models,
        feature_spec=spec,
    )

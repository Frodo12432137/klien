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


@dataclass(frozen=True)
class BlendSelection:
    alpha: float
    n: int
    baseline_mae: float
    hybrid_mae: float
    improvement: float
    reason: str


@dataclass
class HybridModelArtifact:
    estimator: Any
    alpha: float
    calibration_n: int
    calibration_improvement: float
    calibration_reason: str
    train_rows: int = 0
    history_rows: int = 0
    train_start_utc: str | None = None
    train_end_utc: str | None = None
    target: str = "residuum_wzgledem_D3_D14"


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
            # Model uczy korektę do mocnego baseline. MAE jest odporniejsze na
            # pojedyncze skoki niż RMSE i nie wymaga log1p, które zaniżało wolumen.
            loss_function="MAE",
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
        loss="absolute_error",
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


def _cap_training(frame: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None:
        return frame
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


def _fit_residual_model(
    train: pd.DataFrame, spec: FeatureSpec, config: PipelineConfig
):
    usable = train[
        train["wartosc_rzeczywista"].notna()
        & train["wartosc_bazowa"].notna()
    ]
    if len(usable) < config.min_train_rows:
        raise ValueError(
            "Za mało wierszy z naturalnym baseline D-3...D-14 do uczenia korekty."
        )
    model = _make_model(spec, config)
    y = (
        usable["wartosc_rzeczywista"].astype(float)
        - usable["wartosc_bazowa"].astype(float)
    )
    time_limit = None
    if config.model_backend == "catboost" and config.max_fit_seconds is not None:
        time_limit = _CatBoostTimeLimit(config.max_fit_seconds)
        model.fit(usable[spec.all], y, callbacks=[time_limit])
    else:
        model.fit(usable[spec.all], y)
    setattr(model, "_mdd_stopped_by_time", bool(time_limit and time_limit.stopped_by_time))
    return model


def _predict_residual_correction(
    model, frame: pd.DataFrame, spec: FeatureSpec
) -> np.ndarray:
    return np.asarray(model.predict(frame[spec.all]), dtype=float)


def _blend_prediction(
    baseline: pd.Series | np.ndarray,
    correction: pd.Series | np.ndarray,
    alpha: float,
) -> np.ndarray:
    baseline_np = np.asarray(baseline, dtype=float)
    correction_np = np.asarray(correction, dtype=float)
    return np.clip(baseline_np + float(alpha) * correction_np, 0.0, None)


def _select_blend_alpha(
    actual: pd.Series | np.ndarray,
    baseline: pd.Series | np.ndarray,
    correction: pd.Series | np.ndarray,
    *,
    min_rows: int,
    min_improvement: float,
    grid_steps: int,
) -> BlendSelection:
    actual_np = np.asarray(actual, dtype=float)
    baseline_np = np.asarray(baseline, dtype=float)
    correction_np = np.asarray(correction, dtype=float)
    mask = (
        np.isfinite(actual_np)
        & np.isfinite(baseline_np)
        & np.isfinite(correction_np)
    )
    actual_np = actual_np[mask]
    baseline_np = baseline_np[mask]
    correction_np = correction_np[mask]
    n = int(len(actual_np))
    if n < int(min_rows):
        return BlendSelection(0.0, n, np.nan, np.nan, 0.0, "ZA_MALO_KALIBRACJI")

    baseline_prediction = np.clip(baseline_np, 0.0, None)
    baseline_mae = float(np.mean(np.abs(baseline_prediction - actual_np)))
    if not np.isfinite(baseline_mae) or baseline_mae <= 0:
        return BlendSelection(
            0.0,
            n,
            baseline_mae,
            baseline_mae,
            0.0,
            "BASELINE_IDEALNY_LUB_NIEPOPRAWNY",
        )

    steps = max(int(grid_steps), 2)
    alphas = np.linspace(0.0, 1.0, steps)
    maes = np.asarray(
        [
            np.mean(
                np.abs(_blend_prediction(baseline_np, correction_np, alpha) - actual_np)
            )
            for alpha in alphas
        ],
        dtype=float,
    )
    best_index = int(np.nanargmin(maes))
    best_alpha = float(alphas[best_index])
    best_mae = float(maes[best_index])
    improvement = float((baseline_mae - best_mae) / baseline_mae)
    if best_alpha <= 0 or improvement < float(min_improvement):
        return BlendSelection(
            0.0,
            n,
            baseline_mae,
            baseline_mae,
            max(improvement, 0.0),
            "BRAK_POTWIERDZONEJ_POPRAWY",
        )
    return BlendSelection(
        best_alpha,
        n,
        baseline_mae,
        best_mae,
        improvement,
        "HYBRYDA_WYBRANA",
    )


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
    """Baseline bez leakage: naturalne lagi, potem mediany tylko z prefiksu train."""

    baseline = pd.to_numeric(test["wartosc_bazowa"], errors="coerce").copy()
    usable_train = train[train["wartosc_rzeczywista"].notna()]

    def grouped_fallback(columns: list[str]) -> pd.Series:
        medians = usable_train.groupby(columns, dropna=False)[
            "wartosc_rzeczywista"
        ].median()
        keys = pd.MultiIndex.from_frame(test[columns])
        values = medians.reindex(keys).to_numpy()
        return pd.Series(values, index=test.index, dtype=float)

    # Kolejność jest celowa: najpierw profil konkretnego klienta, później coraz
    # szersze grupy. Prawidłowe zero (np. PV nocą) nigdy nie jest traktowane jak brak.
    for columns in (
        ["klient_nazwa", "kierunek_energii_norm", "godzina"],
        ["oddzial_code", "kierunek_energii_norm", "godzina"],
        ["kierunek_energii_norm", "godzina"],
    ):
        baseline = baseline.fillna(grouped_fallback(columns))

    direction_median = usable_train.groupby("kierunek_energii_norm", dropna=False)[
        "wartosc_rzeczywista"
    ].median()
    baseline = baseline.fillna(test["kierunek_energii_norm"].map(direction_median))
    global_median = pd.to_numeric(
        usable_train["wartosc_rzeczywista"], errors="coerce"
    ).median()
    if pd.isna(global_median):
        global_median = 0.0
    return baseline.fillna(float(global_median)).clip(lower=0).astype(float)


def _split_calibration_window(
    train: pd.DataFrame,
    validation_start: pd.Timestamp,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rozdziela prefiks na uczenie i późniejszą kalibrację alfy."""

    calibration_start = validation_start - pd.Timedelta(days=config.calibration_days)
    timestamps = pd.to_datetime(train["model_timestamp_utc"], errors="coerce", utc=True)
    fit = train.loc[timestamps.lt(calibration_start)]
    calibration = train.loc[
        timestamps.ge(calibration_start) & timestamps.lt(validation_start)
    ]
    return fit, calibration


def _fallback_selection(reason: str, n: int = 0) -> BlendSelection:
    return BlendSelection(
        alpha=0.0,
        n=int(n),
        baseline_mae=np.nan,
        hybrid_mae=np.nan,
        improvement=0.0,
        reason=reason,
    )


def run_forecasting(
    frame: pd.DataFrame,
    config: PipelineConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> ForecastResult:
    """Uruchamia uczciwy backtest hybrydy baseline + korekta residualna.

    Model residualny nie zastępuje średniej D-3...D-14. Uczy się jedynie jej
    poprawki, a udział poprawki (`alpha`) jest wybierany na oknie kalibracyjnym
    wcześniejszym od ocenianego OOF. Brak potwierdzonej poprawy oznacza alpha=0.
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
        future_mask = (
            missing_target
            & valid_model_time
            & df["model_timestamp_utc"].gt(forecast_cutoff)
        )
        historical_gap_mask = (
            missing_target
            & valid_model_time
            & df["model_timestamp_utc"].le(forecast_cutoff)
        )
    else:
        future_mask = pd.Series(False, index=df.index)
        historical_gap_mask = missing_target & valid_model_time

    for column in (
        "wartosc_przewidywana",
        "wartosc_bazowa_backtest",
        "wartosc_model_pelny",
        "residuum_rzeczywiste",
        "korekta_ml_surowa",
        "wartosc_ml_przed_blendem",
        "blend_alpha",
        "kalibracja_poprawa_mae",
    ):
        df[column] = np.nan
    df["kalibracja_n"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df["strategia_predykcji"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df["kalibracja_powod"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df["fit_zatrzymany_limitem"] = pd.Series(
        pd.NA, index=df.index, dtype="boolean"
    )
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
    permutation_importance = None
    if config.compute_importance:
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
            history = df.loc[train_mask]
            test = df.loc[test_mask]
            if len(history) < config.min_train_rows or test.empty:
                continue

            baseline_test = _baseline_for_fold(history, test)
            fit_prefix, calibration = _split_calibration_window(
                history, validation_start, config
            )
            fit_train = _cap_training(fit_prefix, config.max_train_rows)
            usable_fit_rows = int(
                (
                    fit_train["wartosc_rzeczywista"].notna()
                    & fit_train["wartosc_bazowa"].notna()
                ).sum()
            )
            model = None
            correction_test = np.zeros(len(test), dtype=float)
            selection = _fallback_selection("ZA_MALO_DANYCH_DO_MODELU")
            stopped_by_time = False

            if usable_fit_rows >= config.min_train_rows:
                fit_started = time.monotonic()
                progress(
                    f"Backtest {fold_no}/{len(splits)}, {direction}: uczę korektę "
                    f"na {usable_fit_rows:,} wierszach; kalibracja {len(calibration):,}..."
                )
                model = _fit_residual_model(fit_train, spec, config)
                stopped_by_time = bool(
                    getattr(model, "_mdd_stopped_by_time", False)
                )
                progress(
                    f"Backtest {fold_no}/{len(splits)}, {direction}: gotowe po "
                    f"{time.monotonic() - fit_started:.1f} s"
                    + (
                        " (zatrzymano limitem czasu)."
                        if stopped_by_time
                        else "."
                    )
                )
                if not calibration.empty:
                    calibration_baseline = _baseline_for_fold(
                        fit_prefix, calibration
                    )
                    calibration_correction = _predict_residual_correction(
                        model, calibration, spec
                    )
                    selection = _select_blend_alpha(
                        calibration["wartosc_rzeczywista"],
                        calibration_baseline,
                        calibration_correction,
                        min_rows=config.min_calibration_rows,
                        min_improvement=config.min_blend_improvement,
                        grid_steps=config.blend_grid_steps,
                    )
                else:
                    selection = _fallback_selection("BRAK_OKNA_KALIBRACYJNEGO")
                correction_test = _predict_residual_correction(model, test, spec)
            else:
                progress(
                    f"Backtest {fold_no}/{len(splits)}, {direction}: za mało danych "
                    "przed kalibracją — używam baseline."
                )

            raw_candidate = _blend_prediction(baseline_test, correction_test, 1.0)
            prediction = _blend_prediction(
                baseline_test, correction_test, selection.alpha
            )
            strategy = (
                "HYBRYDA"
                if selection.alpha > 0
                else "BASELINE_FALLBACK"
            )
            df.loc[test.index, "wartosc_przewidywana"] = prediction
            df.loc[test.index, "wartosc_bazowa_backtest"] = baseline_test
            df.loc[test.index, "residuum_rzeczywiste"] = (
                test["wartosc_rzeczywista"].astype(float).to_numpy()
                - baseline_test.to_numpy(dtype=float)
            )
            df.loc[test.index, "korekta_ml_surowa"] = correction_test
            df.loc[test.index, "wartosc_ml_przed_blendem"] = raw_candidate
            df.loc[test.index, "blend_alpha"] = selection.alpha
            df.loc[test.index, "strategia_predykcji"] = strategy
            df.loc[test.index, "kalibracja_n"] = selection.n
            df.loc[test.index, "kalibracja_poprawa_mae"] = selection.improvement
            df.loc[test.index, "kalibracja_powod"] = selection.reason
            df.loc[test.index, "fit_zatrzymany_limitem"] = stopped_by_time
            df.loc[test.index, "fold"] = fold_no
            df.loc[test.index, "status_predykcji"] = "OOF_BACKTEST"

            # Ważność dotyczy korekty residualnej, nie pełnego poziomu energii.
            importance_test = (
                test[test["wartosc_bazowa"].notna()]
                .sort_values("model_timestamp_utc")
                .tail(config.max_importance_rows)
            )
            if (
                model is not None
                and config.compute_importance
                and len(importance_test) >= 50
            ):
                assert permutation_importance is not None
                residual_target = (
                    importance_test["wartosc_rzeczywista"].astype(float)
                    - importance_test["wartosc_bazowa"].astype(float)
                )
                result = permutation_importance(
                    model,
                    importance_test[spec.all],
                    residual_target,
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
                            "waznosc_permutacyjna_residuum_mae": float(mean_value),
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
        history = df.loc[train_mask]
        train = _cap_training(history, config.max_train_rows)
        predict_mask = df["kierunek_energii_norm"].eq(direction) & df[
            "model_timestamp_utc"
        ].notna()
        if not config.score_full_history:
            predict_mask &= future_mask
        predict_frame = df.loc[predict_mask]

        oof = df[
            df["kierunek_energii_norm"].eq(direction)
            & df["status_predykcji"].eq("OOF_BACKTEST")
        ]
        if oof.empty:
            final_selection = _fallback_selection("BRAK_OOF_DO_KALIBRACJI_FINALNEJ")
        else:
            final_selection = _select_blend_alpha(
                oof["wartosc_rzeczywista"],
                oof["wartosc_bazowa_backtest"],
                oof["korekta_ml_surowa"],
                min_rows=config.min_calibration_rows,
                min_improvement=config.min_blend_improvement,
                grid_steps=config.blend_grid_steps,
            )

        usable_train_rows = int(
            (
                train["wartosc_rzeczywista"].notna()
                & train["wartosc_bazowa"].notna()
            ).sum()
        )
        estimator_train = train.loc[
            train["wartosc_rzeczywista"].notna()
            & train["wartosc_bazowa"].notna()
        ]
        final_model = None
        stopped_by_time = False
        if usable_train_rows >= config.min_train_rows:
            fit_started = time.monotonic()
            progress(
                f"Model końcowy {direction}: uczę korektę na "
                f"{usable_train_rows:,} wierszach; alpha={final_selection.alpha:.2f}..."
            )
            final_model = _fit_residual_model(train, spec, config)
            stopped_by_time = bool(
                getattr(final_model, "_mdd_stopped_by_time", False)
            )
            progress(
                f"Model końcowy {direction}: gotowe po "
                f"{time.monotonic() - fit_started:.1f} s"
                + (
                    " (zatrzymano limitem czasu)."
                    if stopped_by_time
                    else "."
                )
            )
        else:
            final_selection = _fallback_selection(
                "ZA_MALO_DANYCH_DO_MODELU_KONCOWEGO",
                n=final_selection.n,
            )
            progress(
                f"Model końcowy {direction}: brak wystarczających lagów; "
                "prognoza pozostaje baseline."
            )

        artifact = HybridModelArtifact(
            estimator=final_model,
            alpha=final_selection.alpha,
            calibration_n=final_selection.n,
            calibration_improvement=final_selection.improvement,
            calibration_reason=final_selection.reason,
            train_rows=usable_train_rows if final_model is not None else 0,
            history_rows=len(history),
            train_start_utc=(
                None
                if final_model is None or estimator_train.empty
                else str(
                    pd.to_datetime(
                        estimator_train["model_timestamp_utc"],
                        errors="coerce",
                        utc=True,
                    ).min()
                )
            ),
            train_end_utc=(
                None
                if final_model is None or estimator_train.empty
                else str(
                    pd.to_datetime(
                        estimator_train["model_timestamp_utc"],
                        errors="coerce",
                        utc=True,
                    ).max()
                )
            ),
        )
        models[direction] = artifact

        if predict_frame.empty:
            continue

        baseline_full = _baseline_for_fold(history, predict_frame)
        if final_model is None:
            correction_full = np.zeros(len(predict_frame), dtype=float)
        else:
            correction_full = _predict_residual_correction(
                final_model, predict_frame, spec
            )
        raw_full = _blend_prediction(baseline_full, correction_full, 1.0)
        full_prediction = _blend_prediction(
            baseline_full, correction_full, final_selection.alpha
        )
        df.loc[predict_frame.index, "wartosc_model_pelny"] = full_prediction

        future_index = predict_frame.index[
            future_mask.reindex(predict_frame.index, fill_value=False)
        ]
        if len(future_index):
            positions = predict_frame.index.get_indexer(future_index)
            df.loc[future_index, "wartosc_bazowa_backtest"] = baseline_full.loc[
                future_index
            ]
            df.loc[future_index, "korekta_ml_surowa"] = correction_full[positions]
            df.loc[future_index, "wartosc_ml_przed_blendem"] = raw_full[positions]
            df.loc[future_index, "blend_alpha"] = final_selection.alpha
            df.loc[future_index, "strategia_predykcji"] = (
                "HYBRYDA_FINALNA"
                if final_selection.alpha > 0
                else "BASELINE_FALLBACK_FINALNY"
            )
            df.loc[future_index, "kalibracja_n"] = final_selection.n
            df.loc[future_index, "kalibracja_poprawa_mae"] = (
                final_selection.improvement
            )
            df.loc[future_index, "kalibracja_powod"] = final_selection.reason
            df.loc[future_index, "fit_zatrzymany_limitem"] = stopped_by_time
            df.loc[future_index, "wartosc_przewidywana"] = full_prediction[positions]
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
        metric_rows.extend(
            _metrics_by_scope(evaluated, "wartosc_przewidywana", "HYBRYDA_OOF")
        )
        metric_rows.extend(
            _metrics_by_scope(
                evaluated,
                "wartosc_ml_przed_blendem",
                "KOREKTA_ML_SUROWA_OOF",
            )
        )
        metric_rows.extend(
            _metrics_by_scope(
                evaluated,
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

    importance_column = "waznosc_permutacyjna_residuum_mae"
    if importances:
        importance_raw = pd.DataFrame(importances)
        importance = (
            importance_raw.groupby(["kierunek", "cecha"], as_index=False)
            .agg(
                waznosc_permutacyjna_residuum_mae=(importance_column, "mean"),
                odchylenie_miedzy_foldami=(importance_column, "std"),
                liczba_foldow=("fold", "nunique"),
            )
            .sort_values(
                ["kierunek", importance_column], ascending=[True, False]
            )
        )
    else:
        importance = pd.DataFrame(
            columns=[
                "kierunek",
                "cecha",
                importance_column,
                "odchylenie_miedzy_foldami",
                "liczba_foldow",
            ]
        )

    df["blad"] = df["wartosc_przewidywana"] - df["wartosc_rzeczywista"]
    df["blad_bezwzgledny"] = df["blad"].abs()
    df["model_pelny_jest_insample"] = (
        df["wartosc_rzeczywista"].notna()
        & df["wartosc_model_pelny"].notna()
    )
    df["forecast_cutoff_utc"] = forecast_cutoff
    return ForecastResult(
        predictions=df,
        metrics=metrics,
        feature_importance=importance,
        models=models,
        feature_spec=spec,
    )

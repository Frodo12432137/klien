from __future__ import annotations

import platform
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PipelineConfig
from .features import FeatureSpec
from .model import ForecastResult, HybridModelArtifact


BUNDLE_SCHEMA_VERSION = 1


@dataclass
class ForecastBundle:
    """Kompletny, wersjonowany stan potrzebny późniejszemu scoringowi.

    Plik joblib należy ładować wyłącznie z zaufanego źródła. Sam predyktor zostanie
    dodany jako osobny etap; bundle już teraz zamraża modele, cechy, mapowanie,
    historię lagów i fallbacki, więc scoring nie będzie wymagał ponownego uczenia.
    """

    schema_version: int
    created_at_utc: str
    training_cutoff_utc: str | None
    models: dict[str, HybridModelArtifact]
    feature_spec: FeatureSpec
    config: PipelineConfig
    history_tail: pd.DataFrame
    fallback_profiles: dict[str, Any]
    branch_mapping: pd.DataFrame
    oof_metrics: pd.DataFrame
    metadata: dict[str, Any]

    def validate(self) -> None:
        if self.schema_version != BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                "Nieobsługiwana wersja pakietu modelu: "
                f"{self.schema_version}; oczekiwano {BUNDLE_SCHEMA_VERSION}."
            )
        missing_directions = {"POBRANIE", "ODDANIE"} - set(self.models)
        if missing_directions:
            raise ValueError(
                "Pakiet nie zawiera modeli dla kierunków: "
                f"{sorted(missing_directions)}"
            )
        for direction, artifact in self.models.items():
            if not isinstance(artifact, HybridModelArtifact):
                raise TypeError(f"Niepoprawny artefakt modelu dla {direction}.")
            alpha = float(artifact.alpha)
            if pd.isna(alpha) or not 0.0 <= alpha <= 1.0:
                raise ValueError(f"Niepoprawne alpha modelu {direction}: {alpha}.")
            if alpha > 0.0 and artifact.estimator is None:
                raise ValueError(
                    f"Model {direction} ma alpha > 0, ale nie zawiera estymatora."
                )
            if artifact.train_rows < 0 or artifact.history_rows < artifact.train_rows:
                raise ValueError(
                    f"Niespójne liczby wierszy treningowych dla {direction}."
                )

        feature_names = self.feature_spec.all
        if not feature_names or len(feature_names) != len(set(feature_names)):
            raise ValueError("Pakiet ma pustą albo niejednoznaczną listę cech.")

        required_history = {
            "oddzial_code",
            "klient_nazwa",
            "kierunek_energii_norm",
            "doba_handlowa",
            "godzina_handlowa",
            "model_timestamp_utc",
            "wartosc_rzeczywista",
        }
        missing_history = required_history - set(self.history_tail.columns)
        if missing_history:
            raise ValueError(
                "Historia w pakiecie nie zawiera kolumn: "
                f"{sorted(missing_history)}"
            )
        if self.history_tail.empty:
            raise ValueError("Pakiet nie zawiera historii potrzebnej do lagów.")
        required_fallbacks = {
            "klient_kierunek_godzina",
            "oddzial_kierunek_godzina",
            "kierunek_godzina",
            "kierunek",
            "globalna_mediana",
        }
        missing_fallbacks = required_fallbacks - set(self.fallback_profiles)
        if missing_fallbacks:
            raise ValueError(
                "Pakiet nie zawiera profili fallback: "
                f"{sorted(missing_fallbacks)}"
            )
        if not {"oddzial_code", "punkt"}.issubset(self.branch_mapping.columns):
            raise ValueError("Mapowanie w pakiecie nie zawiera oddzial_code i punkt.")
        if self.training_cutoff_utc is None:
            raise ValueError("Pakiet nie ma daty końca danych treningowych.")
        try:
            cutoff = pd.Timestamp(self.training_cutoff_utc)
        except (TypeError, ValueError) as exc:
            raise ValueError("Niepoprawna data końca treningu w pakiecie.") from exc
        if pd.isna(cutoff) or cutoff.tzinfo is None:
            raise ValueError("Data końca treningu musi być poprawnym czasem UTC.")


def _dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "joblib", "catboost", "scikit-learn"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "niezainstalowany"
    return versions


def _fallback_profiles(history: pd.DataFrame) -> dict[str, Any]:
    usable = history[history["wartosc_rzeczywista"].notna()].copy()

    def grouped(columns: list[str]) -> pd.DataFrame:
        return (
            usable.groupby(columns, dropna=False, as_index=False)[
                "wartosc_rzeczywista"
            ]
            .median()
            .rename(columns={"wartosc_rzeczywista": "wartosc_mediana"})
        )

    global_median = pd.to_numeric(
        usable["wartosc_rzeczywista"], errors="coerce"
    ).median()
    return {
        "klient_kierunek_godzina": grouped(
            ["klient_nazwa", "kierunek_energii_norm", "godzina"]
        ),
        "oddzial_kierunek_godzina": grouped(
            ["oddzial_code", "kierunek_energii_norm", "godzina"]
        ),
        "kierunek_godzina": grouped(["kierunek_energii_norm", "godzina"]),
        "kierunek": grouped(["kierunek_energii_norm"]),
        "globalna_mediana": (
            0.0 if pd.isna(global_median) else float(global_median)
        ),
    }


def build_forecast_bundle(
    result: ForecastResult,
    config: PipelineConfig,
    branch_mapping: pd.DataFrame,
) -> ForecastBundle:
    predictions = result.predictions
    timestamps = pd.to_datetime(
        predictions["model_timestamp_utc"], errors="coerce", utc=True
    )
    actual_mask = predictions["wartosc_rzeczywista"].notna() & timestamps.notna()
    cutoff = timestamps.loc[actual_mask].max() if actual_mask.any() else pd.NaT

    # D-14 wymaga 14 dni historii; dodatkowy tydzień bufora ułatwi późniejszy
    # scoring kilku kolejnych uruchomień bez powiększania bundle do całego zbioru.
    history_days = max(config.lag_days) + 7
    if pd.isna(cutoff):
        history_mask = pd.Series(False, index=predictions.index)
    else:
        history_mask = actual_mask & timestamps.ge(
            cutoff - pd.Timedelta(days=history_days)
        )
    history_columns = [
        "oddzial_code",
        "punkt",
        "grupa",
        "klient_nazwa",
        "kierunek_energii_norm",
        "rodzaj",
        "doba_handlowa",
        "godzina_handlowa",
        "valid_timestamp",
        "model_timestamp_utc",
        "wartosc_rzeczywista",
    ]
    history_tail = predictions.loc[
        history_mask, [col for col in history_columns if col in predictions.columns]
    ].copy()
    fallback_columns = [
        "oddzial_code",
        "klient_nazwa",
        "kierunek_energii_norm",
        "godzina",
        "wartosc_rzeczywista",
    ]
    full_history = predictions.loc[actual_mask, fallback_columns]

    bundle = ForecastBundle(
        schema_version=BUNDLE_SCHEMA_VERSION,
        created_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        training_cutoff_utc=None if pd.isna(cutoff) else cutoff.isoformat(),
        models=result.models,
        feature_spec=result.feature_spec,
        config=config,
        history_tail=history_tail,
        fallback_profiles=_fallback_profiles(full_history),
        branch_mapping=branch_mapping.copy(),
        oof_metrics=result.metrics.copy(),
        metadata={
            "bundle_filename": "pakiet_modelu_mdd.joblib",
            "history_buffer_days": history_days,
            "input_rows": int(len(predictions)),
            "actual_rows": int(actual_mask.sum()),
            "history_tail_rows": int(len(history_tail)),
            "forecast_must_be_after_utc": (
                None if pd.isna(cutoff) else cutoff.isoformat()
            ),
            "mapping_source_for_scoring": "embedded_branch_mapping",
            "final_alpha_selected_from_historical_oof": True,
            "target_column": "wartosc_rzeczywista",
            "target_unit": "taka sama jak kolumna Wartość w pliku wejściowym",
            "training_by_direction": {
                direction: {
                    "train_rows": int(artifact.train_rows),
                    "history_rows": int(artifact.history_rows),
                    "train_start_utc": artifact.train_start_utc,
                    "train_end_utc": artifact.train_end_utc,
                    "alpha": float(artifact.alpha),
                    "calibration_n": int(artifact.calibration_n),
                    "calibration_reason": artifact.calibration_reason,
                }
                for direction, artifact in sorted(result.models.items())
            },
            "dependencies": _dependency_versions(),
            "trusted_joblib_only": True,
        },
    )
    bundle.validate()
    return bundle


def load_forecast_bundle(path: str | Path) -> ForecastBundle:
    """Ładuje wyłącznie zaufany pakiet joblib i sprawdza jego wersję."""

    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Do odczytu pakietu modelu wymagany jest joblib.") from exc
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, ForecastBundle):
        raise TypeError("Wskazany plik nie jest pakietem modelu MDD.")
    bundle.validate()
    return bundle

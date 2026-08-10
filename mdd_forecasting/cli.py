from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .artifacts import build_forecast_bundle
from .config import DEFAULT_MAPPING_PATH, PipelineConfig, WEATHER_FEATURES
from .database import (
    DEFAULT_SQL_DATABASE,
    DEFAULT_SQL_DRIVER,
    DEFAULT_SQL_SERVER,
    SqlServerSettings,
    query_weather_sql,
)
from .io import (
    join_prepared_energy_with_weather,
    prepare_energy_dataset,
    prepare_joined_dataset,
)
from .model import run_forecasting
from .report import write_results_workbook


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model godzinowego poboru i oddania MDD z prognozą pogody."
    )
    parser.add_argument("--energy", required=True, help="Excel/CSV z kolumnami A:I ze zdjęcia.")
    weather_source = parser.add_mutually_exclusive_group(required=True)
    weather_source.add_argument(
        "--weather", help="CSV/XLSX z eksportem vPogodaPrognoza."
    )
    weather_source.add_argument(
        "--weather-sql",
        help="Plik SQL wykonywany bezpośrednio przez pyodbc w PGESA_MarketAnalytics.",
    )
    parser.add_argument("--output-dir", required=True, help="Katalog na wyniki CSV i modele.")
    parser.add_argument(
        "--mapping",
        default=str(DEFAULT_MAPPING_PATH),
        help="Edytowalny słownik oddział -> punkt pogodowy.",
    )
    parser.add_argument(
        "--min-lead-hours",
        type=int,
        default=24,
        help="Minimalny odstęp między wydaniem pogody a godziną ważności (domyślnie 24 h).",
    )
    parser.add_argument(
        "--weather-available-from",
        default="2024-10-01",
        help=(
            "Pierwsza doba, dla której prognoza pogody jest dostępna. Starsze wiersze "
            "energii pozostają w modelu bez cech pogodowych (domyślnie 2024-10-01)."
        ),
    )
    parser.add_argument("--validation-days", type=int, default=14)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument(
        "--calibration-days",
        type=int,
        default=7,
        help=(
            "Liczba najnowszych dni historii przed oknem OOF używanych do "
            "kalibracji blendu z baseline."
        ),
    )
    parser.add_argument(
        "--min-calibration-rows",
        type=int,
        default=200,
        help="Minimalna liczba rekordów kalibracyjnych osobno dla kierunku.",
    )
    parser.add_argument(
        "--min-blend-improvement",
        type=float,
        default=0.02,
        help=(
            "Minimalna względna poprawa MAE względem baseline wymagana do włączenia "
            "korekty ML (domyślnie 0.02 = 2%%)."
        ),
    )
    parser.add_argument(
        "--blend-grid-steps",
        type=int,
        default=21,
        help="Liczba sprawdzanych wartości alpha blendu w przedziale 0..1.",
    )
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Opcjonalny limit treningu na kierunek; brak flagi = wszystkie wiersze.",
    )
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument(
        "--execution-profile",
        choices=["standard", "fast_30min", "full_training"],
        default="standard",
    )
    parser.add_argument(
        "--model-backend",
        choices=["hist_gradient_boosting", "catboost"],
        default="hist_gradient_boosting",
    )
    parser.add_argument("--catboost-depth", type=int, default=8)
    parser.add_argument("--max-input-rows", type=int)
    parser.add_argument(
        "--input-row-selection",
        choices=["head", "tail"],
        default="head",
        help="Przy limicie wejścia wybierz pierwsze albo najnowsze wiersze.",
    )
    parser.add_argument(
        "--max-fit-minutes",
        type=float,
        help="Limit czasu pojedynczego fitu CatBoost; sprawdzany po każdej iteracji.",
    )
    parser.add_argument("--model-progress-interval", type=int, default=0)
    parser.add_argument(
        "--skip-full-history-score",
        action="store_true",
        help=(
            "Po treningu nie licz kosztownej predykcji in-sample całej historii; "
            "OOF i przyszłość pozostają bez zmian."
        ),
    )
    parser.add_argument(
        "--oof-output-only",
        action="store_true",
        help="Eksportuj wiersze predykcji tylko dla OOF i przyszłości.",
    )
    parser.add_argument("--skip-importance", action="store_true")
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="Zapisz w predykcjach tylko najważniejsze kolumny, aby skrócić eksport.",
    )
    parser.add_argument(
        "--weather-already-vintaged",
        action="store_true",
        help="Użyj tylko, gdy SQL już wybrał poprawny historyczny vintage bez leakage.",
    )
    parser.add_argument("--valid-from", help="Początek valid time SQL; domyślnie min daty Excela.")
    parser.add_argument(
        "--valid-to-exclusive",
        help="Koniec valid time SQL bez tej chwili; domyślnie dzień po max dacie Excela.",
    )
    parser.add_argument("--sql-server", default=DEFAULT_SQL_SERVER)
    parser.add_argument("--sql-database", default=DEFAULT_SQL_DATABASE)
    parser.add_argument("--sql-driver", default=DEFAULT_SQL_DRIVER)
    parser.add_argument("--sql-connect-timeout", type=int, default=30)
    parser.add_argument("--sql-query-timeout", type=int, default=0)
    parser.add_argument("--sql-owner", default="PGESA")
    parser.add_argument("--sql-weather-type", default="Open Meteo")
    parser.add_argument("--sql-encrypt", action="store_true")
    parser.add_argument("--sql-trust-server-certificate", action="store_true")
    return parser


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )


def _prediction_columns(frame: pd.DataFrame, compact: bool = False) -> list[str]:
    hybrid_audit = [
        "wartosc_bazowa_backtest",
        "liczba_lagow_bazowych",
        "residuum_rzeczywiste",
        "korekta_ml_surowa",
        "wartosc_ml_przed_blendem",
        "blend_alpha",
        "strategia_predykcji",
        "kalibracja_n",
        "kalibracja_poprawa_mae",
        "kalibracja_powod",
        "fit_zatrzymany_limitem",
    ]
    if compact:
        preferred = [
            "source_sheet",
            "source_row",
            "oddzial_code",
            "punkt",
            "grupa",
            "klient_nazwa",
            "kierunek_energii_norm",
            "doba_handlowa",
            "godzina_handlowa",
            "wartosc_rzeczywista",
            "wartosc_przewidywana",
            *hybrid_audit,
            "blad",
            "blad_bezwzgledny",
            "status_predykcji",
            "prognoza_bez_pogody",
            "fold",
            "weather_status",
            "pogoda_dopasowana",
            "liczba_cech_pogodowych",
        ]
        return [col for col in preferred if col in frame.columns]
    preferred = [
        "source_sheet",
        "source_row",
        "oddzial_code",
        "punkt",
        "grupa",
        "klient_nazwa",
        "kierunek_code",
        "kierunek_energii",
        "kierunek_energii_norm",
        "direction_source",
        "doba_handlowa",
        "godzina_handlowa",
        "valid_timestamp",
        "model_timestamp_utc",
        "forecast_cutoff_utc",
        "rodzaj",
        "wartosc_rzeczywista",
        "wartosc_przewidywana",
        *hybrid_audit,
        "wartosc_model_pelny",
        "model_pelny_jest_insample",
        "blad",
        "blad_bezwzgledny",
        "status_predykcji",
        "prognoza_bez_pogody",
        "fold",
        "pogoda_dopasowana",
        "pogoda_dostepna",
        "liczba_cech_pogodowych",
        "weather_status",
        "weather_issue_time",
        "weather_issue_time_utc",
        "weather_valid_time",
        "weather_lead_hours",
        *WEATHER_FEATURES,
    ]
    return [col for col in preferred if col in frame.columns]


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "--validation-days": args.validation_days,
        "--folds": args.folds,
        "--calibration-days": args.calibration_days,
        "--min-calibration-rows": args.min_calibration_rows,
        "--min-train-rows": args.min_train_rows,
        "--max-iter": args.max_iter,
        "--catboost-depth": args.catboost_depth,
    }
    if args.max_train_rows is not None:
        positive["--max-train-rows"] = args.max_train_rows
    if args.max_input_rows is not None:
        positive["--max-input-rows"] = args.max_input_rows
    if args.max_fit_minutes is not None:
        positive["--max-fit-minutes"] = args.max_fit_minutes
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"Parametry muszą być dodatnie: {', '.join(invalid)}")
    if (
        args.max_train_rows is not None
        and args.max_train_rows < args.min_train_rows
    ):
        raise ValueError("--max-train-rows nie może być mniejsze od --min-train-rows.")
    if args.sql_query_timeout < 0:
        raise ValueError("--sql-query-timeout nie może być ujemny.")
    if args.model_progress_interval < 0:
        raise ValueError("--model-progress-interval nie może być ujemny.")
    if not 0 <= args.min_blend_improvement < 1:
        raise ValueError("--min-blend-improvement musi należeć do przedziału [0, 1).")
    if args.blend_grid_steps < 2:
        raise ValueError("--blend-grid-steps musi wynosić co najmniej 2.")


def _apply_execution_profile_defaults(args: argparse.Namespace) -> None:
    """Nadaje realne limity profilowi szybkiemu, jeśli użytkownik ich nie podał."""

    if args.execution_profile == "fast_30min":
        if args.max_input_rows is None:
            args.max_input_rows = 150_000
        if args.max_train_rows is None:
            args.max_train_rows = 100_000


def _infer_sql_date_range(
    energy: pd.DataFrame,
    valid_from_arg: str | None,
    valid_to_arg: str | None,
    weather_available_from: str | None = None,
) -> tuple[pd.Timestamp | None, pd.Timestamp]:
    dates = pd.to_datetime(energy["doba_handlowa"], errors="coerce").dropna()
    if dates.empty and (not valid_from_arg or not valid_to_arg):
        raise ValueError(
            "Nie można wyznaczyć zakresu SQL: brak poprawnej `Doby Handlowej`. "
            "Podaj --valid-from i --valid-to-exclusive."
        )
    requested_from = (
        pd.Timestamp(valid_from_arg) if valid_from_arg else dates.min().normalize()
    )
    valid_to = (
        pd.Timestamp(valid_to_arg)
        if valid_to_arg
        else dates.max().normalize() + pd.Timedelta(days=1)
    )
    if pd.isna(requested_from) or pd.isna(valid_to) or requested_from >= valid_to:
        raise ValueError("Niepoprawny zakres SQL: valid-from musi być wcześniejsze od valid-to.")
    if weather_available_from:
        available_from = pd.Timestamp(weather_available_from).normalize()
        if pd.isna(available_from):
            raise ValueError("Niepoprawna data --weather-available-from.")
        if valid_to <= available_from:
            # Cały Excel jest sprzed początku źródła pogodowego. Model nadal może
            # użyć historii energii, kalendarza i lagów, więc SQL jest zbędny.
            return None, valid_to
        requested_from = max(requested_from, available_from)
    valid_from = requested_from
    return valid_from, valid_to


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _apply_execution_profile_defaults(args)
    _validate_args(args)
    run_started = time.monotonic()

    def progress(message: str) -> None:
        elapsed = int(time.monotonic() - run_started)
        minutes, seconds = divmod(elapsed, 60)
        print(f"[{minutes:02d}:{seconds:02d}] {message}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(
        min_lead_hours=args.min_lead_hours,
        weather_available_from=args.weather_available_from,
        execution_profile=args.execution_profile,
        input_row_selection=args.input_row_selection,
        validation_days=args.validation_days,
        n_splits=args.folds,
        calibration_days=args.calibration_days,
        min_calibration_rows=args.min_calibration_rows,
        min_blend_improvement=args.min_blend_improvement,
        blend_grid_steps=args.blend_grid_steps,
        min_train_rows=args.min_train_rows,
        max_train_rows=args.max_train_rows,
        max_iter=args.max_iter,
        model_backend=args.model_backend,
        catboost_depth=args.catboost_depth,
        max_fit_seconds=(
            args.max_fit_minutes * 60.0
            if args.max_fit_minutes is not None
            else None
        ),
        model_progress_interval=args.model_progress_interval,
        compute_importance=not args.skip_importance,
        score_full_history=not args.skip_full_history_score,
        mapping_path=Path(args.mapping),
        weather_already_vintaged=args.weather_already_vintaged,
    )

    source_info: dict[str, object]
    if args.weather:
        input_scope = (
            f"maks. {args.max_input_rows:,} wierszy energii"
            if args.max_input_rows is not None
            else "cały zbiór energii"
        )
        progress(f"1/6 Czytam {input_scope} i plik pogody...")
        joined, quality, mapping = prepare_joined_dataset(
            args.energy,
            args.weather,
            config=config,
            max_rows=args.max_input_rows,
        )
        source_info = {
            "weather_source": "file",
            "weather_path": str(Path(args.weather)),
            "weather_available_from": config.weather_available_from,
        }
        progress(f"Dane przygotowane: {len(joined):,} wierszy.")
    else:
        progress("1/6 Czytam dane energii z Excela...")
        energy, energy_quality, mapping = prepare_energy_dataset(
            args.energy, config=config, max_rows=args.max_input_rows
        )
        progress(
            f"Odczyt zakończony: {len(energy):,} wierszy "
            f"({config.input_row_selection})."
        )
        valid_from, valid_to = _infer_sql_date_range(
            energy,
            args.valid_from,
            args.valid_to_exclusive,
            config.weather_available_from,
        )
        settings = SqlServerSettings(
            server=args.sql_server,
            database=args.sql_database,
            driver=args.sql_driver,
            encrypt=args.sql_encrypt,
            trust_server_certificate=args.sql_trust_server_certificate,
            connect_timeout_seconds=args.sql_connect_timeout,
        )
        sql_was_skipped = valid_from is None
        if sql_was_skipped:
            progress(
                "Cały zakres energii jest sprzed początku danych pogodowych "
                f"({config.weather_available_from}); pomijam zapytanie SQL."
            )
            weather_raw = pd.DataFrame(
                columns=["punkt", "dataGodzinaCET", "czasDanychZrodlaCET"]
            )
        else:
            progress(
                "2/6 Pobieram pogodę z "
                f"{settings.server} / {settings.database}: "
                f"{valid_from} <= valid time < {valid_to}"
            )
            weather_raw = query_weather_sql(
                args.weather_sql,
                settings=settings,
                valid_from_cet=valid_from,
                valid_to_cet_exclusive=valid_to,
                min_lead_hours=config.min_lead_hours,
                owner=args.sql_owner,
                weather_type=args.sql_weather_type,
                query_timeout_seconds=args.sql_query_timeout,
            )
            progress(f"SQL zakończony: {len(weather_raw):,} wierszy pogody.")
        if weather_raw.empty and not sql_was_skipped:
            raise ValueError("Zapytanie SQL nie zwróciło żadnych rekordów pogody.")
        progress("3/6 Łączę energię z pogodą...")
        joined, quality, mapping = join_prepared_energy_with_weather(
            energy,
            energy_quality,
            mapping,
            weather_raw,
            config,
        )
        progress(f"Łączenie zakończone: {len(joined):,} wierszy.")
        sql_quality = pd.DataFrame(
            [
                ["zrodlo_pogody", len(weather_raw), "SQL Server"],
                [
                    "sql_valid_from",
                    0,
                    "pominięto — cały zakres przed pogodą"
                    if sql_was_skipped
                    else str(valid_from),
                ],
                ["sql_valid_to_exclusive", 0, str(valid_to)],
            ],
            columns=["kontrola", "liczba", "szczegoly"],
        )
        quality = pd.concat([quality, sql_quality], ignore_index=True)
        source_info = {
            "weather_source": "sql_server",
            "sql_server": settings.server,
            "sql_database": settings.database,
            "sql_driver": settings.driver,
            "sql_path": str(Path(args.weather_sql)),
            "valid_from_cet": None if valid_from is None else str(valid_from),
            "valid_to_cet_exclusive": str(valid_to),
            "connection_string_from_env": bool(os.getenv("MDD_SQL_CONNECTION_STRING")),
            "weather_available_from": config.weather_available_from,
        }
        # Po złączeniu pełnego zbioru surowe ramki tylko dublowałyby pamięć
        # podczas budowy cech i uczenia.
        del energy, weather_raw
        gc.collect()
    source_info.update(
        {
            "execution_profile": config.execution_profile,
            "max_input_rows": args.max_input_rows,
            "input_row_selection": config.input_row_selection,
            "compact_output": bool(args.compact_output),
            "oof_output_only": bool(args.oof_output_only),
            "score_full_history": config.score_full_history,
        }
    )
    profile_quality = pd.DataFrame(
        [
            ["profil_uruchomienia", len(joined), config.execution_profile],
            [
                "limit_wierszy_wejscia",
                args.max_input_rows if args.max_input_rows is not None else "BRAK",
                f"wybór={config.input_row_selection}; BRAK oznacza cały plik",
            ],
            [
                "limit_wierszy_treningu_na_kierunek",
                config.max_train_rows if config.max_train_rows is not None else "BRAK",
                "BRAK oznacza wszystkie użyteczne wiersze historyczne",
            ],
            [
                "limit_czasu_pojedynczego_fitu_sekundy",
                int(config.max_fit_seconds or 0),
                "dotyczy CatBoost; 0 oznacza brak limitu",
            ],
        ],
        columns=["kontrola", "liczba", "szczegoly"],
    )
    quality = pd.concat([quality, profile_quality], ignore_index=True)

    progress("4/6 Buduję cechy i uruchamiam uczenie...")
    result = run_forecasting(joined, config, progress_callback=progress)
    del joined
    gc.collect()
    progress("Uczenie i predykcja zakończone.")
    predictions = result.predictions
    hybrid_quality_rows: list[list[object]] = []
    for direction, artifact in sorted(result.models.items()):
        alpha = getattr(artifact, "alpha", None)
        if alpha is None:
            # Zgodność z artefaktami starszego modelu: raport nie zależy twardo od
            # klasy HybridModelArtifact ani nie próbuje zgadywać parametrów.
            continue
        calibration_n = int(getattr(artifact, "calibration_n", 0) or 0)
        train_rows = int(getattr(artifact, "train_rows", 0) or 0)
        history_rows = int(getattr(artifact, "history_rows", 0) or 0)
        improvement = getattr(artifact, "calibration_improvement", None)
        reason = str(getattr(artifact, "calibration_reason", "BRAK_INFORMACJI"))
        target = str(getattr(artifact, "target", "residuum_wzgledem_D3_D14"))
        estimator = getattr(artifact, "estimator", artifact)
        stopped_by_time = bool(getattr(estimator, "_mdd_stopped_by_time", False))
        improvement_text = (
            "brak"
            if improvement is None or pd.isna(improvement)
            else f"{float(improvement):.2%}"
        )
        hybrid_quality_rows.append(
            [
                f"hybryda_kalibracja_{str(direction).lower()}",
                calibration_n,
                (
                    f"alpha={float(alpha):.3f}; poprawa_MAE={improvement_text}; "
                    f"decyzja={reason}; target={target}; "
                    f"wiersze_uczenia={train_rows}; historia={history_rows}; "
                    f"fit_zatrzymany_limitem={stopped_by_time}"
                ),
            ]
        )
        hybrid_quality_rows.append(
            [
                f"pelny_trening_{str(direction).lower()}",
                train_rows,
                (
                    f"użyto {train_rows} z {history_rows} historycznych wierszy; "
                    "pominięte rekordy nie miały naturalnego baseline D-3...D-14"
                ),
            ]
        )
    model_quality = pd.DataFrame(
        [
            ["wiersze_OOF_do_oceny", int(predictions["status_predykcji"].eq("OOF_BACKTEST").sum()), ""],
            [
                "wiersze_prognozy_przyszlej",
                int(predictions["status_predykcji"].eq("PROGNOZA_PRZYSZLA").sum()),
                "",
            ],
            [
                "prognoza_przyszla_bez_pogody",
                int(predictions["prognoza_bez_pogody"].fillna(False).sum()),
                "model użył historii energii i kalendarza, ale nie miał dopasowanej pogody",
            ],
            ["historyczne_braki_targetu", int(predictions["status_predykcji"].eq("HISTORYCZNY_BRAK_TARGETU").sum()), "nie są oznaczane jako prognoza przyszła"],
            ["wiersze_warmup_bez_OOF", int(predictions["status_predykcji"].eq("WARMUP_BEZ_OOF").sum()), ""],
            ["modele_koncowe", len(result.models), ", ".join(sorted(result.models))],
            *hybrid_quality_rows,
        ],
        columns=["kontrola", "liczba", "szczegoly"],
    )
    quality = pd.concat([quality, model_quality], ignore_index=True)

    progress("5/6 Zapisuję wyniki i modele...")
    if args.oof_output_only:
        export_mask = predictions["status_predykcji"].isin(
            ["OOF_BACKTEST", "PROGNOZA_PRZYSZLA"]
        )
        export_predictions = predictions.loc[export_mask].copy()
        prediction_prefix = "predykcje_oof"
    else:
        export_predictions = predictions
        prediction_prefix = "predykcje"
    columns = _prediction_columns(export_predictions, compact=args.compact_output)
    if export_predictions["source_sheet"].nunique(dropna=False) > 1:
        for sheet_name, group in export_predictions.groupby(
            "source_sheet", dropna=False, sort=False
        ):
            safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(sheet_name))
            _write_csv(
                group[columns], output_dir / f"{prediction_prefix}_{safe_name}.csv"
            )
    else:
        _write_csv(
            export_predictions[columns], output_dir / f"{prediction_prefix}.csv"
        )
    _write_csv(result.metrics, output_dir / "metryki.csv")
    _write_csv(result.feature_importance, output_dir / "waznosc_cech.csv")
    _write_csv(quality, output_dir / "kontrola_jakosci.csv")
    _write_csv(mapping, output_dir / "uzyte_mapowanie.csv")

    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Brak joblib, mimo zainstalowanego scikit-learn.") from exc
    for direction, model in result.models.items():
        joblib.dump(model, output_dir / f"model_{direction.lower()}.joblib")
    bundle = build_forecast_bundle(result, config, mapping)
    bundle_path = output_dir / "pakiet_modelu_mdd.joblib"
    joblib.dump(bundle, bundle_path)
    if args.oof_output_only:
        summary_columns = [
            "status_predykcji",
            "wartosc_rzeczywista",
            "pogoda_dopasowana",
            "weather_status",
            "liczba_cech_pogodowych",
            "prognoza_bez_pogody",
        ]
        summary_predictions_for_report = predictions.loc[
            :, [column for column in summary_columns if column in predictions.columns]
        ].copy()
        # Bundle i statystyki są już zbudowane. Przed zapisem Excela zwalniamy
        # szeroką macierz pełnej historii, zostawiając OOF oraz wąskie podsumowanie.
        result.predictions = export_predictions
        del predictions
        gc.collect()
    else:
        summary_predictions_for_report = predictions

    config_payload = asdict(config)
    config_payload["mapping_path"] = str(config.mapping_path)
    manifest = {
        "config": config_payload,
        "source": source_info,
        "categorical_features": result.feature_spec.categorical,
        "numeric_features": result.feature_spec.numeric,
        "model_bundle": {
            "filename": bundle_path.name,
            "schema_version": bundle.schema_version,
            "training_cutoff_utc": bundle.training_cutoff_utc,
            "metadata": bundle.metadata,
        },
        "note": (
            "Metryki HYBRYDA_OOF wykorzystują wyłącznie predykcje poza próbą. "
            "Korekta residualna jest dodawana do baseline D-3...D-14 tylko po "
            "potwierdzeniu minimalnej poprawy na kalibracji; w przeciwnym razie "
            "alpha=0 oznacza bezpieczny fallback do baseline. Końcowe artefakty "
            "uczą się ponownie na wszystkich użytecznych wierszach historii. "
            "Współczynnik alpha zapisanego modelu końcowego jest wybierany na "
            "historycznych wierszach OOF, natomiast raportowane metryki OOF "
            "używają alpha "
            "kalibrowanego wyłącznie przed każdym ocenianym oknem. "
            "Jeżeli kolumna wartosc_model_pelny została wyliczona dla historii, "
            "jest in-sample i nie służy do oceny."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    progress("6/6 Tworzę raport Excel...")
    workbook_path = write_results_workbook(
        predictions=export_predictions,
        summary_predictions=summary_predictions_for_report,
        prediction_columns=columns,
        metrics=result.metrics,
        feature_importance=result.feature_importance,
        quality=quality,
        mapping=mapping,
        models=result.models,
        manifest=manifest,
        output_path=output_dir / "wyniki_mdd.xlsx",
    )
    progress(f"Raport Excel: {workbook_path.resolve()}")
    progress(f"Gotowe. Wyniki zapisano w: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

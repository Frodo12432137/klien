from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

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
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--max-train-rows", type=int, default=1_000_000)
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument(
        "--execution-profile",
        choices=["standard", "fast_30min"],
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
            "wartosc_bazowa_backtest",
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
        "wartosc_bazowa_backtest",
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
        "--min-train-rows": args.min_train_rows,
        "--max-train-rows": args.max_train_rows,
        "--max-iter": args.max_iter,
        "--catboost-depth": args.catboost_depth,
    }
    if args.max_input_rows is not None:
        positive["--max-input-rows"] = args.max_input_rows
    if args.max_fit_minutes is not None:
        positive["--max-fit-minutes"] = args.max_fit_minutes
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"Parametry muszą być dodatnie: {', '.join(invalid)}")
    if args.max_train_rows < args.min_train_rows:
        raise ValueError("--max-train-rows nie może być mniejsze od --min-train-rows.")
    if args.sql_query_timeout < 0:
        raise ValueError("--sql-query-timeout nie może być ujemny.")
    if args.model_progress_interval < 0:
        raise ValueError("--model-progress-interval nie może być ujemny.")


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
        mapping_path=Path(args.mapping),
        weather_already_vintaged=args.weather_already_vintaged,
    )

    source_info: dict[str, object]
    if args.weather:
        progress("1/6 Czytam ograniczony zbiór energii i plik pogody...")
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
    source_info.update(
        {
            "execution_profile": config.execution_profile,
            "max_input_rows": args.max_input_rows,
            "input_row_selection": config.input_row_selection,
            "compact_output": bool(args.compact_output),
        }
    )
    profile_quality = pd.DataFrame(
        [
            ["profil_uruchomienia", len(joined), config.execution_profile],
            [
                "limit_wierszy_wejscia",
                args.max_input_rows or 0,
                f"wybór={config.input_row_selection}; 0 oznacza brak limitu",
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
    progress("Uczenie i predykcja zakończone.")
    predictions = result.predictions
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
        ],
        columns=["kontrola", "liczba", "szczegoly"],
    )
    quality = pd.concat([quality, model_quality], ignore_index=True)

    progress("5/6 Zapisuję kompaktowe CSV i modele...")
    columns = _prediction_columns(predictions, compact=args.compact_output)
    if predictions["source_sheet"].nunique(dropna=False) > 1:
        for sheet_name, group in predictions.groupby("source_sheet", dropna=False, sort=False):
            safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(sheet_name))
            _write_csv(group[columns], output_dir / f"predykcje_{safe_name}.csv")
    else:
        _write_csv(predictions[columns], output_dir / "predykcje.csv")
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

    config_payload = asdict(config)
    config_payload["mapping_path"] = str(config.mapping_path)
    manifest = {
        "config": config_payload,
        "source": source_info,
        "categorical_features": result.feature_spec.categorical,
        "numeric_features": result.feature_spec.numeric,
        "note": (
            "Metryki wykorzystują wyłącznie status OOF_BACKTEST. Kolumna wartosc_model_pelny "
            "dla rekordów historycznych jest in-sample i nie służy do oceny."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    progress("6/6 Tworzę raport Excel...")
    workbook_path = write_results_workbook(
        predictions=predictions,
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

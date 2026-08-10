from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import pandas as pd

from mdd_forecasting.cli import _infer_sql_date_range
from mdd_forecasting.config import PipelineConfig
from mdd_forecasting.database import (
    SqlServerSettings,
    query_weather_sql,
    weather_query_parameters,
)
from mdd_forecasting.generate_demo_data import generate
from mdd_forecasting.features import add_lag_features
from mdd_forecasting.io import (
    join_energy_weather,
    normalize_energy,
    normalize_weather,
    prepare_joined_dataset,
)
from mdd_forecasting.model import run_forecasting
from mdd_forecasting.report import write_results_workbook
from uruchom_model import MODEL_BACKEND, MIN_LEAD_HOURS, build_cli_args


class TestMddForecasting(unittest.TestCase):
    def test_click_launcher_builds_portable_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build_cli_args(
                Path(tmp) / "energia.xlsx",
                Path(tmp) / "wyniki",
            )
        self.assertEqual(args[args.index("--model-backend") + 1], MODEL_BACKEND)
        self.assertEqual(
            args[args.index("--min-lead-hours") + 1], str(MIN_LEAD_HOURS)
        )
        sql_path = Path(args[args.index("--weather-sql") + 1])
        self.assertTrue(sql_path.exists())
        self.assertNotIn("10200871", " ".join(args))

    def test_lag_window_uses_same_hour_from_day_3_to_day_14(self):
        dates = pd.date_range("2025-01-01", periods=15, freq="D")
        frame = pd.DataFrame(
            {
                "oddzial_code": "BIA",
                "grupa": "G",
                "klient_nazwa": "KLIENT",
                "kierunek_energii_norm": "POBRANIE",
                "rodzaj": "4",
                "doba_handlowa": dates,
                "godzina_handlowa": 8,
                "wartosc_rzeczywista": range(1, 16),
            }
        )
        with_lags = add_lag_features(frame)
        last = with_lags.iloc[-1]
        for days in range(3, 15):
            self.assertEqual(last[f"lag_{24 * days}h"], 15 - days)
        self.assertNotIn("lag_24h", with_lags.columns)
        self.assertNotIn("lag_48h", with_lags.columns)
        self.assertEqual(last["lag_srednia_3_14_dni"], 6.5)
        self.assertEqual(last["wartosc_bazowa"], 6.5)

    def test_sql_parameters_and_integrated_connection(self):
        params = weather_query_parameters("2025-01-01", "2025-02-01", 24)
        self.assertEqual(len(params), 5)
        self.assertEqual(params[2], 24)
        self.assertEqual(params[3:], ["PGESA", "Open Meteo"])
        connection = SqlServerSettings().connection_string()
        self.assertIn("SERVER=MISDWHPRD.GKPGE.PL", connection)
        self.assertIn("DATABASE=PGESA_MarketAnalytics", connection)
        self.assertIn("Trusted_Connection=yes", connection)
        self.assertNotIn("PWD=", connection.upper())
        sql_text = (
            Path("mdd_forecasting/sql/pogoda_mdd.sql").read_text(encoding="utf-8")
        )
        self.assertEqual(sql_text.count("?"), 5)
        self.assertIn("[PGESA_MarketAnalytics].[wa].[vPogodaPrognoza]", sql_text)
        self.assertIn("czasDanychZrodlaUTC", sql_text)

    def test_sql_execution_is_parameterized_and_connection_is_closed(self):
        class FakeConnection:
            def __init__(self):
                self.closed = False
                self.timeout = 0

            def close(self):
                self.closed = True

        connection = FakeConnection()
        fake_pyodbc = SimpleNamespace(
            drivers=lambda: ["ODBC Driver 17 for SQL Server"],
            connect=lambda *args, **kwargs: connection,
        )
        expected = pd.DataFrame({"punkt": ["Lublin"]})
        with patch.dict("sys.modules", {"pyodbc": fake_pyodbc}), patch(
            "mdd_forecasting.database.pd.read_sql_query", return_value=expected
        ) as read_query:
            result = query_weather_sql(
                "mdd_forecasting/sql/pogoda_mdd.sql",
                settings=SqlServerSettings(),
                valid_from_cet="2025-01-01",
                valid_to_cet_exclusive="2025-01-03",
                min_lead_hours=24,
                query_timeout_seconds=120,
            )
        self.assertEqual(result.to_dict("records"), [{"punkt": "Lublin"}])
        self.assertTrue(connection.closed)
        self.assertEqual(connection.timeout, 120)
        params = read_query.call_args.kwargs["params"]
        self.assertEqual(len(params), 5)
        self.assertEqual(params[2:], [24, "PGESA", "Open Meteo"])

    def test_sql_range_is_inferred_from_energy_dates(self):
        energy = pd.DataFrame(
            {"doba_handlowa": ["2025-01-05", "2025-01-07", None]}
        )
        valid_from, valid_to = _infer_sql_date_range(energy, None, None)
        self.assertEqual(valid_from, pd.Timestamp("2025-01-05"))
        self.assertEqual(valid_to, pd.Timestamp("2025-01-08"))

    def test_weather_vintage_respects_minimum_lead(self):
        valid = pd.Timestamp("2025-01-03 12:00:00")
        weather = pd.DataFrame(
            {
                "punkt": ["Lublin", "Lublin"],
                "dataGodzinaCET": [valid, valid],
                "czasDanychZrodlaCET": [
                    valid - pd.Timedelta(hours=30),
                    valid - pd.Timedelta(hours=12),
                ],
                "temperatura": [1.0, 99.0],
            }
        )
        selected = normalize_weather(weather, PipelineConfig(min_lead_hours=24))
        self.assertEqual(len(selected), 1)
        self.assertEqual(float(selected.iloc[0]["temperatura"]), 1.0)
        self.assertEqual(float(selected.iloc[0]["weather_lead_hours"]), 30.0)

    def test_end_to_end_demo_creates_oof_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            energy_path, weather_path = generate(Path(tmp), days=20, seed=7)
            config = PipelineConfig(
                min_lead_hours=24,
                validation_days=3,
                n_splits=1,
                min_train_rows=100,
                max_train_rows=50_000,
                max_iter=35,
                compute_importance=False,
            )
            joined, quality, _ = prepare_joined_dataset(
                energy_path, weather_path, config=config
            )
            self.assertEqual(len(joined), 20 * 24 * 4 * 2 * 2)
            self.assertTrue(joined["pogoda_dopasowana"].all())
            historical_gap_index = joined.index[100]
            joined.loc[historical_gap_index, "wartosc_rzeczywista"] = float("nan")
            result = run_forecasting(joined, config)
            evaluated = result.predictions["status_predykcji"].eq("OOF_BACKTEST")
            self.assertGreater(int(evaluated.sum()), 0)
            self.assertFalse(result.metrics.empty)
            self.assertEqual(set(result.models), {"POBRANIE", "ODDANIE"})
            self.assertEqual(
                result.predictions.loc[historical_gap_index, "status_predykcji"],
                "HISTORYCZNY_BRAK_TARGETU",
            )
            workbook_path = write_results_workbook(
                predictions=result.predictions,
                prediction_columns=[
                    "source_sheet",
                    "source_row",
                    "klient_nazwa",
                    "wartosc_rzeczywista",
                    "wartosc_przewidywana",
                    "status_predykcji",
                ],
                metrics=result.metrics,
                feature_importance=result.feature_importance,
                quality=quality,
                mapping=pd.DataFrame(
                    {"oddzial_code": ["BIA"], "punkt": ["Białystok"], "status": ["active"]}
                ),
                models=result.models,
                manifest={"test": True},
                output_path=Path(tmp) / "wyniki_mdd.xlsx",
            )
            self.assertTrue(workbook_path.exists())
            from openpyxl import load_workbook

            workbook = load_workbook(workbook_path, read_only=True, data_only=False)
            try:
                self.assertIn("Podsumowanie", workbook.sheetnames)
                self.assertIn("Metryki", workbook.sheetnames)
                self.assertTrue(any(name.startswith("Pred_") for name in workbook.sheetnames))
                summary = workbook["Podsumowanie"]
                self.assertEqual(summary["A1"].value, "sekcja")
                first_row = next(summary.iter_rows(values_only=True))
                self.assertEqual(len(first_row), 4)
            finally:
                workbook.close()

    def test_dst_repeated_hour_uses_trade_hour_and_utc(self):
        energy_raw = pd.DataFrame(
            [
                ["Dane_01", 2, "BIA", 1, "G", "K", "czynne pobranie", "2025-10-26", 4, 3, 10],
                ["Dane_01", 3, "BIA", 1, "G", "K", "czynne pobranie", "2025-10-26", 4, 25, 11],
            ],
            columns=[
                "source_sheet", "source_row", "oddzial_code", "kierunek_code", "grupa",
                "klient_nazwa", "kierunek_energii", "doba_handlowa", "rodzaj",
                "godzina_handlowa", "wartosc_rzeczywista",
            ],
        )
        energy, _ = normalize_energy(energy_raw, {"bia": "Białystok"})
        local_valid = pd.Timestamp("2025-10-26 02:00:00")
        weather_raw = pd.DataFrame(
            {
                "punkt": ["Białystok", "Białystok"],
                "dataCET": ["2025-10-26", "2025-10-26"],
                "godzinaHandlowa25": [3, 25],
                "dataGodzinaCET": [local_valid, local_valid],
                "dataGodzinaUTC": ["2025-10-26T00:00:00Z", "2025-10-26T01:00:00Z"],
                "czasDanychZrodlaCET": [
                    local_valid - pd.Timedelta(hours=30),
                    local_valid - pd.Timedelta(hours=30),
                ],
                "temperatura": [5, 6],
            }
        )
        weather = normalize_weather(weather_raw, PipelineConfig(min_lead_hours=24))
        joined, _ = join_energy_weather(energy, weather)
        self.assertEqual(len(joined), 2)
        self.assertEqual(joined["model_timestamp_utc"].nunique(), 2)
        self.assertTrue(joined["pogoda_dopasowana"].all())


if __name__ == "__main__":
    unittest.main()

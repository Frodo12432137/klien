from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_SQL_SERVER = "MISDWHPRD.GKPGE.PL"
DEFAULT_SQL_DATABASE = "PGESA_MarketAnalytics"
DEFAULT_SQL_DRIVER = "ODBC Driver 17 for SQL Server"


@dataclass(frozen=True)
class SqlServerSettings:
    """Konfiguracja połączenia bez przechowywania loginu i hasła w kodzie."""

    server: str = DEFAULT_SQL_SERVER
    database: str = DEFAULT_SQL_DATABASE
    driver: str = DEFAULT_SQL_DRIVER
    trusted_connection: bool = True
    encrypt: bool = False
    trust_server_certificate: bool = False
    connect_timeout_seconds: int = 30

    def connection_string(self) -> str:
        override = os.getenv("MDD_SQL_CONNECTION_STRING")
        if override:
            return override
        if not self.server.strip() or not self.database.strip() or not self.driver.strip():
            raise ValueError("Server, database i driver ODBC nie mogą być puste.")
        parts = [
            f"DRIVER={{{self.driver}}}",
            f"SERVER={self.server}",
            f"DATABASE={self.database}",
            f"Trusted_Connection={'yes' if self.trusted_connection else 'no'}",
            f"APP=MDD Forecasting",
        ]
        if self.encrypt:
            parts.append("Encrypt=yes")
            parts.append(
                f"TrustServerCertificate={'yes' if self.trust_server_certificate else 'no'}"
            )
        return ";".join(parts) + ";"


def _sql_datetime(value: object, name: str) -> datetime:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"Niepoprawna data {name}: {value!r}") from exc
    if pd.isna(timestamp):
        raise ValueError(f"Data {name} nie może być pusta.")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Europe/Warsaw").tz_localize(None)
    return timestamp.floor("s").to_pydatetime()


def weather_query_parameters(
    valid_from_cet: object,
    valid_to_cet_exclusive: object,
    min_lead_hours: int,
    owner: str = "PGESA",
    weather_type: str = "Open Meteo",
) -> list[object]:
    """Waliduje wartości przekazywane do pięciu placeholderów `?` pyodbc."""

    lead = int(min_lead_hours)
    if lead < 0 or lead > 720:
        raise ValueError("min_lead_hours musi należeć do zakresu 0–720.")
    valid_from = _sql_datetime(valid_from_cet, "valid_from_cet")
    valid_to = _sql_datetime(valid_to_cet_exclusive, "valid_to_cet_exclusive")
    if pd.Timestamp(valid_from) >= pd.Timestamp(valid_to):
        raise ValueError("valid_from_cet musi być wcześniejsze niż valid_to_cet_exclusive.")
    owner_clean = str(owner).strip()
    weather_type_clean = str(weather_type).strip()
    if not owner_clean or not weather_type_clean:
        raise ValueError("Właściciel i typ prognozy nie mogą być puste.")
    return [valid_from, valid_to, lead, owner_clean, weather_type_clean]


def query_weather_sql(
    sql_path: str | Path,
    settings: SqlServerSettings,
    valid_from_cet: object,
    valid_to_cet_exclusive: object,
    min_lead_hours: int,
    owner: str = "PGESA",
    weather_type: str = "Open Meteo",
    query_timeout_seconds: int = 0,
) -> pd.DataFrame:
    """Wykonuje kontrolowany plik SQL przez pyodbc i zawsze zamyka połączenie."""

    try:
        import pyodbc
    except ImportError as exc:  # pragma: no cover - zależność Windows/ODBC
        raise RuntimeError(
            "Brak pyodbc. Zainstaluj requirements.txt oraz Microsoft ODBC Driver 17/18."
        ) from exc

    if not os.getenv("MDD_SQL_CONNECTION_STRING"):
        available_drivers = set(pyodbc.drivers())
        if settings.driver not in available_drivers:
            raise RuntimeError(
                f"Brak sterownika {settings.driver!r}. Dostępne sterowniki ODBC: "
                f"{sorted(available_drivers)}. Użyj --sql-driver z właściwą nazwą."
            )

    sql_file = Path(sql_path)
    sql_text = sql_file.read_text(encoding="utf-8-sig")
    placeholder_count = sql_text.count("?")
    if placeholder_count == 5:
        params: list[object] | None = weather_query_parameters(
            valid_from_cet=valid_from_cet,
            valid_to_cet_exclusive=valid_to_cet_exclusive,
            min_lead_hours=min_lead_hours,
            owner=owner,
            weather_type=weather_type,
        )
    elif placeholder_count == 0:
        params = None
        warnings.warn(
            "Plik SQL nie ma placeholderów parametrów. Zostanie wykonany bez zmian; "
            "zakres dat, punkty i historyczny vintage muszą być poprawnie ustawione "
            "wewnątrz samej kwerendy.",
            UserWarning,
            stacklevel=2,
        )
    else:
        raise ValueError(
            f"Plik SQL zawiera {placeholder_count} placeholderów ?. Obsługiwane jest "
            "dokładnie 5 (validFrom, validTo, lead, owner, typ) albo 0 dla zwykłego "
            "SQL-a z własnymi DECLARE."
        )
    connection = pyodbc.connect(
        settings.connection_string(),
        timeout=settings.connect_timeout_seconds,
        autocommit=True,
    )
    try:
        if query_timeout_seconds > 0:
            connection.timeout = int(query_timeout_seconds)
        # pandas emituje ogólne ostrzeżenie dla każdego połączenia DBAPI2 innego niż
        # sqlite. pyodbc jest tutaj świadomie testowany; komunikat nie oznacza błędu
        # i tylko niepotrzebnie sugerował użytkownikowi, że model się zatrzymał.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pandas only supports SQLAlchemy connectable.*",
                category=UserWarning,
            )
            if params is None:
                frame = pd.read_sql_query(sql_text, connection)
            else:
                frame = pd.read_sql_query(sql_text, connection, params=params)
    finally:
        connection.close()
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame

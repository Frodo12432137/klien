from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MAPPING_PATH = PACKAGE_DIR / "config" / "oddzial_miasto.csv"


ENERGY_COLUMNS = [
    "oddzial_code",
    "kierunek_code",
    "grupa",
    "klient_nazwa",
    "kierunek_energii",
    "doba_handlowa",
    "rodzaj",
    "godzina_handlowa",
    "wartosc_rzeczywista",
]


WEATHER_ALIASES = {
    "weather_exec_id": ["execid"],
    "weather_issue_time": [
        "czasdanychzrodlacet",
        "czas_danych_zrodla_cet",
        "weather_issue_time",
    ],
    "weather_issue_time_utc": [
        "czasdanychzrodlautc",
        "czas_danych_zrodla_utc",
        "weather_issue_time_utc",
    ],
    "weather_valid_time": [
        "datagodzinacet",
        "data_godzina_cet",
        "weather_valid_time",
    ],
    "weather_valid_time_utc": [
        "datagodzinatutc",
        "data_godzina_utc",
        "weather_valid_time_utc",
    ],
    "weather_trade_date": ["datacet", "data_cet", "weather_trade_date"],
    "weather_trade_hour": [
        "godzinahandlowa25",
        "godzina_handlowa25",
        "weather_trade_hour",
    ],
    "punkt": ["punkt", "miasto", "lokalizacja"],
    "temperatura": ["temperatura"],
    "predkosc_wiatru": ["predkoscwiatru", "predkosc_wiatru"],
    "kierunek_wiatru": ["kierunekwiatru", "kierunek_wiatru"],
    "zachmurzenie": ["zachmurzenie"],
    "opad_konwekcyjny": [
        "intensywnoscopadowkonwekcyjnych",
        "intensywnosc_opadow_konwekcyjnych",
        "opad_konwekcyjny",
    ],
    "widocznosc": ["widocznosc"],
    "promieniowanie_calkowite": [
        "calkowitepromieniowanieslonecznegodzinowe",
        "calkowite_promieniowanie_sloneczne_godzinowe",
        "promieniowanie_calkowite",
    ],
    "promieniowanie_bezposrednie": [
        "bezposredniepromieniowanieslonecznegodzinowe",
        "bezposrednie_promieniowanie_sloneczne_godzinowe",
        "promieniowanie_bezposrednie",
    ],
    "albedo": ["albedoprognozowane", "albedo_prognozowane", "albedo"],
    "warstwa_sniegu": ["warstwasniegu", "warstwa_sniegu"],
}


WEATHER_FEATURES = [
    "temperatura",
    "predkosc_wiatru",
    "kierunek_wiatru",
    "zachmurzenie",
    "opad_konwekcyjny",
    "widocznosc",
    "promieniowanie_calkowite",
    "promieniowanie_bezposrednie",
    "albedo",
    "warstwa_sniegu",
]


@dataclass(frozen=True)
class PipelineConfig:
    """Parametry jawne, aby backtest odpowiadał sposobowi pracy operacyjnej."""

    min_lead_hours: int = 24
    weather_available_from: str = "2024-10-01"
    execution_profile: str = "standard"
    input_row_selection: str = "head"
    lag_days: tuple[int, ...] = tuple(range(3, 15))
    timezone: str = "Europe/Warsaw"
    validation_days: int = 14
    n_splits: int = 3
    min_train_rows: int = 200
    max_train_rows: int = 1_000_000
    max_iter: int = 220
    learning_rate: float = 0.06
    model_backend: str = "hist_gradient_boosting"
    catboost_depth: int = 8
    max_fit_seconds: float | None = None
    model_progress_interval: int = 0
    random_state: int = 42
    compute_importance: bool = True
    max_importance_rows: int = 5_000
    mapping_path: Path = DEFAULT_MAPPING_PATH
    weather_already_vintaged: bool = False

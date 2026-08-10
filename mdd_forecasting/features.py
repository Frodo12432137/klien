from __future__ import annotations

import calendar
import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import WEATHER_FEATURES


DEFAULT_LAG_DAYS = tuple(range(3, 15))
LAG_FEATURES = [f"lag_{24 * days}h" for days in DEFAULT_LAG_DAYS]
LAG_MEAN_FEATURE = "lag_srednia_3_14_dni"


@dataclass(frozen=True)
class FeatureSpec:
    categorical: list[str]
    numeric: list[str]

    @property
    def all(self) -> list[str]:
        return [*self.categorical, *self.numeric]


def _easter_sunday(year: int) -> date:
    """Algorytm Meeusa/Jonesa/Butchera dla kalendarza gregoriańskiego."""

    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _polish_holidays(year: int) -> set[date]:
    easter = _easter_sunday(year)
    fixed = {
        date(year, 1, 1),
        date(year, 1, 6),
        date(year, 5, 1),
        date(year, 5, 3),
        date(year, 8, 15),
        date(year, 11, 1),
        date(year, 11, 11),
        date(year, 12, 25),
        date(year, 12, 26),
    }
    return fixed | {
        easter,
        easter + timedelta(days=1),
        easter + timedelta(days=49),
        easter + timedelta(days=60),
    }


def _last_sunday(year: int, month: int) -> date:
    last_day = calendar.monthrange(year, month)[1]
    result = date(year, month, last_day)
    return result - timedelta(days=(result.weekday() - 6) % 7)


def add_lag_features(
    frame: pd.DataFrame,
    lag_days: tuple[int, ...] = DEFAULT_LAG_DAYS,
) -> pd.DataFrame:
    df = frame.copy()
    if not lag_days or any(int(days) < 1 for days in lag_days):
        raise ValueError("lag_days musi zawierać dodatnie liczby dni.")
    if len(set(lag_days)) != len(lag_days):
        raise ValueError("lag_days nie może zawierać duplikatów.")
    categorical_key = [
        "oddzial_code",
        "grupa",
        "klient_nazwa",
        "kierunek_energii_norm",
        "rodzaj",
    ]
    key_values = []
    for col in categorical_key:
        values = df[col].astype("string").fillna("__BRAK__").str.strip()
        key_values.append(values)
    df["series_id"] = key_values[0]
    for values in key_values[1:]:
        df["series_id"] = df["series_id"] + "|" + values

    history = (
        df.loc[
            df["wartosc_rzeczywista"].notna()
            & df["doba_handlowa"].notna()
            & df["godzina_handlowa"].notna(),
            ["series_id", "doba_handlowa", "godzina_handlowa", "wartosc_rzeczywista"],
        ]
        .groupby(
            ["series_id", "doba_handlowa", "godzina_handlowa"], as_index=False
        )["wartosc_rzeczywista"]
        .mean()
    )
    history_indexed = history.set_index(
        ["series_id", "doba_handlowa", "godzina_handlowa"]
    )["wartosc_rzeczywista"]
    lag_features: list[str] = []
    for days in lag_days:
        hours = 24 * int(days)
        lag_name = f"lag_{hours}h"
        lag_features.append(lag_name)
        # Reindex po kluczu jest znacznie oszczędniejszy pamięciowo niż 12 kolejnych
        # merge całej, coraz szerszej ramki. Odejmujemy dni kalendarzowe, więc klucz
        # godziny handlowej pozostaje odporny na przejścia CET/CEST.
        lookup_keys = pd.MultiIndex.from_arrays(
            [
                df["series_id"],
                df["doba_handlowa"] - pd.to_timedelta(int(days), unit="D"),
                df["godzina_handlowa"],
            ],
            names=history_indexed.index.names,
        )
        df[lag_name] = history_indexed.reindex(lookup_keys).to_numpy(dtype=float)

    mean_feature = (
        LAG_MEAN_FEATURE
        if tuple(lag_days) == DEFAULT_LAG_DAYS
        else f"lag_srednia_{min(lag_days)}_{max(lag_days)}_dni"
    )
    df[mean_feature] = df[lag_features].mean(axis=1, skipna=True)
    df["liczba_lagow_bazowych"] = df[lag_features].notna().sum(axis=1).astype(float)
    df["lag_odchylenie_3_14_dni"] = df[lag_features].std(axis=1, skipna=True)
    recent_lags = [f"lag_{24 * days}h" for days in lag_days if 3 <= int(days) <= 6]
    older_lags = [f"lag_{24 * days}h" for days in lag_days if 10 <= int(days) <= 14]
    if recent_lags and older_lags:
        df["lag_trend_krotki_vs_dlugi"] = (
            df[recent_lags].mean(axis=1, skipna=True)
            - df[older_lags].mean(axis=1, skipna=True)
        )
    else:
        df["lag_trend_krotki_vs_dlugi"] = np.nan
    lag_7d = "lag_168h"
    df["lag_7d_vs_srednia"] = (
        df[lag_7d] - df[mean_feature] if lag_7d in df.columns else np.nan
    )
    # Baseline jest średnią dostępnych analogicznych godzin D-3...D-14. Model dostaje
    # również wszystkie pojedyncze lagi i może nauczyć się dla nich różnych wag.
    df["wartosc_bazowa"] = df[mean_feature]
    return df


def build_features(
    frame: pd.DataFrame,
    lag_days: tuple[int, ...] = DEFAULT_LAG_DAYS,
) -> tuple[pd.DataFrame, FeatureSpec]:
    df = add_lag_features(frame, lag_days=lag_days)
    if "pogoda_dostepna" not in df.columns:
        matched = df.get("pogoda_dopasowana", pd.Series(False, index=df.index))
        df["pogoda_dostepna"] = pd.Series(matched, index=df.index).fillna(False)
    df["pogoda_dostepna"] = pd.to_numeric(
        df["pogoda_dostepna"], errors="coerce"
    ).fillna(0.0).astype("float64")
    ts = pd.to_datetime(df["valid_timestamp"], errors="coerce")
    df["godzina"] = ts.dt.hour
    df["dzien_tygodnia"] = ts.dt.dayofweek
    df["miesiac"] = ts.dt.month
    df["dzien_roku"] = ts.dt.dayofyear
    df["weekend"] = ts.dt.dayofweek.ge(5).astype(float)

    df["godzina_sin"] = np.sin(2 * math.pi * df["godzina"] / 24.0)
    df["godzina_cos"] = np.cos(2 * math.pi * df["godzina"] / 24.0)
    df["dzien_tyg_sin"] = np.sin(2 * math.pi * df["dzien_tygodnia"] / 7.0)
    df["dzien_tyg_cos"] = np.cos(2 * math.pi * df["dzien_tygodnia"] / 7.0)
    df["dzien_roku_sin"] = np.sin(2 * math.pi * df["dzien_roku"] / 365.25)
    df["dzien_roku_cos"] = np.cos(2 * math.pi * df["dzien_roku"] / 365.25)

    years = sorted({int(year) for year in ts.dt.year.dropna().unique()})
    holidays = set().union(*(_polish_holidays(year) for year in years)) if years else set()
    dates = ts.dt.date
    df["swieto_PL"] = dates.isin(holidays).astype(float)
    dst_days = set()
    for year in years:
        dst_days.add(_last_sunday(year, 3))
        dst_days.add(_last_sunday(year, 10))
    df["dzien_zmiany_czasu"] = dates.isin(dst_days).astype(float)

    # `Float64` z pandas przechowuje pd.NA; NumPy nie umie użyć takiej wartości jako
    # warunku w np.where. Zwykłe float64 zachowuje brak jako np.nan i działa w obu
    # backendach modelu.
    temperature = pd.to_numeric(df["temperatura"], errors="coerce").astype("float64")
    wind_speed = pd.to_numeric(df["predkosc_wiatru"], errors="coerce").astype("float64")
    wind_direction = pd.to_numeric(df["kierunek_wiatru"], errors="coerce").astype(
        "float64"
    )
    cloud = pd.to_numeric(df["zachmurzenie"], errors="coerce").astype("float64")
    radiation = pd.to_numeric(
        df["promieniowanie_calkowite"], errors="coerce"
    ).astype("float64")
    precipitation = pd.to_numeric(df["opad_konwekcyjny"], errors="coerce").astype(
        "float64"
    )

    df["stopniogodziny_grzania"] = (18.0 - temperature).clip(lower=0)
    df["stopniogodziny_chlodzenia"] = (temperature - 22.0).clip(lower=0)
    radians = np.deg2rad(wind_direction.mod(360))
    df["kierunek_wiatru_sin"] = np.sin(radians)
    df["kierunek_wiatru_cos"] = np.cos(radians)
    df["wiatr_kwadrat"] = wind_speed.clip(lower=0, upper=40).pow(2)
    df["wiatr_szescian"] = wind_speed.clip(lower=0, upper=40).pow(3)
    df["zachmurzenie_frac"] = np.where(cloud.abs().le(1.5), cloud, cloud / 100.0)
    df["promieniowanie_po_chmurach"] = radiation * (1.0 - df["zachmurzenie_frac"].clip(0, 1))
    df["opad_log1p"] = np.log1p(precipitation.clip(lower=0))

    categorical = [
        "oddzial_code",
        "punkt",
        "grupa",
        "klient_nazwa",
        "kierunek_energii_norm",
        "rodzaj",
    ]
    for col in categorical:
        df[col] = df[col].astype("string").fillna("__BRAK__")

    numeric = [
        *WEATHER_FEATURES,
        "pogoda_dostepna",
        "weather_lead_hours",
        "godzina",
        "dzien_tygodnia",
        "miesiac",
        "weekend",
        "swieto_PL",
        "dzien_zmiany_czasu",
        "godzina_sin",
        "godzina_cos",
        "dzien_tyg_sin",
        "dzien_tyg_cos",
        "dzien_roku_sin",
        "dzien_roku_cos",
        "stopniogodziny_grzania",
        "stopniogodziny_chlodzenia",
        "kierunek_wiatru_sin",
        "kierunek_wiatru_cos",
        "wiatr_kwadrat",
        "wiatr_szescian",
        "zachmurzenie_frac",
        "promieniowanie_po_chmurach",
        "opad_log1p",
        *[f"lag_{24 * days}h" for days in lag_days],
        (
            LAG_MEAN_FEATURE
            if tuple(lag_days) == DEFAULT_LAG_DAYS
            else f"lag_srednia_{min(lag_days)}_{max(lag_days)}_dni"
        ),
        "liczba_lagow_bazowych",
        "lag_odchylenie_3_14_dni",
        "lag_trend_krotki_vs_dlugi",
        "lag_7d_vs_srednia",
    ]
    return df, FeatureSpec(categorical=categorical, numeric=numeric)

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BRANCHES = {
    "BIA": ("Białystok", 0.0),
    "LUB": ("Lublin", 0.8),
    "ŁZE": ("Łódź", 1.5),
    "RZE": ("Rzeszów", 2.0),
}


def generate(output_dir: Path, days: int = 90, seed: int = 42) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2025-01-01", periods=days * 24, freq="h")

    weather_rows = []
    weather_lookup: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
    for city, city_offset in BRANCHES.values():
        for valid in times:
            hour = valid.hour
            doy = valid.dayofyear
            daylight = max(0.0, np.sin(np.pi * (hour - 6) / 12.0))
            temperature = 4 + 9 * np.sin(2 * np.pi * (doy - 30) / 365.25) + 3 * np.sin(2 * np.pi * (hour - 8) / 24) + city_offset
            cloud = float(np.clip(55 + 30 * np.sin(2 * np.pi * doy / 11) + rng.normal(0, 10), 0, 100))
            radiation = float(max(0, 520 * daylight * (1 - 0.006 * cloud)))
            wind_speed = float(max(0, 4.5 + 2 * np.sin(2 * np.pi * doy / 8) + rng.normal(0, 1)))
            wind_direction = float((180 + 80 * np.sin(2 * np.pi * doy / 17) + rng.normal(0, 15)) % 360)
            values = {
                "temperatura": temperature,
                "predkoscWiatru": wind_speed,
                "kierunekWiatru": wind_direction,
                "zachmurzenie": cloud,
                "intensywnoscOpadowKonwekcyjnych": max(0, rng.normal(0.15, 0.3)),
                "widocznosc": max(1000, rng.normal(18000, 2500)),
                "calkowitePromieniowanieSloneczneGodzinowe": radiation,
                "bezposredniePromieniowanieSloneczneGodzinowe": radiation * (1 - cloud / 120),
                "albedoPrognozowane": 0.2,
                "warstwaSniegu": max(0, (0 - temperature) * 0.002),
            }
            weather_lookup[(city, valid)] = values
            # Dwa vintage'y; pipeline z lead=24 h ma wybrać -30 h, a odrzucić -12 h.
            for lead in (30, 12):
                weather_rows.append(
                    {
                        "execId": f"DEMO-{city}-{valid:%Y%m%d%H}-{lead}",
                        "punkt": city,
                        "czasDanychZrodlaCET": valid - pd.Timedelta(hours=lead),
                        "czasDanychZrodlaUTC": (valid - pd.Timedelta(hours=lead))
                        .tz_localize("Europe/Warsaw")
                        .tz_convert("UTC"),
                        "dataGodzinaCET": valid,
                        "dataGodzinaUTC": valid.tz_localize("Europe/Warsaw").tz_convert("UTC"),
                        "dataCET": valid.normalize(),
                        "godzinaHandlowa25": valid.hour + 1,
                        **values,
                    }
                )

    weather = pd.DataFrame(weather_rows)
    energy_rows = []
    customer_no = 0
    for branch, (city, _) in BRANCHES.items():
        for local_customer in range(2):
            customer_no += 1
            customer = f"DEMO_KLIENT_{customer_no:02d}"
            scale = 700 + customer_no * 130
            for valid in times:
                values = weather_lookup[(city, valid)]
                hour = valid.hour
                weekday = valid.dayofweek
                demand_shape = 0.75 + 0.28 * np.exp(-((hour - 9) / 4) ** 2) + 0.35 * np.exp(-((hour - 19) / 3) ** 2)
                weekend = 0.82 if weekday >= 5 else 1.0
                heating = max(0, 17 - values["temperatura"]) * 0.018
                consumption = max(0, scale * weekend * (demand_shape + heating) + rng.normal(0, 35))
                generation = max(
                    0,
                    scale * 0.9 * values["calkowitePromieniowanieSloneczneGodzinowe"] / 520
                    + 12 * values["predkoscWiatru"]
                    - 40
                    + rng.normal(0, 22),
                )
                for code, direction, target in [
                    (1, "czynne pobranie", consumption),
                    (-1, "czynne oddanie", generation),
                ]:
                    energy_rows.append(
                        {
                            "oddzial_code": branch,
                            "kierunek_code": code,
                            "grupa": "DEMO",
                            "klient_nazwa": customer,
                            "kierunek_energii": direction,
                            "doba_handlowa": valid.normalize(),
                            "rodzaj": 4,
                            "godzina_handlowa": hour + 1,
                            "wartosc_rzeczywista": round(target, 3),
                        }
                    )

    energy = pd.DataFrame(energy_rows)
    energy_path = output_dir / "energia_demo.csv"
    weather_path = output_dir / "pogoda_demo.csv"
    energy.to_csv(energy_path, index=False, encoding="utf-8-sig")
    weather.to_csv(weather_path, index=False, encoding="utf-8-sig")
    return energy_path, weather_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generuje wyłącznie syntetyczne dane do smoke testu.")
    parser.add_argument("--output-dir", default="/tmp/mdd_demo")
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    energy, weather = generate(Path(args.output_dir), days=args.days)
    print(energy)
    print(weather)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

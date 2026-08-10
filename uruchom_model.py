"""Klikalny launcher modelu MDD do uruchamiania strzałką ▶ w VS Code.

Plik nie zawiera ścieżek konkretnego użytkownika. Po uruchomieniu pokazuje okna
wyboru pliku energii i katalogu wynikowego, a następnie wywołuje właściwy CLI.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SQL_PATH = BASE_DIR / "mdd_forecasting" / "sql" / "pogoda_mdd.sql"
MODEL_BACKEND = "catboost"
MIN_LEAD_HOURS = 24


def build_cli_args(
    energy_path: str | Path,
    output_dir: str | Path,
    model_backend: str = MODEL_BACKEND,
    min_lead_hours: int = MIN_LEAD_HOURS,
) -> list[str]:
    """Buduje przenośne argumenty bez prywatnych i służbowych ścieżek w kodzie."""

    return [
        "--energy",
        str(Path(energy_path).expanduser().resolve()),
        "--weather-sql",
        str(SQL_PATH),
        "--output-dir",
        str(Path(output_dir).expanduser().resolve()),
        "--model-backend",
        model_backend,
        "--min-lead-hours",
        str(int(min_lead_hours)),
    ]


def _write_error_log(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "blad_uruchomienia.txt"
    log_path.write_text(traceback.format_exc(), encoding="utf-8")
    return log_path


def main() -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError:
        print(
            "Brak tkinter. Na Windows zainstaluj standardową wersję Pythona z python.org."
        )
        return 1

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        energy = filedialog.askopenfilename(
            parent=root,
            title="Wybierz Excel z danymi energii MDD",
            filetypes=[
                ("Excel XLSX", "*.xlsx"),
                ("Excel XLSM", "*.xlsm"),
                ("CSV", "*.csv"),
                ("Wszystkie pliki", "*.*"),
            ],
        )
        if not energy:
            return 0

        output = filedialog.askdirectory(
            parent=root,
            title="Wybierz katalog, do którego zapisać wyniki",
            initialdir=str(Path(energy).resolve().parent),
            mustexist=False,
        )
        if not output:
            return 0

        if not SQL_PATH.exists():
            messagebox.showerror(
                "Brak pliku SQL",
                f"Nie znaleziono pliku:\n{SQL_PATH}\n\nOtwórz w VS Code cały folder repozytorium klien.",
                parent=root,
            )
            return 1

        confirmed = messagebox.askokcancel(
            "Uruchomić model MDD?",
            "Model pobierze pogodę z firmowego SQL Server i rozpocznie uczenie.\n\n"
            f"Dane energii:\n{energy}\n\n"
            f"Wyniki:\n{output}\n\n"
            f"Model: {MODEL_BACKEND}\nMinimalny lead pogody: {MIN_LEAD_HOURS} h",
            parent=root,
        )
        if not confirmed:
            return 0

        args = build_cli_args(energy, output)
        print("Uruchamiam model MDD...")
        print(f"Excel: {energy}")
        print(f"Wyniki: {output}")
        try:
            from mdd_forecasting.cli import main as cli_main

            exit_code = cli_main(args)
        except ModuleNotFoundError as exc:
            messagebox.showerror(
                "Brak bibliotek Pythona",
                "Najpierw w terminalu VS Code wykonaj:\n\n"
                "python -m pip install -r mdd_forecasting\\requirements-production.txt\n\n"
                f"Brakujący moduł: {exc.name}",
                parent=root,
            )
            return 1
        except Exception as exc:
            log_path = _write_error_log(Path(output))
            messagebox.showerror(
                "Model zakończył się błędem",
                f"Przyczyna:\n{type(exc).__name__}: {exc}\n\n"
                "Pełne szczegóły zapisano w pliku:\n"
                f"{log_path}\n\nWyślij ten plik przy zgłaszaniu problemu.",
                parent=root,
            )
            return 1

        if exit_code == 0:
            open_folder = messagebox.askyesno(
                "Gotowe",
                f"Model zakończył pracę.\n\nRaport:\n{Path(output) / 'wyniki_mdd.xlsx'}\n\n"
                "Czy otworzyć katalog wynikowy?",
                parent=root,
            )
            if open_folder and os.name == "nt":
                os.startfile(str(Path(output).resolve()))  # type: ignore[attr-defined]
        return int(exit_code)
    finally:
        root.destroy()


if __name__ == "__main__":
    raise SystemExit(main())

# Jak model pobiera pogodę z SQL Server

## Przepływ

```text
Excel energii
  -> odczyt kolumn A:I
  -> min/max Doby Handlowej
  -> 5 parametrów zapytania SQL
  -> pyodbc.connect(...)
  -> PGESA_MarketAnalytics.wa.vPogodaPrognoza
  -> pandas DataFrame z pogodą
  -> normalizacja i kontrola unikalności
  -> join z energią
  -> cechy i modele POBRANIE / ODDANIE
```

Kod połączenia jest w `database.py`, a zapytanie w `sql/pogoda_mdd.sql`.

## Połączenie

Domyślne ustawienia odpowiadają konfiguracji ze zdjęcia:

```text
DRIVER={ODBC Driver 17 for SQL Server};
SERVER=MISDWHPRD.GKPGE.PL;
DATABASE=PGESA_MarketAnalytics;
Trusted_Connection=yes;
APP=MDD Forecasting;
```

`Trusted_Connection=yes` oznacza, że na komputerze Windows używane jest aktualne
konto domenowe. W kodzie nie ma loginu ani hasła. Konto uruchamiające skrypt musi mieć
prawo `SELECT` do widoku `[PGESA_MarketAnalytics].[wa].[vPogodaPrognoza]`.

Jeżeli środowisko wymaga innego connection stringa, można przekazać go przez zmienną
środowiskową `MDD_SQL_CONNECTION_STRING`. Nie należy commitować haseł do repozytorium.

## Parametry zapytania

Plik SQL przyjmuje pięć parametrów przez placeholdery `?`:

```sql
DECLARE @validFromCET        datetime2(0) = ?;
DECLARE @validToCETExclusive datetime2(0) = ?;
DECLARE @minLeadHours        int          = ?;
DECLARE @wlasciciel          nvarchar(50) = ?;
DECLARE @typPrognozy         nvarchar(50) = ?;
```

Python przekazuje je przez `params=`, a nie przez `re.sub`. Daty są obiektami
`datetime`, lead liczbą całkowitą, a wartości tekstowe nie są wykonywane jako SQL.

Zakres jest automatycznie wyznaczany z całego Excela:

```text
validFromCET        = min(Doba Handlowa)
validToCETExclusive = max(Doba Handlowa) + 1 dzień
```

Można go nadpisać opcjami `--valid-from` i `--valid-to-exclusive`.

## Co robi kwerenda

1. Czyta sześć punktów: Białystok, Lublin, Łódź, Rzeszów, Zamość i
   Skarżysko-Kamienna.
2. Filtruje `wlasciciel = PGESA` i `typ = Open Meteo`.
3. Ogranicza `dataGodzinaCET` do zakresu Excela.
4. Sprawdza w UTC, czy wydanie prognozy było dostępne co najmniej
   `minLeadHours` przed valid time.
5. Numeruje prognozy funkcją `ROW_NUMBER()` osobno dla punktu, daty i godziny
   handlowej.
6. Zachowuje najnowszy dozwolony historyczny vintage (`rn = 1`).

Rozdzielenie czasu jest krytyczne:

- `dataGodzinaUTC/CET` — godzina, dla której prognozowana jest pogoda;
- `czasDanychZrodlaUTC/CET` — godzina wydania prognozy.

Do energii dopasowujemy valid time. Issue time służy tylko do wyboru wersji dostępnej
w chwili sporządzania programu. Dzięki temu historyczny test nie używa poprawionej
prognozy pogody, której wtedy jeszcze nie znano.

## Uruchomienie na komputerze domenowym

```powershell
python -m pip install -r mdd_forecasting\requirements-production.txt

python -m mdd_forecasting `
  --energy "C:\Users\10200871\Desktop\dane_energii.xlsx" `
  --weather-sql "mdd_forecasting\sql\pogoda_mdd.sql" `
  --output-dir "C:\Users\10200871\Desktop\wyniki_mdd" `
  --sql-driver "ODBC Driver 17 for SQL Server" `
  --model-backend catboost `
  --min-lead-hours 24
```

`--weather-sql` i `--weather` są wzajemnie wykluczające. Ten sam model można więc
uruchomić bez bazy na wcześniej wyeksportowanym CSV.

## Szyfrowanie i sterownik

Jeśli firmowy SQL Server ma poprawny certyfikat, należy dodać `--sql-encrypt`.
`--sql-trust-server-certificate` omija weryfikację łańcucha i powinien być używany
wyłącznie zgodnie z polityką administratorów. Dla ODBC 18 można wskazać:

```powershell
--sql-driver "ODBC Driver 18 for SQL Server" --sql-encrypt
```

## Typowe błędy

- `Brak sterownika ODBC` — zainstalować Microsoft ODBC Driver 17/18 lub podać jego
  dokładną nazwę przez `--sql-driver`.
- `Login failed` — konto Windows nie ma dostępu albo skrypt działa pod inną
  tożsamością, np. Jenkins/service account.
- `Certificate chain was not trusted` — poprawić firmowy CA/certyfikat; nie ukrywać
  błędu bez uzgodnienia z administratorem.
- `Zapytanie SQL nie zwróciło rekordów` — sprawdzić zakres dat, punkty, właściciela,
  typ prognozy i `minLeadHours`.
- dużo `pogoda_niedopasowana` — sprawdzić mapowanie oddział–miasto i 25. godzinę.

Połączenie z firmową bazą nie może zostać przetestowane poza siecią/domeną. Lokalne
testy sprawdzają kontrakt parametrów, zakres dat, join, UTC/DST i pełny model na danych
syntetycznych.

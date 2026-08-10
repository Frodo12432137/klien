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
validFromCET        = max(min(Doba Handlowa), 2024-10-01)
validToCETExclusive = max(Doba Handlowa) + 1 dzień
```

Pogoda w źródle jest dostępna od **2024-10-01**, dlatego w trybie pięciu parametrów
dolna granica jest clampowana do tej daty także wtedy, gdy Excel albo opcja
`--valid-from` wskazuje wcześniejszy dzień. `--valid-to-exclusive` nadal określa
górną granicę półotwartego zakresu.

Clamp dotyczy wyłącznie zapytania pogodowego. Wiersze energii sprzed `2024-10-01`
pozostają w danych jako historia klienta, źródło kalendarza oraz materiał do budowy
lagów D-3...D-14.

### Zwykły firmowy SQL bez `?`

Program obsługuje również plik SQL, który nie ma żadnych placeholderów `?` i sam
ustawia zmienne przez `DECLARE`, np. `@data_start` i `@data_stop`. Taki plik jest
wykonywany dokładnie w zapisanej postaci. Wtedy program nie może automatycznie
przekazać zakresu z Excela ani `minLeadHours`, dlatego kwerenda musi samodzielnie:

- ustawić właściwy zakres dat;
- ustawić dolną granicę pogody nie wcześniej niż `2024-10-01` (np. przez
  `CASE`/`IF`, jeżeli własny `DECLARE @data_start` może wskazać wcześniejszy dzień);
- zwrócić wymagane kolumny `punkt` i `dataGodzinaCET`;
- najlepiej zwrócić również `czasDanychZrodlaCET/UTC`, aby Python mógł sprawdzić
  historyczny vintage bez leakage.

Obsługiwane są zatem dwa kontrakty: dokładnie **5** placeholderów dla wersji
parametryzowanej albo dokładnie **0** dla samodzielnego zwykłego SQL-a. Inna liczba
jest traktowana jako niekompletna kwerenda i kończy się czytelnym błędem.

## Okres bez pogody i wartości NULL

Brak pogody przed `2024-10-01` nie usuwa obserwacji energii. Lewostronne łączenie
zachowuje każdy wiersz Excela, również gdy miasto i valid time nie mają rekordu
pogodowego. Takie obserwacje nadal uczestniczą w historii klienta, cechach
kalendarzowych i lagach D-3...D-14.

Kolumn pogodowych z `NULL` nie zerujemy w przygotowanych danych ani raporcie.
`NULL/NaN` oznacza brak dostępnej informacji, natomiast `0` może być prawdziwą
temperaturą, brakiem opadu, brakiem wiatru albo promieniowaniem nocnym. CatBoost
obsługuje brak natywnie; backend sklearn imputuje go dopiero wewnątrz pipeline'u i
dodaje wskaźnik braku. Sam brak wartości pogodowej nie jest warunkiem odrzucenia
wiersza; jego dopasowanie pozostaje widoczne w polach `pogoda_dopasowana`,
`weather_status`, liczbie dostępnych cech oraz kontroli jakości.

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

Launcher `uruchom_model.py` korzysta z profilu szybkiego: zakres SQL wynika wyłącznie
z najnowszych 150 000 wierszy energii, timeout połączenia wynosi 15 sekund, a timeout
zapytania 300 sekund. Terminal pokazuje osobne komunikaty przed i po SQL wraz z liczbą
pobranych rekordów. Ostrzeżenie pandas o bezpośrednim połączeniu DBAPI2 jest ukryte,
ponieważ użycie pyodbc jest tutaj zamierzone i przetestowane.

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

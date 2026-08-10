# Model MDD: pobranie i oddanie energii

To jest działające MVP, które łączy każdy wiersz danych energii z właściwą godzinową
prognozą pogody, wykonuje chronologiczny backtest i uczy dwa modele:

1. `POBRANIE` — prognoza `czynne pobranie`;
2. `ODDANIE` — prognoza `czynne oddanie`.

Modele są globalne, czyli uczą się na wszystkich klientach, ale nazwa klienta,
oddział, miasto i kierunek pozostają cechami. To stabilniejsze niż osobny model dla
każdej nazwy: mały klient może skorzystać z prawidłowości poznanych na całym zbiorze.

## Uruchomienie strzałką ▶ w VS Code

Na komputerze służbowym nie trzeba wpisywać żadnych ścieżek do kodu:

1. Otwórz w VS Code cały folder repozytorium `klien`.
2. Jednorazowo zainstaluj biblioteki:

   ```powershell
   python -m pip install -r mdd_forecasting\requirements-production.txt
   ```

3. Otwórz główny plik `uruchom_model.py`.
4. Kliknij strzałkę **Run Python File** ▶ w prawym górnym rogu.
5. W pierwszym oknie wybierz Excel energii, a w drugim katalog wynikowy.
6. Potwierdź uruchomienie. Po zakończeniu program pokaże lokalizację raportu
   `wyniki_mdd.xlsx`, pakietu `pakiet_modelu_mdd.joblib` i zaproponuje otwarcie
   folderu.

Launcher zawsze odnajduje SQL względem folderu repozytorium, używa CatBoost,
`min-lead-hours=24` i nie przechowuje żadnej prywatnej ani służbowej ścieżki.

### Pełny trening

`uruchom_model.py` uruchamia profil `full_training`. Czyta cały Excel i wszystkie
pasujące arkusze `Dane_*`, bez limitu wierszy wejściowych i bez limitu liczby
rekordów treningowych na kierunek. Chronologiczny OOF nadal służy jako uczciwa
ocena poza próbą: model oraz kalibracja `alpha` korzystają wyłącznie z danych
wcześniejszych od ocenianego okna.

Po backteście powstają dwa końcowe modele — `POBRANIE` i `ODDANIE` — uczone na całej
użytecznej historii. Użyteczna historia oznacza prawidłowe realizacje, dla których
można zbudować baseline z co najmniej jednej analogicznej godziny D-3...D-14.
Pierwsze trzy dni danej serii są więc warm-upem bez baseline, a po 14 dniach może
być dostępne pełne okno 12 lagów.

Pełny przebieg zapisuje `pakiet_modelu_mdd.joblib` do późniejszego, osobnego programu
scoringowego. Obecny `uruchom_model.py` tworzy bundle, lecz jeszcze nie ładuje go w
celu wygenerowania kolejnej prognozy. Przyszły scoring będzie osobnym procesem, bez
ponownego uczenia, ale nadal będzie wymagał bieżących danych i historii D-3...D-14.
Musi używać mapowania osadzonego w pakiecie oraz odrzucać godziny niepóźniejsze niż
zapisany cutoff treningu, aby zachować poprawny moment prognozy.

Wczytanie całego Excela może znacząco zwiększyć zużycie pamięci i całkowity czas
pracy. `max-fit-minutes=3` dotyczy każdego pojedynczego fitu CatBoost; nie ogranicza
całego pipeline'u, zapytania SQL, odczytu Excela, budowy lagów ani zapisu raportu.

## Ziarno danych i mapowanie

Jedna obserwacja modelu to jeden oryginalny wiersz Excela. Arkusze `Dane_01`,
`Dane_02`, ... są scalane w całości, ale zachowane zostają `source_sheet` i
`source_row`.
Kolumny A:I są czytane pozycyjnie, ponieważ na zdjęciach A i D mają ten sam nagłówek
`Nazwa`:

| Excel | Nazwa w modelu | Rola |
|---|---|---|
| A | `oddzial_code` | mapowanie do punktu pogodowego |
| B | `kierunek_code` | kontrola jakości `1/-1`; nie jest głównym kierunkiem |
| C | `grupa` | cecha kategoryczna |
| D | `klient_nazwa` | identyfikator szeregu/cecha kategoryczna |
| E | `kierunek_energii` | źródło prawdy: pobranie albo oddanie |
| F | `doba_handlowa` | data |
| G | `rodzaj` | cecha kategoryczna |
| H | `godzina_handlowa` | godzina 1–24/25 |
| I | `wartosc_rzeczywista` | target modelu |

Potwierdzone mapowanie znajduje się w `config/oddzial_miasto.csv`:

- `BIA -> Białystok`
- `LUB -> Lublin`
- `ŁZE/LZE -> Łódź`
- `RZE/RZ -> Rzeszów`

Kody Zamościa i Skarżyska-Kamiennej są celowo nieaktywne, dopóki nie zostaną
potwierdzone na prawdziwych unikalnych wartościach kolumny A. Pustych oddziałów nie
uzupełniamy metodą `forward fill`, bo mogłaby przypisać klienta do złego miasta.

## Jak łączona jest pogoda

W tabeli pogodowej są dwa różne czasy:

- `dataGodzinaCET` — **valid time**, czyli godzina, której dotyczy pogoda;
- `dataGodzinaUTC` — jednoznaczny wewnętrzny klucz czasu, także podczas 25. godziny;
- `czasDanychZrodlaCET` — **issue time**, czyli chwila utworzenia/dostępności prognozy.

Energię łączymy po mieście i valid time, najlepiej po
`punkt + dataCET + godzinaHandlowa25`. Issue time służy wyłącznie do wybrania
właściwego historycznego wydania prognozy. Przy `min_lead_hours=24` model może użyć
tylko prognozy wydanej co najmniej 24 godziny przed prognozowaną godziną. To usuwa
przeciek informacji z późniejszych aktualizacji pogody.

Gotowy ekstrakt jest w `sql/pogoda_mdd.sql`.

## Zmienne modelu

### Historia energii

- wartości z analogicznej godziny każdego dnia od D-3 do D-14, czyli lagi
  `72, 96, 120, ..., 336 h`;
- średnia ze wszystkich dostępnych analogicznych godzin w tym oknie, czyli
  `wartosc_bazowa`;
- liczba i odchylenie dostępnych lagów;
- trend D-3...D-6 względem D-10...D-14 oraz różnica D-7 względem średniej.

To zwykle najmocniejsze zmienne dla poboru. Lagi są łączone po tej samej godzinie
handlowej 3–14 dni wcześniej, a kolejność walidacji jest oparta na UTC. Model celowo
nie używa D-1 ani D-2, ponieważ zgodnie z dostępnością danych pierwsza pewna
realizacja pochodzi z D-3.

To zakłada codzienną prognozę kroczącą: przy prognozowaniu każdej kolejnej doby
wykonanie D-3 jest już dostępne. Jeżeli jednorazowo prognozowany byłby cały horyzont
14 dni, trzeba zamrozić historię na jednym cutoffie i nie aktualizować lagów wynikami
pojawiającymi się wewnątrz tego horyzontu. Trzeba też potwierdzić, że wartości D-3 są
faktycznie dostępne operacyjnie i nie są późniejszą korektą rozliczeniową.

### Kalendarz

- godzina, dzień tygodnia, weekend, miesiąc i dzień roku;
- cykliczne `sin/cos` godziny, tygodnia i roku;
- polskie święta ruchome i stałe;
- dzień przejścia CET/CEST.

Kalendarz opisuje rytm pracy klientów, sezonowość i różnicę między dniem roboczym a
wolnym.

### Pogoda

| Zmienna | Główne zastosowanie |
|---|---|
| temperatura | ogrzewanie/chłodzenie i pobór |
| stopniogodziny grzania/chłodzenia | nieliniowa reakcja poboru na temperaturę |
| prędkość wiatru | generacja wiatrowa; dodatkowo chłodzenie modułów PV |
| kierunek wiatru jako `sin/cos` | kierunek jest wielkością kołową: 359° leży blisko 0° |
| całkowite i bezpośrednie promieniowanie | najważniejsze wejścia dla PV/oddania |
| zachmurzenie | tłumienie produkcji PV i pośrednio zachowanie odbiorców |
| opad konwekcyjny i widoczność | zmienne pomocnicze/proxy pogody |
| albedo i warstwa śniegu | sezonowe warunki PV i pokrywa śnieżna |
| rzeczywisty lead prognozy pogody | jakość pogody zależy od horyzontu |

Dodatkowo tworzone są interakcje: sześcian prędkości wiatru, promieniowanie po
uwzględnieniu zachmurzenia i `log1p` opadu.

Praktyczna kolejność wdrażania cech:

1. **Obowiązkowe:** klient, kierunek, godzina/dzień tygodnia, lagi D-3...D-14,
   temperatura, promieniowanie, prędkość wiatru i miasto.
2. **Bardzo wartościowe po pozyskaniu:** typ źródła (`PV/wiatr/inne`), moc
   zainstalowana lub umowna, branża/taryfa klienta, godziny pracy oraz pozycja słońca.
3. **Uzupełniające:** opady, widoczność, albedo, śnieg i szczegółowe warstwy chmur.

Najpierw należy udowodnić wartość pierwszej grupy na backteście. Dodawanie wszystkich
kolumn tylko dlatego, że są dostępne, zwiększa ryzyko szumu i niestabilności.

### Kategorie

`klient_nazwa`, `grupa`, `rodzaj`, `oddzial_code`, `punkt` i kierunek energii nie są
traktowane jako zwykłe liczby. CatBoost obsługuje je natywnie jako kategorie, a
awaryjny backend sklearn stosuje target encoding wewnątrz pipeline'u treningowego.

## Algorytm i uczenie

Model jest hybrydą baseline i korekty ML:

```text
baseline(t) = średnia tej samej godziny z D-3...D-14
residuum(t) = wartość_rzeczywista(t) - baseline(t)
korekta(t)  = CatBoost(cechy klienta, czasu, historii i pogody)
prognoza(t) = max(0, baseline(t) + alpha * korekta(t))
```

CatBoost uczy się podpisanego residuum z funkcją straty MAE na oryginalnej skali,
bez `log1p/expm1`. Dzięki temu może zarówno podwyższyć, jak i obniżyć baseline, a
poprzednie systematyczne zaniżanie dużych wolumenów nie jest wzmacniane transformacją.
Awaryjny `HistGradientBoostingRegressor` używa tego samego celu resztowego.

`alpha` jest wybierane oddzielnie dla pobrania i oddania na siedmiodniowym oknie
kalibracyjnym leżącym przed testem OOF. Sprawdzanych jest 21 wartości od 0 do 1.
Korekta zostaje włączona wyłącznie przy co najmniej 200 rekordach i minimum 2%
poprawy MAE względem baseline. W przeciwnym razie `alpha=0`, więc wynik jest dokładnie
baseline. Ten mechanizm ogranicza ryzyko użycia słabszej korekty, ale nie gwarantuje
przewagi w każdym nieznanym przyszłym okresie.

Pobranie i oddanie są rozdzielone, bo mają różną fizykę. Pobranie zależy zwykle od
kalendarza, historii i temperatury. Oddanie zależy mocniej od promieniowania,
zachmurzenia i/lub wiatru. Jeżeli później dostępny będzie typ źródła (`PV`, `wiatr`,
inne), należy go koniecznie dodać; bez niego model uśrednia różne mechanizmy.

## Walidacja

Nie stosujemy losowego `train_test_split`. Backtest ma okna rosnące w czasie:

```text
fold 1: przeszłość ----------> 14 dni testu
fold 2: przeszłość ----------------------> 14 dni testu
fold 3: przeszłość ----------------------------------> 14 dni testu
```

Każdy fold zachowuje kolejność `trening -> kalibracja alpha -> test`. Model residualny
i wybór `alpha` nie widzą targetów z okna testowego. Cechy lagowe są aktualizowane
as-of dla danej prognozowanej godziny, więc używają tylko realizacji sprzed co najmniej
trzech dni. Przy braku naturalnej średniej baseline korzysta kolejno z mediany
klient–kierunek–godzina, oddział–kierunek–godzina, kierunek–godzina, kierunku i całego
prefiksu treningowego.

Raportowane metryki:

- `MAE` — główna, łatwa do interpretacji w jednostce `Wartości`;
- `RMSE` — mocniej karze duże błędy i nietrafione szczyty;
- `bias = średnia(predykcja - realizacja)` — pokazuje systematyczne zawyżanie/zaniżanie;
- `WAPE` i `sMAPE` — porównanie klientów o różnej skali bez klasycznego problemu
  MAPE przy zerach.

Metryki są liczone globalnie, według kierunku, miasta oraz jako średnia makro po
klientach. Raport zawiera trzy porównywalne warianty: `HYBRYDA_OOF`, diagnostyczną
pełną korektę `KOREKTA_ML_SUROWA_OOF` i `SREDNIA_ANALOGICZNYCH_D3_D14`. Wszystkie są
liczone na tych samych wierszach OOF. Predykcje zapisują również baseline, surową
korektę, `blend_alpha`, strategię, liczebność i wynik kalibracji oraz powód fallbacku.
Permutacyjna ważność cech może być liczona dla modelu residuum na danych testowych,
osobno dla pobrania i oddania. Klikalny pełny launcher pomija ten kosztowny etap;
można go włączyć w ręcznym CLI.

## Uruchomienie bezpośrednio z SQL Server

Domyślny tryb odpowiada konfiguracji ze zdjęcia: serwer
`MISDWHPRD.GKPGE.PL`, baza `PGESA_MarketAnalytics`, sterownik ODBC 17 i
`Trusted_Connection=yes`. Zakres zapytania jest wyliczany z min/max `Doby Handlowej`
w Excelu.

```powershell
python -m pip install -r mdd_forecasting\requirements-production.txt

python -m mdd_forecasting `
  --energy "C:\sciezka\dane_energii.xlsx" `
  --weather-sql "mdd_forecasting\sql\pogoda_mdd.sql" `
  --output-dir "C:\sciezka\wyniki" `
  --model-backend catboost `
  --min-lead-hours 24
```

Pełny opis połączenia, parametrów i działania kwerendy znajduje się w
`DATABASE_SQL.md`.

## Uruchomienie z eksportem pogodowym

```bash
python -m pip install -r mdd_forecasting/requirements.txt

python -m mdd_forecasting \
  --energy /sciezka/dane_energii.xlsx \
  --weather /sciezka/pogoda.csv \
  --output-dir /sciezka/wyniki \
  --min-lead-hours 24
```

Szybki test bez danych firmowych:

```bash
python -m mdd_forecasting.generate_demo_data --output-dir /tmp/mdd_demo --days 30
python -m mdd_forecasting \
  --energy /tmp/mdd_demo/energia_demo.csv \
  --weather /tmp/mdd_demo/pogoda_demo.csv \
  --output-dir /tmp/mdd_demo/wyniki \
  --validation-days 3 --folds 2 --max-iter 60 --skip-importance
```

Wynik zawiera gotowy skoroszyt `wyniki_mdd.xlsx` z arkuszami podsumowania,
predykcji, metryk, ważności cech, kontroli jakości, mapowania i konfiguracji. Dla
dużych danych predykcje są dzielone na kolejne arkusze zgodnie z limitem Excela.
Równolegle zapisywane są CSV, modele kierunkowe oraz kompletny
`pakiet_modelu_mdd.joblib`. Pakiet przechowuje oba końcowe modele, ich `alpha`, schemat
cech, konfigurację, mapowanie, profile fallbacku oraz końcówkę historii potrzebną do
lagów D-3...D-14.

`wartosc_przewidywana` w raporcie treningowym służy przede wszystkim do uczciwego
OOF. Brak targetu wewnątrz historii dostaje status `HISTORYCZNY_BRAK_TARGETU`, a nie
fałszywy status przyszłości. Wynik modelu końcowego na historii — jeżeli zostanie
wyliczony w ręcznym CLI — jest in-sample i nie wolno na nim raportować jakości.
Klikalny pełny launcher pomija scoring całej historii, aby ograniczyć czas i pamięć.

Pakiet nie wykonuje prognozy samodzielnie. Ten etap tylko go tworzy; osobny program
scoringowy, który wczyta `pakiet_modelu_mdd.joblib`, bieżące dane, wymaganą historię
D-3...D-14 i pogodę, będzie kolejnym etapem rozwiązania.

## Co trzeba zrobić przed produkcją

1. Dostarczyć rzeczywisty Excel i uruchomić przygotowane połączenie SQL w sieci
   firmowej; lokalnie nie ma danych ani dostępu do hurtowni.
2. Potwierdzić kody Zamościa i Skarżyska-Kamiennej oraz sposób obsługi pustej kolumny A.
3. Potwierdzić jednostkę `Wartości`, numerację godziny handlowej i przypadek 25. godziny.
4. Ustalić operacyjny cutoff, np. kiedy dokładnie w D-1 powstaje program; wtedy zamiast
   uproszczonego leadu 24 h wybierać pogodę dostępną przed tym cutoffem.
5. Dodać typ instalacji i moc umowną/zainstalowaną. Dla PV można wtedy wymusić zero w
   nocy; bez typu źródła taka reguła byłaby ryzykowna dla oddania z wiatru.
6. Po zebraniu wyników porównać boosting z CatBoost i prognozami kwantylowymi
   `P05/P50/P95`. Pipeline ma już opcjonalny backend `--model-backend catboost`;
   jego przewagę trzeba potwierdzić na tych samych chronologicznych foldach.
7. Arkusze ze zdjęć dochodzą do limitu Excela. Produkcyjnie Excel powinien być tylko
   wejściem/raportem, a tabela treningowa powinna trafić do SQL lub Parquet. MVP czyta
   XLSX strumieniowo, ale finalnie składa dane w pamięci, więc przy kilku milionach
   wierszy trzeba dodać etap stagingu i przetwarzanie partiami.

MVP ma służyć jako uczciwy punkt startowy, a nie jako deklaracja jakości bez testu na
rzeczywistych danych.

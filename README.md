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

`uruchom_model.py` działa w profilu `full_training`:

- czyta **cały wskazany Excel** i wszystkie pasujące arkusze `Dane_*`; nie stosuje
  limitu 150 000 wierszy ani limitu liczby rekordów treningowych na kierunek;
- buduje analogiczne lagi D-3...D-14; pierwsze trzy dni danej serii są warm-upem
  bez baseline, a po 14 dniach dostępne może być pełne okno 12 lagów;
- wykonuje chronologiczny OOF, w którym trening i kalibracja zawsze poprzedzają
  oceniane okno — OOF pozostaje uczciwą oceną poza próbą;
- po OOF uczy końcowe, oddzielne modele `POBRANIE` i `ODDANIE` na całej użytecznej
  historii, czyli na prawidłowych realizacjach z dostępnym baseline D-3...D-14;
- zapisuje raport oraz kompletny `pakiet_modelu_mdd.joblib`, przeznaczony do
  późniejszego użycia przez osobny program scoringowy.

Ten launcher tworzy pakiet, ale **nie uruchamia jeszcze późniejszego scoringu z
pakietu**. Dedykowany program będzie osobnym etapem: wczyta bundle, bieżące dane,
historię wymaganą dla lagów D-3...D-14 i pogodę, bez ponownego treningu.
Scoring będzie korzystał z mapowania zapisanego w pakiecie i dopuści wyłącznie
godziny późniejsze od zapisanego końca treningu, aby nie mieszać przyszłości z
realizacjami użytymi podczas uczenia.

Pełny Excel może znacznie zwiększyć zużycie pamięci i czas odczytu, łączenia, budowy
lagów, treningu oraz zapisu raportu. Parametr `max-fit-minutes=3` ogranicza każdy
pojedynczy fit CatBoost osobno; nie jest limitem całego pipeline'u i nie obejmuje
Excela, SQL, tworzenia cech ani eksportu wyników.

## Ziarno danych i mapowanie

Jedna obserwacja modelu to jeden oryginalny wiersz Excela. Launcher pełnego treningu
scala w całości arkusze `Dane_01`, `Dane_02`, ..., zachowując `source_sheet` i
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

Prognoza pogody w źródle jest dostępna od **2024-10-01**. Ta granica ogranicza
wyłącznie zakres pobierania pogody: starsze wiersze Excela pozostają w zbiorze jako
historia klienta, źródło cech kalendarzowych oraz danych do lagów D-3...D-14. Nie
wolno ich usuwać tylko dlatego, że nie mają dopasowanej pogody.

Brakujące wartości pogodowe pozostają w przygotowanych danych i raporcie jako
`NULL/NaN`. Nie są tam zamieniane na zero, bo zero jest prawidłową wartością części
parametrów (np. opadu lub promieniowania nocą) i oznacza coś innego niż brak
pomiaru/prognozy. CatBoost obsługuje te braki natywnie; awaryjny backend sklearn
imputuje je dopiero wewnątrz pipeline'u wraz z osobną flagą braku. Sam `NULL` pogodowy
nie jest powodem usunięcia wiersza energii.

## Zmienne modelu

### Historia energii

- wartości z analogicznej godziny każdego dnia od D-3 do D-14, czyli lagi
  `72, 96, 120, ..., 336 h`;
- średnia ze wszystkich dostępnych analogicznych godzin w tym oknie — bazowa
  prognoza `wartosc_bazowa`;
- liczba dostępnych lagów oraz ich odchylenie standardowe;
- różnica średniej z D-3...D-6 względem D-10...D-14 i różnica D-7 względem
  średniej D-3...D-14.

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

`klient_nazwa`, `grupa`, `rodzaj`, `oddzial_code`, `punkt` i kierunek energii są
kodowane przez cross-fitted target encoding wewnątrz każdego okna treningowego.
Nie są traktowane jako zwykłe liczby.

## Algorytm i uczenie

Model jest hybrydą mocnego, prostego baseline oraz korekty ML. Dla godziny `t` działa
według schematu:

```text
baseline(t) = średnia dostępnych realizacji z tej samej godziny D-3...D-14
residuum(t) = wartość_rzeczywista(t) - baseline(t)
korekta(t)  = CatBoost(cechy klienta, czasu, historii i pogody)
prognoza(t) = max(0, baseline(t) + alpha * korekta(t))
```

CatBoost nie przewiduje więc całej wartości od zera, tylko podpisaną korektę
`residuum_wzgledem_D3_D14`. Uczy się z funkcją straty MAE na skali oryginalnej, bez
transformacji `log1p`, która w poprzednim wariancie sprzyjała zaniżaniu dużych
wolumenów. Jeżeli brakuje naturalnej średniej D-3...D-14, sam baseline korzysta z
historycznej mediany klienta, kierunku i godziny, a następnie z mediany zbioru
treningowego. Awaryjny backend `HistGradientBoostingRegressor` stosuje ten sam cel
resztowy i funkcję straty odporną na pojedyncze skoki.

Współczynnik `alpha` określa, jak dużą część korekty ML wolno dodać do baseline.
Nie jest dobierany na ocenianym oknie. Dla każdego kierunku i folda model wydziela
wcześniejsze okno kalibracyjne, domyślnie 7 dni, i sprawdza siatkę 21 wartości
`alpha` od 0 do 1 według MAE. Korekta zostaje zaakceptowana tylko wtedy, gdy:

- kalibracja zawiera co najmniej 200 prawidłowych obserwacji;
- najlepsza hybryda poprawia MAE baseline o co najmniej 2%.

Jeżeli którykolwiek warunek nie jest spełniony, ustawiane jest `alpha=0`, więc
prognozą staje się baseline D-3...D-14. Powód decyzji jest zapisywany w raporcie,
np. `HYBRYDA_WYBRANA`, `BRAK_POTWIERDZONEJ_POPRAWY` lub
`ZA_MALO_KALIBRACJI`. Ten bezpiecznik chroni przed użyciem korekty, która nie
potwierdziła wartości na wcześniejszych danych; nie stanowi jednak gwarancji, że
hybryda będzie lepsza od baseline w każdym przyszłym okresie.

Pobranie i oddanie są rozdzielone, bo mają różną fizykę. Pobranie zależy zwykle od
kalendarza, historii i temperatury. Oddanie zależy mocniej od promieniowania,
zachmurzenia i/lub wiatru. Jeżeli później dostępny będzie typ źródła (`PV`, `wiatr`,
inne), należy go koniecznie dodać; bez niego model uśrednia różne mechanizmy.

Każdy z dwóch kierunków ma oddzielny model resztowy i oddzielnie skalibrowane
`alpha`. Do CatBoost trafiają pojedyncze lagi D-3...D-14, baseline i diagnostyka
lagów, cechy kalendarzowe, pogodowe i ich interakcje oraz kategorie klienta, grupy,
rodzaju, oddziału i punktu pogodowego. Cechy kierunku wiatru i czasu są kodowane
cyklicznie, a braki pogody pozostają brakami, zamiast być sztucznie zamieniane na zero.

## Walidacja

Nie stosujemy losowego `train_test_split`. Backtest ma okna rosnące w czasie:

```text
fold 1: przeszłość ----------> 14 dni testu
fold 2: przeszłość ----------------------> 14 dni testu
fold 3: przeszłość ----------------------------------> 14 dni testu
```

Każdy fold zachowuje kolejność `trening -> kalibracja alpha -> test`. Ani model
resztowy, ani wybór `alpha` nie widzą targetów z okna testowego. Cechy lagowe są
aktualizowane as-of dla danej prognozowanej godziny, więc mogą użyć tylko realizacji
sprzed co najmniej trzech dni. `wartosc_przewidywana` z wierszy
`OOF_BACKTEST` jest finalną hybrydą albo baseline po bezpiecznym fallbacku i tylko
te wiersze służą do uczciwej oceny. `wartosc_model_pelny` na danych historycznych
jest wynikiem modelu końcowego in-sample i nie wolno z niej raportować jakości.

Raportowane metryki:

- `MAE` — główna, łatwa do interpretacji w jednostce `Wartości`;
- `RMSE` — mocniej karze duże błędy i nietrafione szczyty;
- `bias = średnia(predykcja - realizacja)` — pokazuje systematyczne zawyżanie/zaniżanie;
- `WAPE` i `sMAPE` — porównanie klientów o różnej skali bez klasycznego problemu
  MAPE przy zerach.

Metryki są liczone globalnie, według kierunku, miasta, statusu dostępności pogody oraz
jako średnia makro po klientach. Porównanie błędów „z pogodą / bez pogody” jest
diagnostyką zakresów, a nie dowodem przyczynowego wpływu pogody, bo okresy mogą różnić
się sezonem i składem klientów. Wpływ samych cech pogodowych należy później potwierdzić
testem ablation na identycznych wierszach. Permutacyjna ważność cech może być liczona
na danych testowych osobno dla pobrania i oddania, ale klikalny pełny launcher pomija
ten kosztowny etap. Można go włączyć w ręcznym uruchomieniu CLI.

Raport porównuje finalne prognozy OOF z baseline D-3...D-14 na dokładnie tych samych
wierszach. Dodatkowe kolumny audytowe pozwalają odtworzyć każdą decyzję:

| Kolumna | Znaczenie |
|---|---|
| `wartosc_bazowa_backtest` | baseline policzony bez podglądania okna testowego |
| `liczba_lagow_bazowych` | liczba dostępnych analogicznych godzin D-3...D-14 |
| `residuum_rzeczywiste` | realizacja minus baseline, czyli target modelu korekty |
| `korekta_ml_surowa` | podpisana korekta przewidziana przez model resztowy |
| `wartosc_ml_przed_blendem` | diagnostyczny kandydat baseline + pełna korekta ML |
| `blend_alpha` | przyjęta część korekty, od 0 do 1 |
| `strategia_predykcji` | informacja, czy użyto hybrydy, czy fallbacku do baseline |
| `kalibracja_n` | liczba obserwacji użytych do wyboru `alpha` |
| `kalibracja_poprawa_mae` | względna poprawa MAE na wcześniejszym oknie kalibracyjnym |
| `kalibracja_powod` | powód wyboru hybrydy albo odrzucenia korekty |
| `fit_zatrzymany_limitem` | informacja, czy CatBoost zakończył fit przez limit czasu |

`wartosc_przewidywana` jest finalną hybrydą/fallbackiem na wierszach OOF. Ręczny CLI
może również wyliczyć przyszłe wiersze znajdujące się w tym samym wejściu. Klikalny
pełny launcher pomija kosztowny scoring modelu końcowego na całej historii, dlatego
nie należy oczekiwać wypełnionej `wartosc_model_pelny` dla każdego wiersza. Osobny
scoring z zapisanego bundle nie jest jeszcze częścią tego launchera.

## Uruchomienie bezpośrednio z SQL Server

Domyślny tryb odpowiada konfiguracji ze zdjęcia: serwer
`MISDWHPRD.GKPGE.PL`, baza `PGESA_MarketAnalytics`, sterownik ODBC 17 i
`Trusted_Connection=yes`. Zakres zapytania jest wyliczany z min/max `Doby Handlowej`
w Excelu. Dla parametryzowanego SQL-a z pięcioma placeholderami dolna granica
pobierania pogody jest automatycznie clampowana do `2024-10-01`, czyli wynosi
`max(min(Doba Handlowa), 2024-10-01)`. Nie skraca to historii energii w modelu.
Zwykły firmowy SQL bez `?` nie otrzymuje parametrów od Pythona i musi sam ustawić
dolną granicę co najmniej na `2024-10-01`.

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
`pakiet_modelu_mdd.joblib`. Pakiet zawiera oba końcowe modele, ich `alpha`, schemat
cech, konfigurację, mapowanie, profile fallbacku i końcówkę historii potrzebną dla
lagów D-3...D-14.

`wartosc_przewidywana` w raporcie treningowym służy przede wszystkim do
chronologicznego OOF. Brak targetu wewnątrz historii dostaje status
`HISTORYCZNY_BRAK_TARGETU`, a nie fałszywy status przyszłości. Wynik modelu końcowego
na historii — jeżeli zostanie wyliczony w ręcznym trybie CLI — jest in-sample i nie
wolno na nim raportować jakości. Klikalny pełny launcher pomija taki scoring całej
historii, aby ograniczyć czas i pamięć.

Samo utworzenie `pakiet_modelu_mdd.joblib` nie oznacza jeszcze wykonania przyszłej
prognozy. `uruchom_model.py` nie wczytuje zapisanego bundle ponownie. Późniejszy,
osobny program scoringowy będzie musiał otrzymać ten pakiet oraz bieżące dane z
historią wystarczającą do wyliczenia D-3...D-14. Do czasu dostarczenia tego programu
pakiet jest gotowym artefaktem wejściowym, a nie samodzielnym plikiem wykonywalnym.

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

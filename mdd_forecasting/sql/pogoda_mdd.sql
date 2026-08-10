/*
    Ekstrakt prognozy pogody do modelu MDD (SQL Server).

    Zasada anty-leakage:
      - dataGodzinaCET/UTC       = valid time, czyli godzina, której dotyczy pogoda;
      - czasDanychZrodlaCET/UTC  = issue time, czyli chwila wydania/dostępności prognozy;
      - dla każdego (punkt, valid time) wybierana jest najnowsza prognoza,
        której issue time jest nie późniejszy niż valid time - @minLeadHours.

    Nie wolno zastępować poniższego warunku wyborem prognozy "najbliższej" valid time
    ani używać ABS(DATEDIFF(...)); mogłoby to włączyć rekord wydany już po chwili
    podejmowania decyzji i zawyżyć wynik walidacji modelu.

    Porównanie issue <= valid - lead wykonywane jest po UTC, żeby uniknąć
    niejednoznaczności przy zmianie CET/CEST i 25. godzinie.

    Założenie źródłowe: czasDanychZrodlaUTC jest rzeczywistym czasem dostępności
    rekordu. Jeżeli hurtownia przechowuje osobny czas przyjęcia/ingestii, do testu
    as-of należy użyć późniejszego z issue time i czasu ingestii.
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;

/*
    Parametry wywołania:
      @validFromCET        - początek zakresu valid time (włącznie),
      @validToCETExclusive - koniec zakresu valid time (wyłącznie),
      @minLeadHours        - minimalne wyprzedzenie prognozy, np. 24 dla D-1.

    Zakres półotwarty [od, do) zapobiega zdublowaniu północy przy pobieraniu danych
    partiami. Zakres wybierany jest po lokalnym valid time, natomiast warunek
    anty-leakage i rzeczywisty lead są liczone po UTC.
*/
/* Pięć wartości poniżej jest parametrami pyodbc przekazywanymi przez `params=`. */
DECLARE @validFromCET        datetime2(0) = ?;
DECLARE @validToCETExclusive datetime2(0) = ?;
DECLARE @minLeadHours        int          = ?;
DECLARE @wlasciciel          nvarchar(50) = ?;
DECLARE @typPrognozy         nvarchar(50) = ?;

IF @validFromCET IS NULL OR @validToCETExclusive IS NULL
    THROW 50001, N'Parametry zakresu valid time nie mogą być NULL.', 1;

IF @validFromCET >= @validToCETExclusive
    THROW 50002, N'@validFromCET musi być wcześniejsze niż @validToCETExclusive.', 1;

IF @minLeadHours IS NULL OR @minLeadHours < 0 OR @minLeadHours > 720
    THROW 50003, N'@minLeadHours musi należeć do zakresu 0-720.', 1;

IF NULLIF(LTRIM(RTRIM(@wlasciciel)), N'') IS NULL
   OR NULLIF(LTRIM(RTRIM(@typPrognozy)), N'') IS NULL
    THROW 50004, N'Właściciel i typ prognozy nie mogą być puste.', 1;

;WITH DopuszczalnePrognozy AS
(
    SELECT
        w.[execId],
        w.[punkt],
        w.[dataCET],
        w.[godzinaHandlowa25],
        w.[dataGodzinaCET],
        w.[dataGodzinaUTC],
        w.[czasDanychZrodlaCET],
        w.[czasDanychZrodlaUTC],
        w.[temperatura],
        w.[predkoscWiatru],
        w.[kierunekWiatru],
        w.[zachmurzenie],
        w.[intensywnoscOpadowKonwekcyjnych],
        w.[widocznosc],
        w.[calkowitePromieniowanieSloneczneGodzinowe],
        w.[bezposredniePromieniowanieSloneczneGodzinowe],
        w.[albedoPrognozowane],
        w.[warstwaSniegu],
        ROW_NUMBER() OVER
        (
            PARTITION BY w.[punkt], w.[dataCET], w.[godzinaHandlowa25]
            ORDER BY
                w.[czasDanychZrodlaUTC] DESC,
                w.[execId] DESC
        ) AS [rn]
    FROM [PGESA_MarketAnalytics].[wa].[vPogodaPrognoza] AS w
    WHERE w.[punkt] IN
    (
        N'Białystok',
        N'Rzeszów',
        N'Zamość',
        N'Skarżysko-Kamienna',
        N'Lublin',
        N'Łódź'
    )
      AND w.[wlasciciel] = @wlasciciel
      AND w.[typ] = @typPrognozy
      /* Odrzuca techniczne duplikaty tego samego czasu danych w ramach widoku. */
      AND w.[czyOstatniNaCzasDanych] = 1
      AND w.[dataGodzinaCET] >= @validFromCET
      AND w.[dataGodzinaCET] <  @validToCETExclusive
      AND w.[czasDanychZrodlaUTC] IS NOT NULL
      /* Kluczowy warunek as-of: żadna prognoza wydana po cutoffie nie trafia do modelu. */
      AND w.[czasDanychZrodlaUTC]
          <= DATEADD(HOUR, -@minLeadHours, w.[dataGodzinaUTC])
)
SELECT
    [punkt],
    [dataCET],
    [godzinaHandlowa25],
    [dataGodzinaCET],
    [dataGodzinaUTC],
    [czasDanychZrodlaCET],
    [czasDanychZrodlaUTC],
    CAST(DATEDIFF(MINUTE, [czasDanychZrodlaUTC], [dataGodzinaUTC]) / 60.0
         AS decimal(10, 2)) AS [leadHoursRzeczywiste],
    [temperatura],
    [predkoscWiatru],
    [kierunekWiatru],
    [zachmurzenie],
    [intensywnoscOpadowKonwekcyjnych],
    [widocznosc],
    [calkowitePromieniowanieSloneczneGodzinowe],
    [bezposredniePromieniowanieSloneczneGodzinowe],
    [albedoPrognozowane],
    [warstwaSniegu]
FROM DopuszczalnePrognozy
WHERE [rn] = 1
ORDER BY [punkt], [dataGodzinaCET];

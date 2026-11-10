# Raport rozpoznawczy: gold_dl, adnotacje i metryki jakości

Data: 2026-11-02. Sygnały czytane bezpośrednio z silver EDF (pyedflib), bez funkcji eksportu. Skrypty inspekcji były jednorazowe (scratchpad sesji, nie w repo).

**Najważniejsze:** teza wyjściowa z sekcji 1 się nie potwierdza. Kolumny prob_w..prob_n3 nie są w całości równe 9. 9 pojawia się tylko w 3 pierwszych i 3 ostatnich epokach każdego nagrania.

## 1. caisr_prob

**Kanały w silver** (te same dla wszystkich 15 sprawdzonych pacjentów, po 5 z I0002, I0006 i S0001):

| indeks | nazwa (truth_table) | etykieta w EDF | fs | próbek/epokę | zakres fizyczny |
|---|---|---|---|---|---|
| 24–28 | prob_w, prob_r, prob_n1, prob_n2, prob_n3 | `freq_caisr_prob_` (ucięta do 16 znaków, identyczna dla wszystkich 7) | 1/30 Hz | 1 | 0–9 |
| 29–30 | prob_arous, prob_no-ar | `freq_caisr_prob_` | 2 Hz | 60 | 0–1 |

- Indeksy `CAISR_PROB_INDICES = [24..30]` zgadzają się z `silver_channels` w truth_table.
- Ponieważ etykiety są ucięte i identyczne, kolumny da się przypisać tylko po indeksie.

**Skąd biorą się wartości 9:**
- Dane w gold są identyczne z silver. Maksymalna różnica |gold − średnia z epoki liczona bezpośrednio z silver| wynosi **0.0** u wszystkich 15 pacjentów. Eksport ani niczego nie dodaje, ani nie gubi.
- Odsetek próbek równych 9 w silver wynosi 0,57–0,82% na pacjenta, tyle samo co w gold.
- Wartości poza 9 mieszczą się w zakresie 0–1,0002 (nadwyżka ponad 1 to kwantyzacja EDF).

**Wszyscy 580 pacjenci:**
- **0/580** pacjentów ma wszystkie epoki prob_w..prob_n3 równe 9. Dotyczy to każdego SiteID i każdego splitu (I0002 4/25/4, I0006 14/71/14, S0001 66/316/66 w podziale test/train/val).
- U **każdego** pacjenta 9 występuje w dokładnie **6 epokach**: indeksy 0, 1, 2 oraz N−3, N−2, N−1. Nie ma epok, w których tylko część z 5 kolumn ma 9.
- Prawdopodobna przyczyna twierdzenia „9.0 u sprawdzonych pacjentów” (to tylko hipoteza): podgląd pierwszych wierszy tablicy. To samo twierdzenie jest zapisane w `references/gold_dl_opis.md`, sek. 2.
- W około 50 pacjentach na kolumnę (prob_w 51, n1 50, n2 45, r 11) wartość „9” w gold to 8.99986 albo 8.99992, a nie dokładnie 9.0. Wynika to z kwantyzacji 16-bit przy zakresie 0–9. Test `== 9` takie wartości pomija.

**stage_caisr:**
- Maska `stage_caisr == 9` jest identyczna z maską 9 w prob u **580/580** pacjentów.
- stage_expert w tych 6 skrajnych epokach przyjmuje różne wartości: 9 (1459 epok), 0 (1094), W (512), N2 (220), N1 (95), REM (73), N3 (27).

**Sumy prawdopodobieństw:**
- prob_arous + prob_no-ar (per próbka 2 Hz, wszyscy pacjenci) mieści się w zakresie 0,99948–1,00052.
- prob_w..n3 poza epokami z 9 sumuje się do 1 z maksymalnym odchyleniem 0,0013.

## 2. Kod 0 w stage_expert

**Gdzie występuje 0 i 9 (liczba pacjentów):**

| SiteID | pacjenci | z kodem 0 | z kodem 9 | oba kody |
|---|---|---|---|---|
| I0002 | 33 | 5 | 33 | 5 |
| I0006 | 99 | 99 | 0 | 0 |
| S0001 | 448 | 249 | 420 | 225 |

Ten sam podział według splitu (test/train/val):
- kod 0: I0002 1/4/0, I0006 14/71/14, S0001 37/178/34
- kod 9: I0002 4/25/4, S0001 65/294/61

**Jaką część nagrania zajmuje 0 (353 pacjentów z kodem 0):**
- Mediana 1,33% epok, średnia 7,2%.
- 16 pacjentów ma ≥50% epok z kodem 0.
- **14 pacjentów S0001 ma 100% epok równych 0.** W silver ich kanał ma zakres fizyczny (0, 1), czyli sygnał jest stale zerowy. U 8 z nich arousal, resp i limb też są całkowicie zerowe; u 6 pozostałych zdarzenia są normalnie oznaczone.
  - Stage = 0, zdarzenia obecne (6): sub-S0001111388710 (test), sub-S0001115912412, sub-S0001121415879, sub-S0001121484642 (train), sub-S0001115139512, sub-S0001116852304 (val).
  - Stage = 0, wszystkie zdarzenia = 0 (8): sub-S0001111544884, sub-S0001112558716, sub-S0001114130966, sub-S0001117136199, sub-S0001118058266, sub-S0001121773638 (train), sub-S0001117045059, sub-S0001120186503 (val).
- Pacjent sub-I0002150029713 ma 98,1% epok z kodem 0, w środku nagrania, otoczonych przez 9.

**Położenie w nagraniu** (bez 14 pacjentów z samymi zerami):

| SiteID | wzorzec pacjentów | epoki z 0 |
|---|---|---|
| I0006 | 0 na początku u wszystkich 99 (w tym 8 również na końcu) | początek 4116, koniec 136 |
| S0001 | 214 tylko na końcu, 11 tylko na początku, 8 początek + środek, 2 początek + koniec | koniec 3348, początek 617, środek 129 |
| I0002 | 4 w środku, 1 na końcu | środek 1802, koniec 6 |

**Współwystępowanie z 9:**
- W S0001 typowy układ to 9 na początku nagrania i 0 na końcu. 209 bloków zer kończy się z końcem nagrania i jest poprzedzonych zwykłym stadium. Zera bezpośrednio sąsiadujące z 9 to rzadkość (kilkanaście bloków).
- Epoki z 9 leżą głównie na początku nagrania: S0001 11166 na początku, 280 w środku, 145 na końcu; I0002 692 na początku, 304 na końcu.
- I0006 nie ma kodu 9 wcale. Ma za to 0 na początku nagrania, czyli tam, gdzie pozostałe site'y mają zwykle 9.

**stage_caisr w epokach z expert = 0:**

| SiteID | dominujące wartości stage_caisr |
|---|---|
| I0006 | W 81,9%, 9 7,4%, N2 7,7% |
| S0001 | W 69,8%, 9 17,0%, N2 9,4% |
| I0002 | N2 41,5%, W 38,2% |

**Dokumentacja kodu 0:**
- Katalog `BRONZE_PATH` (`~/.cache/kagglehub/.../versions/2`) **nie istnieje na dysku**, więc nie da się go sprawdzić.
- Nagłówki silver EDF nie opisują kodu 0: pola pacjenta i nagrania są puste, brak adnotacji EDF+, `dimension = 'stage'`.
- `~/Pobrane/expert_annotations.md` wymienia tylko kody 1–5 i 9.

**Co z kodu wynika o pochodzeniu zer:** w `src/bronze_to_silver.py` są dwa miejsca, które mogą wpisać 0:
- `annotation_channel` wykonuje `np.nan_to_num` (NaN → 0);
- brakujący kanał zwraca pusty sygnał, który `fit_to_records` dopełnia zerami.

Bez bronze **nie da się ustalić**, czy 0 jest w danych źródłowych, czy pochodzi z NaN.

## 3. Kody resp i limb

**gold_dl, wszyscy pacjenci.** Format: kod: liczba sub-okien (liczba pacjentów).

- **resp I0002:** 0: 814212 (33), 1: 7574 (25), 2: 6412 (21), 3: 235 (4), 4: 1 (1), 5: 3378 (4), 7: 125 (22), 8: 341 (16), 9: 106157 (32), **10: 183 (12)**, **11: 1612 (13)**
- **resp I0006:** 0: 2312597 (99), 1: 23256 (69), 2: 7786 (45), 3: 1735 (23), 5: 40 (1), 6: 445 (3), 7: 108528 (94), 9: 94713 (90)
- **resp S0001:** 0: 10987204 (448), 1: 255726 (378), 2: 133882 (343), 3: 27558 (131), 4: 2 (1), 5: 216 (3), 7: 450167 (423), 8: 93 (33), 9: 436992 (413)
- **limb I0002:** 0: 913871, 1: 10584 (33), 2: 15773 (20), 3: 1, 4: 1
- **limb I0006:** 0: 2357659, 1: 38141 (57), 2: 153298 (52), 3: 1, 4: 1
- **limb S0001:** 0: 12099357, 1: 12640 (31), 2: 178604 (410), 3: 444 (232), 4: 461 (231), 10: 334 (60)

**Kody poza dokumentacją:** resp 10 i 11 występują tylko w I0002, a dokumentacja przewiduje dla resp kody 0–9. Wszystkie wystąpienia limb mieszczą się w udokumentowanym zakresie 0–12.

**Wartości niecałkowite w silver:**
- Wszystkie kanały adnotacji (arousal, resp, limb i stage, zarówno expert, jak i caisr) mają maksymalne odchylenie od liczby całkowitej około **1e-4**. To kwantyzacja EDF 16-bit przy zakresie fizycznym floor/ceil danych.
- **Zero próbek** ma odchylenie >1e-3.
- `np.rint` nie może więc tworzyć sztucznych kodów na przejściach między zdarzeniami.

**Rzadkie kody** to w większości pojedyncze sub-okna, ale wartości są w silver całkowite (z dokładnością do kwantyzacji):
- resp 8 (I0002): 341/341 serii ma długość 1
- resp 10: 183/183 serii ma długość 1
- resp 11: 1610/1611 serii ma długość 1, sąsiedzi 0/0 (1003) albo 9/9 (300)
- limb 3, 4 i 10 (S0001): prawie zawsze długość 1, sąsiedzi 0/0

Czy pochodzą z resamplingu w bronze, **nie da się ustalić**.

**Obcinanie przez `.astype(int)` (uboczne ustalenie):** część wartości w silver leży tuż poniżej liczby całkowitej (np. 6.9999). `.astype(int)` obcina je w dół, a `rint` zaokrągla poprawnie. gold_dl używa `rint`, więc tam problemu nie ma. Obcinanie występuje w:

- **Notebook** (`resp_pair`/`limb_pair`) na kanałach expert:
  - resp 7→6: S0001 442674 próbek (3,6%), I0006 106287 (4,2%);
  - resp 9→8: I0002 53149 (5,9%);
  - pojedyncze przypadki 5→4, 8→7, 10→9;
  - limb 3→2: S0001 326 próbek u 171 pacjentów.
- **`silver_to_gold.py`** (`read_event_labels`/`read_stage_labels`) na kanałach caisr:
  - resp_caisr 3→2: 0,04–0,11% próbek;
  - stage_caisr i limb_caisr: 0 różnic.
- Dla porównania stage_expert przy obcinaniu dałby 5→4 w 11,4% epok S0001 i 5,3% I0002 (tego kanału te funkcje nie czytają).

**LIMB_MERGE i RESP_MERGE w `signal_quality_analysis.ipynb`:**
- Same mapy są **zgodne** z oficjalnymi.
- Różnią się obsługą kodów spoza mapy:
  - `LIMB_MERGE.get(x, x)` zostawia nieznany kod bez zmian;
  - `RESP_MERGE.get(x, 0)` zamienia nieznany kod na 0, przez co resp 10 i 11 z I0002 stają się 0.
- Mapy dostają kody już po obcięciu `astype(int)` opisanym wyżej.

## 4. Równoważność metryk jakości

**Porównanie logiki:**
- **max_flat_run:** logika równoważna.
  - W obu wersjach run oznacza liczbę równych sąsiednich par + 1.
  - NaN przerywa run (bo NaN ≠ NaN).
  - Epoka o długości 1 lub bez równych par daje 1.
  - Wersja DL jawnie przerywa run na granicy epok, co daje ten sam wynik co liczenie epoka po epoce.
- **fraction_beyond_kmad:** logika różni się tylko przy NaN.
  - `silver_to_gold.py` używa `nanmedian`/`nanmean`; próbki NaN zostają w mianowniku.
  - `dl_silver_to_gold.py` używa `median`/`mean`. Jeden NaN w epoce daje mediana = NaN, MAD = NaN, a wynik **0.0** zamiast NaN.
  - Przykład syntetyczny: 0,0194 w silver_to_gold wobec 0,0 w DL. Przy MAD = 0 i obecnym NaN: silver_to_gold zwraca NaN, DL zwraca 0,0.
  - Gdy MAD = 0 bez NaN, obie wersje zwracają NaN.

**Dane rzeczywiste:** 3 pacjentów (sub-I0002150001401, sub-I0006179000018, sub-S0001111229643) × 6 kanałów EEG = 14886 par (epoka, kanał).
- Macierze wejściowe z `read_epoched_signal` i `read_channel_epochs` są identyczne. W silver EEG nie ma ani jednego NaN.
- max_flat_run i frac_beyond_kmad dają **identyczne wyniki co do bitu**. Wzorzec NaN też jest identyczny (54 epoki z NaN u I0006179000018).
- Przy progach EEG (flat_run ≥ 200, frac_kmad > 0,1) **decyzja zmienia się w 0 epokach**. Dla kontekstu: 85 epok przekracza próg flat, 471 epok przekracza próg kmad.
- Względem zapisanego `quality.npy` (float32): flat_max_run identyczny, frac_kmad różni się maksymalnie o 1,4e-8, decyzja zmienia się w 0 epokach.

## Czego nie da się ustalić

- znaczenia kodu 0 w stage_expert ani tego, czy pochodzi z bronze, czy z NaN zamienionego na 0 (brak bronze na dysku);
- znaczenia kodów resp 10 i 11 w I0002;
- pochodzenia jednosekundowych serii rzadkich kodów przed etapem silver.

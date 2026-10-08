# Predykcja zaburzeń poznawczych na podstawie sygnałów polisomnograficznych

Projekt dyplomowy realizowany na Politechnice Warszawskiej w ramach George B. Moody PhysioNet Challenge 2026. Celem projektu jest zbudowanie modelu uczenia maszynowego przewidującego przyszłe rozpoznanie zaburzeń poznawczych na podstawie zapisu polisomnograficznego (PSG) z jednej nocy.

## Zbiór danych

- 580 pacjentów z trzech ośrodków klinicznych (S0001 - 448, I0006 - 99, I0002 - 33), pozostałych po oczyszczeniu pierwotnego zbioru liczącego 623 pacjentów
- zmienna docelowa `Cognitive_Impairment` jest idealnie zrównoważona (290 / 290)
- podział na zbiory: uczący / walidacyjny / testowy = 412 / 84 / 84, warstwowany względem ośrodka i etykiety, zamrożony w plikach `references/split.json` i `references/assignments.csv`
- wykorzystywane kanały: EEG (F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1), EOG, EMG podbródka, EMG kończyn dolnych, EKG, kanały oddechowe oraz SpO2

## Architektura przetwarzania

Przetwarzanie danych zostało zorganizowane w architekturze warstwowej (medalionowej):

1. **Warstwa brązowa** - surowe zapisy EDF wraz z adnotacjami.
2. **Warstwa srebrna** - ujednolicone nazwy i jednostki kanałów (słownik synonimów w `channel_table.csv`, zasady ujednolicania w `truth_table.json`) oraz ocena jakości epok według progów zapisanych w `quality_thresholds.json`.
3. **Warstwa złota** - tabelaryczny zbiór cech na poziomie pacjenta (`features.csv`, `features_reduced.csv`).

## Ekstrakcja cech

Pełny opis cech wraz ze sposobem liczenia i źródłami znajduje się w pliku `ekstrakcja_cech.ods`. Wyodrębniono następujące grupy cech:

- **struktura snu** - TST, TRT, SPT, SE, SME, SL, latencja REM, WASO, czasy i odsetki faz snu, liczba wybudzeń, wskaźnik fragmentacji snu
- **zdarzenia** - PLMS, PLMSI
- **oksymetria** - średnia i minimalna SpO2, T90
- **EEG w dziedzinie czasu** - parametry Hjortha, kurtoza (również w pasmach), długość linii
- **EEG w dziedzinie częstotliwości** - względne moce w pasmach delta, theta, alfa, sigma i beta, stosunki mocy pasm, wskaźnik spowolnienia

Cechy EEG liczone są osobno dla trzech regionów (czołowy, centralny, potyliczny) i pięciu stanów (W, N1, N2, N3, REM).

Wszystkie cechy zależne od adnotacji wyznaczane są na podstawie etykiet algorytmicznych CAISR, a nie eksperckich. Etykiety eksperckie dostępne są wyłącznie w zbiorze uczącym, więc ich użycie prowadziłoby do niespójności między zbiorami.

## Redukcja cech

Zbiór 296 cech zredukowano do 172 (170 reprezentantów cech EEG i snu po grupowaniu oraz wiek i płeć). Zastosowano korelację Spearmana i hierarchiczne grupowanie z pełnym wiązaniem przy progu ρ = 0,8. Analiza ANOVA nie została użyta jako twardy filtr - selekcja cech odbywa się wewnątrz modeli.

Z modelowania wyłączono:

- `SiteID` - ze względu na silną nierównowagę ośrodków i ryzyko uczenia się cech specyficznych dla ośrodka
- AHI i ODI - ze względu na błąd skalowania w potoku przetwarzania, prowadzący do wartości niemożliwych fizjologicznie

Adnotacje PLMS z CAISR mają ograniczoną wiarygodność, co należy uwzględnić przy interpretacji tych cech.

## Modelowanie

- modele: regresja logistyczna (regularyzacja L1, solver `saga`), wzmacnianie gradientowe, las losowy
- strojenie hiperparametrów: `RepeatedStratifiedKFold` na połączonym zbiorze uczącym i walidacyjnym
- próg decyzyjny: maksymalizacja statystyki J Youdena na predykcjach spoza foldów z połączonego zbioru uczącego i walidacyjnego
- zbiór testowy wykorzystywany jest dokładnie jeden raz, do końcowej oceny

Miary oceny: AUROC (miara główna), czułość, precyzja, macierz pomyłek.

## Wyniki

| Model | AUROC (zbiór testowy) |
|---|---|
| Regresja logistyczna | 0,676 |
| Wzmacnianie gradientowe | 0,704 |
| Las losowy | 0,726 |

Najważniejszą cechą we wszystkich modelach okazała się kurtoza sygnału EEG w regionie czołowym w fazie N2 (`kurtosis_frontal_N2`). Regresja logistyczna przy optymalnej regularyzacji L1 redukuje się do tej jednej cechy, co wskazuje na słaby sygnał liniowy w danych.

## Zawartość repozytorium

```
inzynierka/
├── notebooks/
│   ├── chanel_analysis.ipynb
│   ├── signal_quality_analysis.ipynb
│   ├── feature_analysis.ipynb
│   └── model_analysis.ipynb
├── references/
│   ├── assignments.csv
│   ├── channel_table.csv
│   ├── ekstrakcja_cech.ods
│   ├── quality_thresholds.json
│   ├── split.json
│   └── truth_table.json
└── src/
    ├── main.py
    ├── data_download.py
    ├── preliminary_pipeline.py
    ├── bronze_to_silver.py
    └── silver_to_gold.py
```

Dane nie są częścią repozytorium.

| Ścieżka | Opis |
|---|---|
| `notebooks/chanel_analysis.ipynb` | analiza metadanych kanałów przed ekstrakcją do warstwy srebrnej |
| `notebooks/signal_quality_analysis.ipynb` | analiza jakości sygnałów i decyzje projektowe dla warstwy srebrnej |
| `notebooks/feature_analysis.ipynb` | analiza i redukcja cech |
| `notebooks/model_analysis.ipynb` | uczenie i ocena modeli |
| `references/split.json` | parametry podziału (ziarno losowania, proporcje 70/15/15, zmienne warstwujące) wraz z listami pacjentów w każdym zbiorze |
| `references/assignments.csv` | ten sam podział w postaci tabeli: pacjent, ośrodek, sesja, etykieta i przypisany zbiór |
| `references/truth_table.json` | zasady ujednolicania kanałów w warstwie srebrnej |
| `references/quality_thresholds.json` | progi jakości epok dla poszczególnych klas kanałów |
| `references/channel_table.csv` | słownik synonimów nazw kanałów |
| `references/ekstrakcja_cech.ods` | specyfikacja cech wraz ze źródłami |
| `src/main.py` | główny potok uruchomieniowy |
| `src/data_download.py` | pobieranie danych |
| `src/preliminary_pipeline.py` | ekstrakcja metadanych kanałów |
| `src/bronze_to_silver.py` | przejście z warstwy brązowej do srebrnej |
| `src/silver_to_gold.py` | przejście z warstwy srebrnej do złotej |

## Wymagania

Python, Jupyter, pandas, odfpy, NumPy, SciPy, scikit-learn, YASA.

## Źródła

- American Academy of Sleep Medicine, *AASM Manual for the Scoring of Sleep and Associated Events*, https://aasm.org/clinical-resources/scoring-manual
- *Dementia detection from brain activity during sleep*, Sleep, 2023, https://academic.oup.com/sleep/article/46/3/zsac286/6855026
- *Cognitive Function in a Sleep Clinic Cohort of Patients with Obstructive Sleep Apnea*, https://pubmed.ncbi.nlm.nih.gov/33147067
- *Wearable EEG devices in the detection of mild cognitive impairment: a systematic review*, https://www.nature.com/articles/s41746-026-02342-w
- *Resting-state EEG signatures of Alzheimer's disease are driven by periodic but not aperiodic changes*, Neurobiology of Disease

## Autor

Emil - [stolarczykemil](https://github.com/stolarczykemil)

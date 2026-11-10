# Raport rozpoznawczy: warstwa `data/gold_dl`

## 1. Struktura `gold_dl`

**Układ katalogów:** `data/gold_dl/<patient_id>/` (jeden katalog na pacjenta) + jeden plik zbiorczy `data/gold_dl/patients.parquet`. `patient_id` = nazwa BIDS, np. `sub-I0002150001401`. SiteID jest prefiksem w ID: w zbiorze są **3 site'y** — `I0002` (33 pac.), `I0006` (99 pac.), `S0001` (448 pac.); łącznie **580** katalogów pacjentów.

**Pliki w katalogu pacjenta (9 tablic `.npy`):** `x_200.npy`, `x_100.npy`, `x_25.npy`, `stage_expert.npy`, `arousal.npy`, `resp.npy`, `limb.npy`, `caisr_prob.npy`, `quality.npy`. **Nie ma** osobnego pliku `ch_mask.npy` ani kopii `truth_table.json` w katalogu (patrz sek. 3).

**Kształty i dtype (mmap, 3 pacjenci z różnych site):**

| plik | I0002150001401 | I0006179000018 | S0001111229643 | dtype |
|---|---|---|---|---|
| x_200.npy | (732, 12, 6000) | (812, 12, 6000) | (937, 12, 6000) | float16 |
| x_100.npy | (732, 3, 3000) | (812, 3, 3000) | (937, 3, 3000) | float16 |
| x_25.npy | (732, 1, 750) | (812, 1, 750) | (937, 1, 750) | float16 |
| stage_expert.npy | (732,) | (812,) | (937,) | int8 |
| arousal.npy | (732, 60) | (812, 60) | (937, 60) | int8 |
| resp.npy | (732, 30) | (812, 30) | (937, 30) | int8 |
| limb.npy | (732, 30) | (812, 30) | (937, 30) | int8 |
| caisr_prob.npy | (732, 7) | (812, 7) | (937, 7) | float32 |
| quality.npy | (732, 16, 6) | (812, 16, 6) | (937, 16, 6) | float32 |

**Spójność N_epok:** tak — w obrębie pacjenta N jest identyczne we wszystkich plikach (732 / 812 / 937). N liczone jest z liczby próbek kanału `freq_stage_expert` (funkcja `count_epochs`), więc jest wspólnym mianownikiem.

**Kolejność kanałów** (źródło: słownik `SIGNAL_GROUPS` w `src/dl_silver_to_gold.py`, kolejność jak w liście, nie z `truth_table.json` — truth_table służy tylko do mapowania nazwa→indeks w EDF):
- **x_200 (200 Hz, 12 kan.):** `eeg_f3_m2, eeg_f4_m1, eeg_c3_m2, eeg_c4_m1, eeg_o1_m2, eeg_o2_m1, eog_e1_m2, eog_e2_m1, chin_emg_chin1_chin2, leg_emg_lat, leg_emg_rat, ecg_ecg`
- **x_100 (100 Hz, 3 kan.):** `resp_chest, resp_abd, resp_ptaf`
- **x_25 (25 Hz, 1 kan.):** `resp_spo2`

## 2. Etykiety

**Zapisane etykiety i kształty:**
- `stage_expert.npy` — staging eksperta, `(N,)`, int8, kod surowy (1 na epokę).
- `arousal.npy` — `(N, 60)`, int8 → **sub-epokowe, 0,5 s** (60 pod-okien / 30 s).
- `resp.npy` — `(N, 30)`, int8 → **sub-epokowe, 1 s**.
- `limb.npy` — `(N, 30)`, int8 → **sub-epokowe, 1 s**.
- `caisr_prob.npy` — `(N, 7)`, float32 — prawdopodobieństwa CAISR, uśrednione po epoce (`read_epoch_mean`). Kolumny (indeksy EDF 24–30): `[prob_w, prob_r, prob_n1, prob_n2, prob_n3, prob_arous, prob_no-ar]`.

Czyli etykiety zdarzeń są **sub-epokowe, nie zagregowane** (arousal 0,5 s, resp/limb 1 s).

**Źródło:** wyłącznie kanały *expert* (`freq_arousal_expert`, `freq_limb_expert`, `freq_resp_expert`, `freq_stage_expert`) + prawdopodobieństwa CAISR. Kanały *caisr* dla staging/arousal/resp/limb **nie** są eksportowane do gold_dl.

**Unikalne wartości (3 pacjenci):**
- `stage_expert`: I0002 `{1,2,3,4,5,9}`, I0006 `{0,1,2,3,4,5}`, S0001 `{2,3,4,5,9}`. Mapa kodów (z `silver_to_gold.py`): `1=N3, 2=N2, 3=N1, 4=REM, 5=W, 9=unscored`. **Uwaga:** u I0006 pojawia się kod `0`, którego nie ma w `STAGE_CODE` — nie da się jednoznacznie ustalić jego znaczenia z kodu (w `silver_to_gold.py` byłby mapowany na `"unscored"` przez `.get(code, "unscored")`, ale w gold_dl zapisywany jest kod surowy, więc `0` i `9` to różne wartości liczbowe w tablicy).
- `arousal`: `{0,1}` u wszystkich (binarne).
- `resp`: I0002 `{0,1,2,3,5,7,8,9}`, I0006 `{0,1,2,3,7,9}`, S0001 `{0,7,9}` — wartości surowe, wieloklasowe.
- `limb`: I0002 `{0,1,2}`, I0006 `{0}`, S0001 `{0,2}` — wartości surowe.
- `caisr_prob`: pierwszych 5 kolumn (`w,r,n1,n2,n3`) = **9.0 (sentinel braku)** u wszystkich 3 pacjentów; realne wartości (0–1) tylko w kolumnach `arous`, `no-ar`.

**Mapowania LIMB_MERGE / RESP_MERGE:** **nie zastosowane.** W `dl_silver_to_gold.py` kody zdarzeń są tylko zaokrąglane (`np.rint(...).astype(np.int8)`) i zapisywane surowo (`read_event_codes`). Świadczą o tym surowe zbiory wartości (np. resp `{0,1,2,3,5,7,8,9}`). Agregacja/scalanie klas pozostaje po stronie konsumenta.

**Kodowanie unscored:** sentinel `9` w `stage_expert` (zgodnie z `STAGE_CODE`). Sentinel `9` występuje też w `resp` i w nieużywanych kolumnach `caisr_prob`. Dodatkowo nierozstrzygnięty kod `0` w stagingu I0006 (jw.).

## 3. `ch_mask`

**Nie istnieje jako tablica per-epoka ani plik w katalogu pacjenta.** `ch_mask` to **kolumny w `patients.parquet`**, booleany **per pacjent** (jedna wartość na pacjenta), generowane w `patient_channel_statistics` / `build_patients_table` (`dl_silver_to_gold.py`).

- Kolumny: `ch_mask_<channel>`, tworzone **tylko dla kanałów niekrytycznych** (`critical=false` w `truth_table.json`): `eog_e1_m2, eog_e2_m1, chin_emg_chin1_chin2, resp_chest, resp_abd, resp_spo2`. Dla kanałów krytycznych (m.in. wszystkie EEG) maski **nie ma** — zakłada się ich obecność.
- Znaczenie wartości: `True` = kanał obecny/użyteczny, wyznaczane przez `channel_statistics` jako `isfinite(std) and std > 0` (sygnał nie-stały i skończony). `False` = kanał nieobecny/płaski.
- `patients.parquet`: kształt `(580, 45)`; poza `ch_mask_*` zawiera też `median_<ch>`, `iqr_<ch>` na kanał, `n_epochs`, oraz metadane: `patient_id, SiteID, Cognitive_Impairment, split, Age, Sex`.

## 4. Maska jakości

Trzeba rozróżnić dwie warstwy — pytanie dotyczy `silver_to_gold.py` (warstwa cech ML, katalog `data/gold`), ale metryki jakości per-epoka trafiają do gold_dl z `dl_silver_to_gold.py`. Opisano obie.

### 4a. `silver_to_gold.py` (warstwa ML / `features.parquet`)
- **Funkcje odrzucania epok:**
  - `is_electrophysiological_epoch_valid(epoch, thresholds) -> bool` — decyzja **per (epoka, kanał)**. Kolejne kryteria: brak skończonych wartości; `nanvar < flat_var_min`; `max_flat_run_length >= flat_run_min_samples`; `amplitude_out_of_range_ratio > amp_oor_ratio_max`; `fraction_beyond_kmad > frac_beyond_kmad_max`.
  - Funkcje pomocnicze: `max_flat_run_length(epoch) -> int`, `amplitude_out_of_range_ratio(epoch, amplitude_max_uv) -> float`, `fraction_beyond_kmad(epoch, k) -> float` (NaN gdy MAD=0).
  - `build_region_epochs(channel_epochs, thresholds) -> list` — składa z 2 kanałów danego regionu EEG pojedynczy „użyteczny” epoch (None/1 kanał/średnia), odrzucając niewalidne.
- **Granularność:** walidacja **per epoka i per kanał EEG**, potem agregowana do regionu. Nie jest to pojedynczy per-epoka werdykt dla całej epoki PSG.
- **Użycie `quality_thresholds.json`:** wczytywany w `build_gold`/`preview_patient` jako cały obiekt; w `extract_patient` do walidacji EEG używany jest wyłącznie podsłownik `quality_thresholds["channel_class_thresholds"]["eeg"]`, a do oksymetrii `["channel_class_thresholds"]["spo2"]["physio_min"]`. Nie ma mapowania kanał→klasa (tylko klasa „eeg”/„spo2”).
- **Wynik:** maska per-epoka **nie jest nigdzie zapisywana** — służy wyłącznie do ekstrakcji cech (gold = tablica cech na poziomie pacjenta, nie epok).
- **Importowalność:** bezpieczna. Na poziomie modułu tylko importy i stałe (ścieżki **hardcoded, absolutne**); wywołanie zamknięte w `if __name__ == "__main__": build_gold()`. Import nie czyta EDF ani nie zapisuje plików.

### 4b. `dl_silver_to_gold.py` (quality.npy w gold_dl) — dla kontekstu
- `channel_quality(matrix, thresholds, sampling_rate) -> np.ndarray (N, 6)` — liczy **6 ciągłych metryk per epoka per kanał**: `[variance, flat_max_run, frac_beyond_kmad, amp_oor_ratio, physio_violation_ratio, desat_slope_max]`. Pomocnicze: `max_flat_run`, `fraction_beyond_kmad`, `spo2_to_percent`, `desat_slope_max`.
- `export_patient_quality(...)` składa to po 16 kanałach → `quality.npy` `(N, 16, 6)`, float32.
- **To NIE jest maska binarna** — progi z `quality_thresholds.json` (`load_quality_config`, mapowanie `channel_to_class`) służą jedynie do wyboru, które metryki policzyć (`amplitude_max_uv` → metryki amplitudy; `physio_min` → metryki SpO2) oraz jako wartość odniesienia przy `amp_oor_ratio`. Binaryzacja/odrzucenie epok jest odroczona do konsumenta (Dataset).
- **Importowalność:** bezpieczna — brak kodu wykonywalnego na poziomie modułu (nawet bez `__main__`), ścieżki hardcoded absolutne, EDF czytany tylko wewnątrz funkcji.

## 5. Skrypt eksportu `gold_dl`

- **Lokalizacja/nazwa:** `src/dl_silver_to_gold.py`.
- **Trzy niezależne ścieżki eksportu** (uruchamiane przez `run_export(export_function)` z `ProcessPoolExecutor`):
  - `export_patient_signals` → `x_200/x_100/x_25.npy` (float16, z kontrolą skończoności `to_float16`),
  - `export_patient_annotations` → `stage_expert, arousal, resp, limb, caisr_prob`,
  - `export_patient_quality` → `quality.npy`.
- **Tablica zbiorcza:** `build_patients_table` → `patients.parquet` (metadane z `assignments.csv` + `demographics.csv`: `SiteID, Cognitive_Impairment, split, Age, Sex`; `n_epochs`; per-kanał `ch_mask_*`, `median_*`, `iqr_*`).
- **Czy zapisuje coś poza sygnałami i etykietami:** tak — oprócz sygnałów i etykiet zapisuje `quality.npy` (metryki jakości) oraz `patients.parquet` (metadane, statystyki kanałów, maski obecności). Brak zapisu samej binarnej maski odrzucenia epok.

---

**Nierozstrzygnięte:** znaczenie kodu stagingu `0` u pacjentów site `I0006` (brak w `STAGE_CODE`); z kodu nie wynika, czy to inny wariant „unscored”, czy odrębna klasa.

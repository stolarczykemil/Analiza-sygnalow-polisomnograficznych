import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
from scipy import stats
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, sosfiltfilt, welch

SILVER_DIR = Path("/home/emil/studia/inzynierka/data/silver")
GOLD_DIR = Path("/home/emil/studia/inzynierka/data/gold")
CONFIG_DIR = Path("/home/emil/studia/inzynierka/references")
DEMOGRAPHICS_CSV = SILVER_DIR / "demographics.csv"
TRUTH_TABLE_JSON = CONFIG_DIR / "truth_table.json"
QUALITY_THRESHOLDS_JSON = CONFIG_DIR / "quality_thresholds.json"
GOLD_OUTPUT = GOLD_DIR / "features.parquet"

EPOCH_SECONDS = 30
KMAD_K = 5.0

STAGE_CODE = {1: "N3", 2: "N2", 3: "N1", 4: "REM", 5: "W", 9: "unscored"}
STAGES = ["W", "N1", "N2", "N3", "REM"]
SLEEP_STAGES = ["N1", "N2", "N3", "REM"]

REGIONS = {
    "frontal": ["eeg_f3_m2", "eeg_f4_m1"],
    "central": ["eeg_c3_m2", "eeg_c4_m1"],
    "occipital": ["eeg_o1_m2", "eeg_o2_m1"],
}

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}
TOTAL_BAND = (0.5, 30.0)
KURTOSIS_BANDS = ["delta", "theta", "alpha", "sigma"]

STAGE_CHANNEL = "freq_stage_caisr"
RESP_CHANNEL = "freq_resp_caisr"
LIMB_CHANNEL = "freq_limb_caisr"
SPO2_CHANNEL = "resp_spo2"

RESP_APNEA_HYPOPNEA_CLASSES = {1, 2, 3, 4}
LIMB_PERIODIC_CLASS = 2

DEMOGRAPHICS_PASSTHROUGH = [
    "SiteID", "BDSPPatientID", "CreationTime", "BidsFolder", "SessionID",
    "Age", "Sex", "Race", "Ethnicity", "BMI", "Time_to_Event",
    "Cognitive_Impairment", "Last_Known_Visit_Date", "Time_to_Last_Visit",
]


def load_json(path):
    return json.loads(Path(path).read_text())


def build_channel_index_map(silver_channels):
    return {name: silver_channels.index(name) for name in silver_channels}


def edf_path(patient_id, site_session):
    record = site_session[patient_id]
    return SILVER_DIR / record["SiteID"] / f"{patient_id}_ses-{record['SessionID']}.edf"


def read_stage_labels(reader, stage_index):
    raw = reader.readSignal(stage_index).astype(int)
    return np.array([STAGE_CODE.get(code, "unscored") for code in raw])


def read_event_labels(reader, channel_index):
    return reader.readSignal(channel_index).astype(int)


def read_epoched_signal(reader, channel_index, n_epochs):
    sampling_rate = int(round(reader.getSampleFrequency(channel_index)))
    samples_per_epoch = sampling_rate * EPOCH_SECONDS
    signal = reader.readSignal(channel_index).astype(np.float64)
    usable_epochs = signal.size // samples_per_epoch
    matrix = signal[: usable_epochs * samples_per_epoch]
    matrix = matrix.reshape(usable_epochs, samples_per_epoch)[:n_epochs]
    return matrix, sampling_rate


def max_flat_run_length(epoch):
    if epoch.size < 2:
        return 1
    eq = (epoch[1:] == epoch[:-1]).astype(np.int8)
    z = np.concatenate(([0], eq, [0]))
    edges = np.flatnonzero(np.diff(z))
    if edges.size == 0:
        return 1
    return int((edges[1::2] - edges[0::2] + 1).max())


def amplitude_out_of_range_ratio(epoch, amplitude_max_uv):
    return float(np.nanmean(np.abs(epoch) > amplitude_max_uv))


def fraction_beyond_kmad(epoch, k):
    med = np.nanmedian(epoch)
    mad = np.nanmedian(np.abs(epoch - med))
    if mad == 0:
        return np.nan
    return float(np.nanmean(np.abs(epoch - med) > k * mad))


def is_electrophysiological_epoch_valid(epoch, thresholds):
    if not np.isfinite(epoch).any():
        return False
    if np.nanvar(epoch) < thresholds["flat_var_min"]:
        return False
    if max_flat_run_length(epoch) >= thresholds["flat_run_min_samples"]:
        return False
    if amplitude_out_of_range_ratio(epoch, thresholds["amplitude_max_uv"]) > thresholds["amp_oor_ratio_max"]:
        return False
    if fraction_beyond_kmad(epoch, KMAD_K) > thresholds["frac_beyond_kmad_max"]:
        return False
    return True


def build_region_epochs(channel_epochs, thresholds):
    n_epochs = channel_epochs[0].shape[0]
    region_epochs = []
    for i in range(n_epochs):
        usable = [ch[i] for ch in channel_epochs if is_electrophysiological_epoch_valid(ch[i], thresholds)]
        if len(usable) == 0:
            region_epochs.append(None)
        elif len(usable) == 1:
            region_epochs.append(usable[0])
        else:
            region_epochs.append(np.mean(np.stack(usable), axis=0))
    return region_epochs


def group_region_epochs_by_stage(region_epochs, stage_labels):
    grouped = {stage: [] for stage in STAGES}
    for epoch, stage in zip(region_epochs, stage_labels[: len(region_epochs)]):
        if epoch is None:
            continue
        if stage in grouped:
            grouped[stage].append(epoch)
    return grouped


def compute_sleep_macrostructure(stage_labels):
    sleep_mask = np.isin(stage_labels, SLEEP_STAGES)
    sleep_indices = np.where(sleep_mask)[0]

    TST = int(len(sleep_indices)) * EPOCH_SECONDS
    TRT = len(stage_labels) * EPOCH_SECONDS

    if len(sleep_indices) == 0:
        return {
            "TST": TST, "TRT": TRT, "SPT": np.nan, "SE": 0.0, "SME": np.nan,
            "SL": np.nan, "REM_latency": np.nan, "WASO": np.nan,
            "N1_time": 0, "N2_time": 0, "N3_time": 0, "REM_time": 0,
            "N1_pct": np.nan, "N2_pct": np.nan, "N3_pct": np.nan, "REM_pct": np.nan,
            "NA": 0, "N1_transitions": 0, "SFI": np.nan,
        }

    first_sleep = int(sleep_indices[0])
    last_sleep = int(sleep_indices[-1])

    SPT = (last_sleep - first_sleep + 1) * EPOCH_SECONDS
    SE = TST / TRT * 100
    SME = TST / SPT * 100
    SL = first_sleep * EPOCH_SECONDS
    WASO = TRT - SL - TST

    rem_indices = np.where(stage_labels == "REM")[0]
    REM_latency = (int(rem_indices[0]) - first_sleep) * EPOCH_SECONDS if len(rem_indices) > 0 else np.nan

    N1_time = int(np.sum(stage_labels == "N1")) * EPOCH_SECONDS
    N2_time = int(np.sum(stage_labels == "N2")) * EPOCH_SECONDS
    N3_time = int(np.sum(stage_labels == "N3")) * EPOCH_SECONDS
    REM_time = int(np.sum(stage_labels == "REM")) * EPOCH_SECONDS

    N1_pct = N1_time / TST * 100
    N2_pct = N2_time / TST * 100
    N3_pct = N3_time / TST * 100
    REM_pct = REM_time / TST * 100

    scored = stage_labels[stage_labels != "unscored"]
    NA = int(sum(1 for p, c in zip(scored[:-1], scored[1:]) if p in SLEEP_STAGES and c == "W"))
    N1_transitions = int(sum(
        1 for p, c in zip(scored[:-1], scored[1:])
        if p in ["N2", "N3", "REM"] and c == "N1"
    ))

    total_sleep_hours = TST / 3600.0
    SFI = (NA + N1_transitions) / total_sleep_hours if total_sleep_hours > 0 else np.nan

    return {
        "TST": TST, "TRT": TRT, "SPT": SPT, "SE": SE, "SME": SME,
        "SL": SL, "REM_latency": REM_latency, "WASO": WASO,
        "N1_time": N1_time, "N2_time": N2_time, "N3_time": N3_time, "REM_time": REM_time,
        "N1_pct": N1_pct, "N2_pct": N2_pct, "N3_pct": N3_pct, "REM_pct": REM_pct,
        "NA": NA, "N1_transitions": N1_transitions, "SFI": SFI,
    }


def _count_events(label_array, event_classes):
    is_event = np.isin(label_array, list(event_classes)).astype(np.int8)
    return int(np.sum(np.diff(np.concatenate(([0], is_event))) == 1))


def compute_respiratory_events(resp_labels, total_sleep_hours):
    n_events = _count_events(resp_labels, RESP_APNEA_HYPOPNEA_CLASSES)
    AHI = n_events / total_sleep_hours if total_sleep_hours > 0 else np.nan
    return {"AHI": AHI}


def compute_limb_events(limb_labels, total_sleep_hours):
    is_plm = (limb_labels == LIMB_PERIODIC_CLASS).astype(np.int8)
    n_plms = int(np.sum(np.diff(np.concatenate(([0], is_plm))) == 1))
    PLMSI = n_plms / total_sleep_hours if total_sleep_hours > 0 else np.nan
    return {"PLMS": n_plms, "PLMSI": PLMSI}


def _to_spo2_pct(spo2_signal):
    finite = spo2_signal[np.isfinite(spo2_signal)]
    if len(finite) > 0 and finite.max() <= 2.0:
        return spo2_signal * 100.0
    return spo2_signal


def compute_oximetry(spo2_signal, spo2_sampling_rate, sleep_epoch_mask, thresholds):
    spo2_signal = _to_spo2_pct(spo2_signal)
    physio_min = thresholds["channel_class_thresholds"]["spo2"]["physio_min"]
    samples_per_epoch = spo2_sampling_rate * EPOCH_SECONDS

    n_epochs = len(sleep_epoch_mask)
    n_spo2_epochs = len(spo2_signal) // samples_per_epoch
    valid_epochs = min(n_epochs, n_spo2_epochs)

    epoch_mask_aligned = sleep_epoch_mask[:valid_epochs]
    spo2_aligned = spo2_signal[: valid_epochs * samples_per_epoch]
    sample_mask = np.repeat(epoch_mask_aligned, samples_per_epoch)

    sleep_spo2 = spo2_aligned[sample_mask]
    physiological = (sleep_spo2 >= physio_min) & (sleep_spo2 <= 100.0)
    sleep_spo2_clean = sleep_spo2[physiological]

    nan_result = {"mean_SpO2": np.nan, "minimal_SpO2": np.nan, "ODI": np.nan, "sub90_SpO2": np.nan}
    if len(sleep_spo2_clean) == 0:
        return nan_result

    mean_SpO2 = float(np.mean(sleep_spo2_clean))
    minimal_SpO2 = float(np.min(sleep_spo2_clean))
    sub90_SpO2 = float(np.mean(sleep_spo2_clean < 90.0) * 100)

    baseline_window = max(1, int(120 * spo2_sampling_rate))
    baseline = maximum_filter1d(sleep_spo2_clean, size=baseline_window, mode="nearest")
    desaturated = (sleep_spo2_clean < baseline - 3.0).astype(np.int8)
    n_odi = int(np.sum(np.diff(np.concatenate(([0], desaturated))) == 1))

    total_sleep_hours = float(np.sum(epoch_mask_aligned)) * EPOCH_SECONDS / 3600.0
    ODI = n_odi / total_sleep_hours if total_sleep_hours > 0 else np.nan

    return {"mean_SpO2": mean_SpO2, "minimal_SpO2": minimal_SpO2, "ODI": ODI, "sub90_SpO2": sub90_SpO2}


def _bandpass(epoch, low, high, fs):
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, epoch)


def _epoch_time_features(epoch, fs):
    dx = np.diff(epoch)
    ddx = np.diff(dx)
    var_x = np.var(epoch)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)

    hjorth_activity = float(var_x)
    hjorth_mobility = float(np.sqrt(var_dx / var_x)) if var_x > 0 else np.nan
    if var_x > 0 and var_dx > 0:
        hjorth_complexity = float(np.sqrt(var_ddx / var_dx) / np.sqrt(var_dx / var_x))
    else:
        hjorth_complexity = np.nan

    result = {
        "hjorth_activity": hjorth_activity,
        "hjorth_mobility": hjorth_mobility,
        "hjorth_complexity": hjorth_complexity,
        "kurtosis": float(stats.kurtosis(epoch, fisher=False)),
        "line_length": float(np.sum(np.abs(dx))),
    }
    for band in KURTOSIS_BANDS:
        filtered = _bandpass(epoch, *BANDS[band], fs)
        result[f"{band}_kurtosis"] = float(stats.kurtosis(filtered, fisher=False))
    return result


_TIME_FEAT_KEYS = (
    ["hjorth_activity", "hjorth_mobility", "hjorth_complexity", "kurtosis", "line_length"]
    + [f"{b}_kurtosis" for b in KURTOSIS_BANDS]
)


def compute_eeg_time_domain(region_epochs_by_stage, sampling_rate):
    result = {}
    for region in REGIONS:
        for stage in STAGES:
            epochs = region_epochs_by_stage[region][stage]
            suffix = f"_{region}_{stage}"
            if not epochs:
                for k in _TIME_FEAT_KEYS:
                    result[k + suffix] = np.nan
                continue
            per_epoch = [_epoch_time_features(e, sampling_rate) for e in epochs]
            for k in _TIME_FEAT_KEYS:
                result[k + suffix] = float(np.nanmean([f[k] for f in per_epoch]))
    return result


def _epoch_psd(epoch, fs):
    nperseg = min(4 * fs, len(epoch))
    noverlap = nperseg // 2
    return welch(epoch, fs=fs, nperseg=nperseg, noverlap=noverlap)


_SPECTRAL_FEAT_KEYS = (
    [f"{b}_rel_power" for b in BANDS]
    + ["delta_theta_ratio", "delta_alpha_ratio", "theta_alpha_ratio", "slowing_index"]
)


def compute_eeg_spectral(region_epochs_by_stage, sampling_rate):
    result = {}
    for region in REGIONS:
        for stage in STAGES:
            epochs = region_epochs_by_stage[region][stage]
            suffix = f"_{region}_{stage}"
            if not epochs:
                for k in _SPECTRAL_FEAT_KEYS:
                    result[k + suffix] = np.nan
                continue

            freqs, first_psd = _epoch_psd(epochs[0], sampling_rate)
            psd_stack = np.empty((len(epochs), len(first_psd)))
            psd_stack[0] = first_psd
            for i, e in enumerate(epochs[1:], 1):
                _, psd_stack[i] = _epoch_psd(e, sampling_rate)
            avg_psd = psd_stack.mean(axis=0)

            def bp(low, high):
                idx = (freqs >= low) & (freqs <= high)
                return float(np.trapezoid(avg_psd[idx], freqs[idx]))

            total_power = bp(*TOTAL_BAND)
            band_powers = {b: bp(*BANDS[b]) for b in BANDS}

            for b in BANDS:
                result[f"{b}_rel_power{suffix}"] = (
                    band_powers[b] / total_power if total_power > 0 else np.nan
                )

            delta_p = band_powers["delta"]
            theta_p = band_powers["theta"]
            alpha_p = band_powers["alpha"]
            beta_p = band_powers["beta"]

            result[f"delta_theta_ratio{suffix}"] = delta_p / theta_p if theta_p > 0 else np.nan
            result[f"delta_alpha_ratio{suffix}"] = delta_p / alpha_p if alpha_p > 0 else np.nan
            result[f"theta_alpha_ratio{suffix}"] = theta_p / alpha_p if alpha_p > 0 else np.nan
            result[f"slowing_index{suffix}"] = (
                (alpha_p + beta_p) / (delta_p + theta_p) if (delta_p + theta_p) > 0 else np.nan
            )

    return result


def single_value_feature_columns():
    return [
        "TST", "TRT", "SPT", "SE", "SME", "SL", "REM_latency", "WASO",
        "N1_time", "N2_time", "N3_time", "REM_time",
        "N1_pct", "N2_pct", "N3_pct", "REM_pct",
        "NA", "N1_transitions", "SFI",
        "AHI", "PLMS", "PLMSI",
        "mean_SpO2", "minimal_SpO2", "ODI", "sub90_SpO2",
    ]


def time_domain_feature_columns():
    names = []
    for region in REGIONS:
        for stage in STAGES:
            names += [
                f"hjorth_activity_{region}_{stage}",
                f"hjorth_mobility_{region}_{stage}",
                f"hjorth_complexity_{region}_{stage}",
                f"kurtosis_{region}_{stage}",
                f"line_length_{region}_{stage}",
            ]
            for band in KURTOSIS_BANDS:
                names.append(f"{band}_kurtosis_{region}_{stage}")
    return names


def spectral_feature_columns():
    names = []
    for region in REGIONS:
        for stage in STAGES:
            names += [
                f"delta_rel_power_{region}_{stage}",
                f"theta_rel_power_{region}_{stage}",
                f"alpha_rel_power_{region}_{stage}",
                f"sigma_rel_power_{region}_{stage}",
                f"beta_rel_power_{region}_{stage}",
                f"delta_theta_ratio_{region}_{stage}",
                f"delta_alpha_ratio_{region}_{stage}",
                f"theta_alpha_ratio_{region}_{stage}",
                f"slowing_index_{region}_{stage}",
            ]
    return names


def expected_feature_columns():
    return single_value_feature_columns() + time_domain_feature_columns() + spectral_feature_columns()


def extract_patient(patient_id, site_session, channel_index, quality_thresholds):
    reader = pyedflib.EdfReader(str(edf_path(patient_id, site_session)))
    try:
        stage_labels = read_stage_labels(reader, channel_index[STAGE_CHANNEL])
        n_epochs = len(stage_labels)

        eeg_thresholds = quality_thresholds["channel_class_thresholds"]["eeg"]
        region_epochs_by_stage = {}
        eeg_sampling_rate = None
        for region, channels in REGIONS.items():
            channel_matrices = []
            for channel in channels:
                matrix, eeg_sampling_rate = read_epoched_signal(reader, channel_index[channel], n_epochs)
                channel_matrices.append(matrix)
            region_epochs = build_region_epochs(channel_matrices, eeg_thresholds)
            region_epochs_by_stage[region] = group_region_epochs_by_stage(region_epochs, stage_labels)

        resp_labels = read_event_labels(reader, channel_index[RESP_CHANNEL])
        limb_labels = read_event_labels(reader, channel_index[LIMB_CHANNEL])
        spo2_signal = reader.readSignal(channel_index[SPO2_CHANNEL]).astype(np.float64)
        spo2_sampling_rate = int(round(reader.getSampleFrequency(channel_index[SPO2_CHANNEL])))

        sleep_epoch_mask = np.isin(stage_labels, SLEEP_STAGES)

        macrostructure = compute_sleep_macrostructure(stage_labels)
        total_sleep_hours = macrostructure["TST"] / 3600.0

        row = {}
        row.update(macrostructure)
        row.update(compute_respiratory_events(resp_labels, total_sleep_hours))
        row.update(compute_limb_events(limb_labels, total_sleep_hours))
        row.update(compute_oximetry(spo2_signal, spo2_sampling_rate, sleep_epoch_mask, quality_thresholds))
        row.update(compute_eeg_time_domain(region_epochs_by_stage, eeg_sampling_rate))
        row.update(compute_eeg_spectral(region_epochs_by_stage, eeg_sampling_rate))
        row["BidsFolder"] = patient_id
        return row
    finally:
        reader.close()


def preview_patient(patient_id):
    demographics = pd.read_csv(DEMOGRAPHICS_CSV)
    site_session = demographics.set_index("BidsFolder")[["SiteID", "SessionID"]].to_dict("index")
    truth_table = load_json(TRUTH_TABLE_JSON)
    channel_index = build_channel_index_map(truth_table["silver_channels"])
    quality_thresholds = load_json(QUALITY_THRESHOLDS_JSON)
    row = extract_patient(patient_id, site_session, channel_index, quality_thresholds)
    return pd.DataFrame([row])


def build_gold():
    demographics = pd.read_csv(DEMOGRAPHICS_CSV)
    site_session = demographics.set_index("BidsFolder")[["SiteID", "SessionID"]].to_dict("index")
    truth_table = load_json(TRUTH_TABLE_JSON)
    channel_index = build_channel_index_map(truth_table["silver_channels"])
    quality_thresholds = load_json(QUALITY_THRESHOLDS_JSON)

    rows = []
    for patient_id in demographics["BidsFolder"]:
        rows.append(extract_patient(patient_id, site_session, channel_index, quality_thresholds))

    features = pd.DataFrame(rows)
    gold = demographics[DEMOGRAPHICS_PASSTHROUGH].merge(features, on="BidsFolder", how="left")

    missing = set(expected_feature_columns()) - set(gold.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    gold.to_parquet(GOLD_OUTPUT, index=False)
    return gold


if __name__ == "__main__":
    build_gold()

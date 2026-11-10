import json
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib

from dl.config import CHANNEL_INDEX, EMG_CHANNELS
from dl.quality import load_channel_thresholds, quality_mask

SILVER_DIR = Path("/home/emil/studia/inzynierka/data/silver")
GOLD_DL_DIR = Path("/home/emil/studia/inzynierka/data/gold_dl")
CONFIG_DIR = Path("/home/emil/studia/inzynierka/references")
DEMOGRAPHICS_CSV = SILVER_DIR / "demographics.csv"
TRUTH_TABLE_JSON = CONFIG_DIR / "truth_table.json"
ASSIGNMENTS_CSV = CONFIG_DIR / "assignments.csv"
PATIENTS_PARQUET = GOLD_DL_DIR / "patients.parquet"

EPOCH_SECONDS = 30
STAGE_EXPERT_CHANNEL = "freq_stage_expert"

SIGNAL_GROUPS = {
    "x_200": (200, [
        "eeg_f3_m2", "eeg_f4_m1", "eeg_c3_m2", "eeg_c4_m1", "eeg_o1_m2", "eeg_o2_m1",
        "eog_e1_m2", "eog_e2_m1",
        "chin_emg_chin1_chin2",
        "leg_emg_lat", "leg_emg_rat",
        "ecg_ecg",
    ]),
    "x_100": (100, ["resp_chest", "resp_abd", "resp_ptaf"]),
    "x_25": (25, ["resp_spo2"]),
}

EVENT_ANNOTATIONS = {
    "arousal": ("freq_arousal_expert", 60),
    "limb": ("freq_limb_expert", 30),
    "resp": ("freq_resp_expert", 30),
}
CAISR_PROB_INDICES = [24, 25, 26, 27, 28, 29, 30]

QUALITY_THRESHOLDS_JSON = CONFIG_DIR / "quality_thresholds.json"
QUALITY_METRICS = [
    "variance", "flat_max_run", "frac_beyond_kmad",
    "amp_oor_ratio", "physio_violation_ratio", "desat_slope_max",
]
KMAD_K = 5.0
SPO2_FRACTION_MAX = 1.5


def load_json(path):
    return json.loads(Path(path).read_text())


def build_channel_index_map(silver_channels):
    return {name: silver_channels.index(name) for name in silver_channels}


def load_site_session():
    demographics = pd.read_csv(DEMOGRAPHICS_CSV)
    return demographics.set_index("BidsFolder")[["SiteID", "SessionID"]].to_dict("index")


def edf_path(patient_id, site_session):
    record = site_session[patient_id]
    return SILVER_DIR / record["SiteID"] / f"{patient_id}_ses-{record['SessionID']}.edf"


def count_epochs(reader, channel_index):
    return int(reader.getNSamples()[channel_index[STAGE_EXPERT_CHANNEL]])


def read_channel_epochs(reader, index, sampling_rate, n_epochs):
    actual_rate = reader.getSampleFrequency(index)
    if not np.isclose(actual_rate, sampling_rate):
        raise ValueError(f"Channel {index}: expected {sampling_rate} Hz, got {actual_rate} Hz")
    samples_per_epoch = sampling_rate * EPOCH_SECONDS
    signal = reader.readSignal(index)
    return signal.reshape(n_epochs, samples_per_epoch)


def read_signal_group(reader, channel_index, channels, sampling_rate, n_epochs):
    epochs = [read_channel_epochs(reader, channel_index[name], sampling_rate, n_epochs) for name in channels]
    return np.stack(epochs, axis=1)


def to_float16(array, group_name):
    converted = array.astype(np.float16)
    if not np.isfinite(converted).all():
        raise ValueError(f"{group_name}: non-finite values after float16 cast, max |x| = {np.abs(array).max()}")
    return converted


def export_patient_signals(patient_id, site_session, channel_index):
    out_dir = GOLD_DL_DIR / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = pyedflib.EdfReader(str(edf_path(patient_id, site_session)))
    try:
        n_epochs = count_epochs(reader, channel_index)
        shapes = {}
        for group_name, (sampling_rate, channels) in SIGNAL_GROUPS.items():
            group = read_signal_group(reader, channel_index, channels, sampling_rate, n_epochs)
            np.save(out_dir / f"{group_name}.npy", to_float16(group, group_name))
            shapes[group_name] = group.shape
        return shapes
    finally:
        reader.close()


def samples_per_epoch(reader, index):
    return int(round(reader.getSampleFrequency(index) * EPOCH_SECONDS))


def read_event_codes(reader, index, expected_per_epoch, n_epochs):
    per_epoch = samples_per_epoch(reader, index)
    if per_epoch != expected_per_epoch:
        raise ValueError(f"Channel {index}: expected {expected_per_epoch} samples per epoch, got {per_epoch}")
    codes = np.rint(reader.readSignal(index)).astype(np.int8)
    return codes.reshape(n_epochs, per_epoch)


def read_stage_codes(reader, index, n_epochs):
    return read_event_codes(reader, index, 1, n_epochs).ravel()


def read_epoch_mean(reader, index, n_epochs):
    per_epoch = samples_per_epoch(reader, index)
    values = reader.readSignal(index).reshape(n_epochs, per_epoch)
    return values.mean(axis=1)


def read_caisr_probabilities(reader, n_epochs):
    columns = [read_epoch_mean(reader, index, n_epochs) for index in CAISR_PROB_INDICES]
    return np.stack(columns, axis=1).astype(np.float32)


def export_patient_annotations(patient_id, site_session, channel_index):
    out_dir = GOLD_DL_DIR / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = pyedflib.EdfReader(str(edf_path(patient_id, site_session)))
    try:
        n_epochs = count_epochs(reader, channel_index)
        shapes = {}
        stage = read_stage_codes(reader, channel_index[STAGE_EXPERT_CHANNEL], n_epochs)
        np.save(out_dir / "stage_expert.npy", stage)
        shapes["stage_expert"] = stage.shape
        for name, (channel, per_epoch) in EVENT_ANNOTATIONS.items():
            codes = read_event_codes(reader, channel_index[channel], per_epoch, n_epochs)
            np.save(out_dir / f"{name}.npy", codes)
            shapes[name] = codes.shape
        probabilities = read_caisr_probabilities(reader, n_epochs)
        np.save(out_dir / "caisr_prob.npy", probabilities)
        shapes["caisr_prob"] = probabilities.shape
        return shapes
    finally:
        reader.close()


def max_flat_run(matrix):
    n_epochs, samples = matrix.shape
    signal = matrix.ravel()
    equal = np.equal(signal[1:], signal[:-1])
    if n_epochs > 1:
        equal[np.arange(1, n_epochs) * samples - 1] = False
    padded = np.concatenate(([0], equal.view(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    runs = np.ones(n_epochs, dtype=np.float64)
    if edges.size:
        starts = edges[0::2]
        np.maximum.at(runs, starts // samples, edges[1::2] - starts + 1)
    return runs


def fraction_beyond_kmad(matrix):
    median = np.median(matrix, axis=1, keepdims=True)
    mad = np.median(np.abs(matrix - median), axis=1, keepdims=True)
    fraction = (np.abs(matrix - median) > KMAD_K * mad).mean(axis=1)
    return np.where(mad.ravel() == 0, np.nan, fraction)


def spo2_to_percent(matrix):
    if np.abs(matrix).max() <= SPO2_FRACTION_MAX:
        return matrix * 100.0
    return matrix


def desat_slope_max(matrix, sampling_rate):
    drops = matrix[:, :-sampling_rate] - matrix[:, sampling_rate:]
    return np.maximum(drops.max(axis=1), 0.0)


def channel_quality(matrix, thresholds, sampling_rate):
    metrics = np.full((matrix.shape[0], len(QUALITY_METRICS)), np.nan)
    metrics[:, 0] = matrix.var(axis=1)
    metrics[:, 1] = max_flat_run(matrix)
    if "amplitude_max_uv" in thresholds:
        metrics[:, 2] = fraction_beyond_kmad(matrix)
        metrics[:, 3] = (np.abs(matrix) > thresholds["amplitude_max_uv"]).mean(axis=1)
    if "physio_min" in thresholds:
        percent = spo2_to_percent(matrix)
        metrics[:, 4] = (percent < thresholds["physio_min"]).mean(axis=1)
        metrics[:, 5] = desat_slope_max(percent, sampling_rate)
    return metrics


def load_quality_config():
    config = load_json(QUALITY_THRESHOLDS_JSON)
    classes = config["channel_class_thresholds"]
    return {channel: classes[cls] for channel, cls in config["channel_to_class"].items()}


def export_patient_quality(patient_id, site_session, channel_index):
    channel_thresholds = load_quality_config()
    out_dir = GOLD_DL_DIR / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = pyedflib.EdfReader(str(edf_path(patient_id, site_session)))
    try:
        n_epochs = count_epochs(reader, channel_index)
        per_channel = []
        for sampling_rate, channels in SIGNAL_GROUPS.values():
            for channel in channels:
                matrix = read_channel_epochs(reader, channel_index[channel], sampling_rate, n_epochs)
                per_channel.append(channel_quality(matrix, channel_thresholds[channel], sampling_rate))
        quality = np.stack(per_channel, axis=1).astype(np.float32)
        np.save(out_dir / "quality.npy", quality)
        return {"quality": quality.shape}
    finally:
        reader.close()


def load_export_context():
    site_session = load_site_session()
    channel_index = build_channel_index_map(load_json(TRUTH_TABLE_JSON)["silver_channels"])
    return site_session, channel_index


def preview_patient(patient_id, export_function=export_patient_signals):
    site_session, channel_index = load_export_context()
    return export_function(patient_id, site_session, channel_index)


def safe_export(export_function, patient_id, site_session, channel_index):
    try:
        return patient_id, export_function(patient_id, site_session, channel_index), None
    except Exception as error:
        return patient_id, None, repr(error)


def run_export(export_function, max_workers=8):
    site_session, channel_index = load_export_context()
    patient_ids = list(site_session)

    results = {}
    errors = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        jobs = executor.map(
            safe_export, repeat(export_function), patient_ids, repeat(site_session), repeat(channel_index)
        )
        for done, (patient_id, shapes, error) in enumerate(jobs, 1):
            if error is None:
                results[patient_id] = shapes
            else:
                errors[patient_id] = error
            print(f"{done}/{len(patient_ids)} {patient_id} {'OK' if error is None else 'ERROR'}")

    print(f"Exported: {len(results)}, failed: {len(errors)}")
    for patient_id, error in errors.items():
        print(f"{patient_id}: {error}")
    return results, errors


def channel_statistics(values):
    values = np.asarray(values, dtype=np.float32).ravel()
    std = values.std()
    present = bool(np.isfinite(std) and std > 0)
    if not present:
        return present, np.nan, np.nan
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return present, float(median), float(q75 - q25)


def emg_statistics(values, good_epochs):
    selected = np.asarray(values[good_epochs], dtype=np.float64)
    if selected.size == 0:
        return np.nan, np.nan
    return float(selected.mean()), float(selected.std())


def critical_channels(truth_table):
    return {name for name, config in truth_table["physiological_channels"].items() if config.get("critical")}


def patient_channel_statistics(patient_id, critical, channel_thresholds):
    patient_dir = GOLD_DL_DIR / patient_id
    good = quality_mask(np.load(patient_dir / "quality.npy"), channel_thresholds)
    row = {"patient_id": patient_id}
    for group_name, (_, channels) in SIGNAL_GROUPS.items():
        group = np.load(patient_dir / f"{group_name}.npy", mmap_mode="r")
        row["n_epochs"] = group.shape[0]
        for position, channel in enumerate(channels):
            present, median, iqr = channel_statistics(group[:, position, :])
            if channel not in critical:
                row[f"ch_mask_{channel}"] = present
            row[f"median_{channel}"] = median
            row[f"iqr_{channel}"] = iqr
            if channel in EMG_CHANNELS:
                mean, std = emg_statistics(group[:, position, :], good[:, CHANNEL_INDEX[channel]])
                row[f"mean_{channel}"] = mean
                row[f"std_{channel}"] = std
    return row


def load_patient_metadata():
    assignments = pd.read_csv(ASSIGNMENTS_CSV)[["patient_id", "SiteID", "Cognitive_Impairment", "split"]]
    demographics = pd.read_csv(DEMOGRAPHICS_CSV)[["BidsFolder", "Age", "Sex"]]
    metadata = assignments.merge(demographics, left_on="patient_id", right_on="BidsFolder", validate="one_to_one")
    if len(metadata) != len(assignments):
        raise ValueError(f"Metadata join lost patients: {len(assignments)} -> {len(metadata)}")
    return metadata.drop(columns="BidsFolder")


def build_patients_table(max_workers=8):
    metadata = load_patient_metadata()
    critical = critical_channels(load_json(TRUTH_TABLE_JSON))
    channel_thresholds = load_channel_thresholds()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        rows = list(executor.map(
            patient_channel_statistics, metadata["patient_id"], repeat(critical), repeat(channel_thresholds)
        ))
    patients = metadata.merge(pd.DataFrame(rows), on="patient_id", validate="one_to_one")
    patients.to_parquet(PATIENTS_PARQUET, index=False)
    return patients


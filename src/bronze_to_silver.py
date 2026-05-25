import json
import logging
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import resample_poly
from scipy.stats import kurtosis, skew

logger = logging.getLogger(__name__)

SQI_WINDOW_START_SEC = 1000.0
SQI_WINDOW_LENGTH_SEC = 100.0
DATARECORD_DURATION_SEC = 30.0
ANNOTATION_PREFIX = "freq_"
HUMAN_ANNOTATION_DIRNAME = "human_annotations"
ALGO_ANNOTATION_DIRNAME = "algorithmic_annotations"
HUMAN_ANNOTATION_SUFFIX = "_expert_annotations.edf"
ALGO_ANNOTATION_SUFFIX = "_caisr_annotations.edf"
DEMOGRAPHICS_FILENAME = "demographics.csv"
ZSCORE_DIMENSION = "z"


def load_truth_table(truth_table_path):
    with open(truth_table_path, "r") as handle:
        raw = json.load(handle)
    return {key: value for key, value in raw.items() if not key.startswith("_")}


def clean_label(label):
    return str(label).strip().replace(" ", "_")


def split_source_field(value):
    if value is None:
        return []
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return []
    return [clean_label(part) for part in text.split(";") if clean_label(part) != ""]


def build_channel_index(reader):
    index = {}
    for position, header in enumerate(reader.getSignalHeaders()):
        index.setdefault(clean_label(header["label"]), position)
    return index


def read_window(reader, position, native_fs):
    total = reader.getNSamples()[position]
    if total <= 0:
        return np.empty(0)
    start = int(round(SQI_WINDOW_START_SEC * native_fs))
    length = int(round(SQI_WINDOW_LENGTH_SEC * native_fs))
    if length <= 0:
        length = total
    if start >= total:
        start = 0
    if start + length > total:
        length = total - start
    return reader.readSignal(position, start, length)


def header_frequency(reader, position):
    return float(reader.getSignalHeaders()[position]["sample_frequency"])


def header_dimension(reader, position):
    return str(reader.getSignalHeaders()[position]["dimension"]).strip()


def candidate_variance(reader, index, label):
    position = index[label]
    window = read_window(reader, position, header_frequency(reader, position))
    if window.size == 0:
        return 0.0, 1.0
    nan_ratio = float(np.mean(~np.isfinite(window)))
    finite = window[np.isfinite(window)]
    variance = float(np.var(finite)) if finite.size > 0 else 0.0
    return variance, nan_ratio


def priority_rank(label, priority):
    lowered = label.lower()
    for position, name in enumerate(priority):
        if name.lower() == lowered:
            return float(position)
    for position, name in enumerate(priority):
        if name.lower() in lowered:
            return position + 0.5
    return float(len(priority))


def pick_by_priority(labels, priority):
    ranked = sorted(enumerate(labels), key=lambda item: (priority_rank(item[1], priority), item[0]))
    return ranked[0][1]


def is_globally_flat(reader, index, label):
    position = index[label]
    signal = reader.readSignal(position)
    finite = signal[np.isfinite(signal)]
    return finite.size == 0 or float(np.var(finite)) <= 0.0


def all_globally_flat(candidates, reader, index):
    return all(is_globally_flat(reader, index, label) for label in candidates)


def resolve_variance_priority(conflict, candidates, reader, index):
    nan_ratio_max = conflict.get("nan_ratio_max", 1.0)
    variance_min = conflict.get("variance_min", 0.0)
    priority = conflict.get("priority", [])
    kept = []
    for label in candidates:
        variance, nan_ratio = candidate_variance(reader, index, label)
        if nan_ratio >= nan_ratio_max:
            continue
        if variance <= variance_min:
            continue
        kept.append(label)
    if kept:
        return pick_by_priority(kept, priority)
    if conflict.get("drop_if_all_flat", False) and all_globally_flat(candidates, reader, index):
        return None
    return pick_by_priority(candidates, priority)


def signal_quality_index(reader, index, label):
    position = index[label]
    window = read_window(reader, position, header_frequency(reader, position))
    finite = window[np.isfinite(window)]
    if finite.size == 0 or float(np.var(finite)) <= 0.0:
        return None
    flatness = kurtosis(finite, fisher=False) * abs(skew(finite))
    if not np.isfinite(flatness):
        return None
    return float(flatness)


def resolve_sqi_hos(conflict, candidates, reader, index):
    best_label = None
    best_score = None
    for label in candidates:
        score = signal_quality_index(reader, index, label)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_label = label
    if best_label is not None:
        return best_label
    if conflict.get("drop_if_all_flat", False) and all_globally_flat(candidates, reader, index):
        return None
    return candidates[0]


def resolve_conflict(conflict, candidates, reader, index):
    method = conflict.get("method")
    if method == "variance_priority":
        return resolve_variance_priority(conflict, candidates, reader, index)
    if method == "sqi_hos":
        return resolve_sqi_hos(conflict, candidates, reader, index)
    return candidates[0]


def resolve_field(key, conflict, row, reader, index):
    labels = split_source_field(row.get(f"{key}_SOURCE"))
    candidates = [label for label in labels if label in index]
    if not candidates:
        return None
    if conflict is None or len(candidates) == 1:
        return candidates[0]
    return resolve_conflict(conflict, candidates, reader, index)


def select_source(channel, config, row, reader, index):
    reconstruction = config.get("reconstruction")
    conflict = config.get("conflict")
    if reconstruction == "direct":
        label = resolve_field(channel, conflict, row, reader, index)
        if label is None:
            return None
        return {"kind": "direct", "key": channel, "labels": [label]}
    if reconstruction == "substitution":
        label = resolve_field(channel, conflict, row, reader, index)
        if label is not None:
            return {"kind": "direct", "key": channel, "labels": [label]}
        for alternative in config.get("sources", []):
            label = resolve_field(alternative, conflict, row, reader, index)
            if label is not None:
                return {"kind": "direct", "key": alternative, "labels": [label]}
        return None
    if reconstruction == "bipolar":
        label = resolve_field(channel, None, row, reader, index)
        if label is not None:
            return {"kind": "direct", "key": channel, "labels": [label]}
        sources = config.get("sources", [])
        if len(sources) == 2:
            first = resolve_field(sources[0], None, row, reader, index)
            second = resolve_field(sources[1], None, row, reader, index)
            if first is not None and second is not None:
                return {"kind": "reconstruct", "key": None, "labels": [first, second]}
        return None
    return None


def fill_missing(signal):
    finite = np.isfinite(signal)
    if finite.all():
        return signal
    if not finite.any():
        return np.zeros_like(signal)
    replacement = float(np.mean(signal[finite]))
    filled = signal.copy()
    filled[~finite] = replacement
    return filled


def resample_signal(signal, native_fs, target_fs):
    if abs(native_fs - target_fs) < 1e-9 or signal.size == 0:
        return signal
    ratio = Fraction(target_fs / native_fs).limit_denominator(100000)
    up = max(1, ratio.numerator)
    down = max(1, ratio.denominator)
    return resample_poly(signal, up, down)


def apply_resampling(signal, native_fs, config):
    target_fs = float(config["target_freq_hz"])
    direction = config.get("resample", "both")
    if abs(native_fs - target_fs) < 1e-9:
        return signal, target_fs
    going_down = native_fs > target_fs
    if going_down and direction == "up_only":
        return signal, native_fs
    if not going_down and direction == "down_only":
        return signal, native_fs
    return resample_signal(signal, native_fs, target_fs), target_fs


def apply_units(signal, dimension, config):
    unit = config.get("unit", "none")
    if unit == "zscore":
        finite = signal[np.isfinite(signal)]
        std = float(np.std(finite)) if finite.size > 0 else 0.0
        mean = float(np.mean(finite)) if finite.size > 0 else 0.0
        if std <= 0.0:
            return np.zeros_like(signal), ZSCORE_DIMENSION
        return (signal - mean) / std, ZSCORE_DIMENSION
    if unit == "fix_header":
        header_fix = config.get("header_fix", {})
        new_dimension = header_fix.get("unit_overrides", {}).get(dimension, dimension)
        clip_range = header_fix.get("clip_range")
        if clip_range is not None:
            signal = np.clip(signal, clip_range[0], clip_range[1])
        return signal, new_dimension
    return signal, dimension


def native_frequency_with_override(channel, config, plan, row, reader, index):
    label = plan["labels"][0]
    native_fs = header_frequency(reader, index[label])
    header_fix = config.get("header_fix", {})
    overrides = header_fix.get("freq_overrides", {})
    if plan["key"] is not None and overrides:
        raw_freq = row.get(f"{plan['key']}_FREQ")
        if raw_freq is not None and str(raw_freq) in overrides:
            native_fs = float(overrides[str(raw_freq)])
    return native_fs


def materialize_physiological(channel, config, plan, row, reader, index):
    if plan["kind"] == "reconstruct":
        first_label, second_label = plan["labels"]
        first_position = index[first_label]
        second_position = index[second_label]
        first = fill_missing(reader.readSignal(first_position))
        second = fill_missing(reader.readSignal(second_position))
        first_fs = header_frequency(reader, first_position)
        second_fs = header_frequency(reader, second_position)
        if abs(first_fs - second_fs) > 1e-9:
            second = resample_signal(second, second_fs, first_fs)
        length = min(first.size, second.size)
        signal = first[:length] - second[:length]
        dimension = header_dimension(reader, first_position)
        native_fs = first_fs
    else:
        label = plan["labels"][0]
        position = index[label]
        signal = fill_missing(reader.readSignal(position))
        dimension = header_dimension(reader, position)
        native_fs = native_frequency_with_override(channel, config, plan, row, reader, index)
    signal, output_fs = apply_resampling(signal, native_fs, config)
    signal, output_dimension = apply_units(signal, dimension, config)
    return {"label": channel, "signal": np.nan_to_num(signal), "fs": output_fs, "dimension": output_dimension}


def annotation_channel(channel, row, human_reader, algo_reader, truth_table):
    annotation_channels = truth_table.get("annotation_channels", {})
    label = channel[len(ANNOTATION_PREFIX):] if channel.startswith(ANNOTATION_PREFIX) else channel
    if channel in annotation_channels.get("human", []):
        reader = human_reader
    elif channel in annotation_channels.get("algorithmic", []):
        reader = algo_reader
    else:
        reader = None
    if reader is not None:
        index = build_channel_index(reader)
        if label in index:
            position = index[label]
            signal = np.nan_to_num(reader.readSignal(position))
            return {"label": channel, "signal": signal, "fs": header_frequency(reader, position),
                    "dimension": header_dimension(reader, position)}
    fallback_fs = parse_frequency(row.get(channel))
    return {"label": channel, "signal": np.zeros(0), "fs": fallback_fs, "dimension": "idx"}


def parse_frequency(value):
    try:
        frequency = float(value)
        if frequency > 0:
            return frequency
    except (TypeError, ValueError):
        pass
    return 1.0


def physiological_channels(truth_table):
    return truth_table.get("physiological_channels", {})


def critical_channels(truth_table):
    return [name for name, config in physiological_channels(truth_table).items() if config.get("critical")]


def annotation_channel_names(truth_table):
    annotation_channels = truth_table.get("annotation_channels", {})
    return annotation_channels.get("human", []) + annotation_channels.get("algorithmic", [])


def missing_annotation(row, truth_table):
    for channel in annotation_channel_names(truth_table):
        value = row.get(channel)
        if value is None or pd.isna(value) or str(value).strip() == "":
            return channel
    return None


def open_reader(path):
    try:
        return pyedflib.EdfReader(str(path))
    except (OSError, RuntimeError, ValueError) as error:
        logger.warning("could not open EDF %s: %s", path, error)
        return None


def physiological_path(signals_root, site_id, bids_folder, session_id):
    return Path(signals_root) / site_id / f"{bids_folder}_ses-{session_id}.edf"


def annotation_path(root, dirname, site_id, bids_folder, session_id, suffix):
    return Path(root) / dirname / site_id / f"{bids_folder}_ses-{session_id}{suffix}"


def fit_to_records(signal, samples_per_record, record_count):
    target = samples_per_record * record_count
    if signal.size >= target:
        return signal[:target].astype(np.float64)
    padded = np.zeros(target, dtype=np.float64)
    padded[:signal.size] = signal
    return padded


def physical_bounds(signal):
    if signal.size == 0:
        return -1.0, 1.0
    minimum = float(np.floor(np.min(signal)))
    maximum = float(np.ceil(np.max(signal)))
    if minimum == maximum:
        maximum = minimum + 1.0
    return minimum, maximum


def write_silver_edf(output_path, channels):
    samples_per_record = [max(1, int(round(channel["fs"] * DATARECORD_DURATION_SEC))) for channel in channels]
    record_counts = []
    for channel, spr in zip(channels, samples_per_record):
        if channel["signal"].size >= spr:
            record_counts.append(channel["signal"].size // spr)
    record_count = min(record_counts) if record_counts else 1
    record_count = max(1, record_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pyedflib.EdfWriter(str(output_path), len(channels), file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        writer.setDatarecordDuration(DATARECORD_DURATION_SEC)
        prepared = []
        headers = []
        for channel, spr in zip(channels, samples_per_record):
            fitted = fit_to_records(channel["signal"], spr, record_count)
            minimum, maximum = physical_bounds(fitted)
            fitted = np.clip(fitted, minimum, maximum)
            prepared.append(fitted)
            headers.append({
                "label": channel["label"],
                "dimension": channel["dimension"],
                "sample_frequency": channel["fs"],
                "physical_min": minimum,
                "physical_max": maximum,
                "digital_min": -32768,
                "digital_max": 32767,
                "prefilter": "",
                "transducer": "",
            })
        writer.setSignalHeaders(headers)
        writer.writeSamples(prepared)
    finally:
        writer.close()


def prepare_patient(row, signals_root, annotation_root, truth_table):
    site_id = str(row["SiteID"])
    bids_folder = str(row["BidsFolder"])
    session_id = str(row["SessionID"])
    absent_annotation = missing_annotation(row, truth_table)
    if absent_annotation is not None:
        return None, f"missing critical annotation: {absent_annotation}"
    edf_path = physiological_path(signals_root, site_id, bids_folder, session_id)
    if not edf_path.exists():
        return None, "missing physiological EDF"
    reader = open_reader(edf_path)
    if reader is None:
        return None, "unreadable physiological EDF"
    silver_channels = truth_table["silver_channels"]
    physiological = physiological_channels(truth_table)
    critical = set(critical_channels(truth_table))
    try:
        index = build_channel_index(reader)
        plans = {}
        for channel, config in physiological.items():
            plan = select_source(channel, config, row, reader, index)
            if plan is None and channel in critical:
                return None, f"critical channel unavailable: {channel}"
            plans[channel] = plan
        prepared = []
        human_reader = open_reader(annotation_path(
            annotation_root, HUMAN_ANNOTATION_DIRNAME, site_id, bids_folder, session_id, HUMAN_ANNOTATION_SUFFIX))
        algo_reader = open_reader(annotation_path(
            annotation_root, ALGO_ANNOTATION_DIRNAME, site_id, bids_folder, session_id, ALGO_ANNOTATION_SUFFIX))
        try:
            for channel in silver_channels:
                if channel in physiological:
                    plan = plans.get(channel)
                    if plan is None:
                        prepared.append({"label": channel, "signal": np.zeros(0),
                                         "fs": float(physiological[channel]["target_freq_hz"]),
                                         "dimension": "uV"})
                        logger.info("%s: channel %s missing, writing flat", bids_folder, channel)
                        continue
                    if plan["kind"] == "reconstruct":
                        logger.info("%s: reconstructed %s from %s", bids_folder, channel, plan["labels"])
                    else:
                        logger.info("%s: selected %s for %s", bids_folder, plan["labels"][0], channel)
                    prepared.append(materialize_physiological(channel, physiological[channel], plan, row, reader, index))
                else:
                    prepared.append(annotation_channel(channel, row, human_reader, algo_reader, truth_table))
        finally:
            if human_reader is not None:
                human_reader.close()
            if algo_reader is not None:
                algo_reader.close()
    finally:
        reader.close()
    return prepared, None


def write_silver_demographics(demographics_path, output_root, included_keys):
    demographics = pd.read_csv(demographics_path, dtype=str)
    keep = demographics.apply(
        lambda row: (str(row["BidsFolder"]), str(row["SessionID"])) in included_keys, axis=1)
    filtered = demographics[keep]
    output_path = Path(output_root) / Path(demographics_path).name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)
    return {
        "demographics_total": len(demographics),
        "demographics_kept": int(keep.sum()),
        "demographics_dropped": int((~keep).sum()),
        "demographics_output": str(output_path),
    }


def process_bronze_to_silver(metadata_path, signals_root, output_root, truth_table_path):
    truth_table = load_truth_table(truth_table_path)
    metadata = pd.read_csv(metadata_path, dtype=str)
    annotation_root = Path(signals_root).parent
    demographics_path = annotation_root / DEMOGRAPHICS_FILENAME
    output_root = Path(output_root)
    processed = 0
    skipped = []
    included_keys = set()
    for _, row in metadata.iterrows():
        bids_folder = str(row["BidsFolder"])
        session_id = str(row["SessionID"])
        try:
            prepared, reason = prepare_patient(row, signals_root, annotation_root, truth_table)
        except Exception as error:
            logger.exception("%s: unexpected failure", bids_folder)
            skipped.append({"patient": bids_folder, "reason": f"error: {error}"})
            continue
        if prepared is None:
            logger.info("%s: skipped (%s)", bids_folder, reason)
            skipped.append({"patient": bids_folder, "reason": reason})
            continue
        output_path = physiological_path(output_root, str(row["SiteID"]), bids_folder, session_id)
        try:
            write_silver_edf(output_path, prepared)
        except Exception as error:
            logger.exception("%s: failed writing silver EDF", bids_folder)
            skipped.append({"patient": bids_folder, "reason": f"write error: {error}"})
            continue
        processed += 1
        included_keys.add((bids_folder, session_id))
        logger.info("%s: written %s", bids_folder, output_path)
    summary = {
        "total": len(metadata),
        "processed": processed,
        "skipped": len(skipped),
        "skipped_details": skipped,
    }
    if demographics_path.exists():
        demographics_summary = write_silver_demographics(demographics_path, output_root, included_keys)
        summary.update(demographics_summary)
        logger.info("demographics: kept %s, dropped %s -> %s",
                    demographics_summary["demographics_kept"],
                    demographics_summary["demographics_dropped"],
                    demographics_summary["demographics_output"])
    else:
        logger.warning("demographics file not found at %s, skipping", demographics_path)
    return summary

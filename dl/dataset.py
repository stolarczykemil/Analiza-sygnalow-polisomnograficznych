from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dl.config import (
    ALLOWED_SPLITS,
    AROUSAL_EVENT_CODE,
    CHANNELS,
    EVENT_FILES,
    GOLD_DL_DIR,
    GROUP_SLICES,
    LIMB_NO_EVENT_CODE,
    PASSTHROUGH_CHANNELS,
    PATIENTS_PARQUET,
    RESP_CLASSES,
    RESP_CLASS_CODES,
    ROBUST_CHANNELS,
    SIGNAL_GROUPS,
    SPO2_CENTER,
    SPO2_CHANNEL,
    SPO2_FRACTION_MAX,
    SPO2_PERCENT_FACTOR,
    SPO2_SCALE,
    STAGE_CODE_TO_INDEX,
    STAGE_IGNORE_INDEX,
    STAGE_UNSCORED_CODES,
    STANDARD_CHANNELS,
)
from dl.quality import load_channel_thresholds, quality_mask

DROP_REASONS = ["no_channels", "no_labels", "both"]


def scale_statistic(channel):
    if channel in STANDARD_CHANNELS:
        return "mean", "std"
    return "median", "iqr"


def channel_present(row, channel):
    mask_column = f"ch_mask_{channel}"
    present = bool(row[mask_column]) if mask_column in row.index else True
    if channel not in ROBUST_CHANNELS and channel not in STANDARD_CHANNELS:
        return present
    _, scale_name = scale_statistic(channel)
    scale = row[f"{scale_name}_{channel}"]
    return present and bool(np.isfinite(scale)) and scale > 0


def spo2_factor(spo2_signal):
    if np.abs(spo2_signal).max() <= SPO2_FRACTION_MAX:
        return SPO2_PERCENT_FACTOR
    return 1.0


def patient_normalization(row, spo2_signal):
    center = np.zeros(len(CHANNELS), dtype=np.float32)
    scale = np.ones(len(CHANNELS), dtype=np.float32)
    present = np.array([channel_present(row, channel) for channel in CHANNELS])
    for position, channel in enumerate(CHANNELS):
        if not present[position]:
            continue
        if channel in ROBUST_CHANNELS or channel in STANDARD_CHANNELS:
            center_name, scale_name = scale_statistic(channel)
            center[position] = row[f"{center_name}_{channel}"]
            scale[position] = row[f"{scale_name}_{channel}"]
        elif channel == SPO2_CHANNEL:
            factor = spo2_factor(spo2_signal)
            center[position] = SPO2_CENTER / factor
            scale[position] = SPO2_SCALE / factor
        elif channel not in PASSTHROUGH_CHANNELS:
            raise ValueError(f"No normalization rule for channel {channel}")
    return center, scale, present


def stage_targets(stage_codes):
    lookup = np.full(256, STAGE_IGNORE_INDEX, dtype=np.int64)
    for code, index in STAGE_CODE_TO_INDEX.items():
        lookup[code] = index
    stage = lookup[stage_codes.astype(np.int64) & 0xFF]
    return stage, stage != STAGE_IGNORE_INDEX


def event_targets(arousal, limb, resp):
    arousal_fraction = (arousal == AROUSAL_EVENT_CODE).mean(axis=1)
    limb_fraction = (limb != LIMB_NO_EVENT_CODE).mean(axis=1)
    resp_fractions = np.stack(
        [np.isin(resp, list(RESP_CLASS_CODES[name])).mean(axis=1) for name in RESP_CLASSES], axis=1
    )
    return (
        arousal_fraction.astype(np.float32),
        limb_fraction.astype(np.float32),
        resp_fractions.astype(np.float32),
    )


def patient_event_flags(stage_codes, events):
    if not np.isin(stage_codes, list(STAGE_UNSCORED_CODES)).all():
        return {name: True for name in EVENT_FILES}
    return {name: bool((events[name] != 0).any()) for name in EVENT_FILES}


def build_patient_record(row, channel_thresholds, gold_dir=GOLD_DL_DIR):
    patient_dir = gold_dir / row["patient_id"]
    stage_codes = np.load(patient_dir / "stage_expert.npy")
    events = {name: np.load(patient_dir / f"{name}.npy") for name in EVENT_FILES}
    spo2_signal = np.asarray(np.load(patient_dir / "x_25.npy", mmap_mode="r"), dtype=np.float32)
    quality = np.load(patient_dir / "quality.npy")

    center, scale, present = patient_normalization(row, spo2_signal)
    mask = present[None, :] & quality_mask(quality, channel_thresholds)
    stage, stage_valid = stage_targets(stage_codes)
    arousal, limb, resp = event_targets(events["arousal"], events["limb"], events["resp"])
    event_flags = patient_event_flags(stage_codes, events)

    no_channels = ~mask.any(axis=1)
    no_labels = ~stage_valid & (not any(event_flags.values()))
    keep = ~(no_channels | no_labels)
    drops = {
        "no_channels": int((no_channels & ~no_labels).sum()),
        "no_labels": int((no_labels & ~no_channels).sum()),
        "both": int((no_channels & no_labels).sum()),
    }
    return {
        "patient_id": row["patient_id"],
        "center": center,
        "scale": scale,
        "present": present,
        "mask": mask,
        "stage": stage,
        "stage_valid": stage_valid,
        "arousal": arousal,
        "limb": limb,
        "resp": resp,
        "event_flags": event_flags,
        "keep": np.flatnonzero(keep),
        "drops": drops,
    }


class PSGEpochDataset(Dataset):
    def __init__(self, split, patients_parquet=PATIENTS_PARQUET, gold_dir=GOLD_DL_DIR):
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {ALLOWED_SPLITS}, got {split!r}")
        self.split = split
        self.gold_dir = Path(gold_dir)
        patients = pd.read_parquet(patients_parquet)
        patients = patients[patients["split"] == split].reset_index(drop=True)
        channel_thresholds = load_channel_thresholds()
        self.records = [build_patient_record(row, channel_thresholds, self.gold_dir)
                        for _, row in patients.iterrows()]
        self.patient_ids = [record["patient_id"] for record in self.records]
        self.patient_positions = np.concatenate(
            [np.full(len(record["keep"]), position, dtype=np.int64) for position, record in enumerate(self.records)]
        )
        self.epoch_indices = np.concatenate([record["keep"] for record in self.records])
        self.drop_counts = {reason: sum(record["drops"][reason] for record in self.records) for reason in DROP_REASONS}
        self._arrays = {}

    def __len__(self):
        return len(self.epoch_indices)

    def patient_arrays(self, position):
        if position not in self._arrays:
            patient_dir = self.gold_dir / self.patient_ids[position]
            self._arrays[position] = {
                group_name: np.load(patient_dir / f"{group_name}.npy", mmap_mode="r")
                for group_name in SIGNAL_GROUPS
            }
        return self._arrays[position]

    def normalized_group(self, record, arrays, group_name, epoch):
        channels = GROUP_SLICES[group_name]
        signal = np.asarray(arrays[group_name][epoch], dtype=np.float32)
        normalized = (signal - record["center"][channels, None]) / record["scale"][channels, None]
        return np.where(record["mask"][epoch, channels, None], normalized, np.float32(0.0))

    def __getitem__(self, index):
        position = int(self.patient_positions[index])
        epoch = int(self.epoch_indices[index])
        record = self.records[position]
        arrays = self.patient_arrays(position)
        item = {
            group_name: torch.from_numpy(self.normalized_group(record, arrays, group_name, epoch))
            for group_name in SIGNAL_GROUPS
        }
        item["ch_mask"] = torch.from_numpy(record["mask"][epoch].astype(np.float32))
        item["stage"] = torch.tensor(record["stage"][epoch], dtype=torch.int64)
        item["stage_valid"] = torch.tensor(float(record["stage_valid"][epoch]), dtype=torch.float32)
        item["arousal"] = torch.tensor(record["arousal"][epoch], dtype=torch.float32)
        item["limb"] = torch.tensor(record["limb"][epoch], dtype=torch.float32)
        item["resp"] = torch.from_numpy(record["resp"][epoch])
        for name in EVENT_FILES:
            item[f"{name}_valid"] = torch.tensor(float(record["event_flags"][name]), dtype=torch.float32)
        return item

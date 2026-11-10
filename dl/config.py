from pathlib import Path

REPO = Path("/home/emil/studia/inzynierka")
GOLD_DL_DIR = REPO / "data/gold_dl"
PATIENTS_PARQUET = GOLD_DL_DIR / "patients.parquet"
QUALITY_THRESHOLDS_JSON = REPO / "references/quality_thresholds.json"

ALLOWED_SPLITS = ("train", "val")

EPOCH_SECONDS = 30

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
CHANNELS = [channel for _, channels in SIGNAL_GROUPS.values() for channel in channels]
CHANNEL_INDEX = {channel: position for position, channel in enumerate(CHANNELS)}


def group_slices(signal_groups):
    slices = {}
    start = 0
    for group_name, (_, channels) in signal_groups.items():
        slices[group_name] = slice(start, start + len(channels))
        start += len(channels)
    return slices


GROUP_SLICES = group_slices(SIGNAL_GROUPS)

EMG_CHANNELS = ["chin_emg_chin1_chin2", "leg_emg_lat", "leg_emg_rat"]
ROBUST_CHANNELS = [channel for channel in SIGNAL_GROUPS["x_200"][1] if channel not in EMG_CHANNELS]
STANDARD_CHANNELS = EMG_CHANNELS
PASSTHROUGH_CHANNELS = ["resp_chest", "resp_abd", "resp_ptaf"]
SPO2_CHANNEL = "resp_spo2"
SPO2_CENTER = 90.0
SPO2_SCALE = 10.0
SPO2_FRACTION_MAX = 1.5
SPO2_PERCENT_FACTOR = 100.0

QUALITY_METRICS = [
    "variance", "flat_max_run", "frac_beyond_kmad",
    "amp_oor_ratio", "physio_violation_ratio", "desat_slope_max",
]
QUALITY_METRIC_INDEX = {metric: position for position, metric in enumerate(QUALITY_METRICS)}
SPO2_PHYSIO_VIOLATION_MAX = 0.0
QUALITY_CRITERIA = {
    "flat_var_min": ("variance", "less"),
    "flat_run_min_samples": ("flat_max_run", "greater_equal"),
    "amp_oor_ratio_max": ("amp_oor_ratio", "greater"),
    "frac_beyond_kmad_max": ("frac_beyond_kmad", "greater"),
    "desat_slope_max_pct_per_s": ("desat_slope_max", "greater"),
}
PHYSIO_CRITERION = ("physio_min", "physio_violation_ratio", SPO2_PHYSIO_VIOLATION_MAX)

STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_CODE_TO_NAME = {5: "W", 3: "N1", 2: "N2", 1: "N3", 4: "REM"}
STAGE_CODE_TO_INDEX = {code: STAGES.index(name) for code, name in STAGE_CODE_TO_NAME.items()}
STAGE_UNSCORED_CODES = {0, 9}
STAGE_IGNORE_INDEX = -100

AROUSAL_SUBWINDOWS = 60
LIMB_SUBWINDOWS = 30
RESP_SUBWINDOWS = 30
AROUSAL_EVENT_CODE = 1
LIMB_NO_EVENT_CODE = 0
RESP_CLASSES = ["OA", "CA", "HY"]
RESP_CLASS_CODES = {
    "OA": {1, 3, 8},
    "CA": {2},
    "HY": {4, 5, 6, 9},
}
RESP_NO_EVENT_CODES = {0, 7, 10, 11}
EVENT_FILES = ["arousal", "resp", "limb"]

BRANCH_CONFIG = {
    "x_200": {"stem_kernel": 25, "stem_stride": 2, "block_kernel": 7, "channels": [32, 32, 48, 64, 64]},
    "x_100": {"stem_kernel": 25, "stem_stride": 2, "block_kernel": 9, "channels": [16, 16, 32, 32]},
    "x_25": {"stem_kernel": 7, "stem_stride": 2, "block_kernel": 7, "channels": [8, 8, 16, 16]},
}
BLOCK_STRIDE = 2
EMBEDDING_DIM = 128
EMBEDDING_DROPOUT = 0.3
CHANNEL_DROPOUT = 0.2
EEG_PREFIX = "eeg_"
EEG_CHANNEL_INDICES = [CHANNEL_INDEX[channel] for channel in CHANNELS if channel.startswith(EEG_PREFIX)]
HEAD_OUTPUTS = {"stage": len(STAGES), "arousal": 1, "limb": 1, "resp": len(RESP_CLASSES)}
SCALAR_HEADS = ["arousal", "limb"]

MODELS_DIR = REPO / "models"
LOSS_WEIGHTS = {"stage": 1.0, "arousal": 0.0, "limb": 0.0, "resp": 0.0}
BATCH_SIZE = 256
VAL_BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-2
MAX_PASSES = 10
EVALS_PER_PASS = 4
PATIENCE = 8
SEED = 42
NUM_WORKERS = 2

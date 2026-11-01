import numpy as np
import pyedflib
from tqdm.auto import tqdm
from dl_silver_to_gold import GOLD_DL_DIR, SIGNAL_GROUPS, load_export_context, edf_path, EPOCH_SECONDS

EXPECTED = {
    "x_200": (12, 6000), "x_100": (3, 3000), "x_25": (1, 750),
    "stage_expert": (), "arousal": (60,), "limb": (30,), "resp": (30,),
    "caisr_prob": (7,), "quality": (16, 6),
}

site_session, channel_index = load_export_context()

problems = []
for patient_id in tqdm(site_session, desc="pacjenci"):
    arrays = {name: np.load(GOLD_DL_DIR / patient_id / f"{name}.npy", mmap_mode="r") for name in EXPECTED}
    n = arrays["x_200"].shape[0]
    for name, tail in EXPECTED.items():
        if arrays[name].shape != (n, *tail):
            problems.append((patient_id, name, arrays[name].shape))
print("problemy:", problems if problems else "brak")

patient_id = "sub-I0002150001401"
epoch = 100
reader = pyedflib.EdfReader(str(edf_path(patient_id, site_session)))
for group, (fs, channels) in SIGNAL_GROUPS.items():
    gold = np.load(GOLD_DL_DIR / patient_id / f"{group}.npy", mmap_mode="r")[epoch]
    for position, channel in enumerate(channels):
        start = epoch * fs * EPOCH_SECONDS
        silver = reader.readSignal(channel_index[channel], start, fs * EPOCH_SECONDS)
        error = np.abs(gold[position].astype(np.float64) - silver).max()
        print(f"{channel:22s} max |gold - silver| = {error:.4f}")
reader.close()
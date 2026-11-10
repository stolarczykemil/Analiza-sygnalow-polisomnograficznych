import json
from pathlib import Path

import numpy as np

from dl.config import (
    CHANNELS,
    PHYSIO_CRITERION,
    QUALITY_CRITERIA,
    QUALITY_METRIC_INDEX,
    QUALITY_THRESHOLDS_JSON,
)

COMPARISONS = {
    "less": np.less,
    "greater": np.greater,
    "greater_equal": np.greater_equal,
}


def load_channel_thresholds(path=QUALITY_THRESHOLDS_JSON):
    config = json.loads(Path(path).read_text())
    classes = config["channel_class_thresholds"]
    return [classes[config["channel_to_class"][channel]] for channel in CHANNELS]


def channel_quality_good(metrics, thresholds):
    bad = ~np.isfinite(metrics[:, QUALITY_METRIC_INDEX["variance"]])
    for key, (metric, comparison) in QUALITY_CRITERIA.items():
        if key in thresholds:
            bad |= COMPARISONS[comparison](metrics[:, QUALITY_METRIC_INDEX[metric]], thresholds[key])
    threshold_key, metric, maximum = PHYSIO_CRITERION
    if threshold_key in thresholds:
        bad |= metrics[:, QUALITY_METRIC_INDEX[metric]] > maximum
    return ~bad


def quality_mask(quality, channel_thresholds):
    columns = [channel_quality_good(quality[:, position, :], thresholds)
               for position, thresholds in enumerate(channel_thresholds)]
    return np.stack(columns, axis=1)

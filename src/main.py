from pathlib import Path
from preliminary_pipeline import preliminary_pipeline
from bronze_to_silver import process_bronze_to_silver
from silver_to_gold import build_gold

REPO = Path("/home/emil/studia/inzynierka")
BRONZE_PATH = Path("/home/emil/.cache/kagglehub/datasets/physionet/physionetchallenge2026data/versions/2/training_set")
SIGNALS_ROOT = BRONZE_PATH / "physiological_data"
METADATA_PATH = REPO / "data/bronze/metadata.csv"
MAPPING_PATH = REPO / "references/channel_table.csv"
SILVER_PATH = REPO / "data/silver"
TRUTH_TABLE_PATH = REPO / "references/truth_table.json"


def run():
    df = preliminary_pipeline(BRONZE_PATH, MAPPING_PATH)
    df.to_csv(METADATA_PATH, index=False)
    summary = process_bronze_to_silver(METADATA_PATH, SIGNALS_ROOT, SILVER_PATH, TRUTH_TABLE_PATH)
    build_gold()
    return summary

if __name__ == '__main__':
    print(run())

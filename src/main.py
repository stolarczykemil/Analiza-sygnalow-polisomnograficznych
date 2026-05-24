from pathlib import Path
from src.preliminary_pipeline import preliminary_pipeline

DATA_PATH = Path("/home/emil/.cache/kagglehub/datasets/physionet/physionetchallenge2026data/versions/2/training_set")
OUTPUT_PATH = Path("/home/emil/studia/inzynierka/data/silver/metadata.csv")
MAPPING_PATH =  Path("/references/channel_table.csv")

def run(data_path, mapping_path):
    df = preliminary_pipeline(data_path, mapping_path)
    df.to_csv(OUTPUT_PATH, index=False)

if __name__ == '__main__':
    run(DATA_PATH, MAPPING_PATH)
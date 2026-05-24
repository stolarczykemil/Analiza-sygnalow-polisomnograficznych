import pyedflib
import pandas as pd
import numpy as np


def save_metadata(df, filepath, sample_id):
    with pyedflib.EdfReader(str(filepath)) as reader:
        headers = reader.getSignalHeaders()
        for h in headers:
            clean_chanel_name = h['label'].strip().replace(" ", "_")
            label = f"freq_{clean_chanel_name}"
            fs = h['sample_frequency']
            dim = h['dimension']
            if '_caisr' in str(filepath) or '_expert' in str(filepath):
                combined_value = fs
            else:
                combined_value = f"{fs}|{dim}"
            if sample_id in df.index:
                df.loc[sample_id, label] = combined_value


def preliminary_pipeline(data_path, mapping_path):
    df = pd.read_csv(data_path / "demographics.csv")
    original_headers = df.columns
    df.set_index("BidsFolder", inplace=True)
    for path in data_path.rglob("*.edf"):
        if path.is_file():
            sample_id = path.stem.split("_")[0]
            save_metadata(df, path, sample_id)

    channel_table = pd.read_csv(mapping_path)
    df_cols_lower = {c.lower(): c for c in df.columns}
    mapping = {}

    for _, row in channel_table.iterrows():
        if pd.isna(row['Channel_Names']):
            continue
        synonyms = [s.strip() for s in str(row['Channel_Names']).split(';')]
        std_name = synonyms[0]
        if std_name.endswith('+'):
            std_name = std_name[:-1] + '_PLUS'
        elif std_name.endswith('-'):
            std_name = std_name[:-1] + '_MINUS'
        std_name = std_name.replace('-', '_').replace(' ', '_')
        category = str(row['Category']).strip().replace(' ', '_')
        new_col_name = f"{category}_{std_name}"
        mapping[new_col_name] = synonyms

    new_columns = {}

    for new_col, aliases in mapping.items():
        actual_cols = []
        for alias in aliases:
            alias_clean = alias.strip().replace(" ", "_")
            expected_col = f"freq_{alias_clean}".lower()
            if expected_col in df_cols_lower:
                actual_cols.append(df_cols_lower[expected_col])

        actual_cols = list(set(actual_cols))

        if actual_cols:
            valid_rows = df[actual_cols].notna().any(axis=1)
            source_series = pd.Series(np.nan, index=df.index, dtype='object')
            freq_series = pd.Series(np.nan, index=df.index, dtype='object')
            dim_series = pd.Series(np.nan, index=df.index, dtype='object')

            if valid_rows.any():
                subset = df.loc[valid_rows, actual_cols]

                source_series.loc[valid_rows] = subset.apply(
                    lambda row: ';'.join(row.dropna().index.str.replace('freq_', '', regex=False)),
                    axis=1
                )

                combined_vals = subset.apply(
                    lambda row: ';'.join(row.dropna().astype(str)),
                    axis=1
                )

                freq_series.loc[valid_rows] = combined_vals.apply(
                    lambda x: ';'.join([val.split('|')[0] for val in x.split(';')])
                )

                dim_series.loc[valid_rows] = combined_vals.apply(
                    lambda x: ';'.join([val.split('|')[1] for val in x.split(';')])
                )

            new_columns[f"{new_col}_SOURCE"] = source_series
            new_columns[f"{new_col}_FREQ"] = freq_series
            new_columns[f"{new_col}_DIM"] = dim_series

    cols_to_drop = set()
    for col in df.columns:
        if col not in original_headers and '_caisr' not in col and '_expert' not in col:
            cols_to_drop.add(col)

    df.drop(columns=list(cols_to_drop), inplace=True)
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    df.reset_index(inplace=True)

    return df
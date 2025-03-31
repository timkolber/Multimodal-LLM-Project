import csv
import gc
import logging
import os

import pandas as pd

from data_processing import (
    data_split,
    get_notes,
    process_dataframes,
    processed_data_exists,
    prompt_csv_files_exist,
)
from load_data import load_raw_dataframe
from prompts import build_prompt_dict


def main():
    if prompt_csv_files_exist():
        prompts_df = pd.read_csv("./data/val_prompts.csv")

    elif processed_data_exists() and not prompt_csv_files_exist():
        for split in ["train", "val", "test"]:
            df = pd.read_csv(f"./data/{split}_data.csv")
            prompt_df = pd.DataFrame()
            prompt_df["prompt"] = df.apply(lambda row: build_prompt_dict(row), axis=1)
            prompt_df["prompt"] = prompt_df["prompt"].astype(str)
            prompt_df["label"] = df["TEXT"].apply(lambda x: x[:500])
            prompt_df.to_csv(
                f"./data/{split}_prompts.csv", index=False, quoting=csv.QUOTE_ALL
            )
            del df
            del prompt_df
            gc.collect()

    else:
        patients_df = load_raw_dataframe("PATIENTS")
        train_patients, val_patients, test_patients = data_split(
            patients_df=patients_df, train_pct=0.7, val_pct=0.15, test_pct=0.15
        )
        split_patients = {
            "train": train_patients,
            "val": val_patients,
            "test": test_patients,
        }
        for split in split_patients.keys():
            notes_df = load_raw_dataframe(
                "NOTEEVENTS", patient_ids=split_patients[split], full_table=False
            )
            split_notes_df = get_notes(notes_df, split_patients[split])
            df = process_dataframes(split_notes_df, split_patients[split])

            del notes_df
            gc.collect()

            prompt_df = pd.DataFrame()
            prompt_df["prompt"] = df.apply(lambda row: build_prompt_dict(row), axis=1)
            prompt_df["label"] = df["text"].apply(lambda x: x[:500])
            prompt_df.to_csv(f"./data/{split}_data.csv", index=False)

            del df
            del prompt_df
            gc.collect()


if __name__ == "__main__":
    main()

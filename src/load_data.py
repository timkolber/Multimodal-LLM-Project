import gc
import os
from typing import Optional

import pandas as pd

use_cols_dict = {
    "PATIENTS": ["SUBJECT_ID", "DOB", "GENDER"],
    "PRESCRIPTIONS": ["SUBJECT_ID", "DRUG", "STARTDATE"],
    "LABEVENTS": [
        "SUBJECT_ID",
        "CHARTTIME",
        "VALUE",
        "VALUEUOM",
        "FLAG",
        "ITEMID",
    ],
    "NOTEEVENTS": [
        "SUBJECT_ID",
        "CHARTTIME",
        "CHARTDATE",
        "TEXT",
        "CATEGORY",
        "ROW_ID",
    ],
    "D_LABITEMS": ["ITEMID", "LABEL"],
}


def load_raw_dataframe(
    table_name: str, patient_ids: Optional[pd.DataFrame] = None, full_table=True
) -> pd.DataFrame:
    """
    Load the dataframe from the csv file.
    """
    data_folder = "./data/mimic-iii-clinical-database-1.4"
    file_path = os.path.join(data_folder, f"{table_name}.csv")
    if full_table:
        df = pd.read_csv(file_path, usecols=use_cols_dict[table_name])
    else:
        df = pd.read_csv(file_path, usecols=use_cols_dict[table_name])
        df.drop(columns=["TEXT", "CATEGORY"], inplace=True)

    if table_name == "LABEVENTS":
        labitems_df = load_raw_dataframe("D_LABITEMS")
        df = df.merge(labitems_df, on="ITEMID", how="left")
        df.drop(columns=["ITEMID"], inplace=True)
        del labitems_df
        gc.collect()

    if patient_ids is not None:
        df = df.loc[df["SUBJECT_ID"].isin(patient_ids)]
    return df

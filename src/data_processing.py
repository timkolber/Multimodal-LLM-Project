import gc
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from load_data import load_raw_dataframe


def data_split(
    patients_df: pd.DataFrame,
    train_pct: float,
    val_pct: float,
    test_pct: float,
    random_state: int = 42,
):
    """
    Split the data into training, validation, and test sets.
    """
    patient_ids = patients_df["SUBJECT_ID"].unique()
    train_patient_ids, test_patients_ids = train_test_split(
        patient_ids, test_size=test_pct, random_state=random_state
    )
    train_patient_ids, val_patient_ids = train_test_split(
        train_patient_ids,
        test_size=val_pct / (train_pct + val_pct),
        random_state=random_state,
    )
    return train_patient_ids, val_patient_ids, test_patients_ids


def process_dataframes(
    note_dataframe: pd.DataFrame, patient_ids: pd.DataFrame
) -> pd.DataFrame:
    """
    Process the dataframes to get the desired columns in one dataframe.
    """
    convert_columns_to_datetime(note_dataframe)
    note_dataframe.rename(columns={"CHARTTIME": "NOTETIME"}, inplace=True)
    note_dataframe.rename(columns={"CHARTDATE": "NOTEDATE"}, inplace=True)
    patients_df = load_raw_dataframe("PATIENTS")
    merged_df = note_dataframe.merge(
        patients_df,
        on="SUBJECT_ID",
        how="left",
    )
    del patients_df
    gc.collect()

    prescriptions_df = load_raw_dataframe("PRESCRIPTIONS", patient_ids)
    filtered_prescriptions_df = merge_and_filter(
        dataframe=prescriptions_df,
        note_df=note_dataframe,
        time_col="STARTDATE",
        value_cols=["DRUG"],
        n_chunks=10,
    )
    del prescriptions_df
    gc.collect()

    labevents_df = load_raw_dataframe("LABEVENTS", patient_ids)
    labevents_df["FLAG"].fillna("", inplace=True)
    labevents_df["LabList"] = (
        labevents_df["LABEL"].astype(str)
        + ","
        + labevents_df["VALUE"].astype(str)
        + ","
        + labevents_df["VALUEUOM"].astype(str)
        + labevents_df["FLAG"].apply(lambda x: f",{x}" if x else "")
    )
    labevents_df.drop(columns=["LABEL", "VALUE", "VALUEUOM", "FLAG"], inplace=True)
    filtered_labevents_df = merge_and_filter(
        dataframe=labevents_df,
        note_df=note_dataframe,
        time_col="CHARTTIME",
        value_cols=["LabList"],
        n_chunks=20,
    )
    del labevents_df
    gc.collect()

    past_notes_df = load_raw_dataframe("NOTEEVENTS", patient_ids)
    past_notes_df.drop(columns=["ROW_ID", "CATEGORY"], inplace=True)
    filtered_past_notes_df = merge_and_filter(
        dataframe=past_notes_df,
        note_df=note_dataframe,
        time_col="CHARTTIME",
        value_cols=["TEXT"],
        n_chunks=20,
    )
    filtered_past_notes_df = filtered_past_notes_df.rename(
        columns={"TEXT": "PAST_NOTES"}
    )
    filtered_past_notes_df.drop(columns=["CHARTTIME", "CHARTDATE"], inplace=True)

    del past_notes_df
    del note_dataframe
    gc.collect()

    merged_df = merged_df.merge(
        filtered_prescriptions_df,
        on=["ROW_ID", "NOTEDATE", "NOTETIME", "SUBJECT_ID"],
        how="left",
    )
    merged_df = merged_df.merge(
        filtered_labevents_df,
        on=["ROW_ID", "NOTEDATE", "NOTETIME", "SUBJECT_ID"],
        how="left",
    )

    merged_df["AGE"] = merged_df.apply(
        lambda x: get_age(
            x["DOB"], x["NOTETIME"] if not pd.isna(x["NOTETIME"]) else x["NOTEDATE"]
        ),
        axis=1,
    )

    merged_df = merged_df.merge(
        filtered_past_notes_df,
        on=["ROW_ID", "NOTEDATE", "NOTETIME", "SUBJECT_ID"],
        how="left",
    )

    merged_df.drop(
        columns=[
            "CHARTTIME",
            "NOTETIME",
            "NOTEDATE",
            "STARTDATE",
            "DOB",
        ],
        inplace=True,
    )

    complete_note_df = load_raw_dataframe("NOTEEVENTS")
    merged_df = merged_df.merge(
        complete_note_df,
        on=["ROW_ID", "SUBJECT_ID"],
        how="inner",
    )

    merged_df.drop(
        columns=["CHARTTIME", "SUBJECT_ID", "CHARTDATE"],
        inplace=True,
    )

    del complete_note_df
    del filtered_prescriptions_df
    del filtered_labevents_df
    gc.collect()

    return merged_df


def merge_and_filter(
    dataframe: pd.DataFrame,
    note_df: pd.DataFrame,
    time_col: str,
    value_cols: list[str],
    n_chunks: int = 10,
):
    """
    Merge the dataframe with the labevents dataframe and filter the data.
    """
    dataframe[time_col] = pd.to_datetime(dataframe[time_col], errors="coerce")
    result_list = []
    np_chunks = np.array_split(dataframe, n_chunks)

    for chunk in np_chunks:
        chunk_df = pd.DataFrame(chunk)
        merged_chunk_df = chunk_df.merge(note_df, on="SUBJECT_ID", how="inner")
        filtered_chunk_df = filter_dataframe(merged_chunk_df, time_col, value_cols)
        del merged_chunk_df
        gc.collect()
        result_list.append(filtered_chunk_df)

    result_df = pd.concat(result_list, ignore_index=True)
    del result_list
    gc.collect()

    return result_df


def prompt_csv_files_exist():
    """
    Check if the prompt csv exists.
    """
    for split in ["train", "val", "test"]:
        if f"{split}_prompts.csv" not in os.listdir("./data"):
            return False
    return True


def processed_data_exists():
    """
    Check if the processed data exists.
    """
    for split in ["train", "val", "test"]:
        if f"{split}_data.csv" not in os.listdir("./data"):
            return False
    return True


def filter_dataframe(
    dataframe: pd.DataFrame,
    time_column: str,
    value_cols: list[str],
    time_window: int = 24,
):
    """
    Filter the dataframe based on the time window.
    """
    columns = dataframe.columns
    if time_column not in dataframe.columns:
        return dataframe
    try:
        if time_column == "NOTETIME":
            dataframe[time_column] = dataframe[time_column].fillna(
                pd.to_datetime(dataframe["NOTEDATE"])
            )
        dataframe[time_column] = pd.to_datetime(dataframe[time_column])
    except ValueError:
        return dataframe
    dataframe["REFERENCE_TIME"] = dataframe["NOTETIME"].fillna(dataframe["NOTEDATE"])
    filtered_df = dataframe[
        (
            dataframe["REFERENCE_TIME"] - timedelta(hours=time_window)  # type: ignore
            <= dataframe[time_column]
        )
        & (dataframe[time_column] <= dataframe["REFERENCE_TIME"])
    ]
    # filtered_df = filtered_df.sort_values(by=[time_column], ascending=False)
    # filtered_df = filtered_df.groupby("ROW_ID").head(10)
    filtered_df = filtered_df.groupby("ROW_ID").agg(
        {
            **{col: list for col in value_cols},  # Aggregate selected columns as lists
            **{col: "first" for col in columns if col not in value_cols},
        }
    )
    return filtered_df


def convert_columns_to_datetime(dataframe: pd.DataFrame):
    for column in dataframe.columns:
        if "TIME" in column or "DATE" in column:
            dataframe[column] = pd.to_datetime(dataframe[column])


def get_age(dob, note_time: datetime) -> int | float:
    """
    Get the age of a patient.
    """
    if pd.isna(dob) or pd.isna(note_time):
        return -1
    dob = pd.to_datetime(dob)
    dob = dob.to_pydatetime()
    return round(((note_time - dob).days // 365))


def get_notes(notes_df: pd.DataFrame, patient_ids: list):
    """
    Get the notes for the specified patients.
    """
    return notes_df.loc[notes_df["SUBJECT_ID"].isin(patient_ids)]

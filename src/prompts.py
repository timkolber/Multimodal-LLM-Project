import json
import logging
from math import log
from typing import Dict

import pandas as pd


def build_prompt_dict(df_row: pd.Series) -> str:
    """
    Build and return a json string from the dataframe row.
    """
    prompt_dict = {
        "Hint": df_row["TEXT"][:10],
        "NoteType": df_row["CATEGORY"],
        "Gender": df_row["GENDER"],
        "Age": df_row["AGE"],
        "Medication": df_row["DRUG"],
        "LabResults": df_row["LabList"],
        "PastNotes": df_row["PAST_NOTES"],
    }
    json_string = json.dumps(prompt_dict)
    json_string = json_string.replace('""', '"')
    return json_string[:500]

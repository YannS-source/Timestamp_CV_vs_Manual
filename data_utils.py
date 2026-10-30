"""Loader for the timestamp validation dataset."""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "Original SST data with collapsed procedures.xlsx"

# Full date-times for each case. ground_truth = video annotation,
# rs = computer vision (RoomState), emr = manual entry in the EHR.
TIME_COLS = ["ground_truth_wheels_in", "ground_truth_wheels_out",
             "rs_wheels_in", "rs_wheels_out",
             "emr_wheels_in", "emr_wheels_out"]


def load_data() -> pd.DataFrame:
    """One row per case. Adds signed errors (timestamp minus ground truth,
    in seconds) and the clock hour used for the time-of-day analyses.

    Columns added:
        cv_err_in, cv_err_out    computer vision minus ground truth
        emr_err_in, emr_err_out  manual entry minus ground truth
        hour_in, hour_out        clock hour of the computer vision timestamp
    """
    df = pd.read_excel(DATA)
    for c in TIME_COLS:
        df[c] = pd.to_datetime(df[c], format="ISO8601")
    missing = df[TIME_COLS].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} case(s) with a missing timestamp")

    df["cv_err_in"] = (df["rs_wheels_in"] - df["ground_truth_wheels_in"]).dt.total_seconds()
    df["cv_err_out"] = (df["rs_wheels_out"] - df["ground_truth_wheels_out"]).dt.total_seconds()
    df["emr_err_in"] = (df["emr_wheels_in"] - df["ground_truth_wheels_in"]).dt.total_seconds()
    df["emr_err_out"] = (df["emr_wheels_out"] - df["ground_truth_wheels_out"]).dt.total_seconds()

    df["hour_in"] = df["rs_wheels_in"].dt.hour
    df["hour_out"] = df["rs_wheels_out"].dt.hour
    return df

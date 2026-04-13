"""
data_loader.py
--------------
Responsible for all data ingestion and preprocessing steps.
Functions:
    load_data()               — reads StudentsPerformance.csv
    feature_engineering(df)   — adds average_score, total_score, academic_status
    preprocess_for_clustering(df) — encodes categoricals, drops lunch, scales scores
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ── Load ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame | None:
    """
    Reads the CSV from the data/ folder.
    Cleans column names to lowercase_with_underscores.
    Returns None if the file is not found.

    Functions used:
        pd.read_csv()    — reads a comma-separated file into a DataFrame
        str.lower()      — converts headers to lowercase
        str.replace()    — swaps spaces/slashes with underscores
    """
    try:
        df = pd.read_csv("data/StudentsPerformance.csv")
    except FileNotFoundError:
        st.error("Dataset not found at data/StudentsPerformance.csv")
        return None

    df.columns = (df.columns
                  .str.lower()
                  .str.replace("/", "_", regex=False)
                  .str.replace(" ", "_", regex=False))
    return df


# ── Feature Engineering ───────────────────────────────────────────────
@st.cache_data
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived columns:
        average_score   — mean of the three subject scores
        total_score     — sum of the three subject scores
        academic_status — 'Pass' if average >= 50, else 'Fail'

    Functions used:
        df[cols].mean(axis=1) — row-wise mean across selected columns
        df[cols].sum(axis=1)  — row-wise sum
        np.where(condition, true_val, false_val) — vectorised if/else
    """
    df = df.copy()
    score_cols = ["math_score", "reading_score", "writing_score"]
    df["average_score"]   = df[score_cols].mean(axis=1).round(2)
    df["total_score"]     = df[score_cols].sum(axis=1)
    df["academic_status"] = np.where(df["average_score"] >= 50, "Pass", "Fail")
    return df


# ── Preprocessing for Clustering ─────────────────────────────────────
def preprocess_for_clustering(df: pd.DataFrame):
    """
    Prepares data for K-Means:
        1. Drops 'lunch' — school administrative field, not a learning behaviour.
        2. Label-encodes remaining categorical columns.
        3. Z-score normalises numeric score columns with StandardScaler.

    Functions used:
        LabelEncoder().fit_transform(col) — converts text → integer codes
        StandardScaler().fit_transform(X) — subtracts mean, divides by std dev
                                            so all columns are on the same scale
    Returns:
        ml_df   — processed DataFrame with encoded + scaled columns
        le_dict — dict of fitted LabelEncoders (one per categorical column)
        scaler  — fitted StandardScaler (reused for new student prediction)
    """
    ml_df = df.copy()

    # Drop lunch — it is a school admin field, not a student learning behaviour
    if "lunch" in ml_df.columns:
        ml_df.drop(columns=["lunch"], inplace=True)

    cat_cols = ["gender", "race_ethnicity", "parental_level_of_education",
                "test_preparation_course"]

    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        ml_df[col + "_encoded"] = le.fit_transform(ml_df[col])
        le_dict[col] = le

    # Z-score normalise the three main score columns
    num_cols = ["math_score", "reading_score", "writing_score"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(ml_df[num_cols])
    for i, col in enumerate(num_cols):
        ml_df[f"{col}_scaled"] = scaled[:, i]

    return ml_df, le_dict, scaler


# ── Preprocessed Preview ─────────────────────────────────────────────
def get_preprocessed_preview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a small, readable view of the dataset AFTER preprocessing.
    Encodes categoricals and computes scaled columns so judges can see
    what the data looks like before it enters K-Means.

    Rather than showing every scaled column, we keep the original scores
    alongside the newly derived features (average_score, academic_status)
    and the encoded categoricals — in a single clean table.
    """
    ml_df, _, _ = preprocess_for_clustering(df)

    # Columns we want in the preview (raw scores + new features + encoded cats)
    keep = [
        c for c in ml_df.columns
        if not c.endswith("_scaled")          # exclude scaled columns (too noisy)
    ]
    # Re-attach human-readable derived columns if present
    for col in ["average_score", "total_score", "academic_status"]:
        if col in df.columns and col not in keep:
            ml_df[col] = df[col].values
            keep.append(col)

    return ml_df[keep].head(20)

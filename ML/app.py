from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ------------------------------------------------------------
# Streamlit configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction | ML",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = {0: "Benign (B)", 1: "Malignant (M)"}


# ------------------------------------------------------------
# Dataset discovery
# Supports:
# 1) ML/project.py + ML/DataSet Breast Cancer/data.csv
# 2) ML/project.py + repository-root/DataSet Breast Cancer/data.csv
# 3) ML/project.py + ML/data.csv or repository-root/data.csv
# ------------------------------------------------------------
def locate_dataset():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent

    candidates = [
        script_dir / "DataSet Breast Cancer" / "data.csv",
        repo_dir / "DataSet Breast Cancer" / "data.csv",
        script_dir / "data.csv",
        repo_dir / "data.csv",
        script_dir / "DataSet Breast Cancer" / "Data.csv",
        repo_dir / "DataSet Breast Cancer" / "Data.csv",
    ]

    for path in candidates:
        if path.is_file():
            return path

    # Case-insensitive fallback for folder/file capitalization differences.
    search_roots = [script_dir, repo_dir]
    seen = set()
    for root in search_roots:
        if root in seen or not root.exists():
            continue
        seen.add(root)
        try:
            for path in root.rglob("*"):
                if (
                    path.is_file()
                    and path.name.lower() == "data.csv"
                    and "dataset breast cancer" in str(path.parent).lower()
                ):
                    return path
        except OSError:
            pass

    attempted = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not locate the breast cancer CSV file.\n\n"
        "Expected a file named data.csv in one of these locations:\n"
        f"{attempted}\n\n"
        "Check that data.csv is committed to GitHub and that the folder "
        "name is correct."
    )


@st.cache_data
def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.astype(str).str.strip()

    if "diagnosis" not in df.columns:
        raise ValueError(
            "The CSV does not contain a 'diagnosis' column. "
            "This app expects the Wisconsin Diagnostic Breast Cancer dataset."

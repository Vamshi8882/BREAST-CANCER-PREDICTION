from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_STATE = 42
CLASS_NAMES = {0: "Benign (B)", 1: "Malignant (M)"}


# ============================================================
# 2. DATASET PATH DISCOVERY
# Expected:
# ML/
#   app.py
#   project.py
#   requirements.txt
#   DataSet Breast Cancer/
#       data.csv
# ============================================================
def locate_dataset():
    """Find data.csv relative to this file, not the current working directory."""
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent

    # First try the exact GitHub repository layout.
    candidates = [
        script_dir / "DataSet Breast Cancer" / "data.csv",
        repo_dir / "ML" / "DataSet Breast Cancer" / "data.csv",
        repo_dir / "DataSet Breast Cancer" / "data.csv",
        script_dir / "data.csv",
        repo_dir / "data.csv",
    ]

    for path in candidates:
        if path.is_file():
            return path.resolve()

    # Case-insensitive lookup of the dataset folder and CSV filename.
    # Search only immediate children to avoid traversing .git or other large folders.
    roots = (script_dir, repo_dir)
    seen = set()

    for root in roots:
        root = root.resolve()
        if root in seen or not root.is_dir():
            continue
        seen.add(root)

        try:
            for child in root.iterdir():
                if child.is_dir() and child.name.casefold() == "dataset breast cancer":
                    for csv_file in child.iterdir():
                        if csv_file.is_file() and csv_file.name.casefold() == "data.csv":
                            return csv_file.resolve()
                if child.is_file() and child.name.casefold() == "data.csv":
                    return child.resolve()
        except OSError:
            continue

    # Give a useful diagnostic showing what Streamlit Cloud can actually see.
    visible_items = []
    for root in (script_dir, repo_dir):
        if root.is_dir():
            try:
                visible_items.append(f"Contents of {root}:")
                visible_items.extend(
                    f"  - {item.name}{'/' if item.is_dir() else ''}"
                    for item in sorted(root.iterdir(), key=lambda p: p.name.lower())
                )
            except OSError as exc:
                visible_items.append(f"  Could not list {root}: {exc}")

    checked_paths = "\n".join(f"- {path}" for path in candidates)
    visible_text = "\n".join(visible_items)

    raise FileNotFoundError(
        "Could not locate the breast cancer dataset file.\n\n"
        "Expected the CSV at: ML/DataSet Breast Cancer/data.csv\n\n"
        "Paths checked:\n"
        f"{checked_paths}\n\n"
        "Items visible to the app:\n"
        f"{visible_text}\n\n"
        "Confirm that the folder and data.csv are committed to GitHub. "
        "GitHub paths are case-sensitive."
    )


# ============================================================
# 3. LOAD AND VALIDATE DATASET
# ============================================================
@st.cache_data
def load_dataset(csv_path, modified_time):
    # modified_time changes the cache key when the CSV is updated.
    del modified_time

    df = pd.read_csv(csv_path)
    df.columns = df.columns.astype(str).str.strip()

    if "diagnosis" not in df.columns:
        raise ValueError(
            "The dataset must contain a 'diagnosis' column "
            "with B (Benign) and M (Malignant) values."
        )

    df["diagnosis"] = (
        df["diagnosis"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = df[df["diagnosis"].isin(["B", "M"])].copy()

    if df.empty:
        raise ValueError(
            "No valid B or M diagnosis values were found."
        )

    df["diagnosis"] = df["diagnosis"].map(
        {"B": 0, "M": 1}
    )

    # Remove identifier and empty columns.
    columns_to_drop = [
        column
        for column in df.columns
        if column.lower() in {"id", "unnamed: 32"}
    ]

    df.drop(
        columns=columns_to_drop,
        inplace=True,
        errors="ignore",
    )

    # Use numeric columns as prediction features.
    feature_columns = [
        column
        for column in df.columns
        if column != "diagnosis"
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    if not feature_columns:
        raise ValueError(
            "No numeric feature columns were found in the CSV."
        )

    # Handle infinite values and missing values.
    df[feature_columns] = df[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    medians = df[feature_columns].median()

    # Remove features that contain no valid numeric values.
    feature_columns = [
        column
        for column in feature_columns
        if pd.notna(medians[column])
    ]

    if not feature_columns:
        raise ValueError(
            "All numerical feature columns are empty."
        )

    df[feature_columns] = df[feature_columns].fillna(
        medians[feature_columns]
    )

    df.dropna(subset=["diagnosis"], inplace=True)
    df["diagnosis"] = df["diagnosis"].astype(int)

    if df["diagnosis"].nunique() != 2:
        raise ValueError(
            "The dataset must contain both benign and malignant records."
        )

    return df, feature_columns


# ============================================================
# 4. TRAIN MACHINE LEARNING MODELS
# ============================================================
@st.cache_resource
def train_models(csv_path, modified_time):
    df, feature_columns = load_dataset(
        csv_path,
        modified_time,
    )

    X = df[feature_columns]
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=4,
            random_state=RANDOM_STATE,
        ),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results[name] = {
            "model": model,
            "accuracy": accuracy_score(
                y_test, predictions
            ),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=[0, 1],
            ),
            "report": classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["Benign", "Malignant"],
                output_dict=True,
                zero_division=0,
            ),
        }

    return df, feature_columns, results


# ============================================================
# 5. INITIALIZE APPLICATION
# ============================================================
st.title("🎗️ Breast Cancer Prediction System")

st.markdown(
    """
    A machine-learning demonstration using **Logistic Regression**
    and **Decision Tree classification**.
    """
)

try:
    dataset_path = locate_dataset()

    data, feature_columns, results = train_models(
        str(dataset_path),
        dataset_path.stat().st_mtime,
    )

except Exception as error:
    st.error("Unable to load the dataset or train the models.")
    st.code(str(error))

    st.info(
        "Check the GitHub folder structure, CSV filename, "
        "dataset columns, and requirements.txt."
    )

    st.stop()


# ============================================================
# 6. NAVIGATION
# ============================================================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Home",
        "Predict",
        "Model Evaluation",
        "Dataset Explorer",
        "About",
    ],
)

st.sidebar.divider()
st.sidebar.write(f"Dataset: `{dataset_path.name}`")
st.sidebar.write(f"Records: {len(data):,}")
st.sidebar.write(f"Features: {len(feature_columns)}")


# ============================================================
# 7. HOME PAGE
# ============================================================
if page == "Home":
    st.header("Project Dashboard")

    total_records = len(data)
    benign_count = int((data["diagnosis"] == 0).sum())
    malignant_count = int((data["diagnosis"] == 1).sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Benign Records", f"{benign_count:,}")
    col3.metric("Malignant Records", f"{malignant_count:,}")
    col4.metric("Features", len(feature_columns))

    st.subheader("Model Performance")

    performance = pd.DataFrame({
        name: {
            "Accuracy": result["accuracy"] * 100,
            "Precision": result["precision"] * 100,
            "Recall": result["recall"] * 100,
            "F1 Score": result["f1"] * 100,
        }
        for name, result in results.items()
    })

    st.dataframe(
        performance.round(2),
        use_container_width=True,
    )

    st.bar_chart(performance.loc["Accuracy"])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Diagnosis Distribution")

        diagnosis_counts = (
            data["diagnosis"]
            .map(CLASS_NAMES)
            .value_counts()
        )

        st.bar_chart(diagnosis_counts)

    with col2:
        st.subheader("Model Accuracy Comparison")

        accuracy_data = pd.DataFrame({
            "Model": list(results.keys()),
            "Accuracy (%)": [
                result["accuracy"] * 100
                for result in results.values()
            ],
        })

        st.bar_chart(
            accuracy_data.set_index("Model")
        )

    st.info(
        "This project is intended for education and demonstration. "
        "It is not a medical diagnostic tool."
    )


# ============================================================
# 8. PREDICTION PAGE
# ============================================================
elif page == "Predict":
    st.header("Make a Prediction")

    st.write(
        "Enter numerical feature measurements. "
        "The default values are the corresponding dataset medians."
    )

    selected_model_name = st.selectbox(
        "Select Machine Learning Model",
        list(results.keys()),
    )

    medians = data[feature_columns].median()

    with st.form("prediction_form"):
        st.subheader("Input Features")

        input_values = {}
        columns = st.columns(3)

        for index, feature in enumerate(feature_columns):
            with columns[index % 3]:
                input_values[feature] = st.number_input(
                    feature,
                    value=float(medians[feature]),
                    format="%.6f",
                    key=f"feature_{index}",
                )

        submitted = st.form_submit_button(
            "Predict Diagnosis",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        input_df = pd.DataFrame(
            [
                [
                    input_values[feature]
                    for feature in feature_columns
                ]
            ],
            columns=feature_columns,
        )

        model = results[selected_model_name]["model"]
        prediction = int(model.predict(input_df)[0])

        st.subheader("Prediction Result")

        if prediction == 0:
            st.success("Model prediction: Benign (B)")
        else:
            st.warning("Model prediction: Malignant (M)")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]

            col1, col2 = st.columns(2)

            col1.metric(
                "Benign Model Probability",
                f"{probabilities[0] * 100:.2f}%",
            )

            col2.metric(
                "Malignant Model Probability",
                f"{probabilities[1] * 100:.2f}%",
            )

            st.caption(
                "These are model-generated probabilities, "
                "not clinical certainty."
            )

        st.warning(
            "This result cannot confirm or exclude cancer. "
            "Consult a qualified healthcare professional for "
            "medical assessment."
        )


# ============================================================
# 9. MODEL EVALUATION PAGE
# ============================================================
elif page == "Model Evaluation":
    st.header("Model Evaluation")

    selected_model_name = st.selectbox(
        "Choose a model",
        list(results.keys()),
        key="evaluation_model",
    )

    result = results[selected_model_name]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{result['accuracy'] * 100:.2f}%",
    )
    col2.metric(
        "Precision",
        f"{result['precision'] * 100:.2f}%",
    )
    col3.metric(
        "Recall",
        f"{result['recall'] * 100:.2f}%",
    )
    col4.metric(
        "F1 Score",
        f"{result['f1'] * 100:.2f}%",
    )

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        result["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        ax=ax,
    )

    ax.set_xlabel("Predicted Diagnosis")
    ax.set_ylabel("Actual Diagnosis")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")

    report_df = pd.DataFrame(
        result["report"]
    ).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True,
    )

    st.subheader("Compare Both Models")

    comparison = pd.DataFrame({
        name: {
            "Accuracy": result_item["accuracy"] * 100,
            "Precision": result_item["precision"] * 100,
            "Recall": result_item["recall"] * 100,
            "F1 Score": result_item["f1"] * 100,
        }
        for name, result_item in results.items()
    })

    st.dataframe(
        comparison.round(2),
        use_container_width=True,
    )


# ============================================================
# 10. DATASET EXPLORER PAGE
# ============================================================
elif page == "Dataset Explorer":
    st.header("Dataset Explorer")

    st.subheader("Dataset Preview")

    st.dataframe(
        data.head(25),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Shape")
        st.write(f"Rows: {data.shape[0]}")
        st.write(f"Columns: {data.shape[1]}")

        st.subheader("Missing Values")

        missing_values = (
            data.isnull()
            .sum()
            .rename("Missing Values")
        )

        st.dataframe(
            missing_values,
            use_container_width=True,
        )

    with col2:
        st.subheader("Statistical Summary")

        st.dataframe(
            data[feature_columns].describe().transpose(),
            use_container_width=True,
        )

    st.subheader("Diagnosis Distribution")

    diagnosis_counts = (
        data["diagnosis"]
        .map(CLASS_NAMES)
        .value_counts()
        .rename_axis("Diagnosis")
        .to_frame("Count")
    )

    st.bar_chart(diagnosis_counts)

    st.subheader("Feature Correlation")

    correlation = data[feature_columns].corr()

    fig, ax = plt.subplots(figsize=(14, 10))

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        ax=ax,
    )

    ax.set_title("Correlation Between Numerical Features")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Download Dataset")

    csv_bytes = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Processed CSV",
        data=csv_bytes,
        file_name="processed_breast_cancer_dataset.csv",
        mime="text/csv",
    )


# ============================================================
# 11. ABOUT PAGE
# ============================================================
elif page == "About":
    st.header("About This Project")

    st.markdown(
        """
        ### Breast Cancer Prediction Using Machine Learning

        **Objective:** Demonstrate binary classification using
        numerical features from a breast cancer dataset.

        **Algorithms**
        - Logistic Regression
        - Decision Tree Classifier

        **Technologies**
        - Python
        - Pandas and NumPy
        - Scikit-learn
        - Matplotlib and Seaborn
        - Streamlit

        **Workflow**
        1. Load and validate the CSV dataset.
        2. Prepare the features and diagnosis labels.
        3. Split the data into training and testing sets.
        4. Train both classification models.
        5. Evaluate model performance.
        6. Explore the dataset and test sample inputs.
        """
    )

    st.divider()

    st.warning(
        "This application is for educational purposes only. "
        "Its predictions must not be used to make medical decisions."
    )


# ============================================================
# 12. FOOTER
# ============================================================
st.divider()

st.caption(
    "Breast Cancer Prediction | Machine Learning Project"
)

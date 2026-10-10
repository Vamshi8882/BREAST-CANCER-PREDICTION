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
from sklearn.pipeline import Pipeline
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
    page_title="Breast Cancer Prediction | ML",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = {0: "Benign (B)", 1: "Malignant (M)"}
RANDOM_STATE = 42


# ============================================================
# 2. DATASET DISCOVERY
# Works with ML/project.py and several CSV locations.
# ============================================================
def locate_dataset():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent

    candidates = [
        script_dir / "DataSet Breast Cancer" / "data.csv",
        repo_dir / "DataSet Breast Cancer" / "data.csv",
        script_dir / "data.csv",
        repo_dir / "data.csv",
    ]

    for path in candidates:
        if path.is_file():
            return path

    # Case-insensitive fallback for folder/file capitalization.
    search_roots = [script_dir, repo_dir]
    seen = set()

    for root in search_roots:
        if root in seen or not root.exists():
            continue

        seen.add(root)

        try:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue

                if path.name.lower() != "data.csv":
                    continue

                # Avoid searching unrelated folders unnecessarily.
                if (
                    "dataset breast cancer" in str(path.parent).lower()
                    or path.parent == root
                ):
                    return path
        except OSError:
            continue

    attempted = "\n".join(f"- {path}" for path in candidates)

    raise FileNotFoundError(
        "The breast cancer dataset could not be located.\n\n"
        "Expected data.csv in one of these locations:\n"
        f"{attempted}\n\n"
        "Verify that the CSV is committed to GitHub."
    )


# ============================================================
# 3. LOAD AND VALIDATE DATASET
# ============================================================
@st.cache_data
def load_dataset(csv_path: str, modified_time: float):
    del modified_time  # Used to refresh the cache if the CSV changes.

    df = pd.read_csv(csv_path)
    df.columns = df.columns.astype(str).str.strip()

    if "diagnosis" not in df.columns:
        raise ValueError(
            "Missing 'diagnosis' column. "
            "Expected diagnosis values B and M."
        )

    # Normalize diagnosis values.
    df["diagnosis"] = (
        df["diagnosis"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = df[df["diagnosis"].isin(["B", "M"])].copy()

    if df.empty:
        raise ValueError(
            "No valid diagnosis records were found. "
            "Expected B (Benign) or M (Malignant)."
        )

    # Convert diagnosis into binary labels.
    df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

    # Remove identifiers and empty columns.
    columns_to_drop = [
        column
        for column in df.columns
        if column.lower() in {"id", "unnamed: 32"}
    ]

    df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    # Retain numeric predictor columns only.
    feature_columns = [
        column
        for column in df.columns
        if column != "diagnosis"
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    if not feature_columns:
        raise ValueError("No numeric prediction features were found.")

    # Handle invalid numeric values.
    df[feature_columns] = df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    # Remove rows with missing target values and impute features.
    df.dropna(subset=["diagnosis"], inplace=True)

    medians = df[feature_columns].median()
    feature_columns = [
        column for column in feature_columns
        if pd.notna(medians[column])
    ]

    if not feature_columns:
        raise ValueError("All numeric features are empty.")

    df[feature_columns] = df[feature_columns].fillna(
        medians[feature_columns]
    )

    df["diagnosis"] = df["diagnosis"].astype(int)

    return df, feature_columns


# ============================================================
# 4. TRAIN MACHINE LEARNING MODELS
# ============================================================
@st.cache_resource
def train_models(csv_path: str, modified_time: float):
    df, feature_columns = load_dataset(csv_path, modified_time)

    X = df[feature_columns]
    y = df["diagnosis"]

    if y.nunique() < 2:
        raise ValueError(
            "Training requires both benign and malignant records."
        )

    # Use stratification to preserve class proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Scaling is important for Logistic Regression.
    logistic_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    tree_model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=4,
        random_state=RANDOM_STATE,
    )

    models = {
        "Logistic Regression": logistic_model,
        "Decision Tree": tree_model,
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "predictions": predictions,
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=[0, 1]
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

    return df, feature_columns, X_test, y_test, results


# ============================================================
# 5. LOAD APP DATA
# ============================================================
st.title("🎗️ Breast Cancer Prediction System")
st.caption(
    "Machine Learning application using Logistic Regression "
    "and Decision Tree classification."
)

try:
    dataset_path = locate_dataset()
    modified_time = dataset_path.stat().st_mtime

    data, feature_columns, X_test, y_test, results = train_models(
        str(dataset_path), modified_time
    )

except Exception as error:
    st.error("Unable to load the dataset or train the models.")
    st.code(str(error))
    st.info(
        "Check the GitHub repository structure, CSV filename, "
        "diagnosis column, and requirements.txt."
    )
    st.stop()


# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Home",
        "Predict",
        "Model Evaluation",
        "Dataset Explorer",
        "About",
    ],
)

st.sidebar.divider()
st.sidebar.caption(f"Dataset: {dataset_path.name}")
st.sidebar.caption(f"Records: {len(data):,}")
st.sidebar.caption(f"Features: {len(feature_columns)}")


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

    st.subheader("Model Accuracy")

    accuracy_df = pd.DataFrame(
        {
            "Model": list(results.keys()),
            "Accuracy": [
                result["accuracy"] * 100
                for result in results.values()
            ],
        }
    )

    st.bar_chart(accuracy_df.set_index("Model"))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Diagnosis Distribution")

        fig, ax = plt.subplots()
        sns.countplot(
            x=data["diagnosis"].map(CLASS_NAMES),
            hue=data["diagnosis"].map(CLASS_NAMES),
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("Diagnosis")
        ax.set_ylabel("Number of records")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Model Comparison")

        comparison = pd.DataFrame(
            {
                "Accuracy": {
                    name: result["accuracy"]
                    for name, result in results.items()
                },
                "Precision": {
                    name: result["precision"]
                    for name, result in results.items()
                },
                "Recall": {
                    name: result["recall"]
                    for name, result in results.items()
                },
                "F1 Score": {
                    name: result["f1"]
                    for name, result in results.items()
                },
            }
        )

        st.dataframe(
            (comparison * 100).round(2),
            use_container_width=True,
        )

    st.info(
        "This project demonstrates machine learning classification. "
        "It is not a medical diagnostic tool."
    )


# ============================================================
# 8. PREDICTION PAGE
# ============================================================
elif page == "Predict":
    st.header("Make a Prediction")

    st.write(
        "Enter the numerical measurements below. "
        "Default values are the dataset's median measurements."
    )

    selected_model_name = st.selectbox(
        "Select Model",
        list(results.keys()),
    )

    st.caption(
        "The model expects the same numerical feature definitions "
        "as the training dataset."
    )

    medians = data[feature_columns].median()

    # Group the features into manageable columns.
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
            [[input_values[feature] for feature in feature_columns]],
            columns=feature_columns,
        )

        selected_model = results[selected_model_name]["model"]

        prediction = int(selected_model.predict(input_df)[0])

        st.subheader("Prediction Result")

        if prediction == 0:
            st.success("Model prediction: Benign (B)")
        else:
            st.warning("Model prediction: Malignant (M)")

        if hasattr(selected_model, "predict_proba"):
            probabilities = selected_model.predict_proba(input_df)[0]

            benign_probability = float(probabilities[0]) * 100
            malignant_probability = float(probabilities[1]) * 100

            col1, col2 = st.columns(2)

            col1.metric(
                "Benign Model Probability",
                f"{benign_probability:.2f}%",
            )

            col2.metric(
                "Malignant Model Probability",
                f"{malignant_probability:.2f}%",
            )

            st.caption(
                "These are model probabilities, not clinical certainty "
                "or a substitute for medical evaluation."
            )

        st.warning(
            "Educational demonstration only. This result cannot "
            "confirm or exclude cancer. Consult a qualified healthcare "
            "professional for medical assessment."
        )


# ============================================================
# 9. MODEL EVALUATION PAGE
# ============================================================
elif page == "Model Evaluation":
    st.header("Model Evaluation")

    selected_model_name = st.selectbox(
        "Choose a model to evaluate",
        list(results.keys()),
        key="evaluation_model",
    )

    result = results[selected_model_name]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{result['accuracy'] * 100:.2f}%")
    col2.metric("Precision", f"{result['precision'] * 100:.2f}%")
    col3.metric("Recall", f"{result['recall'] * 100:.2f}%")
    col4.metric("F1 Score", f"{result['f1'] * 100:.2f}%")

    st.subheader("Confusion Matrix")

    cm = result["confusion_matrix"]

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
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

    report_df = pd.DataFrame(result["report"]).transpose()
    st.dataframe(
        report_df.round(4),
        use_container_width=True,
    )

    st.subheader("Compare Both Models")

    metric_df = pd.DataFrame(
        {
            name: {
                "Accuracy": result_item["accuracy"],
                "Precision": result_item["precision"],
                "Recall": result_item["recall"],
                "F1 Score": result_item["f1"],
            }
            for name, result_item in results.items()
        }
    )

    st.dataframe(
        (metric_df * 100).round(2),
        use_container_width=True,
    )


# ============================================================
# 10. DATASET EXPLORER
# ============================================================
elif page == "Dataset Explorer":
    st.header("Dataset Explorer")

    st.subheader("Dataset Preview")
    st.dataframe(data.head(20), use_container_width=True)

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Rows and columns")
        st.write(data.shape)

        st.write("Missing values")
        st.dataframe(
            data.isnull().sum().rename("Missing Values"),
            use_container_width=True,
        )

    with col2:
        st.write("Statistical summary")
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

    st.subheader("Download Processed Dataset")

    csv_bytes = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV",
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

        **Libraries**
        - Python
        - Pandas and NumPy
        - Scikit-learn
        - Matplotlib and Seaborn
        - Streamlit

        **Workflow**
        1. Load and validate the CSV dataset.
        2. Prepare numerical input features and diagnosis labels.
        3. Split the dataset into training and testing sets.
        4. Train both classification models.
        5. Evaluate model performance.
        6. Allow users to explore the dataset and try predictions.
        """
    )

    st.divider()

    st.caption(
        "Important: This application is intended for educational "
        "and demonstration purposes only. Its outputs must not be "
        "used to make medical decisions."
    )

# ============================================================
# 12. FOOTER
# ============================================================
st.divider()

st.caption(
    f"Breast Cancer Prediction | Machine Learning Project | "
    f"Updated {datetime.now().strftime('%d %b %Y')}"
)

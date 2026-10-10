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

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction | ML",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = (
    Path(__file__).resolve().parent
    / "DataSet Breast Cancer"
    / "data.csv"
)

CLASS_NAMES = {
    0: "Benign (B)",
    1: "Malignant (M)",
}


# --------------------------------------------------
# DATA LOADING AND MODEL TRAINING
# --------------------------------------------------
@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}. "
            "Check the folder name, filename, and GitHub repository."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    if "diagnosis" not in df.columns:
        raise ValueError(
            "The dataset must contain a 'diagnosis' column."
        )

    df = df.drop(
        columns=["id", "Unnamed: 32"],
        errors="ignore",
    )

    df["diagnosis"] = (
        df["diagnosis"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"M": 1, "B": 0})
    )

    df = df.dropna(subset=["diagnosis"]).copy()
    df["diagnosis"] = df["diagnosis"].astype(int)

    feature_columns = [
        col for col in df.columns if col != "diagnosis"
    ]

    # All prediction features must be numeric.
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_columns).reset_index(drop=True)

    if not feature_columns:
        raise ValueError("No numerical feature columns were found.")

    if df["diagnosis"].nunique() != 2:
        raise ValueError(
            "The dataset must contain both B and M diagnosis classes."
        )

    if len(df) < 10:
        raise ValueError("The dataset has too few valid rows.")

    return df, feature_columns


@st.cache_resource
def train_models():
    df, feature_columns = load_dataset()

    X = df[feature_columns]
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(max_iter=2000)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)

    dt_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=5,
    )
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)

    predictions = {
        "Logistic Regression": lr_pred,
        "Decision Tree": dt_pred,
    }

    metrics = {}

    for name, pred in predictions.items():
        metrics[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "F1-score": f1_score(
                y_test, pred, zero_division=0
            ),
            "Confusion Matrix": confusion_matrix(
                y_test, pred, labels=[0, 1]
            ),
            "Classification Report": classification_report(
                y_test,
                pred,
                labels=[0, 1],
                target_names=["Benign", "Malignant"],
                output_dict=True,
                zero_division=0,
            ),
        }

    return (
        df,
        feature_columns,
        scaler,
        lr_model,
        dt_model,
        metrics,
        y_test,
    )


# --------------------------------------------------
# LOAD THE APPLICATION
# --------------------------------------------------
try:
    (
        data,
        feature_columns,
        scaler,
        lr_model,
        dt_model,
        metrics,
        y_test,
    ) = train_models()
except Exception as exc:
    st.error("The application could not initialize.")
    st.exception(exc)
    st.info(
        "Verify the dataset path and diagnosis column, "
        "then check the deployment logs."
    )
    st.stop()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("🎗️ Breast Cancer ML")
st.sidebar.caption("Machine Learning Demonstration")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Predict",
        "Model Evaluation",
        "Dataset Explorer",
        "About",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("**Algorithms**")
st.sidebar.write("• Logistic Regression")
st.sidebar.write("• Decision Tree")
st.sidebar.caption(
    "Educational demonstration only. "
    "Not a medical diagnostic tool."
)


# --------------------------------------------------
# REUSABLE METRIC CHART
# --------------------------------------------------
def show_confusion_matrix(name):
    matrix = metrics[name]["Confusion Matrix"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues" if name == "Logistic Regression" else "Greens",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"{name} — Confusion Matrix")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "Home":
    st.title("🎗️ Breast Cancer Prediction")
    st.subheader("Machine Learning Analysis Dashboard")

    st.markdown(
        """
        Explore a machine-learning demonstration using the
        Wisconsin Diagnostic Breast Cancer dataset.

        Compare Logistic Regression and Decision Tree models,
        examine their evaluation metrics, and experiment with
        numerical feature inputs.
        """
    )

    st.warning(
        "This application is for education and software demonstration "
        "only. It is not validated for clinical use and must not be "
        "used to diagnose, rule out, or guide treatment for cancer."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Valid records", f"{len(data):,}")
    col2.metric("Numerical features", len(feature_columns))
    col3.metric(
        "Logistic Regression accuracy",
        f"{metrics['Logistic Regression']['Accuracy']:.1%}",
    )
    col4.metric(
        "Decision Tree accuracy",
        f"{metrics['Decision Tree']['Accuracy']:.1%}",
    )

    st.divider()
    st.subheader("Project Overview")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Logistic Regression")
        st.write(
            "A classification algorithm trained on standardized "
            "features. The scaler is fitted only on the training data."
        )

    with c2:
        st.markdown("### Decision Tree")
        st.write(
            "A tree-based classifier that learns decision rules "
            "from the training dataset."
        )

    st.subheader("Model Comparison")

    chart_data = pd.DataFrame(
        {
            name: {
                metric: values[metric]
                for metric in [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1-score",
                ]
            }
            for name, values in metrics.items()
        }
    )

    st.bar_chart(chart_data)

    st.caption(
        "Metrics shown here are from a held-out test split. "
        "They are not evidence of clinical validity."
    )


# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------
elif page == "Predict":
    st.title("Make an Experimental Prediction")

    st.write(
        "Enter the 30 numerical measurements used by the dataset. "
        "Defaults are the training-data medians, not patient measurements."
    )

    st.info(
        "Use only valid feature measurements in the same units and "
        "definitions as the training dataset. Arbitrary or incorrectly "
        "measured inputs can produce misleading results."
    )

    # Training-data medians are used as form defaults.
    defaults = data[feature_columns].median()

    # Use observed dataset ranges to keep entries numerically bounded.
    minima = data[feature_columns].min()
    maxima = data[feature_columns].max()

    with st.form("prediction_form"):
        st.subheader("Cell Measurement Inputs")

        user_values = {}

        # Three columns make the long form easier to navigate.
        columns = st.columns(3)

        for i, feature in enumerate(feature_columns):
            col = columns[i % 3]

            low = float(minima[feature])
            high = float(maxima[feature])
            default = float(defaults[feature])

            if not np.isfinite(low) or not np.isfinite(high):
                low, high, default = 0.0, 1.0, 0.5

            if low == high:
                low -= 0.001
                high += 0.001

            default = min(max(default, low), high)
            step = max((high - low) / 100.0, 0.0001)

            with col:
                user_values[feature] = st.number_input(
                    feature,
                    min_value=low,
                    max_value=high,
                    value=default,
                    step=step,
                    format="%.5f",
                    help=(
                        "Training-data range: "
                        f"{minima[feature]:.5f} to "
                        f"{maxima[feature]:.5f}"
                    ),
                )

        submitted = st.form_submit_button(
            "Run Both Models",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        input_df = pd.DataFrame(
            [[user_values[col] for col in feature_columns]],
            columns=feature_columns,
        )

        lr_input = scaler.transform(input_df)

        lr_class = int(lr_model.predict(lr_input)[0])
        dt_class = int(dt_model.predict(input_df)[0])

        lr_prob = lr_model.predict_proba(lr_input)[0]
        dt_prob = dt_model.predict_proba(input_df)[0]

        lr_classes = list(lr_model.classes_)
        dt_classes = list(dt_model.classes_)

        lr_confidence = float(
            lr_prob[lr_classes.index(lr_class)]
        )
        dt_confidence = float(
            dt_prob[dt_classes.index(dt_class)]
        )

        st.session_state["last_prediction"] = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lr_class": lr_class,
            "dt_class": dt_class,
            "lr_confidence": lr_confidence,
            "dt_confidence": dt_confidence,
        }

        if "history" not in st.session_state:
            st.session_state["history"] = []

        st.session_state["history"].append(
            {
                "Time": st.session_state["last_prediction"]["time"],
                "Logistic Regression": CLASS_NAMES[lr_class],
                "LR model score": f"{lr_confidence:.1%}",
                "Decision Tree": CLASS_NAMES[dt_class],
                "DT model score": f"{dt_confidence:.1%}",
            }
        )

    result = st.session_state.get("last_prediction")

    if result:
        st.divider()
        st.subheader("Model Outputs")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Logistic Regression")
            st.metric(
                "Predicted dataset class",
                CLASS_NAMES[result["lr_class"]],
            )
            st.metric(
                "Model score for predicted class",
                f"{result['lr_confidence']:.1%}",
            )

        with c2:
            st.markdown("### Decision Tree")
            st.metric(
                "Predicted dataset class",
                CLASS_NAMES[result["dt_class"]],
            )
            st.metric(
                "Model score for predicted class",
                f"{result['dt_confidence']:.1%}",
            )

        st.caption(
            "These scores are model outputs, not clinically validated "
            "cancer probabilities. A high score does not establish a diagnosis."
        )

        if result["lr_class"] != result["dt_class"]:
            st.warning(
                "The models disagree. This illustrates model uncertainty "
                "and is not a basis for a medical decision."
            )

        st.info(
            "The result is a classification from the trained dataset. "
            "Consult a qualified healthcare professional for medical concerns."
        )


# --------------------------------------------------
# MODEL EVALUATION PAGE
# --------------------------------------------------
elif page == "Model Evaluation":
    st.title("Model Evaluation & Comparison")

    st.write(
        "Evaluation uses the same held-out test split for both models."
    )

    rows = []

    for name, values in metrics.items():
        rows.append(
            {
                "Model": name,
                "Accuracy": values["Accuracy"],
                "Precision": values["Precision"],
                "Recall": values["Recall"],
                "F1-score": values["F1-score"],
            }
        )

    metric_df = pd.DataFrame(rows).set_index("Model")

    st.subheader("Performance Metrics")
    st.dataframe(
        metric_df.style.format("{:.3f}"),
        use_container_width=True,
    )

    st.subheader("Compare the Models")
    st.bar_chart(metric_df)

    st.caption(
        "Accuracy is the fraction of correct predictions. Precision, "
        "recall, and F1-score describe different aspects of classification "
        "performance; none alone establishes clinical suitability."
    )

    st.subheader("Confusion Matrices")

    c1, c2 = st.columns(2)

    with c1:
        show_confusion_matrix("Logistic Regression")

    with c2:
        show_confusion_matrix("Decision Tree")

    model_name = st.selectbox(
        "View detailed classification report",
        ["Logistic Regression", "Decision Tree"],
    )

    report = metrics[model_name]["Classification Report"]
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.style.format("{:.3f}"),
        use_container_width=True,
    )


# --------------------------------------------------
# DATASET EXPLORER PAGE
# --------------------------------------------------
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{data.shape[0]:,}")
    c2.metric("Features", len(feature_columns))
    c3.metric(
        "Missing values after cleaning",
        f"{int(data.isna().sum().sum()):,}",
    )

    st.subheader("Preview")
    st.dataframe(data.head(20), use_container_width=True)

    st.subheader("Diagnosis Class Distribution")

    class_counts = (
        data["diagnosis"]
        .map(CLASS_NAMES)
        .value_counts()
        .rename_axis("Class")
        .to_frame("Records")
    )

    st.bar_chart(class_counts)

    st.subheader("Feature Distributions")

    selected_feature = st.selectbox(
        "Select a feature",
        feature_columns,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=data,
        x=selected_feature,
        hue="diagnosis",
        bins=30,
        kde=True,
        ax=ax,
    )
    ax.set_title(f"Distribution of {selected_feature}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation Heatmap")

    correlation = data[feature_columns].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        ax=ax,
    )
    ax.set_title("Feature Correlations")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Download cleaned dataset",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_breast_cancer_dataset.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# PREDICTION HISTORY PAGE SECTION
# --------------------------------------------------
if page == "Predict":
    st.divider()
    st.subheader("Current Session — Prediction History")

    history = st.session_state.get("history", [])

    if history:
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, use_container_width=True)

        st.download_button(
            "Download session history",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )

        if st.button("Clear session history"):
            st.session_state["history"] = []
            st.session_state.pop("last_prediction", None)
            st.rerun()
    else:
        st.caption("No predictions have been made in this session yet.")


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------
elif page == "About":
    st.title("About This Project")

    st.markdown(
        """
        ### Project objective
        Demonstrate how supervised machine-learning classifiers can
        analyze the Wisconsin Diagnostic Breast Cancer dataset.

        ### Algorithms
        - Logistic Regression
        - Decision Tree Classifier

        ### Workflow
        1. Load and validate the dataset.
        2. Remove unnecessary columns and encode diagnosis labels.
        3. Split data into training and testing sets.
        4. Standardize the Logistic Regression features.
        5. Train and evaluate both models.
        6. Display metrics, charts, and experimental predictions.

        ### Limitations
        This project is an educational prototype. Its test-set performance
        does not establish clinical validity, generalizability, or safety.
        It has not been validated for real-world medical use.
        """
    )

    st.markdown(
        "[Dataset information — UCI Machine Learning Repository]"
        "(https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)"
    )

st.sidebar.divider()
st.sidebar.caption("Breast Cancer Prediction • ML Project")

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Health & Lifestyle Classification", page_icon="🩺", layout="wide")

st.title("🩺 Health & Lifestyle Classification")
st.write(
    "Upload the health/lifestyle CSV used in your notebook. "
    "The app trains four classification models and predicts the target class."
)
st.info(
    "This is an educational machine-learning demo, not a medical diagnosis. "
    "Do not use its prediction to make healthcare decisions."
)

DROP_COLUMNS = [
    "survey_code", "environmental_risk_score", "gene_marker_flag",
    "electrolyte_level", "bmi_estimated", "bmi_corrected", "bmi_scaled",
    "job_type", "occupation", "education_level", "pet_owner"
]

TARGET = "target"

uploaded_file = st.file_uploader("Upload health_lifestyle_classification.csv", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload your CSV file to start.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if TARGET not in raw_df.columns:
    st.error("The uploaded CSV must contain a 'target' column.")
    st.stop()

df = raw_df.drop(columns=[c for c in DROP_COLUMNS if c in raw_df.columns]).copy()
df = df.dropna(subset=[TARGET])

if df.empty:
    st.error("No rows remain after removing rows with a missing target.")
    st.stop()

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(str)

# Remove columns that contain no usable values.
empty_cols = [c for c in X.columns if X[c].isna().all()]
if empty_cols:
    X = X.drop(columns=empty_cols)

# Detect column types from the uploaded dataset.
categorical_cols = X.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
numeric_cols = [c for c in X.columns if c not in categorical_cols]

numeric_pipeline = SkPipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = SkPipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ],
    remainder="drop"
)

st.subheader("Dataset overview")
m1, m2, m3 = st.columns(3)
m1.metric("Rows", f"{len(df):,}")
m2.metric("Input features", len(X.columns))
m3.metric("Target classes", y.nunique())

with st.expander("Preview dataset and target distribution"):
    st.dataframe(df.head(10), use_container_width=True)
    st.write("Target class counts")
    st.bar_chart(y.value_counts())

if y.nunique() < 2:
    st.error("The target column must contain at least two different classes.")
    st.stop()

if y.value_counts().min() < 2:
    st.error("Each target class needs at least two rows for a stratified train/test split.")
    st.stop()

test_size = st.sidebar.slider("Test data percentage", 10, 40, 20, 5) / 100
use_smote = st.sidebar.checkbox("Use SMOTE to balance training data", value=True)
st.sidebar.caption("SMOTE is applied only to the training split.")

if st.button("🚀 Train and compare models", type="primary", use_container_width=True):
    with st.spinner("Training models. This may take a little time for a large CSV..."):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            model_definitions = {
                "Logistic Regression": LogisticRegression(max_iter=1500),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(
                    n_estimators=150, random_state=42, n_jobs=-1
                ),
                "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
            }

            trained_models = {}
            rows = []

            for model_name, classifier in model_definitions.items():
                steps = [("preprocessor", preprocessor)]
                if use_smote:
                    # Avoid SMOTE failures when a small class has too few examples.
                    min_class_count = int(y_train.value_counts().min())
                    if min_class_count >= 2:
                        steps.append(("smote", SMOTE(
                            random_state=42, k_neighbors=min(5, min_class_count - 1)
                        )))
                    else:
                        st.warning("SMOTE skipped: a training class has fewer than 2 rows.")
                steps.append(("classifier", classifier))
                model = ImbPipeline(steps=steps)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                rows.append({
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_test, predictions),
                    "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
                    "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
                    "F1 Score": f1_score(y_test, predictions, average="weighted", zero_division=0)
                })
                trained_models[model_name] = model

            results = pd.DataFrame(rows).set_index("Model").sort_values("F1 Score", ascending=False)
            st.session_state["trained_models"] = trained_models
            st.session_state["results"] = results
            st.session_state["X_columns"] = X.columns.tolist()
            st.session_state["categorical_cols"] = categorical_cols
            st.session_state["numeric_cols"] = numeric_cols
            st.session_state["X_reference"] = X.copy()
            st.session_state["y_test"] = y_test
            st.session_state["test_size"] = test_size

        except Exception as exc:
            st.error(f"Training failed: {exc}")
            st.exception(exc)

if "results" in st.session_state:
    st.divider()
    st.subheader("Model comparison")
    st.dataframe(
        st.session_state["results"].style.format("{:.3f}"),
        use_container_width=True
    )
    st.bar_chart(st.session_state["results"][["Accuracy", "Precision", "Recall", "F1 Score"]])

    st.subheader("Confusion matrix")
    selected_model_name = st.selectbox(
        "Choose a model",
        list(st.session_state["trained_models"].keys())
    )
    selected_model = st.session_state["trained_models"][selected_model_name]
    X_reference = st.session_state["X_reference"]
    # Recreate the same deterministic test split used for training.
    X_train_ref, X_test_ref, y_train_ref, y_test_ref = train_test_split(
        X_reference, y, test_size=st.session_state["test_size"],
        random_state=42, stratify=y
    )
    y_pred_ref = selected_model.predict(X_test_ref)
    labels = selected_model.named_steps["classifier"].classes_
    cm = confusion_matrix(y_test_ref, y_pred_ref, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {selected_model_name}")
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Classification report"):
        st.text(classification_report(y_test_ref, y_pred_ref, zero_division=0))

    st.divider()
    st.subheader("Make a prediction")
    st.write("Enter values for one new record. The app uses the same preprocessing learned during training.")

    reference = st.session_state["X_reference"]
    categorical = st.session_state["categorical_cols"]
    numeric = st.session_state["numeric_cols"]
    form_values = {}

    with st.form("prediction_form"):
        left, right = st.columns(2)
        for idx, col in enumerate(st.session_state["X_columns"]):
            container = left if idx % 2 == 0 else right
            with container:
                if col in categorical:
                    choices = sorted(reference[col].dropna().astype(str).unique().tolist())
                    if not choices:
                        choices = ["Unknown"]
                    form_values[col] = st.selectbox(col.replace("_", " ").title(), choices)
                else:
                    series = pd.to_numeric(reference[col], errors="coerce").dropna()
                    default = float(series.median()) if not series.empty else 0.0
                    minimum = float(series.min()) if not series.empty else default - 1
                    maximum = float(series.max()) if not series.empty else default + 1
                    if minimum == maximum:
                        minimum -= 1
                        maximum += 1
                    form_values[col] = st.number_input(
                        col.replace("_", " ").title(),
                        min_value=minimum,
                        max_value=maximum,
                        value=min(max(default, minimum), maximum),
                        format="%.3f"
                    )
        submitted = st.form_submit_button("Predict target", type="primary")

    if submitted:
        input_df = pd.DataFrame([form_values], columns=st.session_state["X_columns"])
        prediction = selected_model.predict(input_df)[0]
        st.success(f"Predicted target: **{prediction}**")
        if hasattr(selected_model, "predict_proba"):
            probabilities = selected_model.predict_proba(input_df)[0]
            probability_df = pd.DataFrame({
                "Class": selected_model.named_steps["classifier"].classes_,
                "Probability": probabilities
            }).sort_values("Probability", ascending=False)
            st.write("Prediction probabilities")
            st.dataframe(probability_df.style.format({"Probability": "{:.1%}"}),
                         use_container_width=True)

    st.caption("Model metrics describe performance on the held-out test split. They do not guarantee accuracy on future records.")

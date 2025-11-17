# -*- coding: utf-8 -*-
"""Final Streamlit App - Fully Validated"""

# ===================== IMPORTS =====================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import joblib
import zipfile
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sns.set_theme(style="whitegrid")
st.set_page_config(layout="wide", page_title="YouTube Engagement Dashboard")


# ============================================================
# ZIP + CSV LOADER (VALIDATED)
# ============================================================
def load_file(uploaded_file):
    """Load CSV or extract CSV from ZIP."""
    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower() if hasattr(uploaded_file, "name") else "repo_zip"

    # Case 1: Direct CSV upload
    if filename.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file, low_memory=True, on_bad_lines="skip")
        except Exception as e:
            st.error(f"❌ Could not read CSV: {e}")
            return None

    # Case 2: ZIP upload
    try:
        z = zipfile.ZipFile(uploaded_file)
        inside = z.namelist()

        csv_files = [f for f in inside if f.lower().endswith(".csv")]
        if len(csv_files) == 0:
            st.error("❌ ZIP contains no CSV.")
            return None

        csv_name = csv_files[0]
        st.success(f"📄 Loaded CSV from ZIP: {csv_name}")

        with z.open(csv_name) as f:
            return pd.read_csv(f, low_memory=True, on_bad_lines="skip")

    except Exception as e:
        st.error(f"❌ ZIP extract error: {e}")
        return None


# ============================================================
# AUTO LOAD DATASET FROM REPO
# ============================================================
REPO_ZIP_PATH = "youtube recommendation dataset.zip"

def load_default_dataset():
    """Load ZIP dataset stored inside GitHub repo."""
    try:
        if os.path.exists(REPO_ZIP_PATH):
            with open(REPO_ZIP_PATH, "rb") as f:
                st.info("📦 Automatically loading dataset from repository...")
                return load_file(f)
        else:
            st.error(f"❌ ZIP not found in repo: {REPO_ZIP_PATH}")
    except Exception as e:
        st.error(f"❌ Error loading default dataset: {e}")
    return None


# ============================================================
# DATA CLEANING FUNCTIONS
# ============================================================
def normalize_cols(df):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def bin_map(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ['1','yes','y','true','t','liked','subscribed','subscribe']:
        return 1
    if s in ['0','no','n','false','f','not liked']:
        return 0
    try:
        v = float(s)
        if v in [0, 1]:
            return int(v)
    except:
        pass
    return np.nan


@st.cache_data(show_spinner=False)
def clean_df(df):
    df = df.copy()
    df = normalize_cols(df)

    # Object cleanup
    for c in df.select_dtypes(include="object"):
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "none": np.nan})

    # Numeric cleanup
    for c in ["video_duration", "watch_time", "watch_percent"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Binary flags
    for c in ["liked", "commented", "subscribed_after", "recommended", "clicked"]:
        if c in df:
            df[c] = df[c].apply(bin_map).astype("float")

    # Category/device cleanup
    if "category" in df:
        df["category"] = df["category"].astype(str).str.lower() \
            .str.replace(r"[^a-z0-9\s\-]", "", regex=True)

    if "device" in df:
        df["device"] = df["device"].astype(str).str.lower() \
            .str.replace(r"[^a-z0-9\s\-]", "", regex=True) \
            .replace({
                "mobilephone": "mobile",
                "mobile phone": "mobile",
                "iphone": "mobile",
                "ipad": "tablet"
            })

    # Timestamp cleanup
    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["watch_hour"] = df["timestamp"].dt.hour

        def tod(h):
            if pd.isna(h): return "unknown"
            if 5 <= h < 12: return "morning"
            if 12 <= h < 17: return "afternoon"
            if 17 <= h < 21: return "evening"
            return "night"

        df["watch_time_of_day_inferred"] = df["watch_hour"].apply(tod)
    else:
        df["watch_time_of_day_inferred"] = "unknown"

    # Compute watch percent if missing
    if set(["watch_percent", "watch_time", "video_duration"]).issubset(df.columns):
        mask = df["watch_percent"].isna() & df["video_duration"].gt(0)
        df.loc[mask, "watch_percent"] = (
            df.loc[mask, "watch_time"] / df.loc[mask, "video_duration"]
        ) * 100

    # Fix fractional %
    if "watch_percent" in df:
        frac = (df["watch_percent"].dropna() <= 1).mean()
        if frac > 0.5:
            df["watch_percent"] *= 100

        df["watch_percent"] = df["watch_percent"].clip(0, 100)
        df["watch_percent"] = df["watch_percent"].fillna(df["watch_percent"].median())

    df = df.drop_duplicates()

    df["liked"] = df["liked"].fillna(0).astype(int)
    df["subscribed_after"] = df["subscribed_after"].fillna(0).astype(int)

    df["engagement_score"] = (
        df["watch_percent"] + df["liked"] + df["subscribed_after"]
    )

    return df


# ============================================================
# ML MODEL TRAINING
# ============================================================
@st.cache_data(show_spinner=False)
def train_xgb(X, y, sample_size=50000):
    if X.shape[0] > sample_size:
        idx = np.random.choice(X.index, size=sample_size, replace=False)
        Xs, ys = X.loc[idx], y.loc[idx]
    else:
        Xs, ys = X, y

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, ys, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "model": model,
        "y_test": y_test,
        "y_pred": preds,
        "mae": mean_absolute_error(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "r2": r2_score(y_test, preds),
        "fi": pd.Series(model.feature_importances_, index=X_train.columns)
    }


# ============================================================
# START APP
# ============================================================
st.title("📊 YouTube Engagement Analysis Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload CSV or ZIP", type=["csv", "zip"])

# Priority: user upload → else auto-load repo ZIP
if uploaded_file:
    df_raw = load_file(uploaded_file)
else:
    df_raw = load_default_dataset()

if df_raw is None:
    st.error("❌ Could not load dataset. Upload a file or check ZIP in repo.")
    st.stop()

st.success("Dataset loaded successfully!")
st.subheader("Raw Data Preview")
st.dataframe(df_raw.head())


# CLEANING
st.header("🔧 Data Cleaning")
df = clean_df(df_raw)
st.dataframe(df.head())

# BASIC EDA
st.header("📈 Exploratory Data Analysis")

fig, ax = plt.subplots()
sns.histplot(df['watch_percent'], kde=True, bins=30, ax=ax)
st.pyplot(fig)

# RETENTION
st.header("📌 Retention Analysis")
if "category" in df:
    retention = df.groupby("category")["watch_percent"].mean().sort_values(ascending=False)
    st.dataframe(retention.head(15))

# ENGAGEMENT
st.header("⭐ Engagement Score")
fig, ax = plt.subplots()
sns.histplot(df["engagement_score"], kde=True, bins=40, ax=ax)
st.pyplot(fig)

# MODELING
st.header("🧠 XGBoost Model")

numeric = ["watch_time","video_duration","watch_percent","liked","subscribed_after"]
categorical = ["category","device","watch_time_of_day_inferred"]

num = [c for c in numeric if c in df]
cat = [c for c in categorical if c in df]

sel_num = st.multiselect("Select numeric features", num, default=num)
sel_cat = st.multiselect("Select categorical features", cat, default=cat)

model_df = df[sel_num + sel_cat + ["engagement_score"]].dropna()
model_df = pd.get_dummies(model_df, columns=sel_cat, drop_first=True)

X = model_df.drop("engagement_score", axis=1)
y = model_df["engagement_score"]

if st.button("Train Model"):
    res = train_xgb(X, y)

    st.metric("MAE", round(res["mae"], 4))
    st.metric("RMSE", round(res["rmse"], 4))
    st.metric("R²", round(res["r2"], 4))

    fig, ax = plt.subplots(figsize=(8, 5))
    res["fi"].sort_values().tail(20).plot(kind="barh", ax=ax)
    st.pyplot(fig)


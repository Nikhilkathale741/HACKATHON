# -*- coding: utf-8 -*-
"""Streamlit YouTube Engagement Dashboard"""

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
# ZIP + CSV LOADER
# ============================================================
def load_file(uploaded_file):
    """Load CSV or extract CSV from ZIP."""
    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower() if hasattr(uploaded_file, "name") else "repo_zip"

    # CASE 1 → CSV
    if filename.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file, low_memory=True, on_bad_lines="skip")
        except Exception as e:
            st.error(f"❌ Could not read CSV: {e}")
            return None

    # CASE 2 → ZIP
    if filename.endswith(".zip") or True:
        try:
            z = zipfile.ZipFile(uploaded_file)
            files = z.namelist()

            csv_files = [f for f in files if f.lower().endswith(".csv")]

            if len(csv_files) == 0:
                st.error("❌ No CSV file found inside ZIP.")
                return None

            csv_name = csv_files[0]
            st.success(f"📄 Loaded CSV from ZIP: {csv_name}")

            with z.open(csv_name) as f:
                return pd.read_csv(f, low_memory=True, on_bad_lines="skip")

        except:
            pass

    return None


# ============================================================
# AUTO LOAD DATASET FROM REPO
# ============================================================
REPO_ZIP_PATH = "youtube recommendation dataset.zip"

def load_default_dataset():
    """Load dataset ZIP stored in GitHub repo."""
    try:
        if os.path.exists(REPO_ZIP_PATH):
            with open(REPO_ZIP_PATH, "rb") as f:
                st.info("📦 Automatically loading dataset from repository...")
                return load_file(f)
        else:
            st.error(f"❌ Dataset ZIP not found in repo: {REPO_ZIP_PATH}")
    except Exception

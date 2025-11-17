# ============================================================
# STEP 2 — EXACT EDA OUTPUT (MIRRORING YOUR COLAB SCRIPT)
# ============================================================

st.header("📊 EDA Visualizations")

# 1. Watch Percent Distribution
st.subheader("Distribution of Watch Percent")
fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(df['watch_percent'], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# 2. Avg watch_time by category
if 'category' in df.columns:
    st.subheader("Average Watch Time by Category (Top 20)")
    fig, ax = plt.subplots(figsize=(10,5))
    df.groupby("category")["watch_time"].mean().sort_values(ascending=False).head(20).plot(kind="bar", ax=ax)
    st.pyplot(fig)

# 3. Device distribution
if 'device' in df.columns:
    st.subheader("Device Distribution")
    fig, ax = plt.subplots(figsize=(6,6))
    df['device'].value_counts().plot(kind='pie', autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

# 4. Time of Day distribution
st.subheader("Views by Time of Day")
fig, ax = plt.subplots(figsize=(8,4))
df['watch_time_of_day_inferred'].value_counts().plot(kind='bar', color="purple", ax=ax)
st.pyplot(fig)


# ============================================================
# STEP 3 — RETENTION ANALYSIS (EXACT)
# ============================================================

st.header("📌 Retention Analysis — Which Categories Keep Users Watching Longer?")

if 'category' in df.columns:
    retention = df.groupby("category")["watch_percent"].mean().sort_values(ascending=False)

    st.subheader("Top 20 Categories by Average Watch %")
    st.dataframe(retention.head(20))

    fig, ax = plt.subplots(figsize=(10,5))
    retention.head(20).plot(kind="bar", color="green", ax=ax)
    ax.set_ylabel("Avg Watch %")
    st.pyplot(fig)


# ============================================================
# STEP 4 — ENGAGEMENT SCORE DISTRIBUTION
# ============================================================

st.header("⭐ Engagement Score Distribution (Your Exact Output)")
fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(df['engagement_score'], bins=40, ax=ax)
st.pyplot(fig)


# ============================================================
# STEP 5 — XGBOOST MODEL (EXACT AS YOUR COLAB CODE)
# ============================================================

st.header("🧠 XGBoost Model — Predict Engagement Score (Auto-Run)")

st.write("### Numeric features used:")
numeric_features = ['watch_time','video_duration','watch_percent','liked','commented','subscribed_after','recommended']
st.write(numeric_features)

st.write("### Categorical features used:")
cat_features = ['category','device','watch_time_of_day','watch_time_of_day_inferred']
st.write(cat_features)

# Build model dataframe
model_df = df[numeric_features + cat_features + ['engagement_score']].dropna()

# One-hot encode
model_df = pd.get_dummies(model_df, columns=cat_features, dummy_na=True, drop_first=True)

X = model_df.drop("engagement_score", axis=1)
y = model_df["engagement_score"]

# Sampling
RANDOM_STATE = 42
if X.shape[0] > 50000:
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(X.index, size=50000, replace=False)
    X = X.loc[sample_idx]
    y = y.loc[sample_idx]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

st.write(f"**Train shape:** {X_train.shape} — **Test shape:** {X_test.shape}")

# Train XGBoost (fixed)
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=RANDOM_STATE
)

xgb.fit(X_train, y_train)

# Predict
y_pred = xgb.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance Metrics")
st.write("MAE:", mae)
st.write("RMSE:", rmse)
st.write("R²:", r2)

# Feature Importance
st.subheader("Top 20 Important Features")
fi = pd.Series(xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10,6))
fi.head(20).plot(kind='barh', color='skyblue', ax=ax)
ax.invert_yaxis()
st.pyplot(fig)

# Actual vs Predicted
st.subheader("Actual vs Predicted Engagement Score")
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y_test, y_pred, alpha=0.3)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
st.pyplot(fig)

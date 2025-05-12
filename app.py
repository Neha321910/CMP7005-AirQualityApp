# Save as app.py, then run: streamlit run app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_air_quality.csv", parse_dates=['datetime'], index_col='datetime')
    return df

df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "EDA", "Model & Prediction"])

# --- Page 1: Data Overview ---
if page == "Data Overview":
    st.title("Data Overview")
    st.dataframe(df.head())
    st.write(f"Shape: {df.shape}")
    st.write("Missing Values:")
    st.write(df.isnull().sum())

# --- Page 2: EDA ---
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    st.subheader("PM2.5 Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['PM2.5'], bins=50, kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("PM2.5 vs Temperature")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x='TEMP', y='PM2.5', data=df, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    corr_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM']
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

# --- Page 3: Model & Prediction ---
elif page == "Model & Prediction":
    st.title("🤖 Model & Prediction")

    st.write("We use a **Linear Regression model** to predict PM2.5 using weather and gas pollutants.")

    # Select features & target
    features = ['TEMP', 'PRES', 'DEWP', 'WSPM', 'SO2', 'NO2', 'CO', 'O3']
    target = 'PM2.5'

    # Drop NaNs in features and target
    model_df = df[features + [target]].dropna()

    # Train/test split
    X = model_df[features]
    y = model_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.success(f"Model trained. R² Score: {r2:.2f}, MAE: {mae:.2f}")

    # --- Prediction Interface ---
    st.subheader("Make a Prediction")
    user_input = {}
    for feature in features:
        min_val = float(X[feature].min())
        max_val = float(X[feature].max())
        default_val = float(X[feature].mean())
        user_input[feature] = st.slider(f"{feature}", min_value=min_val, max_value=max_val, value=default_val)

    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.markdown(f"### Predicted PM2.5: **{prediction:.2f} µg/m³**")

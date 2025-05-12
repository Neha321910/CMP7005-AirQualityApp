# CMP7005 – Programming for Data Analysis

## 📘 Assessment: From Data to Application Development (PRAC1)

This project addresses the challenge of air pollution prediction using real-world data from Beijing, China. The task involved processing and analyzing large-scale environmental data, building a machine learning model, and creating an interactive GUI application.

---

## 📁 Contents

- `CMP7005_PRAC1_20322354.ipynb`: Jupyter Notebook with all code and commentary for Task 1–3
- `app.py`: Streamlit-based GUI for interactive data analysis and prediction (Task 4)
- `cleaned_air_quality.csv`: Cleaned and merged dataset used throughout
- `screenshots_git_task5/`: Screenshots of version control commits for Task 5
- `README.md`: Project summary and usage instructions

---

## ✅ Tasks Summary

### 🔹 Task 1: Data Handling
- Selected 4 monitoring stations representing urban, suburban, rural, and industrial zones
- Merged hourly air quality data from 2013–2017
- Cleaned and saved merged dataset

### 🔹 Task 2: Exploratory Data Analysis (EDA)
- Investigated missing values and data types
- Created visualizations for PM2.5 distribution, correlations, and scatter plots
- Engineered new features like season

### 🔹 Task 3: Model Building
- Implemented Linear Regression to predict PM2.5
- Evaluated model using R² and MAE
- Used meteorological and gas pollutants as features

### 🔹 Task 4: Application Development
- Built a multi-page GUI using Streamlit
- GUI includes:
  - Data Overview
  - EDA Visualizations
  - Live Prediction using user input

### 🔹 Task 5: Version Control
- Project managed using Git and GitHub
- Frequent commits with meaningful messages
- Screenshots included as evidence

---

## 🚀 How to Run the Streamlit App

Install required packages:
pip install streamlit pandas scikit-learn seaborn matplotlib

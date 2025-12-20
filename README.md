🧠 Digital Wellbeing & Cognitive Fatigue Tracker
An Intelligent Machine Learning Dashboard for Human–Computer Interaction Analysis
🚀 What This Project Does

The Digital Wellbeing & Cognitive Fatigue Tracker is a machine learning–powered application that studies how digital habits affect cognitive health.

Using 525 real-world survey samples, the system learns relationships between screen time, social media usage, sleep quality, stress, and focus to predict cognitive fatigue levels.

All insights are presented through a modern, dark-themed Streamlit dashboard, making complex ML results easy to understand and act upon.

📊 Smart Exploratory Data Analysis

Before training models, the data is carefully explored to uncover meaningful patterns:

🔥 Correlation Heatmaps – Reveal strong links between stress, sleep, and fatigue

🎯 Multi-Feature Scatter Plots – Highlight high-risk digital behavior zones

⚠️ Outlier Detection – Identify extreme usage patterns via box plots

🌊 Violin Plots – Show fatigue density across different stress levels

🧠 Machine Learning Intelligence
🔹 Supervised Learning

Linear Regression – Predicts fatigue score (1–5)

Logistic Regression – Classifies users as Healthy or High Risk

KNN (Optimized) – Tested across multiple K values for best accuracy

SVM – Visualizes user separation in PCA space

Decision Tree – Provides interpretable, rule-based predictions

🔹 Unsupervised Learning

K-Means Clustering – Groups users into digital behavior personas

PCA – Compresses 5D habit data into intuitive 2D visual clusters

🔮 Live Fatigue Predictor

The dashboard includes a real-time fatigue prediction engine.

Users interact with sliders for:

Screen Time (hrs)

Social Media Usage (hrs)

Sleep Quality (1–5)

Stress Level (1–5)

Focus Level (1–5)

✨ The system instantly calculates a Fatigue Risk Percentage using the optimized KNN model.

🛠️ Setup & Run
git clone https://github.com/your-username/digital-wellbeing-tracker.git
cd digital-wellbeing-tracker
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
streamlit run app.py

🎯 Why This Project Matters

This project showcases a complete end-to-end ML pipeline:

Data preprocessing → Advanced EDA → Model training & tuning → Real-time deployment

It proves that digital behavior is a measurable predictor of mental fatigue and demonstrates how machine learning can support healthier digital lifestyles.

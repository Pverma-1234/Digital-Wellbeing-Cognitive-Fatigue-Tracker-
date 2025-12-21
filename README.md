🧠 Digital Wellbeing & Cognitive Fatigue Tracker

A machine learning–based Streamlit application that analyzes smartphone usage patterns and their impact on digital wellbeing, cognitive fatigue, stress, focus, and sleep quality among students and young adults.

The project uses survey data (525 responses, 17 features) and applies multiple machine learning models to generate insights, predictions, and behavior clusters.

🌟 Features

📊 Predict cognitive fatigue levels

🎯 Classify stress & wellbeing states (Confusion Matrix)

🧩 Identify digital behavior patterns using clustering

📈 Interactive visualizations

🌐 Streamlit-based web dashboard

📱 Responsive UI (desktop & mobile)

🛠️ Tech Stack
Machine Learning & Data

Python

scikit-learn

pandas

NumPy

Matplotlib

Seaborn

Dashboard

Streamlit

Plotly

Data Collection

Google Forms (Anonymous Survey)

📋 Prerequisites

Python 3.8 or higher

pip package manager

Modern web browser

🚀 Method 1: Quick Start
'''
git clone https://github.com/yourusername/Digital-Wellbeing-Cognitive-Fatigue-Tracker.git
cd Digital-Wellbeing-Cognitive-Fatigue-Tracker
pip install -r requirements.txt
streamlit run app.py
'''

The app will open automatically at:
'''http://localhost:8501'''

🔧 Method 2: Manual Setup
'''
# 1. Install dependencies
pip install -r requirements.txt

# 2. Load / prepare dataset
# Ensure dataset.csv is present inside the data/ directory

# 3. Start the Streamlit application
streamlit run app.py

# 4. Open the application in your browser
# http://localhost:8501
'''

📁 Project Structure
Digital_Wellbeing_Tracker/
├── app.py                     # Main Streamlit application
├── data/
│   └── dataset.csv            # Survey dataset (525 × 17)
├── models/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── svm_model.py
│   ├── knn_model.py
│   └── clustering.py
├── utils/
│   ├── preprocessing.py
│   └── visualization.py
├── requirements.txt
└── README.md

🎯 Usage Guide
streamlit run app.py


View dataset insights

Run ML models

Analyze predictions

Explore behavior clusters

🤖 Machine Learning Models
Linear Regression

Predicts cognitive fatigue score

Logistic Regression

Classifies stress / fatigue level
Includes confusion matrix

Support Vector Machine (SVM)

Wellbeing state classification

K-Nearest Neighbors (KNN)

Finds similar users based on digital behavior

K-Means Clustering

Identifies digital behavior groups

📊 Dataset Information

Source: Google Forms

Records: 525

Features: 17

Includes:

Screen time

Social media usage

Phone unlock frequency

Mental fatigue (1–5 scale)

Sleep quality

Focus & stress indicators

🔍 Troubleshooting
pip install streamlit


Ensure dataset path is correct

Verify dependencies installation

Restart Streamlit server

🔮 Future Scope

Real-time smartphone usage tracking

Wearable device integration

Deep learning models (LSTM)

Personalized wellbeing recommendations

Cloud deployment

📝 License

MIT License

👨‍💻 Author

Prince Verma
Machine Learning & Data Science

📩 Feedback & Contributions
⭐ Star the repository
🐛 Open issues
🔧 Create pull requests


Feedback is always welcome 🙌

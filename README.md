🧠 Digital Wellbeing & Cognitive Fatigue Tracker

A machine learning–powered Streamlit dashboard designed to analyze smartphone usage patterns and their impact on digital wellbeing, cognitive fatigue, stress, focus, and sleep quality among students and young adults.

This project applies regression, classification, and clustering algorithms on real-world survey data to uncover behavioral insights and predict mental fatigue.

🌟 Features

📊 Predict cognitive fatigue levels using ML models
🎯 Classify stress & wellbeing states with confusion matrix
🧩 Discover digital behavior patterns using clustering
📈 Interactive visualizations and analytics
🌐 Streamlit-based interactive web dashboard
📱 Responsive UI (desktop & mobile)

🛠️ Technologies Used
🧠 Machine Learning & Data Science

Python

scikit-learn

pandas

NumPy

Matplotlib

Seaborn

🌐 Dashboard / Frontend

Streamlit

Plotly

📄 Data Collection

Google Forms (Anonymous Survey)

📋 Prerequisites

Python 3.8+

pip package manager

Modern web browser

Internet connection (optional, for deployment)

🚀 Quick Start
Method 1: Basic Setup (Recommended)
# Clone the repository
git clone https://github.com/yourusername/Digital-Wellbeing-Cognitive-Fatigue-Tracker.git
cd Digital-Wellbeing-Cognitive-Fatigue-Tracker

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py


The app will open at:
👉 http://localhost:8501

🔧 Method 2: Manual Setup
# 1. Install dependencies
pip install -r requirements.txt

# 2. Load / prepare dataset
# Ensure dataset.csv is available inside the data/ directory

# 3. Start the Streamlit application
streamlit run app.py

# 4. Open the application in your browser
# Streamlit will automatically open at:
# http://localhost:8501

📁 Project Structure
Digital_Wellbeing_Tracker/
├── app.py                     # Main Streamlit application
├── data/
│   └── dataset.csv            # Survey dataset (525 × 17)
├── models/
│   ├── linear_regression.py   # Fatigue prediction
│   ├── logistic_regression.py # Classification model
│   ├── svm_model.py           # Support Vector Machine
│   ├── knn_model.py           # KNN model
│   └── clustering.py          # K-Means clustering
├── utils/
│   ├── preprocessing.py       # Data cleaning & encoding
│   └── visualization.py       # Charts & plots
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

🎯 Usage Guide
Using the Dashboard

Launch the app with streamlit run app.py

Explore dataset insights

Run ML models interactively

View predictions, confusion matrix & clusters

Functionalities

📈 Fatigue Prediction

🎯 Stress & Wellbeing Classification

📊 Confusion Matrix Visualization

🧩 Behavior Clustering Analysis

📉 Screen Time vs Focus Trends

🤖 Machine Learning Models
1️⃣ Linear Regression

Purpose: Predict cognitive fatigue score
Metrics: R² Score, Mean Squared Error

2️⃣ Logistic Regression

Purpose: Classify stress/fatigue levels
Metrics: Accuracy, Precision, Recall
Visualization: Confusion Matrix

3️⃣ Support Vector Machine (SVM)

Purpose: Wellbeing state classification

4️⃣ K-Nearest Neighbors (KNN)

Purpose: Identify similar users based on digital behavior

5️⃣ K-Means Clustering

Purpose: Discover hidden digital behavior patterns
Visualization: Cluster plots, Elbow method

📊 Dataset Description

Source: Google Forms

Records: 525

Features: 17

Includes:

Screen time & social media usage

Phone unlock frequency

Mental fatigue (1–5 scale)

Sleep quality & focus levels

Stress & productivity indicators

Fatigue management strategies

🔧 Configuration
Example: Change Clustering Parameters
KMeans(n_clusters=4, random_state=42)

🔍 Troubleshooting
Common Issues

Streamlit not opening

Install Streamlit: pip install streamlit

Dataset not found

Check data/dataset.csv path

Model errors

Verify dependencies using pip install -r requirements.txt

🔮 Future Enhancements

🚀 Real-time smartphone usage tracking
⌚ Wearable device integration
🧠 Deep learning models (LSTM)
📱 Mobile app version
🎯 Personalized wellbeing recommendations

🤝 Contributing

Fork the repository

Create a new branch

Commit your changes

Push to GitHub

Open a Pull Request

📝 License

This project is licensed under the MIT License.

👨‍💻 Author

Prince Verma
Machine Learning & Data Science

📩 Support

If you face any issues:

Check troubleshooting section

Verify dataset & dependencies

Review Streamlit logs

Happy building healthy digital habits 🧠📱✨

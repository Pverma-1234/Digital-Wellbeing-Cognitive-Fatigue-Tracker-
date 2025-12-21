🧠 Digital Wellbeing & Cognitive Fatigue Tracker

A comprehensive machine learning–powered Streamlit dashboard for analyzing digital behavior, cognitive fatigue, stress levels, focus, and overall digital wellbeing among students and young adults.

This project applies regression, classification, and clustering algorithms on real-world survey data to uncover usage patterns and predict mental fatigue caused by smartphone overuse.

🌟 Features

📊 Fatigue Prediction using regression models
🎯 Wellbeing Classification with confusion matrix evaluation
🔍 Behavior Clustering to identify digital usage patterns
🧠 Multi-Model ML Analysis (Regression, Classification, Clustering)
🌐 Interactive Streamlit Dashboard for real-time insights
📱 Responsive UI – works on desktop & mobile
📈 Visual Analytics with charts and graphs

🛠️ Technologies Used
🧠 Machine Learning & Data Science

Python

scikit-learn – ML algorithms

pandas – Data processing

NumPy – Numerical computation

Matplotlib & Seaborn – Visualization

🌐 Frontend / Dashboard

Streamlit – Interactive web application

Plotly – Dynamic visualizations

📄 Data Collection

Google Forms – Anonymous survey data

📋 Prerequisites

Python 3.8 or higher

pip package manager

Internet connection (for Streamlit hosting)

Modern web browser

🚀 Quick Start
Method 1: Automated Setup (Recommended)
# Clone the repository
git clone https://github.com/yourusername/Digital-Wellbeing-Cognitive-Fatigue-Tracker.git
cd Digital-Wellbeing-Cognitive-Fatigue-Tracker

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

Method 2: Manual Setup
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit dashboard
streamlit run app.py


The app will open automatically at:
👉 http://localhost:8501

📁 Project Structure
Digital_Wellbeing_Tracker/
├── app.py                     # Main Streamlit application
├── data/
│   └── dataset.csv            # Survey dataset (525 × 17)
├── models/
│   ├── linear_regression.py   # Fatigue prediction
│   ├── logistic_regression.py # Classification model
│   ├── svm_model.py           # Support Vector Machine
│   ├── knn_model.py           # KNN classification
│   └── clustering.py          # K-Means clustering
├── utils/
│   ├── preprocessing.py       # Data cleaning & encoding
│   └── visualization.py       # Charts & plots
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

🎯 Usage Guide
Running the Dashboard

Launch the Streamlit app using streamlit run app.py

Upload or load the survey dataset

Explore interactive ML results and visualizations

Dashboard Functionalities

📈 Predict Cognitive Fatigue

🎯 Classify Stress & Wellbeing Levels

📊 View Confusion Matrix

🧩 Analyze User Behavior Clusters

📉 Explore Screen Time & Focus Trends

🤖 Machine Learning Models
1️⃣ Linear Regression

Purpose:
Predict cognitive fatigue score based on screen usage patterns

Metrics:

R² Score

Mean Squared Error

2️⃣ Logistic Regression

Purpose:
Classify users into low/high fatigue or stress levels

Metrics:

Accuracy

Precision

Recall

Confusion Matrix

3️⃣ Support Vector Machine (SVM)

Purpose:
Separate users based on wellbeing and focus patterns

Metrics:

Accuracy

Classification Report

4️⃣ K-Nearest Neighbors (KNN)

Purpose:
Identify similar users based on digital behavior

5️⃣ K-Means Clustering

Purpose:
Discover hidden digital usage patterns and behavior groups

Visualization:

Cluster plots

Elbow method

📊 Data Features

The dataset includes:

Total daily screen time

Social media usage duration

Phone unlock frequency

Mental fatigue level (1–5)

Sleep quality rating

Focus level during work/study

Stress & productivity indicators

Cognitive fatigue management strategies

🔧 Configuration
Modify Features or Models

You can adjust model parameters in the models/ directory:

# Example: Change K-Means clusters
KMeans(n_clusters=4, random_state=42)

🔍 Troubleshooting
Common Issues

Streamlit app not opening

Ensure Streamlit is installed

Run: pip install streamlit

Dataset not loading

Check file path in app.py

Ensure dataset.csv exists

Model errors

Verify all dependencies are installed

Check data preprocessing steps

🔮 Future Enhancements

✅ Real-time smartphone usage integration
✅ Wearable device data support
✅ Deep learning models (LSTM)
✅ Personalized wellbeing recommendations
✅ Mobile app integration
✅ Cloud deployment

🤝 Contributing

Fork the repository

Create a new branch (git checkout -b feature-name)

Commit your changes

Push to your branch

Open a pull request

📝 License

This project is open source and available under the MIT License.

👨‍💻 Author

Prince Verma
Machine Learning & Data Science Enthusiast

📩 Support

If you face any issues:

Check the troubleshooting section

Verify dependencies

Review Streamlit logs

Happy learning & building healthy digital habits 🧠📱✨

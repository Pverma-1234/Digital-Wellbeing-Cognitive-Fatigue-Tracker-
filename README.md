# 🧠 Digital Wellbeing & Cognitive Fatigue Tracker
## 🚀 Live Demo  👉 https://fatiguetracker01pv.streamlit.app/

## 📌 About Project
A **machine learning–based Streamlit application** that analyzes smartphone usage patterns and their impact on **digital wellbeing, cognitive fatigue, stress, focus, and sleep quality** among students and young adults.

The project leverages **survey data (525 responses, 17 features)** and applies multiple ML models to generate **predictions, insights, and behavioral clusters**.

---

## 🌟 Features

- 📊 Predict cognitive fatigue levels  
- 🎯 Classify stress & wellbeing states (with confusion matrix)  
- 🧩 Identify digital behavior patterns using clustering  
- 📈 Interactive visualizations  
- 🌐 Streamlit-based web dashboard  
- 📱 Fully responsive UI (desktop & mobile)  

---

## 🛠️ Tech Stack

### 🔹 Machine Learning & Data
- Python  
- scikit-learn  
- pandas  
- NumPy  
- Matplotlib  
- Seaborn  

### 🔹 Dashboard
- Streamlit  
- Plotly  

### 🔹 Data Collection
- Google Forms (Anonymous Survey)

---

## 📦 Dependencies

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
```

---

## 🚀 Installation & Setup

### Method 1: Quick Start

```
git clone https://github.com/yourusername/Digital-Wellbeing-Cognitive-Fatigue-Tracker.git
cd Digital-Wellbeing-Cognitive-Fatigue-Tracker
pip install -r requirements.txt
streamlit run app.py
```

Open in browser:
```
http://localhost:8501
```

---

### Method 2: Manual Setup

```
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
Digital_Wellbeing_Tracker/
│── app.py
│── data/
│   └── dataset.csv
│── models/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── svm_model.py
│   ├── knn_model.py
│   ├── decision_tree.py
│   └── clustering.py
│── utils/
│   ├── preprocessing.py
│   └── visualization.py
│── requirements.txt
│── README.md
```

---

## 🤖 Machine Learning Models

### 🔹 Regression
- Linear Regression → Predicts cognitive fatigue score  

### 🔹 Classification
- KNN → Finds similar behavior patterns  
- SVM → Wellbeing classification  
- Logistic Regression → Stress/fatigue classification  
- Decision Tree → Rule-based classification  

### 🔹 Clustering
- K-Means Clustering → Identifies digital behavior groups  

---

## 📊 Model Performance

| Model                  | Accuracy | Rank |
|----------------------|----------|------|
| Logistic Regression  | 0.9400   | 🥇 1 |
| SVM                  | 0.9300   | 🥈 2 |
| KNN                  | 0.9100   | 🥉 3 |
| Random Forest        | 0.8800   |  4  |
| Decision Tree        | 0.7429   |  5  |

---

## 📊 Dataset Information

- Source: Google Forms  
- Records: 520+  
- Features: 17  

Includes:
- Screen time  
- Social media usage  
- Phone unlock frequency  
- Mental fatigue (1–5 scale)  
- Sleep quality  
- Focus & stress indicators  

---

## 🎯 Usage Guide

```
streamlit run app.py
```

---

## 🔍 Troubleshooting

```
pip install streamlit
```

- Ensure dataset path is correct  
- Verify dependencies installation  
- Restart Streamlit server  

---

## 🔮 Future Scope

- Real-time smartphone usage tracking  
- Wearable device integration  
- Deep learning models (LSTM)  
- Personalized wellbeing recommendations  
- Cloud deployment  

---

## 📝 License

MIT License  

---

## 👨‍💻 Author

Prince Verma  
Machine Learning & Data Science  

---

## 📩 Feedback & Contribution

- ⭐ Star the repository  
- 🐛 Open issues  
- 🔧 Create pull requests  

Feedback is always welcome 🙌

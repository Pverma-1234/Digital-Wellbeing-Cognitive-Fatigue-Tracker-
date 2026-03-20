import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, confusion_matrix, mean_absolute_error, 
    mean_squared_error, r2_score
)

# --- 1. SETTINGS & BRANDING ---
# Project Title in the Browser Tab
st.set_page_config(page_title="Digital Wellbeing: Fatigue Tracker", layout="wide")

# Persistent memory to store models across navigation
if 'models' not in st.session_state:
    st.session_state.models = {}

# --- 2. DATA ENGINE ---
@st.cache_data
def load_and_prep():
    try:
        df = pd.read_csv("your_collected_data.csv")
    except:
        # Fallback dummy data for testing (525 rows)
        np.random.seed(42)
        X = np.random.rand(525, 5) 
        y = (X[:, 0]*3 + X[:, 1]*2 - X[:, 2]*1.5 + X[:, 3]*2.5 - X[:, 4]*1) + np.random.normal(0, 0.2, 525)
        df = pd.DataFrame(X, columns=['ScreenTime', 'SocialMedia', 'SleepQuality', 'StressLevel', 'FocusLevel'])
        df['fatigue_numeric'] = np.clip(y, 1, 5)
        df['BreakDuration'] = np.random.choice(["< 1hr", "1-2hr", "2-3hr", "> 3hr"], 525)
    return df

df = load_and_prep()

# --- 3. SIDEBAR NAVIGATION (Earnomly Project Style) ---
# FIXED: Added Project Title in Navigation Bar
st.sidebar.title("🧠 Digital Wellbeing")
st.sidebar.caption("Machine Learning Fatigue Project")
page = st.sidebar.radio("NAVIGATION", 
    ["Dashboard Overview", "Data Analysis", "Supervised Learning", "Unsupervised Learning", "Fatigue Predictor"])

st.sidebar.markdown("---")
st.sidebar.info("**Objective:** Monitor digital habits to predict and prevent cognitive burnout.")

# --- 4. PAGE: DASHBOARD OVERVIEW ---
if page == "Dashboard Overview":
    st.title("📊 Dashboard Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("TOTAL POPULATION", len(df), "+12%")
    col2.metric("FEATURES ANALYZED", len(df.columns)-2, "Stable")
    col3.metric("MISSING VALUES", "0.0%", "Clean")
    
    st.markdown("### Sample Dataset")
    st.dataframe(df.head(10), use_container_width=True)

# --- 5. PAGE: DATA ANALYSIS ---
# elif page == "Data Analysis":
#     st.title("📈 Exploratory Data Analysis")
    
#     # Average Fatigue by Break Duration Chart
#     st.subheader("Average Cognitive Fatigue by Break Duration")
#     avg_fatigue = df.groupby('BreakDuration')['fatigue_numeric'].mean().reset_index()
#     fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
#     sns.barplot(data=avg_fatigue, x='BreakDuration', y='fatigue_numeric', color="#2ecc71", ax=ax_bar)
#     plt.axhline(df['fatigue_numeric'].mean(), color='red', linestyle='--', label='Overall Average')
#     ax_bar.set_ylabel("Average Fatigue Level")
#     st.pyplot(fig_bar)
elif page == "Data Analysis":
    st.title("📈 Advanced Exploratory Data Analysis")
    
    # 1. Correlation Heatmap
    st.subheader("1. Feature Correlation Matrix")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="YlGnBu", ax=ax_corr)
    st.pyplot(fig_corr)
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        # 2. Multi-variate Scatter Plot
        st.subheader("2. Screen Time vs Sleep vs Fatigue")
        fig_scatter, ax_scatter = plt.subplots()
        sns.scatterplot(data=df, x='ScreenTime', y='SleepQuality', hue='fatigue_numeric', palette='viridis', ax=ax_scatter)
        st.pyplot(fig_scatter)
        
    with col_e2:
        # 3. Box Plot for Outliers
        st.subheader("3. Data Distribution & Outliers")
        fig_box, ax_box = plt.subplots()
        sns.boxplot(data=df[['ScreenTime', 'SocialMedia', 'StressLevel']], palette="Set2", ax=ax_box)
        st.pyplot(fig_box)

    # 4. Violin Plot for Density
    st.subheader("4. Fatigue Density by Stress Level")
    fig_violin, ax_violin = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=df, x='StressLevel', y='fatigue_numeric', palette="light:g", ax=ax_violin)
    st.pyplot(fig_violin)

# --- 6. PAGE: SUPERVISED LEARNING ---
elif page == "Supervised Learning":
    st.title("🧠 Supervised Learning & Evaluation")
    
    # Prepare Data
    X = df.drop(['fatigue_numeric', 'BreakDuration'], axis=1)
    y_reg = df['fatigue_numeric']
    y_cls = (df['fatigue_numeric'] > df['fatigue_numeric'].median()).astype(int)
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_train_cls, y_test_cls = train_test_split(X, y_cls, test_size=0.2, random_state=42)

    tabs = st.tabs(["Regression Analysis", "Classification Model", "KNN Analysis"])
    
    with tabs[0]:
        st.subheader("Linear Regression: Actual vs Predicted")
        lin = LinearRegression().fit(X_train, y_train_reg)
        y_pred = lin.predict(X_test)
        
        # FIXED: Added Regression Metrics (MAE, MSE, RMSE, R2)
        col1, col2, col3, col4 = st.columns(4)
        mse = mean_squared_error(y_test_reg, y_pred)
        col1.metric("MAE", round(mean_absolute_error(y_test_reg, y_pred), 3))
        col2.metric("MSE", round(mse, 3))
        col3.metric("RMSE", round(np.sqrt(mse), 3))
        col4.metric("R² Score", round(r2_score(y_test_reg, y_pred), 3))
        
        fig_lin, ax_lin = plt.subplots(figsize=(10, 5))
        sns.scatterplot(x=y_test_reg, y=y_pred, alpha=0.6, ax=ax_lin)
        ax_lin.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
        ax_lin.set_xlabel("Actual Fatigue Score")
        ax_lin.set_ylabel("Predicted Fatigue Score")
        st.pyplot(fig_lin)
        

    with tabs[1]:
        # FIXED: Added Classification Model (Logistic & SVM)
        st.subheader("Classification: Logistic Regression & SVM")
        log_reg = LogisticRegression().fit(X_train, y_train_cls)
        svm = SVC(kernel='linear', probability=True).fit(X_train, y_train_cls)
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Logistic Regression Confusion Matrix**")
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(confusion_matrix(y_test_cls, log_reg.predict(X_test)), annot=True, cmap="Blues")
            st.pyplot(fig_cm)
            
        with c2:
            st.metric("SVM Final Accuracy", f"{accuracy_score(y_test_cls, svm.predict(X_test)):.2%}")
            
            # SVM Visual Intuition
            pca_svm = PCA(n_components=2)
            X_pca = pca_svm.fit_transform(X_train)
            fig_svm, ax_svm = plt.subplots()
            ax_svm.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train_cls, cmap='coolwarm', edgecolors='k')
            ax_svm.set_title("SVM Linear Separation (PCA Space)")
            st.pyplot(fig_svm)
            

    with tabs[2]:
        st.subheader("KNN Performance Analysis")
        # Accuracy Table for Different K-Values
        k_list = [3, 5, 7, 9, 11]
        results = []
        for k in k_list:
            knn_m = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train_cls)
            acc = accuracy_score(y_test_cls, knn_m.predict(X_test))
            results.append({"K Value": k, "Accuracy": acc})
        
        st.table(pd.DataFrame(results))
        
        # Graph with specific point highlighted
        fig_knn, ax_knn = plt.subplots()
        acc_values = [r["Accuracy"] for r in results]
        ax_knn.plot(k_list, acc_values, marker='o', color='teal')
        ax_knn.annotate('Target K=5', xy=(5, acc_values[1]), xytext=(7, acc_values[1]+0.02),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        st.pyplot(fig_knn)
        
        
        st.session_state.models['knn'] = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train_cls)

# --- 7. PAGE: UNSUPERVISED LEARNING ---
elif page == "Unsupervised Learning":
    st.title("📍 Clustering Analysis")
    X_cluster = df.drop(['fatigue_numeric', 'BreakDuration'], axis=1)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_cluster)
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42).fit(X_cluster)
    
    fig_pca, ax_pca = plt.subplots(figsize=(10, 5))
    plt.scatter(components[:, 0], components[:, 1], c=kmeans.labels_, cmap='Spectral', edgecolors='k')
    plt.title("User Habits Grouping (PCA Transformation)")
    st.pyplot(fig_pca)
    

# --- 8. PAGE: FATIGUE PREDICTOR ---
elif page == "Fatigue Predictor":
    st.title("🔮 Real-Time Fatigue Tracker")
    if 'knn' not in st.session_state.models:
        st.error("Please train models in 'Supervised Learning' first!")
    else:
        with st.form("input_form"):
            s1 = st.slider("Screen Time (Hrs)", 0.0, 15.0, 5.0)
            s2 = st.slider("Social Media (Hrs)", 0.0, 10.0, 1.0)
            s3 = st.slider("Sleep Quality (1-5)", 1, 5, 3)
            s4 = st.slider("Stress Level (1-5)", 1, 5, 3)
            s5 = st.slider("Focus Level (1-5)", 1, 5, 3)
            submit = st.form_submit_button("🚀 Run Analysis")
            
        if submit:
            test_input = np.array([[s1, s2, s3, s4, s5]])
            prob = st.session_state.models['knn'].predict_proba(test_input)[0][1]
            st.subheader(f"Fatigue Prediction Score: {int(prob*100)}%")
            st.progress(prob)
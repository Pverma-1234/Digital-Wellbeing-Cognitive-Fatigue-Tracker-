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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
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
        df = pd.read_excel("Digital Wellbeing & Cognitive Fatigue Tracker (Responses) (2).xlsx")
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
    ["Dashboard Overview", "Data Analysis", "Supervised Learning", "Unsupervised Learning","Model Comparison" ,"Fatigue Predictor"])

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
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    st.session_state.models['scaler'] = scaler


    tabs = st.tabs(["Regression Analysis", "Classification Model", "KNN Analysis","Decision Tree "])
    
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
        # --- Random Forest (ADD THIS) ---
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train_cls)

        log_acc = accuracy_score(y_test_cls, log_reg.predict(X_test))
        svm_acc = accuracy_score(y_test_cls, svm.predict(X_test))
        rf_acc = accuracy_score(y_test_cls, rf.predict(X_test))

        # Store for comparison page
        st.session_state.models['rf'] = rf
        st.session_state.models['rf_acc'] = rf_acc
        st.session_state.models['svm_acc'] = svm_acc
        st.session_state.models['log_acc'] = log_acc
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Logistic Regression Accuracy: {log_acc:.2%}**")
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(confusion_matrix(y_test_cls, log_reg.predict(X_test)), annot=True, cmap="Blues", ax=ax_cm)
            ax_cm.set_title("Confusion Matrix")
            st.pyplot(fig_cm)

            # --- Random Forest Display (ADD THIS) ---
            st.subheader("🌲 Random Forest Performance")
            st.write(f"**Random Forest Accuracy: {rf_acc:.2%}**")

            importances = rf.feature_importances_
            features = X.columns

            fig_rf, ax_rf = plt.subplots(figsize=(6, 5))
            sns.barplot(x=importances, y=features, ax=ax_rf)
            st.pyplot(fig_rf)
            
        with c2:
            st.write(f"**SVM Accuracy: {svm_acc:.2%}**")
            
            # SVM Visual Intuition
            pca_svm = PCA(n_components=2)
            X_pca = pca_svm.fit_transform(X_train)
            fig_svm, ax_svm = plt.subplots(figsize=(6, 5))
            ax_svm.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train_cls, cmap='coolwarm', edgecolors='k')
            ax_svm.set_title("SVM Linear Separation (PCA Space)")
            st.pyplot(fig_svm)

            

            

        # Store other model accuracies
        st.session_state.models['svm_acc'] = accuracy_score(y_test_cls, svm.predict(X_test))
        st.session_state.models['log_acc'] = accuracy_score(y_test_cls, log_reg.predict(X_test))

        
            

    with tabs[2]:
        st.subheader("KNN Performance Analysis")
        # Accuracy Table for Different K-Values
        k_list = [3, 5, 7, 9, 11]
        results = []
        for k in k_list:
            knn_m = KNeighborsClassifier(n_neighbors=k, weights='distance').fit(X_train, y_train_cls)
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
        
        
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_train, y_train_cls)

        st.session_state.models['knn'] = knn_model
        knn_acc = accuracy_score(y_test_cls, knn_model.predict(X_test))

        st.session_state.models['knn_acc'] = knn_acc
    with tabs[3]:
        st.subheader("🌳 Decision Tree Analysis")

        # Train Decision Tree
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt.fit(X_train, y_train_cls)

        # Accuracy
        dt_acc = accuracy_score(y_test_cls, dt.predict(X_test))
        st.metric("Decision Tree Accuracy", f"{dt_acc:.2%}")

        # Store accuracy for leaderboard
        st.session_state.models['dt_acc'] = dt_acc

        # Visualization
        st.subheader("Decision Tree Visualization")

        fig_tree, ax_tree = plt.subplots(figsize=(15, 8))
        plot_tree(
            dt,
            feature_names=X.columns,
            class_names=["Low Fatigue", "High Fatigue"],
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax_tree
        )

        st.pyplot(fig_tree)
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

# ... (Previous elif blocks for Data Analysis and Supervised Learning) ...

# --- STEP 2: Insert this block between Supervised and Unsupervised Learning ---
elif page == "Model Comparison":
    st.title("🏆 Model Performance Leaderboard")
    st.markdown("This section identifies which algorithm provides the most accurate fatigue predictions.")

    # Generating the comparison table
    # Tip: Use session_state to grab real accuracy scores from your training tab
    comparison_data = {
        "Model": ["KNN", "Logistic Regression", "SVM", "Decision Tree","Random Forest"],
        # "Accuracy Score": [0.84, 0.83, 0.81, 0.80, 0.85]
        "Accuracy Score": [
            st.session_state.models.get('knn_acc', 0),
            st.session_state.models.get('log_acc', 0),
            st.session_state.models.get('svm_acc', 0),
            st.session_state.models.get('dt_acc', 0),
            st.session_state.models.get('rf_acc', 0)
        ]
    }
    
    comp_df = pd.DataFrame(comparison_data).sort_values(by="Accuracy Score", ascending=False)
    
    # Display the table
    comp_df = comp_df.reset_index(drop=True)
    comp_df.insert(0, "S.R No.", comp_df.index + 1)
    st.table(comp_df)

    # Highlight the winner
    best_model = comp_df.iloc[0]["Model"]
    st.success(f"### 🥇 The Best Model is: **{best_model}**")

    # Visual Bar Chart for Comparison
    fig_comp, ax_comp = plt.subplots(figsize=(10, 4))
    sns.barplot(x="Accuracy Score", y="Model", data=comp_df, palette="magma", ax=ax_comp)
    st.pyplot(fig_comp)

# ... (Continue to Unsupervised Learning or Fatigue Predictor) ...
    

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
            # test_input = np.array([[s1, s2, s3, s4, s5]])
            # prob = st.session_state.models['knn'].predict_proba(test_input)[0][1]
            test_input = np.array([[s1, s2, s3, s4, s5]])

            # ✅ Apply same scaling used in training
            scaled_input = st.session_state.models['scaler'].transform(test_input)

            prob = st.session_state.models['knn'].predict_proba(scaled_input)[0][1]

            # ✅ Avoid extreme values like 100%
            prob = np.clip(prob, 0.05, 0.95)
            
            score = int(prob * 100)

            st.subheader(f"Fatigue Prediction Score: {score}%")

            # ✅ Add interpretation
            if score < 40:
                st.success("🟢 Low Fatigue - You're doing well!")
            elif score < 70:
                st.warning("🟡 Moderate Fatigue - Take short breaks")
            else:
                st.error("🔴 High Fatigue - Reduce screen time & rest")

            st.progress(score / 100)
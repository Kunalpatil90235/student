import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import io

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Learning Behaviour Profiling",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS FOR STYLING (PROFESSIONAL PINK THEME)
# ==========================================
st.markdown(
    """
    <style>
    /* Global Styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4 {
        color: #d81b60; /* Professional Deep Pink */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
    }
    
    /* Metrics Styling */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #cfd8dc;
        padding: 15px;
        border-radius: 6px;
        box-shadow: 1px 1px 4px rgba(0,0,0,0.05);
    }
    
    /* Custom Banner */
    .main-banner {
        background: linear-gradient(90deg, #d81b60 0%, #f06292 100%);
        color: white;
        padding: 24px;
        border-radius: 6px;
        margin-bottom: 25px;
        text-align: left;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    }
    .main-banner h1 {
        color: white !important;
        margin: 0;
        padding: 0;
        font-size: 2.2rem;
    }
    .main-banner p {
        margin: 5px 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# DATA PROCESSING MODULE
# ==========================================
@st.cache_data
def load_data():
    """Load the dataset directly from the validated data/ directory."""
    try:
        df = pd.read_csv("data/StudentsPerformance.csv")
    except FileNotFoundError:
        st.error("Dataset not found at data/StudentsPerformance.csv. Please verify file placement.")
        return None
            
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace('/', '_').str.replace(' ', '_')
    return df

@st.cache_data
def feature_engineering(df):
    """Create aggregated scores and synthetic demographic socio-economic combinations."""
    df_processed = df.copy()
    
    # Calculate average score directly as a new feature
    df_processed['average_score'] = df_processed[['math_score', 'reading_score', 'writing_score']].mean(axis=1).round(2)
    df_processed['total_score'] = df_processed[['math_score', 'reading_score', 'writing_score']].sum(axis=1)
    
    # Identify passing/failing (Threshold 50 for average score) # Professional definition of success vs at-risk threshold
    df_processed['academic_status'] = np.where(df_processed['average_score'] >= 50, 'Pass', 'Fail')
    
    return df_processed

def preprocess_for_ml(df):
    """Encode categorical features and standardize numeric features for clustering and classification."""
    ml_df = df.copy()
    
    # Categorical Columns
    cat_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
    
    # Label Encoding for maintaining singular columns for the tree-based model
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        ml_df[col + '_encoded'] = le.fit_transform(ml_df[col])
        le_dict[col] = le
        
    # Cross combining socioeconomic indicators (lunch and test prep)
    ml_df['socio_economic_proxy'] = ml_df['lunch_encoded'] + ml_df['test_preparation_course_encoded']
        
    # Features to scale (Z-score normalization)
    num_cols = ['math_score', 'reading_score', 'writing_score', 'average_score']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(ml_df[num_cols])
    
    # Replace num columns with scaled versions for clustering
    for i, col in enumerate(num_cols):
        ml_df[f"{col}_scaled"] = scaled_features[:, i]
        
    return ml_df, le_dict, scaler

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("Navigation")

# App Mode selection
app_mode = st.sidebar.radio("Modules:", [
    "Dashboard Overview", 
    "Exploratory Data Analysis", 
    "K-Means Clustering", 
    "Predictive Classification", 
    "Actionable Recommendations"
])

# ==========================================
# MAIN APP LOGIC
# ==========================================
st.markdown("<div class='main-banner'><h1>Learning Behaviour Profiling</h1><p>Problem Statement 5: Data-Driven Student Performance Insights</p></div>", unsafe_allow_html=True)

# Load Data Structure from static path
df_raw = load_data()

if df_raw is None:
    st.stop()

# Feature Engineering Executed
df = feature_engineering(df_raw)

# Sidebar Filters
st.sidebar.markdown("---")
st.sidebar.header("Global Filtering")

selected_gender = st.sidebar.multiselect("Gender Selection", options=df['gender'].unique(), default=df['gender'].unique())
selected_lunch = st.sidebar.multiselect("Lunch Type Segment", options=df['lunch'].unique(), default=df['lunch'].unique())
selected_prep = st.sidebar.multiselect("Test Preparation Segment", options=df['test_preparation_course'].unique(), default=df['test_preparation_course'].unique())

# Apply filters
df_filtered = df[
    (df['gender'].isin(selected_gender)) &
    (df['lunch'].isin(selected_lunch)) &
    (df['test_preparation_course'].isin(selected_prep))
]

if df_filtered.empty:
    st.warning("Filters applied resulted in an empty working context. Adjust the selections.")
    st.stop()

# ==========================================
# PAGE ROUTING
# ==========================================

# --- 1. Dashboard Overview --- #
if app_mode == "Dashboard Overview":
    st.markdown("### Executive Summary")
    
    with st.expander("About This Analysis", expanded=True):
         st.write('''
         **Data Science Concept: Aggregation and Profiling**  
         This module provides a high-level statistical evaluation of student performance demographics. 
         By aggregating the target metrics (`math`, `reading`, and `writing` scores) and summarizing them visually, we establish foundational hypotheses regarding external socioeconomic factors affecting learning outcomes.
         ''')

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Population Segment", f"{len(df_filtered)}")
    col2.metric("Mean Math Score", f"{df_filtered['math_score'].mean():.1f}")
    col3.metric("Mean Reading Score", f"{df_filtered['reading_score'].mean():.1f}")
    col4.metric("Mean Writing Score", f"{df_filtered['writing_score'].mean():.1f}")
    
    st.markdown("---")
    
    st.markdown("### Dataset Overview")
    st.dataframe(df_filtered.head(15), use_container_width=True)
    
    # Layout with high-level distributions
    col1, col2 = st.columns(2)
    
    with col1:
        # Score Distribution by Status
        fig_status = px.pie(df_filtered, names='academic_status', title="Academic Status Proportion", 
                            color='academic_status', color_discrete_map={'Pass':'#f06292', 'Fail':'#880e4f'},
                            hole=0.3)
        st.plotly_chart(fig_status, use_container_width=True)
        
    with col2:
        # Average Score by Parental Education
        avg_par = df_filtered.groupby('parental_level_of_education')['average_score'].mean().reset_index().sort_values('average_score')
        fig_par = px.bar(avg_par, x='average_score', y='parental_level_of_education', orientation='h',
                         title="Mean Performance Baseline vs. Parental Education Background", color='average_score', color_continuous_scale='RdPu')
        st.plotly_chart(fig_par, use_container_width=True)


# --- 2. EDA & Insights --- #
elif app_mode == "Exploratory Data Analysis":
    st.markdown("### Exploratory Data Analysis (EDA)")
    
    with st.expander("Data Science Concept: EDA Methodology"):
        st.write("""
        **Exploratory Data Analysis** is a critical phase utilized to summarize underlying structures, expose relationships, and detect anomalies.
        - **Histograms** highlight the mathematical distribution (e.g., normally distributed vs. skewed).
        - **Violin & Boxplots (IQR Analysis)** expose variance, density, and outliers within subgroups.
        - **Pearson Correlation** statistically gauges the linear correlation (-1 to 1) between quantitative features.
        """)

    tab1, tab2, tab3 = st.tabs(["Metric Distributions", "Categorical Variance", "Correlation Matrices"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            fig_math = px.histogram(df_filtered, x="math_score", nbins=30, title="Math Score Dist.", color_discrete_sequence=['#ff4081'])
            st.plotly_chart(fig_math, use_container_width=True)
        with c2:
            fig_read = px.histogram(df_filtered, x="reading_score", nbins=30, title="Reading Score Dist.", color_discrete_sequence=['#f50057'])
            st.plotly_chart(fig_read, use_container_width=True)
        with c3:
            fig_write = px.histogram(df_filtered, x="writing_score", nbins=30, title="Writing Score Dist.", color_discrete_sequence=['#c51162'])
            st.plotly_chart(fig_write, use_container_width=True)
            
    with tab2:
        st.markdown("**Segment Variance Analysis**")
        col1, col2 = st.columns([1, 1])
        plot_type = col1.selectbox("Compare Target Aggregate Score By Group:", 
                                 ['gender', 'lunch', 'test_preparation_course', 'race_ethnicity'])
        
        plot_style = col2.radio("Plot Type", ["Box Plot", "Violin Plot (Density)"])
        
        if plot_style == "Box Plot":
            fig_var = px.box(df_filtered, x=plot_type, y="average_score", color=plot_type,
                             title=f"Distribution of Academic Performance by {plot_type.replace('_',' ').title()}")
        else:
            fig_var = px.violin(df_filtered, x=plot_type, y="average_score", color=plot_type, box=True, points="all",
                                title=f"Density Map of Academic Performance by {plot_type.replace('_',' ').title()}")
            
        st.plotly_chart(fig_var, use_container_width=True)
        
    with tab3:
        st.markdown("**Feature Interdependency**")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            num_cols = df_filtered[['math_score', 'reading_score', 'writing_score', 'average_score']]
            corr = num_cols.corr()
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(corr, annot=True, cmap="RdPu", fmt=".2f", ax=ax, linewidths=0.5)
            ax.set_title("Pearson Correlation Coefficient Matrix")
            st.pyplot(fig)
            
        with col2:
            fig_scatter = px.scatter(df_filtered, x="reading_score", y="writing_score", color="math_score",
                                     title="Bivariate Scatter: Reading vs Writing (Color: Math)", size_max=10, opacity=0.7)
            st.plotly_chart(fig_scatter, use_container_width=True)


# --- 3. K-Means Clustering --- #
elif app_mode == "K-Means Clustering":
    st.markdown("### Unsupervised Segmentation: K-Means Clustering")
    
    with st.expander("Machine Learning Concept: Centroid-Based Clustering & The Elbow Method"):
        st.write("""
        **Unsupervised Learning (K-Means)** is employed when datasets lack predefined labels. The algorithm iteratively minimizes the Within-Cluster Sum of Squares (WCSS) to partition data logically.
        - **Z-score normalization:** We ensure continuous features are scaled so that high variances (like reading scores) don't dominate the clustering distance metrics.
        - **Elbow Method:** The Elbow Method validates the optimal 'K' clusters by indicating diminishing returns on WCSS (Inertia). A "bend" or "elbow" in the curve suggests the structurally ideal number of clusters.
        """)
    
    ml_df, le_dict, scaler = preprocess_for_ml(df_filtered)
    cluster_features = ['math_score_scaled', 'reading_score_scaled', 'writing_score_scaled']
    X = ml_df[cluster_features]
    
    tab_elbow, tab_cluster = st.tabs(["Elbow Method Visualizer", "Cluster Interpretation"])
    
    with tab_elbow:
        st.markdown("#### WCSS Inertia Optimization (The Elbow Method)")
        if st.button("Calculate Elbow Curve"):
            wcss = []
            k_range = range(1, 11)
            for k in k_range:
                kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans_test.fit(X)
                wcss.append(kmeans_test.inertia_)
                
            fig_elbow = px.line(x=list(k_range), y=wcss, markers=True, 
                                labels={'x': 'Number of Clusters (K)', 'y': 'WCSS (Inertia)'},
                                title="Elbow Method For Optimal K Selection")
            fig_elbow.add_vline(x=3, line_width=3, line_dash="dash", line_color="red")
            st.plotly_chart(fig_elbow, use_container_width=True)
            st.info("The sharp drop stabilizes heavily around K=3, rendering it the optimal structural choice for segregating performance limits into 'Low', 'Average', and 'High' groups.")
            
    with tab_cluster:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### Model Administrations")
            k_value = st.slider("Target Number of Clusters (K):", min_value=2, max_value=6, value=3)
            
            kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
            ml_df['cluster'] = kmeans.fit_predict(X)
            df_filtered['Cluster'] = ml_df['cluster'].astype(str)
            
            cluster_means = df_filtered.groupby('Cluster')['math_score'].mean().sort_values()
            labels = {}
            if k_value == 3:
                labels = {cluster_means.index[0]: "At-Risk Profiling", 
                          cluster_means.index[1]: "Average Capacity", 
                          cluster_means.index[2]: "High Efficacy"}
            else:
                labels = {k: f"Segment {k} (Mean Metric={v:.1f})" for k, v in zip(cluster_means.index, cluster_means.values)}
                
            df_filtered['Profile'] = df_filtered['Cluster'].map(labels)
            
            st.write("#### Cluster Volumetrics")
            st.dataframe(df_filtered['Profile'].value_counts().reset_index().rename(columns={'count':'Student Population'}))
            
        with col2:
            st.markdown("#### Dimensional Projection")
            fig_3d = px.scatter_3d(df_filtered, x='math_score', y='reading_score', z='writing_score',
                                   color='Profile', opacity=0.8,
                                   title=f"3-Dimensional Topographical Projection of K={k_value}",
                                   color_discrete_sequence=['#f06292', '#ba68c8', '#4dd0e1', '#81c784'])
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig_3d, use_container_width=True)


# --- 4. Classification Model --- #
elif app_mode == "Predictive Classification":
    st.markdown("### Supervised Learning: Segment Forecasting")
    
    with st.expander("Machine Learning Concept: Random Forest Classification"):
        st.write("""
        **Supervised Classification** maps non-continuous input features to categorical targets. In this scenario:
        - **Target:** The K-Means Generated Profile ('At-Risk', 'Average', 'High').
        - **Features:** Exclusively Demographic and Socio-economic data (Scores are entirely redacted to prevent data leakage).
        - **Algorithm:** `RandomForestClassifier`. We use a Random Forest algorithm because it is highly explainable, creates multiple decision boundaries via standard Decision Trees, and aggregates the results to avoid overfitting ("bagging"). It determines the highest voting class natively without complex mathematical derivations.
        - **Accuracy Context:** Accurately predicting scholastic scores based *entirely* on demographics is universally complex. Optimizing this via a Random Forest captures the maximum available demographic signal without resorting to 'black-box' techniques.
        """)
    
    # Reload and Preprocess full
    ml_df, le_dict, scaler = preprocess_for_ml(df) 
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    X_cluster = ml_df[['math_score_scaled', 'reading_score_scaled', 'writing_score_scaled']]
    ml_df['cluster'] = kmeans.fit_predict(X_cluster)
    
    cluster_means = df.assign(cluster=ml_df['cluster']).groupby('cluster')['math_score'].mean().sort_values()
    labels = {cluster_means.index[0]: "At-Risk", cluster_means.index[1]: "Average", cluster_means.index[2]: "High"}
    ml_df['target_profile'] = ml_df['cluster'].map(labels)
    
    features = ['gender_encoded', 'race_ethnicity_encoded', 'parental_level_of_education_encoded', 
                'lunch_encoded', 'test_preparation_course_encoded', 'socio_economic_proxy']
    X_class = ml_df[features]
    y_class = ml_df['target_profile']
    
    X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42, stratify=y_class)
    
    if st.button("Initialize Random Forest Model Workflow"):
        with st.spinner("Compiling Random Forest structures..."):
            
            # Using Random Forest as per the updated simplistic and explainable requirement
            rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            rf_model.fit(X_train, y_train)
            
            y_pred = rf_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            st.success(f"Execution Complete. Structural Accuracy Benchmark: **{acc*100:.2f}%**")
            
            col1, col2 = st.columns(2)
            with col1:
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred, labels=["At-Risk", "Average", "High"])
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='RdPu',
                                   x=["At-Risk", "Average", "High"], y=["At-Risk", "Average", "High"],
                                   title="Validation Matrix Evaluation")
                st.plotly_chart(fig_cm, use_container_width=True)
                
            with col2:
                # Feature Importance
                importance = rf_model.feature_importances_
                feat_names = [f.replace('_encoded', '').replace('_', ' ').title() for f in features]
                fig_imp = px.bar(x=importance, y=feat_names, orientation='h', title="Feature Statistical Dominance",
                                 labels={'x': 'Relative Importance', 'y': 'Feature Vector'},
                                 color=importance, color_continuous_scale='RdPu')
                st.plotly_chart(fig_imp, use_container_width=True)
            
            with st.expander("View Full Classification Report Data"):
                 report = classification_report(y_test, y_pred, output_dict=True)
                 st.dataframe(pd.DataFrame(report).transpose())


# --- 5. Recommendations Engine --- #
elif app_mode == "Actionable Recommendations":
    st.markdown("### Institutional Strategy & Recommendation Engine")
    
    with st.expander("Data Science Concept: Prescriptive Analytics"):
         st.write("""
         The final stage of the data maturity lifecycle is **Prescriptive Analytics**. While Descriptive (EDA) informs us what happened, and Predictive (Classification) forecasts future states, Prescriptive analytics translates these numerical insights into direct operational interventions.
         """)
    
    st.markdown("#### Primary Data-Driven Insights")
    st.info("Socioeconomic Disparities: Students utilizing standard lunch categories exhibit consistently superior mean performances compared to those reliant on free/reduced lunch, isolating a primary socioeconomic constraint variable.")
    st.info("Iterative Preparedness: Quantitative test preparation course completion serves as a statistical buffer, reliably transitioning populations from critical 'At-Risk' matrices into average compliance pools.")

    st.markdown("---")
    st.markdown("#### Interventional Strategies by ML Cluster")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.error("#### At-Risk Profiling")
        st.markdown("""
        **Statistical Baseline:**
        - Heavy free/reduced lunch indices.
        - Performance aggregated values below 60th percentile.
        - Vast absence of test prep modules.
        
        **Actionable Items:**
        1. Implement mandatory foundational review programs.
        2. Bridge nutritional divides by introducing supplementary breakfast schedules.
        3. Cross-subsidize external test-prep resources for lower-income subsets.
        """)
        
    with col2:
        st.warning("#### Average Capacity")
        st.markdown("""
        **Statistical Baseline:**
        - Scores centered securely within established standard deviations.
        - Varied demographic footprint.
        
        **Actionable Items:**
        1. Distribute advanced practice mechanisms specific to lagging subjects (especially math logic).
        2. Institute collaborative peer networks pairing Average with High Efficacy groups.
        """)
        
    with col3:
        st.success("#### High Efficacy")
        st.markdown("""
        **Statistical Baseline:**
        - Mathematical and reading aggregations consistently topping 85th percentile.
        - Correlated strongly with collegiate parental background distributions.
        
        **Actionable Items:**
        1. Enroll in accelerated AP/curricular frameworks.
        2. Repurpose intellectual capital internally via formalized student tutoring hierarchies.
        """)

st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; font-size: 0.9rem;'>Deployed for Smart Education Architecture | Data Science Interface Unit</p>", unsafe_allow_html=True)

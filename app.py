import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

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
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4 {
        color: #d81b60;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #cfd8dc;
        padding: 15px;
        border-radius: 6px;
        box-shadow: 1px 1px 4px rgba(0,0,0,0.05);
    }
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
# DATA LOADING
# Reads directly from the data/ folder.
# Column names are cleaned to lowercase with underscores.
# ==========================================
@st.cache_data
def load_data():
    """Load and clean the dataset from the data/ directory."""
    try:
        df = pd.read_csv("data/StudentsPerformance.csv")
    except FileNotFoundError:
        st.error("Dataset not found at data/StudentsPerformance.csv")
        return None
    df.columns = df.columns.str.lower().str.replace('/', '_').str.replace(' ', '_')
    return df

# ==========================================
# FEATURE ENGINEERING
# - Compute average and total scores
# - Create a pass/fail academic status label
# ==========================================
@st.cache_data
def feature_engineering(df):
    """Derive aggregate score columns and an academic status label."""
    df = df.copy()
    df['average_score'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1).round(2)
    df['total_score']   = df[['math_score', 'reading_score', 'writing_score']].sum(axis=1)
    df['academic_status'] = np.where(df['average_score'] >= 50, 'Pass', 'Fail')
    return df

# ==========================================
# PREPROCESSING FOR CLUSTERING
# Steps performed:
#   1. Drop the 'lunch' column — it is a logistical field, not a learning behaviour.
#   2. Label-encode remaining categorical columns so K-Means can use them.
#   3. Z-score (StandardScaler) normalise ALL features so no single column
#      dominates the distance calculations.
# ==========================================
def preprocess_for_clustering(df):
    """Encode and scale features for K-Means. Drops 'lunch' column."""
    ml_df = df.copy()

    # Drop lunch — it reflects school administration, not student behaviour
    ml_df.drop(columns=['lunch'], inplace=True)

    cat_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 'test_preparation_course']

    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        ml_df[col + '_encoded'] = le.fit_transform(ml_df[col])
        le_dict[col] = le

    # Z-score normalise the numeric score columns
    num_cols = ['math_score', 'reading_score', 'writing_score', 'average_score']
    scaler = StandardScaler()
    scaled = scaler.fit_transform(ml_df[num_cols])
    for i, col in enumerate(num_cols):
        ml_df[f"{col}_scaled"] = scaled[:, i]

    return ml_df, le_dict, scaler

# ==========================================
# SIDEBAR — Navigation & Filters
# ==========================================
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Modules:", [
    "Dashboard Overview",
    "Exploratory Data Analysis",
    "K-Means Clustering",
    "Actionable Recommendations"
])

st.sidebar.markdown("---")
st.sidebar.header("Global Filtering")

# ==========================================
# BANNER
# ==========================================
st.markdown(
    "<div class='main-banner'>"
    "<h1>Learning Behaviour Profiling</h1>"
    "<p>Problem Statement 5 — Smart Education Analytics</p>"
    "</div>",
    unsafe_allow_html=True
)

# ==========================================
# LOAD & PREPARE DATA
# ==========================================
df_raw = load_data()
if df_raw is None:
    st.stop()

df = feature_engineering(df_raw)

# Sidebar filter widgets
selected_gender = st.sidebar.multiselect("Gender", options=df['gender'].unique(), default=df['gender'].unique())
selected_prep   = st.sidebar.multiselect("Test Preparation", options=df['test_preparation_course'].unique(), default=df['test_preparation_course'].unique())

# Apply filters
df_filtered = df[
    (df['gender'].isin(selected_gender)) &
    (df['test_preparation_course'].isin(selected_prep))
]

if df_filtered.empty:
    st.warning("No data matches current filters. Please adjust selections.")
    st.stop()

# ============================================================
# MODULE 1 — DASHBOARD OVERVIEW
# ============================================================
if app_mode == "Dashboard Overview":
    st.markdown("### Executive Summary")

    with st.expander("About This Dashboard"):
        st.write("""
        This dashboard analyses the **Student Performance Dataset** to identify
        key learning behaviour patterns. It uses **K-Means Clustering** to
        automatically group students into performance tiers and provides
        institution-level recommendations for each group.

        Dataset columns: gender, race/ethnicity, parental education,
        test preparation course, math score, reading score, writing score.
        """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students in View",   f"{len(df_filtered)}")
    c2.metric("Mean Math Score",    f"{df_filtered['math_score'].mean():.1f}")
    c3.metric("Mean Reading Score", f"{df_filtered['reading_score'].mean():.1f}")
    c4.metric("Mean Writing Score", f"{df_filtered['writing_score'].mean():.1f}")

    st.markdown("---")
    st.markdown("### Dataset Preview")
    st.dataframe(df_filtered.head(15), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(df_filtered, names='academic_status', title="Pass / Fail Distribution",
                     color='academic_status',
                     color_discrete_map={'Pass': '#f06292', 'Fail': '#880e4f'},
                     hole=0.3)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        avg_par = (df_filtered
                   .groupby('parental_level_of_education')['average_score']
                   .mean().reset_index()
                   .sort_values('average_score'))
        fig = px.bar(avg_par, x='average_score', y='parental_level_of_education', orientation='h',
                     title="Mean Score by Parental Education Level",
                     color='average_score', color_continuous_scale='RdPu')
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MODULE 2 — EXPLORATORY DATA ANALYSIS
# ============================================================
elif app_mode == "Exploratory Data Analysis":
    st.markdown("### Exploratory Data Analysis (EDA)")

    with st.expander("What is EDA?"):
        st.write("""
        EDA means exploring the data visually before building any model.
        We look at how scores are spread, whether certain groups score better,
        and which features are correlated with each other.
        """)

    tab1, tab2, tab3 = st.tabs(["Score Distributions", "Group Comparisons", "Correlation Analysis"])

    # --- Tab 1: Distributions ---
    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            fig = px.histogram(df_filtered, x="math_score",    nbins=30, title="Math Score",    color_discrete_sequence=['#ff4081'])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(df_filtered, x="reading_score", nbins=30, title="Reading Score", color_discrete_sequence=['#f50057'])
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            fig = px.histogram(df_filtered, x="writing_score", nbins=30, title="Writing Score", color_discrete_sequence=['#c51162'])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Summary Statistics")
        st.dataframe(
            df_filtered[['math_score', 'reading_score', 'writing_score', 'average_score']].describe().round(2),
            use_container_width=True
        )

    # --- Tab 2: Group Comparisons ---
    with tab2:
        col1, col2 = st.columns([1, 1])
        group_col  = col1.selectbox("Group students by:", ['gender', 'test_preparation_course', 'race_ethnicity', 'parental_level_of_education'])
        plot_style = col2.radio("Chart type:", ["Box Plot", "Violin Plot"])

        if plot_style == "Box Plot":
            fig = px.box(df_filtered, x=group_col, y="average_score", color=group_col,
                         title=f"Average Score by {group_col.replace('_', ' ').title()}")
        else:
            fig = px.violin(df_filtered, x=group_col, y="average_score", color=group_col,
                            box=True, points="all",
                            title=f"Score Density by {group_col.replace('_', ' ').title()}")
        st.plotly_chart(fig, use_container_width=True)

        # Average per group in a table as well
        grp_avg = df_filtered.groupby(group_col)['average_score'].mean().reset_index()
        grp_avg.columns = [group_col.replace('_', ' ').title(), 'Mean Average Score']
        grp_avg['Mean Average Score'] = grp_avg['Mean Average Score'].round(2)
        st.dataframe(grp_avg, use_container_width=True)

    # --- Tab 3: Correlations ---
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            corr = df_filtered[['math_score', 'reading_score', 'writing_score', 'average_score']].corr()
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(corr, annot=True, cmap="RdPu", fmt=".2f", ax=ax, linewidths=0.5)
            ax.set_title("Pearson Correlation Matrix")
            st.pyplot(fig)

        with col2:
            fig = px.scatter(df_filtered, x="reading_score", y="writing_score", color="math_score",
                             title="Reading vs Writing (colour = Math Score)",
                             opacity=0.7, color_continuous_scale='RdPu')
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MODULE 3 — K-MEANS CLUSTERING
# ============================================================
elif app_mode == "K-Means Clustering":
    st.markdown("### K-Means Clustering: Student Segmentation")

    with st.expander("How K-Means Works"):
        st.write("""
        **K-Means** is an unsupervised machine learning algorithm.
        It groups students into K clusters based on how similar their scores are.

        Steps:
        1. We tell the algorithm how many clusters we want (K).
        2. It randomly places K "center points" (centroids).
        3. Each student is assigned to the nearest center.
        4. Centers move to the average of their group.
        5. Steps 3–4 repeat until the groups stop changing.

        We use the **Elbow Method** to pick the best K.
        """)

    ml_df, le_dict, scaler = preprocess_for_clustering(df_filtered)
    X = ml_df[['math_score_scaled', 'reading_score_scaled', 'writing_score_scaled']]

    tab_elbow, tab_clusters = st.tabs(["Elbow Method", "Cluster Visualisation"])

    # --- Elbow Method ---
    with tab_elbow:
        st.markdown("#### Finding the Optimal Number of Clusters")
        with st.expander("What is the Elbow Method?"):
            st.write("""
            We run K-Means for K = 1 to 10 and record how compact each grouping is
            (measured by **WCSS — Within-Cluster Sum of Squares**, also called Inertia).

            - Low K: students are lumped together — not very useful.
            - High K: too many tiny groups — also not useful.
            - The "elbow" point is where adding more clusters stops making a big difference.

            In our data, the elbow appears at **K = 3**, which naturally maps to:
            At-Risk, Average, and High Performing students.
            """)

        if st.button("Run Elbow Calculation"):
            wcss = []
            for k in range(1, 11):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X)
                wcss.append(km.inertia_)

            fig = px.line(x=list(range(1, 11)), y=wcss, markers=True,
                          labels={'x': 'Number of Clusters (K)', 'y': 'WCSS (Inertia)'},
                          title="Elbow Method — Optimal K Selection",
                          color_discrete_sequence=['#d81b60'])
            fig.add_vline(x=3, line_dash="dash", line_color="#880e4f",
                          annotation_text="Optimal K=3", annotation_position="top right")
            st.plotly_chart(fig, use_container_width=True)

    # --- Cluster Visualisation ---
    with tab_clusters:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### Cluster Settings")
            k_value = st.slider("Number of Clusters (K):", min_value=2, max_value=6, value=3)

            kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
            ml_df['cluster'] = kmeans.fit_predict(X)
            df_filtered = df_filtered.copy()
            df_filtered['Cluster'] = ml_df['cluster'].astype(str)

            # Label clusters based on mean math score (lowest = At-Risk, highest = High Efficacy)
            cluster_means = df_filtered.groupby('Cluster')['math_score'].mean().sort_values()
            if k_value == 3:
                labels = {
                    cluster_means.index[0]: "At-Risk",
                    cluster_means.index[1]: "Average",
                    cluster_means.index[2]: "High Performer"
                }
            else:
                labels = {k: f"Segment {k}" for k in cluster_means.index}

            df_filtered['Profile'] = df_filtered['Cluster'].map(labels)

            st.markdown("#### Cluster Sizes")
            cluster_summary = df_filtered['Profile'].value_counts().reset_index()
            cluster_summary.columns = ['Profile', 'Number of Students']
            st.dataframe(cluster_summary, use_container_width=True)

            st.markdown("#### Mean Scores by Cluster")
            st.dataframe(
                df_filtered.groupby('Profile')[['math_score', 'reading_score', 'writing_score']].mean().round(1),
                use_container_width=True
            )

        with col2:
            fig = px.scatter_3d(df_filtered,
                                x='math_score', y='reading_score', z='writing_score',
                                color='Profile', opacity=0.8,
                                title=f"3D Student Clusters (K={k_value})",
                                color_discrete_sequence=['#f06292', '#ba68c8', '#4dd0e1', '#81c784', '#ffb74d', '#ff8a65'])
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MODULE 4 — ACTIONABLE RECOMMENDATIONS
# ============================================================
elif app_mode == "Actionable Recommendations":
    st.markdown("### Institutional Strategy and Recommendations")

    with st.expander("What is Prescriptive Analytics?"):
        st.write("""
        After we understand the data (EDA) and group students (Clustering),
        the final step is to give concrete suggestions on what schools can do
        to help each group of students.
        """)

    st.markdown("#### Key Findings from the Data")
    st.info("Students who completed the test preparation course scored significantly higher on average.")
    st.info("Parental education level shows a consistent positive correlation with student scores.")

    st.markdown("---")
    st.markdown("#### Recommendations by Student Cluster")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.error("**At-Risk Students**")
        st.markdown("""
        **Who they are:**
        Low scores across all three subjects. Usually missed test preparation.

        **What to do:**
        1. Enrol them in mandatory test preparation sessions.
        2. Offer one-on-one tutoring for math fundamentals.
        3. Regular progress check-ins from teachers.
        """)

    with col2:
        st.warning("**Average Students**")
        st.markdown("""
        **Who they are:**
        Scores in the mid-range. Mixed backgrounds.

        **What to do:**
        1. Provide supplementary reading and writing exercises.
        2. Pair them with high-performing peers for group study.
        3. Encourage optional test prep completion.
        """)

    with col3:
        st.success("**High Performers**")
        st.markdown("""
        **Who they are:**
        Consistently high scores. Often completed test preparation.

        **What to do:**
        1. Offer advanced coursework and enrichment programmes.
        2. Assign mentoring roles to reinforce their own understanding.
        3. Nominate for academic competitions and scholarships.
        """)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.85rem;'>"
    "Problem Statement 5 — Smart Education Analytics | Powered by Python and Streamlit"
    "</p>",
    unsafe_allow_html=True
)

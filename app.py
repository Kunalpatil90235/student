"""
app.py
------
Main entry point for the Learning Behaviour Profiling dashboard.

This file is deliberately thin — it handles only:
    1. Page configuration and global CSS
    2. Sidebar navigation and filters
    3. Data loading (via modules/data_loader.py)
    4. Routing to the correct module based on the selected page

All heavy logic lives in the modules/ folder.

Run with:
    streamlit run app.py
"""

import streamlit as st
from modules.data_loader import load_data, feature_engineering, get_preprocessed_preview
from modules.eda import render_eda
from modules.clustering import render_clustering
from modules.risk_checker import render_risk_checker
from modules.recommendations import render_recommendations

# ── Page configuration ────────────────────────────────────────────────
st.set_page_config(
    page_title="Learning Behaviour Profiling",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (dark blue professional theme) ─────────────────────────
st.markdown("""
<style>
/* Remove default Streamlit top padding */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Headings */
h1, h2, h3, h4 {
    color: #38bdf8;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-weight: 600;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background-color: #1e293b;
    border: 1px solid #334155;
    padding: 14px;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
}

/* Hero banner */
.main-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e40af 60%, #0369a1 100%);
    color: white;
    padding: 28px 32px;
    border-radius: 10px;
    margin-bottom: 24px;
    border: 1px solid #1e40af;
    box-shadow: 0 4px 20px rgba(56,189,248,0.15);
}
.main-banner h1 { color: #38bdf8 !important; margin: 0; font-size: 2.1rem; }
.main-banner p  { margin: 6px 0 0 0; color: #94a3b8; font-size: 1.05rem; }

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 500;
    color: #94a3b8 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* Expanders */
details summary { color: #38bdf8; }

/* Sidebar */
.css-1d391kg, section[data-testid="stSidebar"] > div {
    background-color: #0f172a !important;
}
</style>
""", unsafe_allow_html=True)

# ── Banner ────────────────────────────────────────────────────────────
st.markdown(
    "<div class='main-banner'>"
    "<h1>Learning Behaviour Profiling</h1>"
    "<p>Problem Statement 5 — Smart Education Analytics | "
    "Unsupervised K-Means Clustering with PCA Visualisation</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Sidebar navigation ────────────────────────────────────────────────
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("Go to:", [
    "Dashboard Overview",
    "Exploratory Data Analysis",
    "K-Means Clustering",
    "Student Risk Assessment",
    "Recommendations",
])

st.sidebar.markdown("---")
st.sidebar.markdown("## Filters")

# ── Load and engineer data ────────────────────────────────────────────
df_raw = load_data()
if df_raw is None:
    st.stop()
df = feature_engineering(df_raw)

# Sidebar filters
sel_gender = st.sidebar.multiselect(
    "Gender",
    options=df["gender"].unique(),
    default=df["gender"].unique(),
)
sel_prep = st.sidebar.multiselect(
    "Test Preparation",
    options=df["test_preparation_course"].unique(),
    default=df["test_preparation_course"].unique(),
)

df_filtered = df[
    df["gender"].isin(sel_gender) &
    df["test_preparation_course"].isin(sel_prep)
]

if df_filtered.empty:
    st.warning("No data matches the current filters. Please adjust your selections.")
    st.stop()

# ── Show active record count in sidebar ──────────────────────────────
st.sidebar.markdown(f"**Active Records:** `{len(df_filtered)}`")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:#64748b;'>Problem Statement 5<br>Smart Education Domain</small>",
    unsafe_allow_html=True,
)

# ── Page routing ─────────────────────────────────────────────────────

# ── Dashboard Overview ────────────────────────────────────────────────
if page == "Dashboard Overview":
    st.markdown("### Executive Summary")

    with st.expander("About This Project — Objective and Approach"):
        st.write("""
        **Objective:** Understand student learning patterns by identifying key learning factors
        (test preparation, parental education, gender, race/ethnicity) and using those factors
        to cluster students into distinct learning profiles.

        **What the project does:**
        - Loads and cleans the Student Performance Dataset (1,000 students, 8 features).
        - Performs Exploratory Data Analysis to surface which factors most influence performance.
        - Applies **K-Means Clustering** (unsupervised ML) to automatically segment students
          into three learning profiles: **At-Risk, Average, and High Performer**.
        - Uses **PCA** (Principal Component Analysis) to visualise the clusters in 2D.
        - Provides an interactive **Student Risk Assessment** — enter any student's scores
          and instantly see which learning profile they belong to.
        - Generates institutional recommendations per cluster so educators can take action.

        **Type of project:** Clustering (Unsupervised Machine Learning).  
        There is no train/test split and no traditional accuracy metric.
        We validate cluster quality using the **Elbow Method (WCSS)**.
        """)

    # KPI metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students",     f"{len(df_filtered)}")
    c2.metric("Mean Math Score",    f"{df_filtered['math_score'].mean():.1f}")
    c3.metric("Mean Reading Score", f"{df_filtered['reading_score'].mean():.1f}")
    c4.metric("Mean Writing Score", f"{df_filtered['writing_score'].mean():.1f}")

    st.markdown("---")
    c5, c6 = st.columns(2)
    pass_rate = (df_filtered["academic_status"] == "Pass").mean() * 100
    avg_score = df_filtered["average_score"].mean()
    c5.metric("Pass Rate",     f"{pass_rate:.1f}%")
    c6.metric("Overall Average", f"{avg_score:.1f}/100")

    st.markdown("---")
    st.markdown("### Raw Dataset Preview (first 15 rows)")
    st.dataframe(df_filtered.head(15), use_container_width=True)

    st.markdown("---")
    st.markdown("### Data After Preprocessing")
    with st.expander("What changed during preprocessing?"):
        st.write("""
        The table below shows the dataset **after** the following steps:
        1. **Dropped** the `lunch` column (administrative field, not a learning behaviour).
        2. **Label Encoded** categorical columns — text like `male`/`female` becomes `0`/`1`.
        3. **Derived features** added: `average_score`, `total_score`, `academic_status`.
        Scaled columns (`_scaled`) are used internally by K-Means but are hidden here for readability.
        """)
    preprocessed_df = get_preprocessed_preview(df)
    st.dataframe(preprocessed_df, use_container_width=True)
    st.caption("Showing first 20 rows after Label Encoding and feature derivation (scaled columns hidden).")

    import plotly.express as px
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(df_filtered, names="academic_status",
                     title="Pass / Fail Distribution",
                     color="academic_status",
                     color_discrete_map={"Pass": "#34d399", "Fail": "#f87171"},
                     hole=0.35)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        avg_par = (df_filtered
                   .groupby("parental_level_of_education")["average_score"]
                   .mean().reset_index()
                   .sort_values("average_score"))
        fig = px.bar(avg_par, x="average_score", y="parental_level_of_education",
                     orientation="h",
                     title="Mean Score by Parental Education Level",
                     color="average_score", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

# ── EDA ──────────────────────────────────────────────────────────────
elif page == "Exploratory Data Analysis":
    render_eda(df_filtered)

# ── Clustering ───────────────────────────────────────────────────────
elif page == "K-Means Clustering":
    kmeans_model, scaler_model, cluster_label_map = render_clustering(df_filtered, df)
    # Store in session state so Risk Checker can reuse them
    st.session_state["kmeans_model"]     = kmeans_model
    st.session_state["scaler_model"]     = scaler_model
    st.session_state["cluster_label_map"] = cluster_label_map

# ── Risk Assessment ──────────────────────────────────────────────────
elif page == "Student Risk Assessment":
    # Build the model on the full dataset if not already in session state
    if "kmeans_model" not in st.session_state:
        from modules.clustering import render_clustering as _rc
        import io, contextlib
        # Silently build the model without rendering the full clustering UI
        from modules.data_loader import preprocess_for_clustering
        from sklearn.cluster import KMeans as _KM
        ml_df, _, scaler_s = preprocess_for_clustering(df)
        X_s = ml_df[["math_score_scaled", "reading_score_scaled", "writing_score_scaled"]]
        km_s = _KM(n_clusters=3, random_state=42, n_init=10)
        km_s.fit(X_s)
        df_tmp = df.copy()
        df_tmp["Cluster"] = km_s.labels_.astype(str)
        cm = df_tmp.groupby("Cluster")["math_score"].mean().sort_values()
        lmap = {cm.index[0]: "At-Risk", cm.index[1]: "Average", cm.index[2]: "High Performer"}
        st.session_state["kmeans_model"]      = km_s
        st.session_state["scaler_model"]      = scaler_s
        st.session_state["cluster_label_map"] = lmap

    render_risk_checker(
        st.session_state["kmeans_model"],
        st.session_state["scaler_model"],
        st.session_state["cluster_label_map"],
    )

# ── Recommendations ──────────────────────────────────────────────────
elif page == "Recommendations":
    render_recommendations(df_filtered)

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.82rem;'>"
    "Problem Statement 5 — Smart Education Analytics | "
    "K-Means Clustering + PCA | Python + Streamlit"
    "</p>",
    unsafe_allow_html=True,
)

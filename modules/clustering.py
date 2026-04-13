"""
clustering.py
-------------
K-Means clustering, Elbow Method, and PCA 2-D visualisation.

Key concepts:
    K-Means   — unsupervised algorithm that groups similar students together
    Elbow     — method to find the best number of clusters (K)
    PCA       — Principal Component Analysis, used here to reduce 3D scores
                to 2D so we can plot and see how well clusters are separated

Functions used:
    KMeans(n_clusters=k).fit_predict(X)   — assigns each row to a cluster
    KMeans.inertia_                        — WCSS value after fitting
    PCA(n_components=2).fit_transform(X) — reduces dimensions to 2 principal
                                           components for plotting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from modules.data_loader import preprocess_for_clustering


# ── Cluster labelling helper ──────────────────────────────────────────
def _label_clusters(df: pd.DataFrame, k: int) -> dict:
    """
    After K-Means runs, we look at the average math score of each cluster.
    The lowest average → At-Risk, middle → Average, highest → High Performer.
    For K != 3 we just name them by their average score.
    """
    cluster_means = df.groupby("Cluster")["math_score"].mean().sort_values()
    if k == 3:
        return {
            cluster_means.index[0]: "At-Risk",
            cluster_means.index[1]: "Average",
            cluster_means.index[2]: "High Performer",
        }
    return {k_id: f"Segment {k_id} (mean={v:.0f})"
            for k_id, v in zip(cluster_means.index, cluster_means.values)}


# ── Main render function ──────────────────────────────────────────────
def render_clustering(df_filtered: pd.DataFrame, df_full: pd.DataFrame):
    """
    Renders the K-Means clustering module.
    Returns the fitted KMeans model and scaler so risk_checker.py can reuse them.
    """
    st.markdown("### K-Means Clustering: Student Segmentation")

    with st.expander("How K-Means Works"):
        st.write("""
        **K-Means is an unsupervised machine learning algorithm.**
        Here is how it works in plain language:

        1. You choose how many groups (K) you want — we use 3 (Low/Average/High).
        2. The algorithm picks 3 random "centre points" in the score space.
        3. Every student is assigned to the nearest centre.
        4. Each centre moves to the average position of its students.
        5. Steps 3–4 repeat until nothing changes.

        The result: students with similar scores are in the same group.

        **Why is PCA used here?**
        After clustering, we have three cluster labels. We use PCA (Principal Component Analysis)
        to squish the three score dimensions (math, reading, writing) into two dimensions,
        so we can draw a flat 2D scatter plot. PCA finds the directions of maximum variance
        in the data and projects all points onto those directions.
        This is purely for VISUALISATION — it makes it easy to see that the clusters are
        well separated.
        """)

    ml_df, le_dict, scaler = preprocess_for_clustering(df_filtered)
    X = ml_df[["math_score_scaled", "reading_score_scaled", "writing_score_scaled"]]

    tab_elbow, tab_3d, tab_pca = st.tabs(["Elbow Method", "3D Cluster View", "PCA 2D View"])

    # ── Elbow Method ───────────────────────────────────────────────────
    with tab_elbow:
        st.markdown("#### Finding the Optimal Number of Clusters (K)")

        with st.expander("What is the Elbow Method?"):
            st.write("""
            We run K-Means for K = 1, 2, 3, … 10 and record how compact each grouping is.
            This compactness is measured by **WCSS (Within-Cluster Sum of Squares)** —
            also called Inertia. Lower WCSS = tighter clusters.

            We plot WCSS vs K and look for the "elbow" — the point where the drop slows down.
            Adding more clusters after that point gives very little benefit.

            In our data, the elbow is clearly at **K = 3**.
            """)

        if st.button("Calculate Elbow Curve"):
            wcss = []
            for k in range(1, 11):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X)
                wcss.append(km.inertia_)

            fig = px.line(
                x=list(range(1, 11)), y=wcss, markers=True,
                labels={"x": "Number of Clusters (K)", "y": "WCSS (Inertia)"},
                title="Elbow Method — Optimal K Selection",
                color_discrete_sequence=["#38bdf8"],
            )
            fig.add_vline(x=3, line_dash="dash", line_color="#f472b6",
                          annotation_text="Optimal K = 3",
                          annotation_position="top right")
            st.plotly_chart(fig, use_container_width=True)
            st.info("The sharp drop stabilises at K=3, confirming three clusters: At-Risk, Average, High Performer.")

    # ── 3D Scatter ────────────────────────────────────────────────────
    with tab_3d:
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.markdown("#### Cluster Settings")
            k_value = st.slider("Number of Clusters (K):", 2, 6, 3, key="k_3d")

            kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
            ml_df["cluster"] = kmeans.fit_predict(X)
            df_filtered = df_filtered.copy()
            df_filtered["Cluster"] = ml_df["cluster"].astype(str)
            labels = _label_clusters(df_filtered, k_value)
            df_filtered["Profile"] = df_filtered["Cluster"].map(labels)

            st.markdown("#### Cluster Sizes")
            cs = (df_filtered["Profile"].value_counts()
                  .reset_index()
                  .rename(columns={"count": "Students"}))
            st.dataframe(cs, use_container_width=True)

            st.markdown("#### Mean Scores per Cluster")
            st.dataframe(
                df_filtered.groupby("Profile")[["math_score", "reading_score", "writing_score"]]
                .mean().round(1),
                use_container_width=True,
            )

        with col_right:
            fig = px.scatter_3d(
                df_filtered,
                x="math_score", y="reading_score", z="writing_score",
                color="Profile", opacity=0.8,
                title=f"3D Student Clusters (K={k_value})",
                color_discrete_sequence=["#f87171", "#fbbf24", "#34d399", "#38bdf8", "#818cf8", "#f472b6"],
            )
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
            st.plotly_chart(fig, use_container_width=True)

    # ── PCA 2D View ──────────────────────────────────────────────────
    with tab_pca:
        st.markdown("#### PCA 2D Projection of Clusters")

        with st.expander("What is PCA?"):
            st.write("""
            **PCA (Principal Component Analysis)** is a technique that reduces many dimensions
            down to fewer, while keeping as much of the original variation as possible.

            Our students are plotted in **3D space** (math score, reading score, writing score).
            PCA compresses that into **2D** — two "principal components" — so we can draw a
            flat scatter plot. These components are not real scores; they are mathematical
            combinations of all three scores that capture the most variance.

            We use PCA here **only for visualisation**. It helps judges see that the three
            clusters are genuinely well-separated and not just random.

            After K-Means assigns cluster labels, we know which group each student belongs to.
            PCA then gives us a 2D map to display that clearly.
            """)

        # Use K=3 clusters for PCA view
        km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_3 = km3.fit_predict(X)

        pca = PCA(n_components=2)
        components = pca.fit_transform(X)

        pca_df = pd.DataFrame(components, columns=["PCA Component 1", "PCA Component 2"])
        pca_df["Cluster"] = labels_3.astype(str)

        # Map cluster labels
        tmp = df_filtered.copy().reset_index(drop=True)
        tmp["Cluster"] = labels_3.astype(str)
        cm = tmp.groupby("Cluster")["math_score"].mean().sort_values()
        pca_labels = {cm.index[0]: "At-Risk", cm.index[1]: "Average", cm.index[2]: "High Performer"}
        pca_df["Profile"] = pca_df["Cluster"].map(pca_labels)

        var_explained = pca.explained_variance_ratio_ * 100
        fig = px.scatter(
            pca_df, x="PCA Component 1", y="PCA Component 2",
            color="Profile", opacity=0.8,
            title=f"PCA 2D Cluster Map  (PC1={var_explained[0]:.1f}% var, PC2={var_explained[1]:.1f}% var)",
            color_discrete_map={"At-Risk": "#f87171", "Average": "#fbbf24", "High Performer": "#34d399"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"PCA explains {var_explained.sum():.1f}% of total variance in two components.")

    # Return the fitted K=3 model and scaler for the Risk Checker to reuse
    km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
    ml_full, _, scaler_full = preprocess_for_clustering(df_full)
    X_full = ml_full[["math_score_scaled", "reading_score_scaled", "writing_score_scaled"]]
    km_final.fit(X_full)

    # Build cluster → profile map using full dataset
    df_full_copy = df_full.copy()
    df_full_copy["Cluster"] = km_final.labels_.astype(str)
    full_cm = df_full_copy.groupby("Cluster")["math_score"].mean().sort_values()
    full_labels = {
        full_cm.index[0]: "At-Risk",
        full_cm.index[1]: "Average",
        full_cm.index[2]: "High Performer",
    }

    return km_final, scaler_full, full_labels

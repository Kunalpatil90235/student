"""
risk_checker.py
---------------
Interactive student risk assessment tool.

The user enters a student's math, reading, and writing scores via sliders.
We scale those inputs using the same StandardScaler that was fitted on the
training dataset, then pass them to the trained K-Means model.
K-Means.predict() assigns the new student to one of the 3 clusters.
We then show which performance profile the student falls into, along with
a specific recommendation.

Functions used:
    np.array([[m, r, w]])       — creates a 1-row 2D array for the model
    scaler.transform(X_new)     — scales new data using the SAME mean and std
                                  that were calculated on the training data
    kmeans.predict(X_scaled)    — assigns the new point to the nearest centroid
"""

import streamlit as st
import numpy as np


def render_risk_checker(kmeans, scaler, cluster_labels: dict) -> None:
    """
    Renders the interactive Student Risk Assessment form.

    Parameters
    ----------
    kmeans         : fitted KMeans model (from clustering.py)
    scaler         : fitted StandardScaler (from data_loader.py preprocessing)
    cluster_labels : dict mapping cluster-id string → profile name
    """
    st.markdown("### Student Risk Assessment")

    st.info("""
    Enter a student's expected or actual scores below.
    The trained K-Means model will predict which performance cluster
    this student belongs to, and give a personalised recommendation.
    """)

    with st.expander("How does this prediction work?"):
        st.write("""
        1. You enter three scores (math, reading, writing).
        2. We scale those three numbers using the **same StandardScaler**
           that was fitted on all 1000 students in the dataset.
           This is important — we must use the same scale, otherwise the
           distances would be meaningless.
        3. We pass the scaled scores to the trained **K-Means model**.
           K-Means.predict() finds which of the 3 cluster centres is
           closest to this new point and returns that cluster's ID.
        4. We map the cluster ID to a label (At-Risk / Average / High Performer)
           and show the result.

        This is still **unsupervised** — we are not predicting a fixed label
        that we defined. We are checking which naturally discovered group
        a new student is most similar to.
        """)

    st.markdown("#### Enter Student Scores")
    col1, col2, col3 = st.columns(3)
    math_score    = col1.slider("Math Score",    0, 100, 60, step=1)
    reading_score = col2.slider("Reading Score", 0, 100, 60, step=1)
    writing_score = col3.slider("Writing Score", 0, 100, 60, step=1)

    average = round((math_score + reading_score + writing_score) / 3, 1)
    st.markdown(f"**Computed Average Score:** `{average}`")

    if st.button("Assess Student Risk"):
        # Scale the input the same way the training data was scaled
        X_new = np.array([[math_score, reading_score, writing_score]])
        X_scaled = scaler.transform(X_new)

        # K-Means predict — returns the cluster ID closest to this point
        cluster_id = str(kmeans.predict(X_scaled)[0])
        profile = cluster_labels.get(cluster_id, "Unknown")

        st.markdown("---")
        st.markdown("#### Assessment Result")

        if profile == "At-Risk":
            st.error(f"**Profile: {profile}**")
            st.markdown("""
            **What this means:** This student's scores fall in the lowest-performing cluster.

            **Recommended Actions:**
            - Enrol in a test preparation course immediately.
            - Set up regular one-on-one tutoring sessions, especially for math.
            - Monitor progress weekly and adjust support as needed.
            """)
        elif profile == "Average":
            st.warning(f"**Profile: {profile}**")
            st.markdown("""
            **What this means:** This student is performing in the middle range.

            **Recommended Actions:**
            - Provide supplementary practice materials for weaker subjects.
            - Encourage peer-study groups with higher-performing classmates.
            - Consider optional test preparation to push into the High Performer tier.
            """)
        else:
            st.success(f"**Profile: {profile}**")
            st.markdown("""
            **What this means:** This student is in the highest-performing cluster.

            **Recommended Actions:**
            - Offer access to advanced or AP-level coursework.
            - Assign mentoring roles to reinforce leadership and deeper understanding.
            - Nominate for academic competitions or enrichment programmes.
            """)

        # Show a score bar chart for quick visual
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=["Math", "Reading", "Writing"],
            y=[math_score, reading_score, writing_score],
            marker_color=["#38bdf8", "#818cf8", "#34d399"],
        ))
        fig.update_layout(
            title="Entered Score Breakdown",
            yaxis=dict(range=[0, 100]),
            plot_bgcolor="#1e293b",
            paper_bgcolor="#1e293b",
            font_color="#e2e8f0",
        )
        st.plotly_chart(fig, use_container_width=True)

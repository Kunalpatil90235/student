"""
recommendations.py
------------------
Prescriptive analytics panel — translates cluster findings into
actionable institutional strategies.

This is the final stage of the analytics pipeline:
    Descriptive  (EDA)      → what happened
    Diagnostic   (Clusters) → why / which groups
    Prescriptive (here)     → what to do about it
"""

import pandas as pd
import streamlit as st



def render_recommendations(df_filtered: "pd.DataFrame") -> None:
    """Renders the Recommendations and Insights panel."""

    st.markdown("### Institutional Recommendations")

    with st.expander("What is Prescriptive Analytics?"):
        st.write("""
        After we understand the data (EDA) and group the students (Clustering),
        the final step is to turn those insights into **concrete actions**.
        This is called Prescriptive Analytics — it answers the question:
        *"Now that we know this, what should we actually do?"*
        """)

    st.markdown("#### Key Findings from the Data")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Test Preparation**  
        Students who completed the test preparation course scored on average
        **8–12 points higher** across all three subjects compared to those who did not.
        This is the single strongest lever schools can pull.
        """)
    with col2:
        st.markdown("""
        **Parental Education**  
        A clear positive trend exists between a parent's highest education level
        and their child's academic performance. Students whose parents hold a
        bachelor's or master's degree consistently outperform those from
        high-school-only household backgrounds.
        """)

    st.markdown("---")
    st.markdown("#### Recommendations by Performance Cluster")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.error("**At-Risk Students**")
        st.markdown("""
        *Who they are:* Low scores across all subjects.
        Usually absent from test preparation.

        **Actions:**
        1. Make test preparation mandatory or subsidised.
        2. One-on-one math tutoring sessions.
        3. Weekly teacher progress check-ins.
        4. Connect parents with school support resources.
        """)

    with c2:
        st.warning("**Average Students**")
        st.markdown("""
        *Who they are:* Mid-range scores. Mixed backgrounds.

        **Actions:**
        1. Supplementary reading and writing practice packs.
        2. Peer-study groups paired with High Performers.
        3. Optional test preparation to push into the top tier.
        4. Subject-specific workshops for the weakest area.
        """)

    with c3:
        st.success("**High Performers**")
        st.markdown("""
        *Who they are:* Consistently high scores. Usually completed test prep.

        **Actions:**
        1. Enrol in advanced / AP-level coursework.
        2. Assign peer-mentoring roles.
        3. Nominate for academic competitions.
        4. Scholarship and enrichment programme referrals.
        """)

    # ── Summary table ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Dataset Snapshot for Current Filter Selection")
    summary = df_filtered.groupby("academic_status").agg(
        Count=("average_score", "count"),
        Mean_Math=("math_score", "mean"),
        Mean_Reading=("reading_score", "mean"),
        Mean_Writing=("writing_score", "mean"),
        Mean_Average=("average_score", "mean"),
    ).round(1).reset_index()
    st.dataframe(summary, use_container_width=True)

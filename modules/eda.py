"""
eda.py
------
All visualisations for the Exploratory Data Analysis module.
Called by app.py when the user selects 'Exploratory Data Analysis'.

Functions used in this module:
    px.histogram()   — bar chart where bars represent frequency in a range
    px.box()         — box-and-whisker plot showing median, IQR, outliers
    px.violin()      — like a box plot but also shows data density shape
    px.bar()         — standard bar chart
    px.pie()         — pie / donut chart
    px.scatter()     — scatter plot of two continuous variables
    df.groupby()     — groups rows by a column so we can aggregate per group
    df.describe()    — gives count, mean, std, min, max, quartiles
    df.corr()        — Pearson correlation matrix between numeric columns
    sns.heatmap()    — colour-coded grid for the correlation matrix
"""

import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


def render_eda(df_filtered: "pd.DataFrame") -> None:
    """Renders all five EDA tabs for the given (filtered) DataFrame."""

    st.markdown("### Exploratory Data Analysis")

    with st.expander("What is EDA?"):
        st.write("""
        EDA is the step where we **explore and understand the data before doing any modelling**.
        We create charts to find patterns, spot outliers, and test whether our assumptions are correct.
        For example: Do students who completed test prep score higher? (Yes.) Does parental education matter? (Yes, to some degree.)
        """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Score Distributions",
        "Group Comparisons",
        "Subject Breakdown",
        "Categorical Counts",
        "Correlation Analysis",
    ])

    # ── Tab 1 · Score Distributions ────────────────────────────────────
    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            fig = px.histogram(df_filtered, x="math_score", nbins=30,
                               title="Math Score Distribution",
                               color_discrete_sequence=["#38bdf8"])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(df_filtered, x="reading_score", nbins=30,
                               title="Reading Score Distribution",
                               color_discrete_sequence=["#818cf8"])
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            fig = px.histogram(df_filtered, x="writing_score", nbins=30,
                               title="Writing Score Distribution",
                               color_discrete_sequence=["#34d399"])
            st.plotly_chart(fig, use_container_width=True)

        # Average score with a box plot in the margin
        fig = px.histogram(df_filtered, x="average_score", nbins=30,
                           title="Overall Average Score Distribution",
                           marginal="box",
                           color_discrete_sequence=["#f472b6"])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Descriptive Statistics")
        st.dataframe(
            df_filtered[["math_score", "reading_score", "writing_score", "average_score"]]
            .describe().round(2),
            use_container_width=True,
        )

    # ── Tab 2 · Group Comparisons ───────────────────────────────────────
    with tab2:
        col1, col2 = st.columns(2)
        group_col = col1.selectbox("Group students by:", [
            "gender", "test_preparation_course",
            "race_ethnicity", "parental_level_of_education",
        ], key="grp_cmp")
        plot_style = col2.radio("Chart type:", ["Box Plot", "Violin Plot"], key="grp_style")

        if plot_style == "Box Plot":
            fig = px.box(df_filtered, x=group_col, y="average_score", color=group_col,
                         title=f"Average Score by {group_col.replace('_', ' ').title()}")
        else:
            fig = px.violin(df_filtered, x=group_col, y="average_score", color=group_col,
                            box=True, points="all",
                            title=f"Score Density by {group_col.replace('_', ' ').title()}")
        st.plotly_chart(fig, use_container_width=True)

        grp_avg = (df_filtered.groupby(group_col)["average_score"]
                   .mean().reset_index()
                   .rename(columns={group_col: group_col.replace("_", " ").title(),
                                    "average_score": "Mean Average Score"}))
        grp_avg["Mean Average Score"] = grp_avg["Mean Average Score"].round(2)
        st.dataframe(grp_avg, use_container_width=True)

    # ── Tab 3 · Subject-Wise Breakdown ─────────────────────────────────
    with tab3:
        st.markdown("Compare each subject's mean score across a demographic group.")
        subj_group = st.selectbox("Select group:", [
            "gender", "test_preparation_course",
            "race_ethnicity", "parental_level_of_education",
        ], key="subj_grp")

        # Grouped bar — math / reading / writing side by side per category
        subj_df = (
            df_filtered
            .groupby(subj_group)[["math_score", "reading_score", "writing_score"]]
            .mean().reset_index()
            .melt(id_vars=subj_group, var_name="Subject", value_name="Mean Score")
        )
        fig = px.bar(subj_df, x=subj_group, y="Mean Score", color="Subject",
                     barmode="group",
                     title=f"Mean Subject Scores by {subj_group.replace('_', ' ').title()}",
                     color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399"])
        st.plotly_chart(fig, use_container_width=True)

        # Stacked Pass / Fail bar per group
        st.markdown("#### Pass / Fail Count by Group")
        pass_df = (df_filtered
                   .groupby([subj_group, "academic_status"])
                   .size().reset_index(name="Count"))
        fig = px.bar(pass_df, x=subj_group, y="Count", color="academic_status",
                     barmode="stack",
                     title=f"Pass / Fail Breakdown by {subj_group.replace('_', ' ').title()}",
                     color_discrete_map={"Pass": "#34d399", "Fail": "#f87171"})
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4 · Categorical Counts & Proportions ────────────────────────
    with tab4:
        st.markdown("Understand the demographic composition of the dataset.")
        cat_col = st.selectbox("Select category:", [
            "gender", "race_ethnicity",
            "parental_level_of_education", "test_preparation_course",
        ], key="cat_cnt")

        col1, col2 = st.columns(2)
        with col1:
            count_df = df_filtered[cat_col].value_counts().reset_index()
            count_df.columns = [cat_col.replace("_", " ").title(), "Count"]
            fig = px.bar(count_df, x=cat_col.replace("_", " ").title(), y="Count",
                         title=f"Student Count by {cat_col.replace('_', ' ').title()}",
                         color="Count", color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.pie(df_filtered, names=cat_col,
                         title=f"Proportion — {cat_col.replace('_', ' ').title()}",
                         color_discrete_sequence=px.colors.sequential.Blues_r)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Average Score: Gender × Test Preparation")
        combo = (df_filtered
                 .groupby(["gender", "test_preparation_course"])["average_score"]
                 .mean().reset_index())
        fig = px.bar(combo, x="gender", y="average_score",
                     color="test_preparation_course", barmode="group",
                     title="Average Score by Gender and Test Preparation Status",
                     color_discrete_sequence=["#38bdf8", "#818cf8"])
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5 · Correlation Analysis ────────────────────────────────────
    with tab5:
        col1, col2 = st.columns(2)
        with col1:
            corr = df_filtered[
                ["math_score", "reading_score", "writing_score", "average_score"]
            ].corr()
            fig_h, ax = plt.subplots(figsize=(6, 5))
            fig_h.patch.set_facecolor("#1e293b")
            ax.set_facecolor("#1e293b")
            sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f",
                        ax=ax, linewidths=0.5,
                        annot_kws={"color": "white"})
            ax.tick_params(colors="white")
            ax.set_title("Pearson Correlation Matrix", color="white")
            st.pyplot(fig_h)

        with col2:
            fig = px.scatter(df_filtered, x="reading_score", y="writing_score",
                             color="math_score",
                             title="Reading vs Writing (colour = Math Score)",
                             opacity=0.7, color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Math vs Average Score by Gender (with Trend Line)")
        fig = px.scatter(df_filtered, x="math_score", y="average_score",
                         color="gender", trendline="ols",
                         title="Math Score vs Average Score",
                         color_discrete_map={"male": "#38bdf8", "female": "#f472b6"})
        st.plotly_chart(fig, use_container_width=True)

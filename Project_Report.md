# Educational Learning Behaviour Profiling
## Project Documentation & Report

### 1. Executive Summary
This project addresses **Problem Statement 5: Student Learning Behaviour Analysis** within the Smart Education domain. The primary objective is to profile learning behaviours, identify defining factors of scholastic success, and deliver actionable institutional recommendations. The end product is a robust, interactive dashboard driven by Machine Learning.

### 2. Dataset and Features 
The pipeline utilizes the **Student Performance Dataset**, containing demographic variables and scaled scholastic indicators.
**Variables utilized**:
- **Categorical:** Gender, Race/Ethnicity, Parental Level of Education, Lunch type (proxy for socioeconomic boundaries), and Test Preparation Course completion.
- **Continuous (Scores):** Math, Reading, Writing, and composite Total/Average scores computationally derived.

### 3. Data Science Concepts Applied

#### A. Exploratory Data Analysis (EDA)
EDA involves investigating the statistical properties of the dataset to discover patterns and test hypotheses.
*   **Distribution Analysis:** We leverage histograms to understand probability distributions indicating whether scholastic outputs exhibit normal ("bell-curve") behaviour.
*   **Variance Evaluation Structure (Boxplots & Violins):** Utilized to isolate Interquartile Ranges (IQR) and underlying density populations to identify how varying demographics statically shift median scores and influence outlier generations. 
*   **Pearson Correlation:** A bivariate analysis technique visualizing multi-collinearity among standard continuous scores (e.g., verifying if excellence in reading is linearly co-dependent on excellence in writing).

#### B. Feature Engineering
Raw data rarely suffices for complex modeling.
*   We synthesized **composite scoring features** (Total/Average Score) to act as absolute thresholds.
*   **Socio-Economic Proxy Formulation:** We mathematically merged 'Lunch' status with 'Test Preparation' accessibility to formulate a unified socioeconomic integer that amplifies decision boundaries for tree-based models. Ensure continuous variables were processed via **Standard Scaler (Z-score normalization)** to conform numerical magnitudes for geometric distance calculations.

#### C. Unsupervised Learning: K-Means Clustering
Clustering identifies hidden groupings within unstructured data. To identify distinct academic personas (At-Risk, Average, High Efficacy), we invoked:
*   **K-Means Algorithm:** Operates by iteratively assigning students to `K` centroids based on Euclidean distance, optimizing the Within-Cluster Sum of Squares (WCSS).
*   **The Elbow Method:** A mathematical technique visualizing WCSS plotting to definitively isolate the ideal geometry fragmentation structure, locating optimal stability natively at K=3.
*   We visualize these results dimensionally via a 3D scatter projection, enabling stakeholders to observe geometric clustering of different student capacities.

#### D. Supervised Learning: Random Forest Classification
To predict academic profiles exclusively via disjoint demographic data, we constructed an active supervised model:
*   **Random Forest Classifier:** A highly resilient ensemble paradigm that constructs expansive multitudes of individual standard decision trees to circumvent internal overfitting (Bagging), enabling highly explainable node traversal output voting mechanics. The choice of RF optimizes execution transparency over black-box methodologies.
*   **Stratification & Train-Test Splitting:** Implementing an 80/20 train/test protocol ensures rigorous isolation of validation sets, structurally prohibiting feature leakage.
*   **Evaluation:** Extracted Feature Importances inform which demographics exert the most structural leverage, measured critically via the Confusion Matrix and Classification Reporting constructs.

### 4. Recommendations and Insights (Prescriptive Analytics)
Extracted statistics dictate real-world interventions:
1.  **Socioeconomic Support:** The data highlights `Lunch Status` as a critical predictor; we suggest expansive resource redistribution regarding auxiliary meals and subsidized curriculum support.
2.  **Structural Preparedness:** Mandatory integration of standard test-prep environments bridges immediate gaps for at-risk profiles.

### 5. Architectural Technologies 
- **Python Data Ecosystem:** `pandas`, `numpy`.
- **Machine Learning Subsystem:** `scikit-learn` (KMeans, RandomForestClassifier, Processing).
- **Interactive Visualizations:** `plotly`, `seaborn`, `matplotlib`.
- **Frontend Container:** `streamlit`.

*Internal Document for Hackathon Deliberation Panel.*

# Presentation Slides: Learning Behaviour Profiling
*Problem Statement 5: Student Learning Behaviour Analysis*

---

## Slide 1: Introduction
**Title:** Learning Behaviour Profiling
**Subtitle:** Data-Driven Student Segmentation & Recommendations
* Domain: Smart Education
* Problem Statement 5: Student Learning Behaviour Analysis
* **Objective:** Understand student learning patterns by identifying key learning factors (like test preparation, parental education, gender) and clustering students into distinct learning profiles to provide targeted institutional recommendations.
* Our dashboard identifies "At-Risk" students early without relying on subjective biases.

---

## Slide 2: Project Workflow & Steps
**Title:** Our Methodology
1. **Data Ingestion:** Loaded the Student Performance Dataset (1,000 records).
2. **Preprocessing:** Cleaned data, dropped irrelevant fields (like lunch type), derived new features (average score), and applied Label Encoding / Z-Score Normalization.
3. **Exploratory Data Analysis (EDA):** Visualized distributions, demographics, and found strong correlations (e.g., Test Preparation drastically improves scores).
4. **Machine Learning:** Used Unsupervised Learning to segment students based solely on performance.
5. **Prescriptive Analytics:** Mapped the discovered clusters to specific, actionable recommendations (Risk Assessment Tool).

---

## Slide 3: Algorithms & Techniques Used
**Title:** Algorithms & Data Science Techniques
* **K-Means Clustering:** The core Unsupervised ML algorithm used to group students into 3 performance profiles (At-Risk, Average, High Performer) based strictly on data patterns, rather than pre-defined labels.
* **The Elbow Method (WCSS):** Used to mathematically validate that K=3 is the optimal number of groups for this dataset.
* **Principal Component Analysis (PCA):** A dimensionality reduction technique used to compress 3D scores into 2D, allowing us to visually prove that the clusters are well-separated.
* **StandardScaler & LabelEncoder:** Crucial for standardizing numerical data and converting text demographics into ML-readable formats.

---

## Slide 4: Future Scope
**Title:** Future Scope & Enhancements
* **Real-time Database Integration:** Connect the model directly to schools' internal grading databases for live updates and continuous risk assessment.
* **Predictive Grading system:** Add Supervised Learning modules to predict end-of-semester scores based on early-term quizzes and demographic profiles.
* **Automated Alert System:** Send automated emails or dashboard alerts to teachers when a student drops into the "At-Risk" cluster.
* **Granular Subject Pathways:** Generate subject-specific customized practice sheets based on individual feature weaknesses.

---

## Slide 5: Conclusion
**Title:** Conclusion & Impact
* We successfully built a robust, scalable clustering dashboard.
* **Impact for Educators:** Data-backed insights reduce guesswork. Teachers can allocate resources (like tutoring or advanced AP classes) precisely where they are needed most.
* **Impact for Students:** Identifies struggling students *before* they fail a final exam, and provides High Performers with the correct pathways for enrichment.
* *Smart Education is not just about collecting data, it is about acting on it.*

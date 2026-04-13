# Mini Project Report

**Name and PRN no:** [Insert Name] - [Insert PRN]  
**Name and PRN no:** [Insert Name] - [Insert PRN]  
**Name and PRN no:** [Insert Name] - [Insert PRN]  
**Name and PRN no:** [Insert Name] - [Insert PRN]  

---

### 1. Title of the Project
**Learning Behaviour Profiling: A Smart Education Analytics Dashboard**

### 2. Problem Statement
The project addresses **Problem Statement 5: Student Learning Behaviour Analysis**. In modern education, identifying students who require early intervention is challenging. This project aims to analyze learning factors (demographics, preparation habits) and performance data to discover natural student segments and provide data-driven recommendations.

### 3. Objective
*   To understand student learning patterns and identifying key factors (like test preparation and parental education) that influence academic success.
*   To implement an **Unsupervised Machine Learning** model to group students into distinct learning profiles (At-Risk, Average, High Performer).
*   To provide educators with an interactive tool for **Risk Assessment** and institutional recommendations.

### 4. Methodology
The project follows a standard Data Science pipeline:
1.  **Data Ingestion:** Loading the Student Performance dataset.
2.  **Data Preprocessing:** Cleaning headers, dropping irrelevant administrative features (like lunch type), and engineering new features (Average Score, Academic Status).
3.  **Exploratory Data Analysis (EDA):** Performing multivariate analysis using Histograms, Box Plots, and Correlation Heatmaps to find patterns.
4.  **Feature Transformation:** Converting categorical text into numeric IDs (Label Encoding) and normalizing scores to a standard scale (StandardScaler).
5.  **K-Means Clustering:** Applying the K-Means algorithm to segment the data.
6.  **Visualization:** Using Principal Component Analysis (PCA) to reduce data dimensions for visual cluster verification.
7.  **Prescriptive Analytics:** Generating actionable recommendations based on cluster profiles.

### 5. Dataset, Tools & Technologies Used
*   **Dataset:** Student Performance Dataset (1,000 students, background information, and scores in Math, Reading, and Writing).
*   **Programming Language:** Python 3.x
*   **Libraries:** 
    *   `Pandas` & `NumPy` (Data Manipulation)
    *   `Scikit-Learn` (Machine Learning: KMeans, PCA, Scaler, Encoder)
    *   `Plotly` (Interactive 3D and 2D Visualizations)
    *   `Seaborn` & `Matplotlib` (Heatmaps and Statistical Plots)
*   **Frontend Framework:** Streamlit (Web UI)
*   **Version Control:** Git & GitHub

### 6. Implementation
The project is implemented using a **Modular Architecture**:
*   `app.py`: The main orchestrator and UI layout.
*   `data_loader.py`: Handles data cleaning and z-score normalization.
*   `eda.py`: Contains 5 specialized tabs for comprehensive data exploration.
*   `clustering.py`: Implements the **Elbow Method** to find optimal K and fits the K-Means model.
*   `risk_checker.py`: An interactive module where users can input scores to "predict" the cluster of a new student.
*   `recommendations.py`: Logic for translating cluster insights into institutional strategies.

### 7. Results
*   **Optimal Clusters:** The Elbow Method mathematically confirmed **K=3** as the optimal number of student groups.
*   **Segments Identified:** 
    *   **Cluster 1 (At-Risk):** Students with low scores, typically lacking test preparation.
    *   **Cluster 2 (Average):** Mid-tier performers with mixed backgrounds.
    *   **Cluster 3 (High Performer):** Students with consistently high scores across all subjects.
*   **Visual Proof:** The **PCA 2D Plot** successfully showed clear separation between the three identified segments, validating the model's reliability.
*   **Interactive Testing:** The Risk Assessment tool allows for real-time profiling of individual student scores.

### 8. Conclusion
The project successfully demonstrates how Unsupervised Learning can be used to profile student behavior without pre-existing labels. By using K-Means clustering and PCA visualization, we have created a tool that allows educators to move from "Guesswork" to "Data-Driven Intervention." This approach ensures that At-Risk students are identified early and High Performers are provided with the necessary enrichment pathways.

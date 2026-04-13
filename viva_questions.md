# Hackathon Viva Defense Guide
# Problem Statement 5: Student Learning Behaviour Analysis

---

## CRITICAL — UNDERSTAND YOUR PROJECT FIRST (READ THIS BEFORE ANYTHING)

**What kind of project is this?**
This is a **Clustering project using Unsupervised Machine Learning (K-Means)**.

It is NOT a prediction model. It does NOT predict a label from labelled training data.
It DISCOVERS natural groups in the data.

**Do we have a training set and testing set?**
No. K-Means is unsupervised — it does not learn from labelled examples.
There is no train/test split because there is nothing to "test accuracy" against.

**Do we have accuracy?**
No, not in the traditional sense. We do not predict something we already know the answer to.
Instead we validate our choice of K using the Elbow Method (WCSS / Inertia).

**What exactly does the app do?**
1. Loads and cleans the dataset (data preprocessing).
2. Performs EDA — charts to understand the data patterns.
3. Runs K-Means to group students into At-Risk / Average / High Performer.
4. Visualises clusters in 3D and 2D (via PCA).
5. Lets users enter scores and get the student's cluster profile (Risk Assessment).
6. Gives institutional recommendations per cluster.

**What is the Risk Assessment tool then?**
It uses `kmeans.predict()` — this takes a new student's scaled scores and assigns them
to the nearest cluster centre. It is still clustering-based, not a separate trained model.
It's the same K-Means model, just being used on a new data point.

---

## SECTION 1: Project Overview

**Q: What is your project about in one sentence?**

A: We analysed a dataset of 1000 students and used K-Means clustering to automatically group them into three learning profiles — At-Risk, Average, and High Performers — based on their subject scores, and then provided targeted educational recommendations for each group.

---

**Q: Which problem statement does it address?**

A: Problem Statement 5 — Student Learning Behaviour Analysis, under the Smart Education domain. The goal is to identify learning factors and cluster students, then use those clusters to generate personalised recommendations.

---

**Q: What is the dataset?**

A: The Student Performance Dataset with 1000 rows and 8 columns:
- Categorical: gender, race/ethnicity, parental level of education, lunch, test preparation course
- Numerical: math score, reading score, writing score (all 0–100)

We dropped the lunch column in preprocessing (explained below).

---

## SECTION 2: Data Preprocessing

**Q: Walk me through exactly what data preprocessing you did.**

A:

**Step 1 — Loaded the CSV**
Used `pd.read_csv("data/StudentsPerformance.csv")` to read the file into a pandas DataFrame.

**Step 2 — Cleaned column names**
Used `df.columns.str.lower().str.replace(" ", "_")` to convert all headers like "parental level of education" into clean "parental_level_of_education". This prevents errors in code.

**Step 3 — Dropped the lunch column**
The lunch column (standard vs free/reduced) is a school administration decision based on family income. It does not tell us anything about how a student studies or behaves as a learner. Our project is specifically about "Learning Behaviour Profiling", so we only kept features relevant to learning behaviour. Lunch does not qualify — we dropped it.

**Step 4 — Created new features (Feature Engineering)**
- `average_score = mean(math, reading, writing)` — gives us a single overall performance metric
- `total_score = sum(math, reading, writing)` — for dashboard display
- `academic_status = "Pass" if average >= 50 else "Fail"` — binary label for EDA charts

**Step 5 — Label Encoding**
Used `LabelEncoder().fit_transform(column)` on categorical columns (gender, race, parental education, test prep). This converts text like "male"/"female" into numbers like 0/1. Machine learning algorithms only understand numbers.

**Step 6 — Standard Scaling (Z-score Normalization)**
Used `StandardScaler().fit_transform(score_columns)` on the three numeric score columns.

Z-score formula: (value − mean) / standard deviation

This puts all columns on the same scale. Without this, math scores (0–100) would dominate over encoded categoricals (0–5) just because the numbers are bigger, which would bias the distance calculations in K-Means.

---

**Q: Why did you drop the lunch column?**

A: Lunch type (standard vs free/reduced) is assigned by school administration based on family income — it is not something the student decides and it does not reflect any learning behaviour. Since our project is called "Learning Behaviour Profiling", we keep only columns that actually describe how a student engages with their education. Lunch does not qualify, so we removed it to keep the model focused.

---

**Q: What is Label Encoding and why not One-Hot Encoding?**

A: Label Encoding maps each category to a number (e.g. "male"=0, "female"=1). One-Hot Encoding creates a new binary column for each category.

We chose Label Encoding because K-Means works on euclidean distances. Adding many extra binary columns from One-Hot Encoding can distort distance calculations in high-dimensional space. Label Encoding keeps the number of features manageable and works fine for our tree-free clustering approach.

---

**Q: What is Standard Scaling? Why do you need it for K-Means?**

A: Standard Scaling (Z-score normalization) converts all values so they have mean = 0 and standard deviation = 1.

K-Means measures the distance between points. If math scores range 0–100 but an encoded category column ranges 0–4, K-Means would mostly pay attention to math scores and largely ignore the other columns — just because the numbers are bigger. Scaling removes this unfair bias and makes every feature contribute equally to the distance calculation.

The function `scaler.transform()` is also used when a new student score is entered in the Risk Assessment tool — we must use the SAME scaler (same mean and std) to scale new input so distances are consistent.

---

## SECTION 3: EDA

**Q: What is EDA and what did you do in it?**

A: EDA stands for Exploratory Data Analysis. It is the step where we explore and visualise the data before doing any modelling to understand patterns.

We created:
- **Histograms** — to see the distribution of math, reading, and writing scores (are they normally distributed? skewed?)
- **Box Plots** and **Violin Plots** — to compare score spreads across gender, race, parental education, and test prep groups. Box plots show median and IQR; violin plots also show the density shape.
- **Grouped Bar Charts** — to compare all three subjects side by side per group
- **Stacked Bar Charts** — to show pass/fail counts per demographic group
- **Pie Charts** — demographic composition of the dataset
- **Correlation Heatmap** — to see which features are most related to each other
- **Scatter Plots with trend lines** — to visualise relationships between two continuous variables

---

**Q: What were your main EDA findings?**

A:
1. Students who completed the test preparation course scored around 8–12 points higher on average across all subjects — this is the strongest factor.
2. Parental education level shows a clear positive relationship with student performance.
3. Math, reading, and writing scores are highly correlated (~0.80–0.95 Pearson correlation). If a student is good at reading, they are likely good at writing too.
4. The score distributions are roughly bell-shaped (normal), with most students scoring between 55 and 80.

---

**Q: What is a Pearson Correlation?**

A: A number between -1 and +1 measuring how linearly related two variables are.
- +1 = as one goes up, the other always goes up
- 0 = no relationship
- -1 = as one goes up, the other goes down

We used it to show that reading and writing scores are very strongly correlated (around 0.95), meaning students who read well also tend to write well.

---

## SECTION 4: K-Means Clustering

**Q: What is K-Means and why did you choose it?**

A: K-Means is an unsupervised machine learning algorithm that groups data points into K clusters based on similarity (Euclidean distance).

We chose it because:
1. We had no pre-existing labels — no one told us who is "At-Risk" or "High Performer". K-Means discovers these groups by itself.
2. It is simple, fast, and easy to explain and visualise.
3. The results are intuitive — students with similar scores end up together.

---

**Q: How does K-Means work step by step?**

A:
1. Choose K (we chose 3).
2. Randomly place K centre points (centroids) in the score space.
3. Assign each student to the nearest centroid using Euclidean distance.
4. Move each centroid to the average position of all its assigned students.
5. Repeat steps 3 and 4 until the centroids stop moving.

The result is K groups where students within a group are similar to each other.

**Function used:** `KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)`
- `n_clusters=3` — we want 3 groups
- `random_state=42` — so results are reproducible every run
- `n_init=10` — runs the algorithm 10 times with different starting points, keeps the best
- `fit_predict(X)` — fits the model and immediately returns the cluster ID for each row

---

**Q: Why K=3? How did you decide?**

A: We used the **Elbow Method**.

We ran K-Means for K=1 through K=10 and recorded the WCSS (Within-Cluster Sum of Squares), also called Inertia, for each. We plotted this and looked for the "elbow" — the point where the decrease in WCSS slows down. Before that point, adding clusters helps a lot. After it, it barely helps.

In our data, the elbow is clearly at K=3. It also makes logical sense: students naturally fall into Low / Middle / High performance groups.

---

**Q: What is WCSS / Inertia?**

A: WCSS stands for Within-Cluster Sum of Squares. It measures how compact (tight) the clusters are — the sum of squared distances from each point to its cluster centroid.

Lower WCSS = students within each cluster are more similar to each other = better clustering.
As K increases, WCSS always decreases. But the decrease slows down at the optimal K. That's the elbow.

**Code:** `km.inertia_` after calling `km.fit(X)` gives the WCSS value.

---

**Q: Is K-Means supervised or unsupervised learning?**

A: Unsupervised. We give it no labels or correct answers. It finds groups by itself using only the feature values (scores). The labels "At-Risk / Average / High Performer" are applied by us AFTER clustering — we just look at which cluster has the lowest average math score and call it At-Risk, and so on.

---

**Q: Do you have training data and testing data?**

A: No. K-Means is unsupervised — the concepts of train/test split and accuracy do not apply in the traditional sense. We validate our choice of K using the Elbow Method instead of accuracy.

---

## SECTION 5: PCA

**Q: What is PCA and why did you use it?**

A: PCA stands for Principal Component Analysis. It is a dimensionality reduction technique — it takes many dimensions and compresses them into fewer while keeping as much variance (information) as possible.

We used it purely for visualisation. Our students exist in 3-dimensional space (math, reading, writing scores). PCA reduces that to 2 dimensions (two Principal Components) so we can draw a 2D scatter plot showing how well the clusters are separated.

**Function used:** `PCA(n_components=2).fit_transform(X)` — fits PCA on the scaled scores and returns 2D coordinates.

**Important:** PCA is used here ONLY for visualisation, not for clustering. The K-Means algorithm still runs on the original 3D scaled scores.

---

**Q: Why can you use PCA here? Aren't labels needed?**

A: PCA itself is also unsupervised — it does not need labels. It just finds the directions of maximum variance in the data. After K-Means assigns cluster labels, we colour the PCA scatter plot by those labels to show the cluster separation visually. The labels are used for colouring only, not for PCA computation.

---

## SECTION 6: Risk Assessment Tool

**Q: You have a Risk Assessment tool — isn't that a prediction model?**

A: It looks like a prediction, but it is still K-Means based — not a separate trained model. Here is how it works:

1. We train K-Means on all 1000 students (using their scores).
2. When a user enters new scores, we scale those scores using the same StandardScaler.
3. We call `kmeans.predict(X_new)` which finds the nearest cluster centroid to the new scores.
4. We show the profile label for that cluster.

`kmeans.predict()` does not predict a class label like a supervised model does. It just finds the closest cluster centre. It is still clustering — we are asking "which existing group is this new student most similar to?"

---

**Q: Do you have accuracy for the Risk Assessment?**

A: No, because we do not have ground truth labels to compare against. The original dataset has no "At-Risk" or "High Performer" column — those labels were created by us based on the clusters. So there is nothing to measure accuracy against in the traditional sense.

---

## SECTION 7: Technical Implementation (What Functions Did You Use?)

**Q: What Python functions and libraries did you use?**

A:

**Libraries:**
- `pandas` — for reading CSV files and data manipulation (groupby, filtering, merging)
- `numpy` — for numerical operations (np.where for conditional column creation, np.array for new data points)
- `scikit-learn` — for all ML operations:
  - `LabelEncoder` — encodes text categories to numbers
  - `StandardScaler` — Z-score normalization
  - `KMeans` — the clustering algorithm
  - `PCA` — dimensionality reduction for 2D visualisation
- `plotly.express` — for all interactive charts (bar, pie, scatter, violin, histogram, scatter_3d)
- `matplotlib + seaborn` — for the correlation heatmap
- `streamlit` — for the entire web dashboard (UI, widgets, layout)

**Key functions used:**
| Function | What it does |
|----------|-------------|
| `pd.read_csv()` | Reads the CSV file into a DataFrame |
| `df.columns.str.lower()` | Cleans column names |
| `df[cols].mean(axis=1)` | Row-wise average across selected columns |
| `np.where(condition, a, b)` | Creates a column with conditional values |
| `LabelEncoder().fit_transform(col)` | Converts text to integer codes |
| `StandardScaler().fit_transform(X)` | Scales features to mean=0, std=1 |
| `KMeans(n_clusters=3).fit_predict(X)` | Fits K-Means and returns cluster IDs |
| `KMeans.inertia_` | Returns WCSS for the fitted model |
| `kmeans.predict(X_new)` | Assigns a new data point to a cluster |
| `PCA(n_components=2).fit_transform(X)` | Reduces to 2D for visualisation |
| `df.groupby(col).mean()` | Groups rows and computes averages |
| `df.corr()` | Computes Pearson correlation matrix |
| `df.describe()` | Computes count, mean, std, quartiles |
| `st.slider()` | Creates a slider widget in Streamlit |
| `st.session_state` | Stores model across Streamlit page changes |

---

## SECTION 8: General Data Science Questions

**Q: What is supervised vs unsupervised learning?**

A:
- **Supervised**: You give the model labelled data (input + correct answer). It learns to predict the answer. Example: given exam scores, predict if a student passes.
- **Unsupervised**: No labels. The model finds patterns by itself. Example: group students by score similarity without telling it what the groups should be.

We used **unsupervised** (K-Means) because we had no pre-existing cluster labels.

**Q: What is overfitting?**

A: Overfitting is when a model memorises the training data so well that it fails on new data. It is less of a concern in K-Means since we are not building a predictive model. But in K-Means, using too many clusters (high K) could be seen as "overfitting" — every student becomes their own cluster, which is meaningless.

**Q: What is Streamlit?**

A: Streamlit is a Python library that converts Python scripts into interactive web dashboards. You write Python functions and it auto-generates a web page with charts, tables, sliders, and buttons — no HTML, CSS, or JavaScript needed beyond optional customisation.

**Q: What is modular code structure?**

A: We split the application into separate Python files inside a `modules/` folder. Each file handles one responsibility:
- `data_loader.py` — loading and preprocessing data
- `eda.py` — all EDA charts
- `clustering.py` — K-Means, Elbow, PCA
- `risk_checker.py` — user input risk assessment
- `recommendations.py` — institutional recommendations

`app.py` is the main entry point that imports and calls these modules. This makes the code easier to read, debug, and explain to judges.

# Hackathon Viva Defense Guide
# Problem Statement 5: Student Learning Behaviour Analysis

---

## FIRST — UNDERSTAND WHAT YOUR PROJECT IS

This is a **Clustering project**, NOT a prediction project.

- We do NOT predict anything.
- We do NOT have a training set and a testing set.
- We do NOT have accuracy percentages.
- We are using **K-Means**, which is an **Unsupervised Machine Learning** algorithm.
- This means we are just FINDING groups (clusters) in the data that already exist — students who naturally score similarly get grouped together.
- There is no "correct answer" we are trying to match. We are discovering patterns.

If a judge asks "What is your accuracy?" — say:
> "Since we are using K-Means Clustering, which is unsupervised learning, accuracy in the traditional sense does not apply. Instead, we evaluate cluster quality using the Elbow Method and WCSS (inertia) to confirm K=3 is the optimal number of clusters."

---

## SECTION 1: About the Project

**Q: What is your project about?**

A: Our project is called "Learning Behaviour Profiling" under the Smart Education domain. We took a real dataset of 1000 students with their scores in math, reading, and writing, along with some background details like gender, parental education, and whether they completed a test preparation course. We used machine learning to automatically group students into three categories — At-Risk, Average, and High Performers — without any prior labels. Then we gave specific recommendations for each group so schools know exactly how to help their students.

---

**Q: What is the problem you are solving?**

A: The problem is that schools don't always know which students need help until exam results come out — and by then it might be too late. Our system uses data to identify struggling students early based on their background and performance patterns. This allows teachers and schools to take action before it becomes a bigger problem.

---

**Q: What dataset did you use and what does it contain?**

A: We used the Student Performance Dataset. It has 1000 rows (one per student) and 8 columns:
- `gender`: male or female
- `race/ethnicity`: group A, B, C, D, or E
- `parental level of education`: from high school to master's degree
- `lunch`: standard or free/reduced (dropped — explained below)
- `test preparation course`: completed or none
- `math score`: 0–100
- `reading score`: 0–100
- `writing score`: 0–100

---

## SECTION 2: Data Preprocessing

**Q: What data preprocessing did you perform?**

A: We did the following steps in order:

1. **Cleaned column names** — We converted all column names to lowercase with underscores so the code runs cleanly without special character issues.

2. **Dropped the lunch column** — The lunch column tells us whether a student gets standard or free/reduced lunch, which is a school administration decision. It does not reflect any learning behaviour of the student. Since our project is specifically about "Learning Behaviour Profiling", we only kept columns that are relevant to how a student learns. So we removed lunch.

3. **Created new features** — We calculated `average_score` (mean of math, reading, writing) and `total_score` (sum of all three). We also created `academic_status` — Pass or Fail — based on whether the average was above 50. This helped us in visualization and analysis.

4. **Label Encoding** — Categorical text columns like gender, race, parental education, and test prep course were converted to numbers using Label Encoding. For example, "male" becomes 0, "female" becomes 1. Machine learning models work only with numbers, not text.

5. **Z-score Normalization (StandardScaler)** — We scaled all the score columns so they all have the same mean (0) and standard deviation (1). This is critical for K-Means because K-Means uses distance. If one column has values 0–100 and another has values 0–5, the big column will dominate and unfairly bias the grouping. Scaling removes this problem and makes all features equally important.

---

**Q: Why did you drop the lunch column?**

A: We specifically dropped it because our project is about "Learning Behaviour" — we want to understand how students study and perform based on their academic environment. Lunch type is a school administration detail. A student does not choose their lunch type; it is assigned based on family income. It does not tell us anything about how they behave as a learner. So we removed it to keep our analysis focused on genuine learning behaviour indicators.

---

**Q: What is Label Encoding?**

A: Label Encoding converts text categories into numbers. For example:
- "none" test prep → 0, "completed" → 1
- "male" → 0, "female" → 1

This allows machine learning algorithms to process these columns because they only understand numbers, not words.

---

**Q: What is Z-score Normalization / Standard Scaling?**

A: It is a way to put all numbers on the same scale.

Formula: (value - mean) / standard deviation

For example, if math scores have an average of 66 and standard deviation of 15, a score of 81 becomes (81-66)/15 = 1.0. This means "1 standard deviation above average."

Without this, K-Means would mostly look at math scores (big range) and ignore other smaller-value encoded columns. Scaling makes everything fair.

---

## SECTION 3: Exploratory Data Analysis (EDA)

**Q: What is EDA and why did you do it?**

A: EDA stands for Exploratory Data Analysis. It is the step where we understand the data before doing any modeling. We look at charts, graphs, and statistics to see patterns. For example, we check if students who completed test preparation score better (they do), or if parental education matters (it does, to some extent). EDA helps us form hypotheses and validate our assumptions.

---

**Q: What did you find in EDA?**

A: Our key findings were:
- **Test preparation** is the strongest factor — students who completed it scored noticeably higher on average across all three subjects.
- **Parental education** has a clear positive trend — the higher the parent's education, the better the student tends to score.
- **Math, reading, and writing scores are highly correlated** — a student who is strong in one subject tends to be strong in all three. The Pearson correlation between reading and writing was around 0.95.
- Score distributions are roughly bell-shaped (normal distribution), with most students scoring between 50 and 80.

---

**Q: What charts did you create?**

A:
- **Histograms** for each subject score to see the distribution (how many students got each score range)
- **Box Plots and Violin Plots** to compare score spreads across gender, parental education, and test prep groups
- **Grouped Bar Charts** to compare math, reading, and writing scores side by side for each group
- **Stacked Bar Charts** showing pass/fail count across groups
- **Pie Charts** to show the proportion of students in each demographic category
- **Correlation Heatmap** to show which scores are most related to each other
- **Scatter Plots** with trend lines to look at relationships between two scores

---

**Q: What is a Pearson Correlation?**

A: It is a number between -1 and 1 that tells you how strongly two variables are related.
- +1 means as one goes up, the other always goes up.
- 0 means no relationship.
- -1 means as one goes up, the other goes down.

In our data, reading and writing had a correlation of ~0.95, which means they are very strongly related.

---

## SECTION 4: K-Means Clustering

**Q: Why did you use K-Means?**

A: Because we did not have any pre-existing labels saying who is "At-Risk" or "High Performer". We needed the algorithm to discover these groups on its own just by looking at the score patterns. K-Means is designed exactly for this — it is unsupervised, meaning it finds natural groupings in the data without needing any labels. It is also simple, fast, and the results are easy to explain.

---

**Q: What is K-Means in simple words?**

A: Imagine you have 1000 students plotted in a 3D space (math, reading, writing scores as three axes). K-Means says: "I will group these into K clusters." Here is how it works:

1. Pick K random points as initial cluster centers.
2. Assign every student to the nearest center (using Euclidean distance).
3. Move each center to the average position of its assigned students.
4. Repeat steps 2–3 until the centers stop moving.

The result is K groups where students within each group are similar to each other and different from students in other groups.

---

**Q: Why K=3? How did you decide?**

A: We used the **Elbow Method**. We ran K-Means for K=1, 2, 3, ... all the way to 10 and recorded the WCSS (Within-Cluster Sum of Squares) for each. We plotted this and found that the WCSS drops sharply up to K=3, and then the decrease slows down significantly. The point where the curve "bends" like an elbow is the optimal K. In our data that was clearly K=3. Also, K=3 makes logical sense — students naturally fall into Low, Average, and High performance groups.

---

**Q: What is WCSS / Inertia?**

A: WCSS stands for Within-Cluster Sum of Squares. It measures how similar students within the same cluster are to each other. The lower the WCSS, the tighter and more compact the clusters. As we increase K, WCSS always goes down, but at some point adding more clusters gives very little benefit. That "elbow" point tells us the best K to use.

---

**Q: Is K-Means supervised or unsupervised?**

A: It is **unsupervised**. That means we do not give it any labels or correct answers. It discovers the groupings by itself purely based on the data patterns. Our labels ("At-Risk", "Average", "High Performer") were applied by us AFTER clustering — we just looked at which cluster had the lowest and highest average math scores and labelled them accordingly.

---

**Q: What is the difference between supervised and unsupervised learning?**

A: 
- **Supervised learning**: You give the model labeled data (input + correct answer). The model learns to predict the answer. Example: given scores, predict if a student will pass.
- **Unsupervised learning**: There are no labels. The model finds patterns on its own. Example: grouping students by similarity without telling it what the groups should be.

We used unsupervised learning (K-Means) because we did not have pre-defined groups in our data.

---

**Q: Do you have training data and testing data?**

A: No, because K-Means is unsupervised learning. The concept of train/test split is used in supervised learning where you train a model to predict something and then test it on new data to measure accuracy. In K-Means we just run the algorithm on all the data to find the natural groups. There is no prediction, so there is no need to split the data.

---

**Q: What is your model accuracy?**

A: We do not have an accuracy metric because this is a clustering project, not a classification or regression project. In unsupervised learning, we do not predict a known answer, so there is nothing to compare against. Instead, we validated our choice of K=3 using the Elbow Method, which confirmed that 3 clusters provide the best balance between simplicity and cluster quality.

---

**Q: How did you label the clusters?**

A: After K-Means ran, we had 3 unlabeled clusters. We then calculated the mean math score for each cluster. The cluster with the lowest mean was labelled "At-Risk", the middle one was "Average", and the highest was "High Performer". This is a simple and logical way to assign meaningful names to what the algorithm found mathematically.

---

## SECTION 5: Recommendations

**Q: How did you generate recommendations?**

A: Based on what the data showed us about each cluster, we created targeted suggestions:
- **At-Risk students** have low scores and usually missed test prep → we recommend mandatory tutoring and test preparation enrollment.
- **Average students** are in the middle → we suggest supplementary exercises and peer study groups.
- **High Performers** consistently score above 80 → we recommend advanced coursework and mentoring roles.

These are directly derived from the data, not made up.

---

**Q: Is this a prediction model?**

A: No. We are not predicting anything. We are **grouping** students based on their existing data. This is called clustering or segmentation. A prediction model would say "given this new student's background, I predict they will fall in this category." We are not doing that — we are analyzing existing patterns and identifying groups that already exist in the data.

---

## SECTION 6: General Data Science Questions

**Q: What is a decision tree?**

A: A decision tree is a supervised learning algorithm that splits data into branches based on feature values, like a flowchart. We did not use it in this project but it is often used for classification problems.

**Q: What is overfitting?**

A: Overfitting is when a model learns the training data so well that it fails on new, unseen data. It memorizes instead of generalizing. In K-Means this is less of a concern because we are not training on one set and predicting on another.

**Q: What libraries did you use?**

A:
- `pandas` — for loading and manipulating the dataset
- `numpy` — for numerical operations
- `scikit-learn` — for K-Means, StandardScaler, and LabelEncoder
- `plotly` — for interactive charts
- `matplotlib` and `seaborn` — for the correlation heatmap
- `streamlit` — for the web dashboard interface

**Q: What is Streamlit?**

A: Streamlit is a Python library that lets you build interactive web dashboards very easily. You write Python code and it automatically creates a web page with charts, tables, and widgets. We used it to build the entire front-end without any HTML or JavaScript.

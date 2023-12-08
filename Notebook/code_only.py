# %% [markdown]
# # Evaluation Exam

# %% [markdown]
# ### Theory Section
# 
# - Define the following terms:
# 
#     - Supervised learning: 
#     
#         This is a subset of machine learning where the training datasets are pre-labeled. These labels guide the algorithms in classifying data or predicting outcomes accurately. This method is particularly useful when the desired output is known, and the algorithm needs to learn the mapping from input to output.
#     
#     - Unsupervised learning:  
# 
#         Contrary to supervised learning, this approach involves working with datasets that are not pre-labeled. In this scenario, the algorithms autonomously identify hidden patterns or clusters within the data, without any human intervention. 
#         
#     - Classification:
# 
#         Classification is a task that involves predicting the label or class of a given unlabeled data point. Formally speaking, a classifier is a function $M$ that predicts the class label $\hat{y}$ for a given input example $x$ in $\hat{y} = M(x)$, where $\hat{y} = \{c_1, c_1,...,c_k\}$ (each $c_i$ is a categorical attribute value). 
#         
#         As a form of supervised learning, this approach requires a training process where the model is exposed to training set, which includes data points already assigned with correct class labels. Through this training, the model learns to identify and categorize these points. Once the training is complete, the model is capable of automatically predicting the class for any new, unlabeled data point
# 
#     - Regression:
# 
#         Given a set of variables $X_1, X_2, ..., X_d$, called the predictor or independent variables, and given a variable $Y$ where $Y \in \mathbb{R}$, called the response or dependent variable, the goal of the regression is to predict the response variable based on the independent variables. In general:
# 
#         $Y = f(X_1, X_2, ..., X_d) + \epsilon = f(\textbf{X}) + \epsilon$
# 
#         Where $\textbf{X}$ is the multivariate random variable, and $\epsilon$ is a random error. Also, depending on the type of regression model (linear, logistic, etc.), the data used in these models need to adhere to certain statistical assumptions.
# 
#     - Clustering:
# 
#         Formally speaking, given a dataset $\textbf{D}$ with $n$ points $\textbf{x}_i$, and given the number of $k$ clusters, the goal of clustering is to partition the dataset into $k$ groups without using any kind of pre-labeled points, which differentiates it from classification tasks. The clusters are denoted by $C = \{C_1, C_2,..., C_k\}$ and, for each $C_i$, there exists a representative point defined as the mean or centroid $\mu_i$:
# 
#         $\mu_i = \frac{1}{n_i} \sum_{x_j \in C_i} \textbf{x}_j$
# 
#     - Dimensionality reduction: 
#     
#         Dimensionality reduction involves transforming data from a high-dimensional space into a lower-dimensional space, aiming to retain the most meaningful properties of the original dataset. Formally speaking, the goal of dimensionality reduction is to find an $r$-dimensional basis that aproximates $\textbf{x}'_i$ over all the points of $\textbf{x}_i \in \textbf{D}$ while minimizing the error $\epsilon_i = \textbf{x}_i - \textbf{x}_i'$.
# 
# - Explain the bias-variance tradeoff in machine learning.
# 
#     This concept describes the relationship between a model's complexity and the accuracy of its predictions. Generally, increasing the number of parameters in a model can lead to a better fit on the training set, thereby reducing bias. However, the trade-off is that such models often exhibit higher variance. In other words, when exposed to different sets of samples, these more complex models are likely to produce varying results.
# 
# - Describe the steps involved in the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology.
# 
#     - Business Understanding: Define the project objectives and requirements from a business perspective.
#     - Data Understanding: Collect and familiarize with the data, identifying quality issues and initial insights.
#     - Data Preparation: Clean and preprocess the data to create a final dataset for modeling.
#     - Modeling: Select and apply various modeling techniques, tuning them to optimal parameters.
#     - Evaluation: Assess the model against the business objectives, ensuring it meets the required goals.
#     - Deployment: Implement the data mining solution, plan for its maintenance, and monitor its performance.

# %% [markdown]
# ### Practical Section
# 
# - Business Case: You are working for a retail company, and they want to identify customer segments for targeted marketing campaigns. They have provided you with a dataset containing customer information such as age, gender, income, and purchase history. Your task is to perform customer segmentation using supervised learning techniques.
# 
# Steps to follow:
# 
# 1. Which suitable supervised learning algorithm to cluster the customers would you choose? Give at least three options.
# 2. Choose the best and run it.
# 3. Analyze and interpret the results of the clustering.
# 4. Provide recommendations on how the company can utilize the customer segments for targeted marketing campaigns.
# 
# 
# 

# %% [markdown]
# ### Exploratory Data Analysis
# 
# **1. To propose three suitable supervised learning algorithms, we should first conduct an exploratory data analysis to uncover meaningful insights that could inform our choice of the most appropriate model.**

# %%
# Library for data manipulation
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Load the data
df = pd.read_csv("data.csv", index_col="customer_id")

# %%
df.head()

# %% [markdown]
# Since the data has already been cleaned, further improvements are not required at this stage.

# %%
df.info()

# %% [markdown]
# According to the description, there is no target variable since this is a clustering task, suggesting that all columns are features. However, the dataset contains mixed types of data: categorical (`gender and` and `purchase_history`) and numerical (`age` and `income`), with the latter being continuous variables.

# %%
df.shape

# %% [markdown]
# The dataset contains at least 100 samples but fewer than 10,000, indicating the need for an algorithm that is effective with smaller datasets.

# %%
# Select numerical columns
numerical_columns = df.select_dtypes(include=['int64']).columns

# %%
# Adjusting the layout to place one plot beside the other
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Plotting 'income' histogram and KDE in the first subplot
sns.histplot(df["income"], kde=True, kde_kws=dict(cut=3), stat="density", color="blue", ax=axs[0])
axs[0].set_title('Income Distribution')

# Plotting 'age' histogram and KDE in the second subplot
sns.histplot(df["age"], kde=True, kde_kws=dict(cut=3), stat="density", color="red", ax=axs[1])
axs[1].set_title('Age Distribution')

plt.tight_layout()
plt.show()

# %% [markdown]
# Income Distribution (Blue Plot):
# 
# - The distribution of income appears to be right-skewed, suggesting that a larger number of individuals earn less, with fewer high earners.
# - The peak of the density curve is on the lower end of the income scale, which indicates the mode of the income distribution is lower than the mean.
# - There is a tail extending towards the higher income values, indicating the presence of individuals with significantly higher incomes than the average.
# 
# Age Distribution (Red Plot):
# 
# - The age distribution appears to be slightly left-skewed, indicating that there are more young people in the dataset and a smaller number of older individuals.
# - The tallest bar is around the 35-40 age range, which may indicate the mode of the distribution is within this age bracket.
# - The skewness is not as pronounced in the age distribution as it is in the income distribution, but it still suggests a younger demographic overall.

# %%
# Plot for 'gender' value counts
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
df['gender'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')

# Plot for 'purchase_history' value counts
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
df['purchase_history'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Purchase History Distribution')
plt.xlabel('Purchase History')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# %%
# Encoding categorical data
df['gender_encoded'] = LabelEncoder().fit_transform(df['gender'])
df['purchase_history_encoded'] = LabelEncoder().fit_transform(df['purchase_history'])

# Selecting numerical columns for the correlation matrix
numerical_data = df[['age', 'income', 'gender_encoded', 'purchase_history_encoded']]

# Calculating the correlation matrix
correlation_matrix = numerical_data.corr()

# Plotting the correlation matrix
plt.figure(figsize=(5, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# %% [markdown]
# Key observations:
# 
# - The chosen algorithm must be capable of effectively handling datasets with a small number of samples (fewer or equal to 100).
# - The dataset comprises a mix of categorical and numerical data types.
# - The gender distribution within the dataset is notably balanced, with nearly equal numbers of female and male participants.
# - We don't previously know the number of categories.
# - There is no significant correlation among the variables, which likely indicates the absence of collinearity as well.
# 
# Some algorithms that could be used are:
# 
# - KMeans.
# - KModes.
# - K-Prototype.
# - MeanShift.
# - VBGMM.
# - MiniBatch KMeans.
# 
# However, in this case, the most suitable algorithm for our problem is the K-Prototype algorithm. This method is an improvement of both the K-Means and K-Modes clustering algorithms, designed to handle mixed data types. The only downside is that we need to pre-determine the optimal number of categories to assign, but we can use the Elbow Method to handle this.

# %% [markdown]
# ### K-Prototype implementation

# %%
from kmodes.kprototypes import KPrototypes

# %%
# Generate intervals for the age
classes = df['age'].value_counts(bins = 3, sort = False)
classes.index

# %%
# Convert age to bins
df['age_bins'] = pd.cut(df['age'], bins=(25,33,40,47), labels=["26-33","34-40","40+"])
df[['age','age_bins']].drop_duplicates(subset=['age_bins']).reset_index(drop=True)

# %%
features = df[['income', 'gender', 'age_bins', 'purchase_history']].reset_index(drop=True)

# %%
# Get the position of categorical columns
categorical_colums_index = [features.columns.get_loc(col) for col in list(features.select_dtypes(include=['object', 'category']).columns)]

print('Categorical columns           : {}'.format(list(df.select_dtypes(include=['object', 'category']).columns)))
print('Categorical columns position  : {}'.format(categorical_colums_index))

# %%
# Elbow method
cost = []
for cluster in range(1, 10):
    try:
        kprototype = KPrototypes(n_jobs = -1, n_clusters = cluster, init = 'Huang', random_state = 0)
        kprototype.fit_predict(features, categorical = categorical_colums_index)
        cost.append(kprototype.cost_)
    except:
        break

plt.plot(cost)
plt.xlabel('K')
plt.ylabel('c')

# %%
# Library for the locator
from kneed import KneeLocator

# %%
# Cost (sum distance): confirm visual clue of elbow plot
# KneeLocator class will detect elbows if curve is convex; if concave, will detect knees
cost_knee_c3 = KneeLocator(
        x=range(1,10), 
        y=cost, 
        S=0.1, curve="convex", direction="decreasing", online=True)

K_cost_c3 = cost_knee_c3.elbow   
print("Elbow at k =", f'{K_cost_c3:.0f} clusters')

# %% [markdown]
# **2. Model run and results:**

# %%
# Model run with k = 3
kproto_clusters = KPrototypes(n_clusters=3, init = 'Huang', random_state = 0)
result_cluster = kproto_clusters.fit_predict(features, categorical=categorical_colums_index)

# %%
features['Clusters'] = result_cluster
features.head()

# %%
features['Clusters'].value_counts()

# %%
# The volume of each cluster
features['Clusters'].value_counts().plot(kind='bar')

# %%
# Mode of each feature grouped by clusters 
features.groupby(['Clusters']).agg(lambda x: pd.Series.mode(x).iat[0])[['income', 'gender', 'age_bins', 'purchase_history']]

# %%
# Mean of the income feature grouped by clusters
features.groupby(['Clusters'])['income'].mean().round(2)

# %% [markdown]
# **3. Key insights of the results:**
# 
# - Cluster 0: Comprises predominantly younger individuals, characterized by lower average income and purchase history, with a majority being female.
# - Cluster 1: Also composed mainly of females, this cluster falls into a slightly higher age bracket of 34-40 years old, with an average income of $48,000, and a similar medium purchase history. 
# - Cluster 2: This segment is made up of males in the age range of 34-40, with a higher average income of $60,000, and a high purchase history. 
# - Clusters 1 and 2 consist of individuals within the same age range. However, they differ in terms of average income and gender composition; the cluster with a predominantly male demographic exhibits a higher average income.
# - While a higher income generally suggests a more extensive purchase history, this requires further analysis for confirmation. A statistical method such as ANOVA could be appropriate to investigate these relationships in more depth.

# %% [markdown]
# ### Recommendations
# 
# **4. Some general recommendations for targeted marketing campaigns:**
# 
# - Clusters 1 and 2:
#     - Age-Based Marketing: Since both clusters fall within the same age range, consider marketing campaigns that feature products or services relevant to mid-life milestones.
#     - Gender-Specific Approaches: Develop marketing materials that cater to the predominantly female group of Cluster 1 and the male group of Cluster 2.
# 
# - Cluster 0:
#     - Marketing Channels: Leverage social media platforms that are popular among younger demographics.
# 
# - Further analysis:
#     - Income vs. Purchase History Analysis: Conduct an ANOVA test or similar statistical analyses to better understand the relationship between income levels and purchase history.
#     - Segment Deep Dive: Perform a deeper analysis of the clusters to identify sub-segments or niche markets within each cluster, which could reveal more targeted opportunities.

# %% [markdown]
# ### Further analysis: ANOVA

# %% [markdown]
# The subsequent analysis was performed in R and can be found in the `anova.ipynb` file. Our aim is to test the following hypotheses:
# 
# For `age_bins`:
# - $H_0$: There is no significant difference in the average income across different age bins.
# - $H_1$: There is a significant difference in the average income across different age bins.
# 
# For `gender`:
# - $H_0$: There is no significant difference in the average income between genders.
# - $H_1$: There is a significant difference in the average income between genders.
# 
# For `purchase_history`:
# - $H_0$: There is no significant difference in the average income across different levels of purchase history.
# - $H_1$: There is a significant difference in the average income across different levels of purchase history.
# 
# For Interaction Terms (e.g., `age_bins:gender`):
# - $H_0$: There is no significant interaction effect on average income between the categorical variables (e.g., age bins and gender).
# - $H_1$: There is a significant interaction effect on average income between the categorical variables (e.g., age bins and gender).
# 
# In this model, the dependent variable is the income, and the remaining variables serve as the factors.
# 
# Additionally, the ANOVA model must adhere to the following assumptions:
# - Independence of observations.
# - The normality of the residuals are normal.
# - Homoscedasticity.
# 
# Let's assume that the all observations are independent of each other. Now, to verify the normality of the residuals we are going to use the Shapiro-Wilk test with a level of significance of $\alpha = 0.05$:
# 
# | Test                       | Data            | W       | p-value |
# |----------------------------|-----------------|---------|---------|
# | Shapiro-Wilk normality test | residuals(anova) | 0.98858 | 0.5522  |
# 
# Given that the p-value is greater than $\alpha$, we cannot reject the null hypothesis: there's enough evidence that the residuals could be normally distributed. Finally, we are going to verify the homoscedasticity of the data by using the Levene test with a level of significance of $\alpha = 0.05$:
# 
# | <!--/--> | Df &lt;int&gt; | F value &lt;dbl&gt; | Pr(&gt;F) &lt;dbl&gt; |
# |---|---|---|---|
# | group | 17 | 0.6159254 | 0.8702271 |
# | <!----> | 82 |        NA |        NA |
# 
# Since the p-value is greater than $\alpha$, we cannot reject the null hypothesis: there's enough evidence that the variances across different groups are statistically similar. We will now proceed to the ANOVA results:
# 
# | Factor                           | Df |   Sum Sq   |  Mean Sq   | F value | Pr(>F) |
# |----------------------------------|----|------------|------------|---------|--------|
# | age_bins                         |  2 | 2.959e+08  | 147973409  |  1.289  | 0.2812 |
# | gender                           |  1 | 2.874e+08  | 287417989  |  2.503  | 0.1175 |
# | purchase_history                 |  2 | 2.806e+08  | 140314292  |  1.222  | 0.3000 |
# | age_bins:gender                  |  2 | 5.939e+08  | 296953935  |  2.586  | 0.0814 |
# | age_bins:purchase_history        |  4 | 2.166e+08  | 54146998   |  0.472  | 0.7565 |
# | gender:purchase_history          |  2 | 6.473e+07  | 32366436   |  0.282  | 0.7551 |
# | age_bins:gender:purchase_history |  4 | 1.986e+08  | 49641984   |  0.432  | 0.7849 |
# | Residuals                        | 82 | 9.416e+09  | 114827438  |         |        |
# 
# From the ANOVA table, we observe that all p-values are greater than $\alpha$, leading us to fail to reject the null hypothesis. This implies there is no statistically significant difference between the average incomes of the groups.

# %% [markdown]
# ### Model evaluation

# %% [markdown]
# We will treat the clusters as labels and build a classification model on top of them. Ideally, if the clusters are of high quality, the classification model should be able to predict them accurately. Additionally, the model should utilize various combinations of features, indicating that the clusters are not overly simplistic.
# 
# We will use the following tools:
# 
# - The $F_1$ score to measure the distinctiveness of the clusters.
# - SHAP for assessing feature importances.

# %%
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import shap

# %%
# Transform categorical features into the appropriate type
for c in features.columns:
    col_type = features[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        features[c] = features[c].astype('category')

# %%
x = features.loc[:, features.columns != 'Clusters']
clf_kp = lgb.LGBMClassifier(colsample_bytree=0.8, verbose=-1)
cv_scores_kp = cross_val_score(clf_kp, x, features['Clusters'], scoring='f1_weighted')

# %%
print(f'F1 score for K-Prototypes clusters is {round(np.mean(cv_scores_kp), 4)}')

# %% [markdown]
# Since the $F_1$ score is close to 1, it means that the K-Prototype algorithm has produced clusters that are easily distinguishable.

# %%
clf_kp.fit(x, features['Clusters'])
explainer_kp = shap.TreeExplainer(clf_kp)
shap_values_kp = explainer_kp.shap_values(x)
shap.summary_plot(shap_values_kp, x, plot_type="bar", plot_size=(12, 8))

# %% [markdown]
# LightGBM considers income as the most important feature for predicting the labels, but it ignores the other features, suggesting that our model is not very informative. Ideally, we should compare multiple models, but this requires more time and computational resources. Overall, the model is useful, but has some limitations.

# %% [markdown]
# ### References
# 
# - Jia, Z., & Song, L. (2020). Weighted K-Prototypes clustering algorithm based on the hybrid dissimilarity coefficient. Mathematical Problems in Engineering, 2020, 1–13. https://doi.org/10.1155/2020/5143797
# - Miller, I., & Miller, M. (2018). John E. Freund’s Mathematical Statistics with Applications.
# - scikit-learn: machine learning in Python — scikit-learn 1.3.2 documentation. (n.d.). https://scikit-learn.org/stable/
# - Zaki, M. J., & Meira, W. (2020). Data Mining and Machine Learning: Fundamental Concepts and Algorithms. https://dx.doi.org/10.1017/9781108564175
# - Nicodv. (n.d.). GitHub - nicodv/kmodes: Python implementations of the k-modes and k-prototypes clustering algorithms, for clustering categorical data. GitHub. https://github.com/nicodv/kmodes



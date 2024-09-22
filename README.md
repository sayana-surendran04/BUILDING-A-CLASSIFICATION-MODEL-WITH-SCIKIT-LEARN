MINI PROJECT 2: 
Building a Classification Model with scikit-learn

Objective
The primary objective of this mini-project is to build and evaluate classification models using scikit-learn on a real-world dataset. This will involve data preprocessing, model selection, training, evaluation, and potentially hyperparameter tuning.

Dataset Selection
For this project, I'll use the Iris dataset, a built-in dataset in scikit-learn. This dataset is a classic example for classification tasks and is well-suited for understanding the basics of machine learning.

Data Loading and Exploration

from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["target"] = iris.target

print(data.head())

Data Preprocessing
Since the Iris dataset is relatively clean, with no missing values or categorical features, minimal preprocessing is needed.

Data Splitting
from sklearn.model_selection import train_test_split

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Model Selection and Training
Let’s train two classification models: Logistic Regression and Decision Tree.

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

Model Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Logistic Regression
y_pred_logreg = logreg.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
precision_logreg = precision_score(y_test, y_pred_logreg, average="weighted")
recall_logreg = recall_score(y_test, y_pred_logreg, average="weighted")
f1_logreg = f1_score(y_test, y_pred_logreg, average="weighted")
roc_auc_logreg = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])

# Decision Tree
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, average="weighted")
recall_dt = recall_score(y_test, y_pred_dt, average="weighted")
f1_dt = f1_score(y_test, y_pred_dt, average="weighted")
roc_auc_dt = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])

print("Logistic Regression:")
print("Accuracy:", accuracy_logreg)
print("Precision:", precision_logreg)
print("Recall:", recall_logreg)
print("F1-score:", f1_logreg)
print("ROC AUC:", roc_auc_logreg)

print("\nDecision Tree:")
print("Accuracy:", accuracy_dt)
print("Precision:", precision_dt)
print("Recall:", recall_dt)
print("F1-score:", f1_dt)
print("ROC AUC:", roc_auc_dt)

Conclusion
Based on the evaluation metrics, you can compare the performance of Logistic Regression and Decision Tree. You can also explore other classification algorithms and experiment with different hyperparameters to find the best model for your specific dataset.





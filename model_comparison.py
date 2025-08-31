# Task 1: Create a NumPy Array
import pandas as pd
import numpy as np

# Load dataset (replace 'your_dataset.csv' with your actual dataset file)
data = pd.read_csv('your_dataset.csv')

Y = data['Class'].to_numpy()
X = data.drop('Class', axis=1)

# Task 2: Standardize the Data
from sklearn import preprocessing

transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)

# Task 3: Split Data into Training and Testing Sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Task 4: Create and Fit a Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

parameters = {"C": [0.01, 0.1, 1], 'penalty': ['l2'], 'solver': ['liblinear']}
lr_model = LogisticRegression(solver='liblinear')
logreg_cv = GridSearchCV(lr_model, parameters, cv=10)
logreg_cv.fit(X_train, Y_train)

# Task 5: Calculate Accuracy for Logistic Regression
logreg_accuracy = logreg_cv.score(X_test, Y_test)
print(f"Logistic Regression Accuracy on test data: {logreg_accuracy}")

# Task 6: Create and Fit a Support Vector Machine (SVM) Model
from sklearn.svm import SVC

parameters = {'kernel':('linear', 'rbf','poly','sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma': np.logspace(-3, 3, 5)}
svm = SVC()
svm_cv = GridSearchCV(svm, parameters, cv=10)
svm_cv.fit(X_train, Y_train)

# Task 7: Calculate Accuracy for SVM
svm_accuracy = svm_cv.score(X_test, Y_test)
print(f"SVM Accuracy on test data: {svm_accuracy}")

# Task 8: Create and Fit a Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

parameters = {"criterion": ["gini", "entropy"],
              "splitter": ["best", "random"],
              "max_depth": [2*n for n in range(1,10)],
              "max_features": ["sqrt", None],
              "min_samples_leaf": [1, 2, 4],
              "min_samples_split": [2, 5, 10]}
tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree, parameters, cv=10)
tree_cv.fit(X_train, Y_train)

# Task 9: Calculate Accuracy for Decision Tree
tree_accuracy = tree_cv.score(X_test, Y_test)
print(f"Decision Tree Accuracy on test data: {tree_accuracy}")

# Task 10: Create and Fit a k-Nearest Neighbors (KNN) Model
from sklearn.neighbors import KNeighborsClassifier

parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, parameters, cv=10)
knn_cv.fit(X_train, Y_train)

# Task 11: Calculate Accuracy for KNN
knn_accuracy = knn_cv.score(X_test, Y_test)
print(f"KNN Accuracy on test data: {knn_accuracy}")

# Task 12: Find the Best Performing Method
accuracies = {
    "Logistic Regression": logreg_accuracy,
    "SVM": svm_accuracy,
    "Decision Tree": tree_accuracy,
    "KNN": knn_accuracy
}

best_model = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_model]

print("-" * 50)
print(f"The best performing method is: {best_model} with an accuracy of {best_accuracy:.2f}")

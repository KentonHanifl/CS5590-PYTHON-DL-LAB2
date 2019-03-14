# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

# Read in the file with pandas
diabetes = pd.read_csv('diabetes.csv')

# Drop any nan values if any exist
diabetes_data = diabetes.select_dtypes(include=[np.number]).interpolate().dropna()

# If the column is non-numeric, dummify the data
for column in diabetes_data:
    if np.issubdtype(diabetes_data[column].dtype, np.number) == False:
        diabetes_data = pd.get_dummies(
            diabetes_data,
            columns=[column]
        )

# Determine data(X) vs target(y)
X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

# Split the data set into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# LinearSVC accuracy
clf = LinearSVC().fit(X_train, y_train)
print("SVM accuracy:{:.2f}".format(clf.score(X_test, y_test)))

# Gaussian Bayes accuracy
gn = GaussianNB().fit(X_train, y_train)
print("Bayes accuracy:{:.2f}".format(gn.score(X_test, y_test)))

# KNN accuracy
knn = KNeighborsClassifier().fit(X_train, y_train)
print("KNN accuracy:{:.2f}".format(knn.score(X_test, y_test)))


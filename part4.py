# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn import linear_model
import pandas as pd
import numpy as np

# Load in boston housing dataset and convert it into a data frame for easier manipulation
boston_data = datasets.load_boston()
df_boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
df_boston['target'] = pd.Series(boston_data.target)
df_boston.head()

y = df_boston.CRIM
X = df_boston[["LSTAT", "TAX"]]

X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)

# Evaluate the regression model before EDA
print("Before EDA")

lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

# Evaluate the performance and visualize results
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

# Drop any nan values if any exist
boston_data_frame = df_boston.select_dtypes(include=[np.number]).interpolate().dropna()

# If the column is non-numeric, dummify the data
for column in boston_data_frame:
    if np.issubdtype(boston_data_frame[column].dtype, np.number) == False:
        boston_data_frame = pd.get_dummies(
            boston_data_frame,
            columns=[column]
        )

##Build the reggression model
y = boston_data_frame.CRIM
X = boston_data_frame[["LSTAT", "NOX"]]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)

lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)


# Evaluate the regression model after EDA
print("\nAfter EDA")

lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

# Evaluate the performance and visualize results
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

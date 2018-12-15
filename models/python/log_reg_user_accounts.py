# Logistic Regression on user account meta data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Load users, pay attention to the path here
f1x = pd.read_csv("users.csv")
g1x = f1x[["id", "name", "screen_name", "statuses_count", "followers_count", "friends_count", "lang", "default_profile", "protected", "verified", "description", "contributors_enabled"]].copy()
g1x['label'] = 1

f2x = pd.read_csv("users1.csv")
g2x = f2x[["id", "name", "screen_name", "statuses_count", "followers_count", "friends_count", "lang", "default_profile", "protected", "verified", "description", "contributors_enabled"]].copy()
g2x['label'] = 1

f3x = pd.read_csv("users2.csv")
g3x = f3x[["id", "name", "screen_name", "statuses_count", "followers_count", "friends_count", "lang", "default_profile", "protected", "verified", "description", "contributors_enabled"]].copy()
g3x['label'] = 1

f4x = pd.read_csv("users3.csv")
g4x = f4x[["id", "name", "screen_name", "statuses_count", "followers_count", "friends_count", "lang", "default_profile", "protected", "verified", "description", "contributors_enabled"]].copy()
g4x['label'] = 0

# make a single data frame
gX = pd.concat([g1x, g2x, g3x, g4x], ignore_index=True)
gX = gX.sample(frac=1).reset_index(drop=True)
gY = gX[['label']]
gX.drop(['label'], 1, inplace=True)

# Strings to Numeric
gX = gX.apply(pd.to_numeric, errors='coerce')
gX = gX.fillna(0)
print(gX.shape)
print(gY.shape)

# Split into test and train sets
x_train, x_test, y_train, y_test = train_test_split(gX, gY, test_size=0.25, random_state=0)

# Initialize the classifier
logistic_regressor = LogisticRegression()
logistic_regressor.fit(x_train, y_train)

# Predict Labels
predictions = logistic_regressor.predict(x_test)

# Print Results
print("Confusion Matrix:")
print(classification_report(y_test, predictions))
score = accuracy_score(y_test, predictions)
print("Accuracy:", score)

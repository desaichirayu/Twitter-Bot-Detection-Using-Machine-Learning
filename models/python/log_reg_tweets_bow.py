# This is the logistic regression model for classification based on tweet content with TF-IDF features
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# The directory containing the dataset, you may have to change it
data_dir = "E:/100NOKIA/cresci-2017/datasets_full.csv/Tweets/"

# Load data sets
ds_s = pd.read_csv(data_dir + "all_tweets_s.csv")
ds_g = pd.read_csv(data_dir + "all_tweets_g.csv")

# Aggregate Tweets by user
ds_s1 = ds_s.groupby(['user_id'])['text'].apply(', '.join).reset_index()
ds_s2 = ds_g.groupby(['user_id'])['text'].apply(', '.join).reset_index()

# Label Them
ds_s1['label'] = 1
ds_s2['label'] = 0

gX = pd.concat([ds_s1, ds_s2], ignore_index=True)
gX = gX.sample(frac=1).reset_index(drop=True)
gY = gX[['label']]
gX.drop(['label'], 1, inplace=True)
gX.drop(['user_id'], 1, inplace=True)

gX = gX['text'].str.lower()

# Split into test and train sets
x_train, x_test, y_train, y_test = train_test_split(gX, gY, test_size=0.25, random_state=0)

# Convert to Bag of Words Like feature vectors
vectorizer = CountVectorizer()
xx_train = vectorizer.fit_transform(x_train)
xx_test = vectorizer.transform(x_test)

# Initialize the model
logistic_regressor = LogisticRegression()
logistic_regressor.fit(xx_train, y_train)

# Predict Labels
predictions = logistic_regressor.predict(xx_test)

# Get the confidence in prediction
predictions_prob = logistic_regressor.predict_proba(xx_test)
df = pd.DataFrame()
df['bin'] = predictions
df['prob'] = pd.Series(predictions_prob.tolist())

# Print the Results
print("Confusion Matrix:")
print(classification_report(y_test, predictions))
score = accuracy_score(y_test, predictions)
print("Accuracy:", score)
print("Prediction  Confidences:")
print(df)

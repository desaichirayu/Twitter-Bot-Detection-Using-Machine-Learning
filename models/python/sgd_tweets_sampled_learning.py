# Stochastic gradient descent classifier learning from chunks of data set with TF-IDF
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


# The directory containing the dataset, you may have to change it
data_dir = "E:/100NOKIA/cresci-2017/datasets_full.csv/Tweets/"

# Load data sets
ds_s = pd.read_csv(data_dir + "all_tweets_s.csv")
ds_g = pd.read_csv(data_dir + "all_tweets_g.csv")

# Aggregate tweets by user
ds_s1 = ds_s.groupby(['user_id'])['text'].apply(', '.join).reset_index()
ds_s2 = ds_g.groupby(['user_id'])['text'].apply(', '.join).reset_index()

ds_test = ds_s.groupby(['user_id'])['text'].apply(list).reset_index()

# Label samples
ds_s1['label'] = 1
ds_s2['label'] = 0

gX = pd.concat([ds_s1, ds_s2], ignore_index=True)
gX = gX.sample(frac=1).reset_index(drop=True)
gY = gX[['label']]
gX.drop(['label'], 1, inplace=True)
gX.drop(['user_id'], 1, inplace=True)

gX = gX['text'].str.lower()

# Split into test and train set
x_train, x_test, y_train, y_test = train_test_split(gX, gY, test_size=0.25, random_state=0)

# Initialize classifier
text_clf = SGDClassifier(max_iter=100, loss="log")

# Divide data into 4 chunks
x_trains = np.array_split(x_train, 4)
y_trains = np.array_split(y_train, 4)
vectorizer = TfidfVectorizer()
vectorizer.fit(x_train)

# get results
for i in range(len(x_trains)):
    xx_train = vectorizer.transform(x_trains[i])
    xx_test = vectorizer.transform(x_test)

    text_clf = text_clf.partial_fit(xx_train, y_trains[i], classes=np.unique(y_trains[i]))

    predictions = text_clf.predict(xx_test)
    predictions_prob = text_clf.predict_proba(xx_test)
    df = pd.DataFrame()
    df['bin'] = predictions
    df['prob'] = pd.Series(predictions_prob.tolist())
    print("Batch no: ", str(i))
    # Print the Results
    print("Confusion Matrix:")
    print(classification_report(y_test, predictions))
    score = accuracy_score(y_test, predictions)
    print("Accuracy:", score)
    print("Prediction  Confidences:")
    print(df)

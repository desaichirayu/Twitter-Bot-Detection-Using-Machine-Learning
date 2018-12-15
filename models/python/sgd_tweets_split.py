# Train the SGD classifier with incremental Stream snapshots of data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def get_splits_25(string):
    """
    Helper function to split aggregated tweet content
    """
    str_len = len(string)
    one_fourth = str_len // 4
    return string[:one_fourth-1]


def get_splits_50(string):
    """
        Helper function to split aggregated tweet content
    """
    str_len = len(string)
    half = str_len // 2
    return string[:half-1]


def get_splits_75(string):
    """
        Helper function to split aggregated tweet content
    """
    str_len = len(string)
    three_quarter = (str_len * 3) // 4
    return string[:three_quarter-1]


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

# Create 4 Snapshots of aggregated tweet content for each user
gX_t = gX.copy()
gX_t = gX_t.to_frame().reset_index()
gX_t['text_25'] = gX_t['text'].apply(get_splits_25)
gX_t['text_50'] = gX_t['text'].apply(get_splits_50)
gX_t['text_75'] = gX_t['text'].apply(get_splits_75)


def classify(gX, gY):
    """
    Reusable Function to do Classification
    :param gX: The Samples
    :param gY: The Labels
    """
    # Split into test and train set
    x_train, x_test, y_train, y_test = train_test_split(gX, gY, test_size=0.25, random_state=0)

    vectorizer = TfidfVectorizer()
    xx_train = vectorizer.fit_transform(x_train)
    xx_test = vectorizer.transform(x_test)

    logisticRegr = SGDClassifier(loss='log', n_jobs=5)
    logisticRegr.fit(xx_train, y_train)
    predictions = logisticRegr.predict(xx_test)
    predictions_prob = logisticRegr.predict_proba(xx_test)
    df = pd.DataFrame()
    df['bin'] = predictions
    df['prob'] = pd.Series(predictions_prob.tolist())
    print("Confusion Matrix:")
    print(classification_report(y_test, predictions))
    score = accuracy_score(y_test, predictions)
    print("Accuracy:", score)
    print("Prediction  Confidences:")
    print(df)


# Print the process
print("Running at 25% Knowledge:")
classify(gX_t['text_25'], gY)
print()
print()
print()
print("Running at 50% Knowledge:")
classify(gX_t['text_50'], gY)
print()
print()
print()
print("Running at 75% Knowledge:")
classify(gX_t['text_75'], gY)
print()
print()
print()
print("Running at 100% Knowledge:")
classify(gX_t['text'], gY)
# Logistic regression on tweet content with 10 fold cross validation
import collections
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


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

x = gX
y = gY

# The Number of Folds
kf = KFold(n_splits=10)

# Initialize the classifier
classifier = LogisticRegression(n_jobs=5)

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()


def parse_classification_report(clfreport):
    """
    Parse a sklearn classification report into a dict keyed by class name
    and containing a tuple (precision, recall, fscore, support) for each class
    """
    lines = clfreport.split('\n')
    # Remove empty lines
    lines = list(filter(lambda l: not len(l.strip()) == 0, lines))

    # Starts with a header, then score for each class and finally an average
    header = lines[0]
    cls_lines = lines[1:-1]
    avg_line = lines[-1]

    print(header)
    print(cls_lines)
    print(avg_line)
    assert header.split() == ['precision', 'recall', 'f1-score', 'support']
    assert avg_line.split()[1] == 'avg'

    # We cannot simply use split because class names can have spaces. So instead
    # figure the width of the class field by looking at the indentation of the
    # precision header
    cls_field_width = len(header) - len(header.lstrip())

    # Now, collect all the class names and score in a dict

    def parse_line(l):
        """Parse a line of classification_report"""
        cls_name = l[:cls_field_width].strip()
        precision, recall, fscore, support = l[cls_field_width:].split()
        precision = float(precision)
        recall = float(recall)
        fscore = float(fscore)
        support = int(support)
        return (cls_name, precision, recall, fscore, support)

    data = collections.OrderedDict()
    for l in cls_lines:
        ret = parse_line(l)
        cls_name = ret[0]
        scores = ret[1:]
        data[cls_name] = scores

    # average
    data['avg'] = parse_line(avg_line)[1:]

    return data


ind = 0
prec = dict()
rec = dict()

# Run kFold CV
for train_indices, test_indices in kf.split(x, y):
    print("iteration: " + str(ind))
    x_train, x_test = x.iloc[train_indices], x.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    # x_train, x_test, y_train, y_test = x[train_indices], x[test_indices], y[train_indices], y[test_indices]
    train_x_vector = vectorizer.fit_transform(x_train)
    test_X_vector = vectorizer.transform(x_test)
    classifier.fit(train_x_vector, y_train)
    guess = classifier.predict(test_X_vector)
    # print(len(guess))
    # print(confusion_matrix(y_test, guess))
    rep = classification_report(y_test, guess)
    print(rep)
    a, b, c, d = dict(parse_classification_report(rep))['avg']
    prec[ind] = a
    rec[ind] = b
    ind = ind + 1
    print(dict(parse_classification_report(rep)))
    print(classification_report(y_test, guess))

p, r = (float(sum(prec.values())) / 10), (float(sum(rec.values())) / 10)
print(p * 100, " ", r * 100)
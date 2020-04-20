import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

phrases = ["The quick brown fox jumped over the lazy dog",
           "How quickly daft jumping zebras vex"]

vect = CountVectorizer()
vect.fit(phrases)

#print("Vocabulary size: {}".format(len(vect.vocabulary_)))
#print("Vocabulary Content:\n {}".format(vect.vocabulary_))

bag_of_words = vect.transform(phrases)
#print(bag_of_words)

#print("bag_words as an array : \n{}".format(bag_of_words.toarray()))

#print(vect.get_feature_names())

# 1 is positive and 0 is negative
data = pd.read_csv(r"C:\Users\rohit\Desktop\Datasets\labeledTrainData.tsv", delimiter = "\t")
#print(data.head())

#print("Samples per class: {}".format(np.bincount(data.sentiment)))

#Instead of train test split, we're writing a simple split function

def simple_split(data, y, length, split_mark=0.7):
    if split_mark > 0. and split_mark < 1.0:
        n = int(split_mark*length)
    else:
        n = int(split_mark)
    X_train = data[:n].copy()
    X_test = data[n:].copy()
    y_train = y[:n].copy()
    y_test = y[n:].copy()
    return X_train,X_test,y_train,y_test

vectorizer = CountVectorizer()
X_train,X_test,y_train,y_test = simple_split(data.review,data.sentiment, len(data))
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

#print("Samples per class: {}".format(np.bincount(y_train)))
#print("Samples per class: {}".format(np.bincount(y_test)))

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

#print(vectorizer.vocabulary_)

#Model 1
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Training Set Score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test Set Score: {:.3f}".format(logreg.score(X_test,y_test)))

#Printing the confusion Matrix
pred_logreg = logreg.predict(X_test)
confusion = confusion_matrix(y_test, pred_logreg)
print("Confusion Matrix:\n{}".format(confusion))

#Model 2
nb = MultinomialNB()
nb.fit(X_train, y_train)
print("Training Set Score: {:.3f}".format(nb.score(X_train, y_train)))
print("Test Set Score: {:.3f}".format(nb.score(X_test,y_test)))

#Confusion Matrix
pred_nb = nb.predict(X_test)
confusion = confusion_matrix(y_test, pred_nb)
print("Confusion Matrix:\n{}".format(confusion))

#Model 3
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print("Training Set Score: {:.3f}".format(rf.score(X_train, y_train)))
print("Test Set Score: {:.3f}".format(rf.score(X_test,y_test)))

# OUT OF BOX REVIEWS

review = "greatest movie ever made"

#Predict using the above models
print(logreg.predict(vectorizer.transform([review]))[0])
print(rf.predict(vectorizer.transform([review]))[0])
print(nb.predict(vectorizer.transform([review]))[0])

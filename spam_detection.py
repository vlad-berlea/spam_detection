#!/usr/bin/env python3
import gzip

import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

 

 

def load_ham(filename="src/ham.txt.gz"):
    with gzip.open(filename) as f:
        lines = f.readlines()
    return lines

 

def load_spam(filename="src/spam.txt.gz"):
    with gzip.open(filename) as f:
        lines = f.readlines()
    return lines

 

def spam_detection(random_state=0, fraction=1.0):
    vec = CountVectorizer()
    ham = load_ham()
    spam = load_spam()
    ham = ham[:int(fraction*len(ham))]
    spam = spam[:int(fraction*len(spam))]
    X = vec.fit_transform(ham+spam)
    n1 = len(ham)
      n2 = len(spam)

    if False:   # Print some info. From first two (ham) messages, show counts of common words.
        print(X.shape)
        temp = X[0:2, :].toarray()   # Vectorizer returns sparse array, convert to dense array
        idx = temp[:, :] != 0
        idx = temp.all(axis=0)
        names = vec.get_feature_names()
        df = pd.DataFrame(temp[:, idx], columns=np.array(names)[idx])
        print(df.T)
    y = np.hstack([[0]*n1, [1]*n2])

 

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, train_size=0.75, test_size=0.25)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_fitted = model.predict(X_test)
    acc = model.score(X_test, y_test)
    return acc, len(y_test), (y_test != y_fitted).sum()

 

def main():

    accuracy, total, misclassified = spam_detection()

    print("Accuracy score:", accuracy)

    print(f"{misclassified} messages miclassified out of {total}")

 

if __name__ == "__main__":

    main()

 
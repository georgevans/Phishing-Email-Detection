import os
import sys
import pickle
import csv

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

csv.field_size_limit(sys.maxsize)

from models.DataCleaner import (
    X_train_vectors,
    X_test_vectors,
    y_train,
    y_test,
    vectorizer
)

def train_and_save_model(model_path):
    print("Running function (train_and_save_model)...")
    model = MultinomialNB()
    print("X_train_vectors type:", type(X_train_vectors))
    print("y_train type:", type(y_train))
    print("X_train_vectors:", X_train_vectors)
    print("y_train:", y_train)

    model.fit(X_train_vectors, y_train)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return model

def load_model(model_path=None):
    if model_path is None or not os.path.exists(model_path):
        print("load_model: First if triggered")
        return train_and_save_model(os.path.join(os.path.dirname(__file__), "mymodel.pkl"))
    else:
        print("load_model: first if not triggered")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model

def load_vectorizer():
    return vectorizer

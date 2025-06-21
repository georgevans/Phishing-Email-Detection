import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

with open('data/Ling.csv', mode='r') as file:
    File = csv.reader(file)
    next(File)
    data = {
        'subject': [],
        'text': [],
        'label': []
    }
    for lines in File:
        data['subject'].append(lines[0])
        data['text'].append(lines[1])
        data['label'].append(int(lines[2]))

df = pd.DataFrame(data)

print(df)

#X = df['subject'] + " " + df['text']
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_test)

vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

__all__ = ['X_train_vectors', 'X_test_vectors', 'y_train', 'y_test', 'vectorizer']

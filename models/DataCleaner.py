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



#X = df['subject'] + " " + df['text']
Nphish = df[df['label'] == 0].sample(n=458, random_state=42)
phish = df[df['label'] == 1]

balanced_df = pd.concat([Nphish, phish])
X = balanced_df['text']
y = balanced_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

__all__ = ['X_train_vectors', 'X_test_vectors', 'y_train', 'y_test', 'vectorizer']

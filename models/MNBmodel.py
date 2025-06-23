import pandas as pd
import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from DataCleaner import X_train_vectors, X_test_vectors, y_train, y_test, vectorizer


model = MultinomialNB()
model.fit(X_train_vectors, y_train)

y_pred = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification = classification_report(y_test, y_pred)
print(confusion)
print(classification)
print(f"Accuracy: {accuracy * 100:.2f}%\n")

with open('data/HamTest.csv', mode='r') as file:
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

for i in range(0, len(data['text'])):
    custom_text = [data['text'][0]]
    custom_vector = vectorizer.transform(custom_text)
    custom_pred = model.predict(custom_vector)
    print(f"Custom prediction: {custom_pred[0]} (1 = Spam, 0 = Not Spam)")

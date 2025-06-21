import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from DataCleaner import X_train_vectors, X_test_vectors, y_train, y_test, vectorizer


model = MultinomialNB()
model.fit(X_train_vectors, y_train)

y_pred = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")


custom = {
        'subject': ['Today'],
        'text': ["Unfortunately, no Direct Debit has been set up by Ben Byrne for their share of the utilities package at 5 Branksome Terrace. It is extremely important that this is resolved as soon as possible, otherwise they may be removed from the utility account and payments may be increased to cover their share. To set up the remaining Direct Debits, please refer outstanding housemates to their own welcome and reminder emails. Remember the Direct Debit links are unique to the individual. Many thanks, UniHomes Billing Team"]    
    }

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

custom_text = [custom['text'][0]]
custom_vector = vectorizer.transform(custom_text)
custom_pred = model.predict(custom_vector)
print(f"Custom prediction: {custom_pred[0]} (1 = Spam, 0 = Not Spam)")

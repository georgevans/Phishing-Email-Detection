import pandas as pd
import csv
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from DataCleaner import X_train_vectors, X_test_vectors, y_train, y_test, vectorizer

csv.field_size_limit(sys.maxsize)

model = MultinomialNB() #lower alpha made 1 precision worse
model.fit(X_train_vectors, y_train)

# y_pred = model.predict(X_test_vectors)
# accuracy = accuracy_score(y_test, y_pred)
# confusion = confusion_matrix(y_test, y_pred)
# classification = classification_report(y_test, y_pred)
# print(confusion)
# print(classification)
# print(f"Accuracy: {accuracy * 100:.2f}%\n")

# Load and clean external test CSV
with open('test/spam_or_not_spam.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    texts = []
    labels = []
    for row in reader:
        if len(row) >= 2:
            texts.append(row[0])
            labels.append(int(row[1]))  # Make sure it's int

# Transform test texts with the same vectorizer
X_custom = vectorizer.transform(texts)
y_custom_true = labels

# Predict
y_custom_pred = model.predict(X_custom)

# Print evaluation
print("Custom Test Set Results:")
print(confusion_matrix(y_custom_true, y_custom_pred))
print(classification_report(y_custom_true, y_custom_pred))
print(f"Accuracy: {accuracy_score(y_custom_true, y_custom_pred) * 100:.2f}%")

# Optional: Save predictions
output_df = pd.DataFrame({
    'text': texts,
    'actual_label': y_custom_true,
    'predicted_label': y_custom_pred
})
output_df.to_csv("predictions_on_spam_or_not_spam.csv", index=False)
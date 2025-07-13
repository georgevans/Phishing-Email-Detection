import pandas as pd
import csv
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from DataCleaner import X_train_vectors, X_test_vectors, y_train, y_test, vectorizer

csv.field_size_limit(sys.maxsize)

# Train the model
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Evaluate on DataCleaner test set
y_pred = model.predict(X_test_vectors)
print("Test Set Results from DataCleaner split:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# OPTIONAL: Load and evaluate an external CSV
# Uncomment if you want to test additional data
# with open('test/FinalTest.csv', mode='r') as file:
#     reader = csv.reader(file)
#     next(reader)
#     texts = []
#     labels = []
#     for row in reader:
#         if len(row) >= 2:
#             texts.append(row[0])
#             labels.append(int(row[1]))

#     X_custom = vectorizer.transform(texts)
#     y_custom_pred = model.predict(X_custom)
#     print("\nCustom Test Set Results:")
#     print(confusion_matrix(labels, y_custom_pred))
#     print(classification_report(labels, y_custom_pred))
#     print(f"Accuracy: {accuracy_score(labels, y_custom_pred) * 100:.2f}%")

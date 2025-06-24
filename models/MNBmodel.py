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

# with open('test/HamTest.csv', mode='r') as file:
#     File = csv.reader(file)
#     next(File)
#     data = {
#         'subject': [],
#         'text': [],
#         'label': []
#     }
#     for lines in File:
#         data['subject'].append(lines[0])
#         data['text'].append(lines[1])
#         data['label'].append(int(lines[2]))

# df = pd.DataFrame(data)

data = {
    "subject": [
        "Your Account Has Been Locked",
        "Important Notice Regarding Your Tax Return",
        "Update Your Payment Details Immediately",
        "Claim Your Unused Rewards Now",
        "Final Warning: Your Subscription Will Be Cancelled"
    ],
    "text": [
        "Dear Customer,\n\nWe've detected suspicious activity in your account. Please log in immediately to verify your identity.\n\nClick here to restore access: http://secure-verify-login.com\n\nThank you,\nSecurity Team",
        "Dear Taxpayer,\n\nWe noticed discrepancies in your recent tax filing. Please review your information within 48 hours to avoid penalties.\n\nAccess your report here: http://gov-taxsecure.net\n\nHMRC",
        "Dear User,\n\nYour payment method was declined. Please update your billing information to avoid service interruption.\n\nUpdate Now: http://billing-update-alerts.com\n\nThanks,\nBilling Department",
        "You've earned $250 in reward points, but they will expire soon!\n\nRedeem your rewards now: http://reward-claim-center.net\n\nAct fast!\nRewards Team",
        "Hi there,\n\nYour subscription is expiring. Please confirm your payment information to continue enjoying our service.\n\nRenew now: http://subscription-secure.net\n\nSincerely,\nSupport"
    ],
    "label": [1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
df.to_csv("test_phishing_emails.csv", index=False)


for i in range(0, len(data['text'])):
    custom_text = [data['text'][0]]
    custom_vector = vectorizer.transform(custom_text)
    custom_pred = model.predict(custom_vector)
    print(f"Custom prediction: {custom_pred[0]} (1 = Spam, 0 = Not Spam)")

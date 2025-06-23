# Phishing-Email-Detection
A program to detect Phishing emails using a Multinomial naive bayes ML model 

# Phishing email detection project
## Overview
This project will be a python based tool to detect phishing emails using ML. 
Project goals
- Train a basic ML model on a real dataset
- Classify an email as phishing or legit
- Build a interface to test emails

### Classify an email as phishing or legit
In order to do this I will need to train a ML model to look for signs of a phishing scam. These include things like urgent phrasing, spelling and grammar mistakes, generic greetings, threatening language, requests for personal information and unusual email addresses of links.

I plan on detecting each of these features separately and calculating a percentage score for each one to see how much the model thinks the email is guilty of having that trait. This will be to avoid false positives such as a situation involving a sale, where an email could raise a 95% for urgent phrasing ( as a email about a discount often contains words like “quick” and “now” which are often used in phishing emails ), but only raise a 5% for threatening language and have <5% for spelling and grammar mistakes then we can use this to judge that the email is not phishing despite it having a similar trait.

## Searching for and cleaning data
I found this kaggle dataset to use: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

It contains over 82000 emails to use with the following columns:
- Sender Email
- Receiver Email
- Date Time
- Subject
- Body
- Label
- URLs

I will also need to clean the text and convert it into a number format to ensure the ML model can recognise patterns and actually classify if a email is phishing or legitimate. To achieve this I plan on processing the data in 3 stages:
- Clean text
- - This will involve removing all punctuation, lowercases and stripping whitespace
- Tokenization
- Converting text to numbers

To clean the text I can use built-in python features and to tokenize and convert the text I will use the nltk module as well as the TfidfVectorizer from the sklearn module

And finally I will need to split data into training and testing data (I plan on an 80/20 split for this)
## Model training
I plan on using a Multinomial Naive Bayes model to classify my emails as phishing or legitimate. The reasoning behind the decision for this model is that its built for classification, fast to train, and it works well with small/medium datasets.


# Development

## Cleaning data
So in order to make this data usable in a Multinomial naive bayes model I will need to clean it and format it in a way that I have training and test data split by subject, text and the label (0, for non spam and 1 for spam). 

To do this I have thought of two potential apporaches.
1. Converting the .csv file into a object format then converting to a pandas dataframe
- - This would be done by reading the csv file in and then using a loop to format it into an object
2. The alternative is converting directly from csv to pandas
- - This is a much faster method but gives me less control over the inital format of the data which could be relavant later down the line.

I intend on implementing both methods to see which gives the best results.

## Issue in first iteration of development
I've noticed an issue with the program in which the model reports an accuracy of 99%> however when tested using emails from outside of the phishing dataset, I get the wrong classification 4 out of 5 times. I'm unsure what could be causing this but planning on analysing individual parts of my code to get a better idea. My inital assumption is the model being overfitted to the data causing the high accuracy score for the testing data but unable to correcly classify non-training or non-testing data.

### Potential causes of issue
- Data leakage  
   - This could be caused by the preprocessing stage exposing label information which allows the model to "cheat"  
   - I do think this is unlikely as my code (as shown below) correctly preprocesses the data in a way that avoids data leakage:  
     ```python
     vectorizer = CountVectorizer()
     X_train_vectors = vectorizer.fit_transform(X_train)
     X_test_vectors = vectorizer.transform(X_test)
     ```
  - Given this information I think its unlikely that data leakage is causing the issue 
- Over fitting of the model
  - This could be caused by the model "memorizing" patterns in the training data that doesnt extend to new emails
  - A common sign of overfitting is high accuracy with training/test data but poor performance on real emails
  - Often caused by a dataset that is either too small or not diverse.
  - This I believe is the most likely culprit as the data set only contains 2859 emails which may not be sufficient in training this kind of     model
  - A potential fix beside finding an entirely new dataset would be to add regularization which is something I will test in the future.
- If the test set is not representative of the training set we could have exposed the model to weak generalization
- Vocabulary mismatch
  - New emails may contain words not seen in training set, this could cause the model to be unable to interpret them.
 
I'm predicting the issue is overfitting as it would explain the high accuracy on testing data (of the same data set) but the poor results when classifying emails outside the dataset.

In order to fix this issue I want to try a few things:
- Check class balance
- Look at adding more diverse data
- Switch to TfidVectorizer for better feature extraction
- Potentially look at changing the model type entirely (this would be a last resort)

Initially I ran the below code in an attempt to see if my data set was biased towards either pishing or non-phishing emails:
```python
print(df['label'].value_counts())
```
This returned the following:
```bash
label
0    2401
1     458
```
This confirmed my suspicion that the issue was relating to the dataset being used and potential bias. As this result shows a clear bias towards non-phishing emails.

To gain more confirmation of this being the issue, I will run the following code to undersample my non-phishing emails in the dataset.
```
Nphish = df[df['label'] == 0].sample(n=458, random_state=42)
phish = df[df['label'] == 1]

balanced_df = pd.concat([Nphish, phish])
X = balanced_df['text']
y = balanced_df['label']
``` 
This caused a decrease in accuracy to 97.5% but still falsely identified 4 of the 5 test emails from outside the dataset.
However we can likely attribute this to further reducing the dataset size to only 458 emails which is unlikley to be sufficient data to train the model.

I added a confusion matrix to the model evaluation to see how it was classifying, the model had:
- 138 True negatives
- 7 False positives
- 0 False negatives 
- 130 True positives

So after running these tests we can see the issue is most likley overfitting as the model is functioning correclty but only on data relating to the training and testing set.

So in order to address these issues I'll need a much larger and more diverse dataset.


# Phishing-Email-Detection
A program to detect Phishing emails using a Multinomial naive bayes ML model 

# Phishing email detection project
## Overview
This project will be a python based tool to detect phishing emails using ML. 
Project goals
- Train a basic ML model on a real dataset
- Classify an email as phishing or legit
- Build a interface to test emails

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

## Introducing a larger dataset
In order to fix my issue with potential overfitting or data leakage I have decided to create a larger more diverse dataset using more varied phishing and non-phishing emails.

To combine multiple csv files I need to address the issue of formatting as some datasets I found online use different formats to store their csv information.

To get around this I need to standardize the format, I did this using the following:
```python
def load_csv(file_path):
    df = pd.read_csv(file_path)

    if {'subject', 'body', 'label'}.issubset(df.columns):
        df = df[['subject', 'body', 'label']]
    elif {'sender','receiver','date','subject','body','urls','label'}.issubset(df.columns):
        df = df[['subject', 'body', 'label']]
    elif {'sender','receiver','date','subject','body','label','urls'}.issubset(df.columns):
        df = df[['sender', 'body', 'label']]
    else:
        raise ValueError(f"Unexpected format in file: {file_path}")
    return df
```
This function is able to create a dataframe of the same format despite the differences in formats of the csv files being combined

To combine the files:
```python 
def combine_csvs(folder_path):
    combined_df = pd.DataFrame(columns=['text', 'label'])
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                df = load_csv(file_path)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return combined_df
```

This results in a larger more diverse database with a much better distribution of phishing and non-phishing emails with around 25583 phishing emails and 23804 non-phishing emails. This should stop the model over guessing phishing emails.

After testing this did not immidelty fix the issue, it did have a positive impact on the model accuracy (increasing it by 0.2%, to around 99.73%) however it did not have a meaningful effect on external test data which still resulted in 4 out of 5 false positives. 

## Alternate fix idea
I plan on using TfidVectorizer as oppose to countvectorizer as this helps reduce the impact of commonly occuring words and places more emphasis on meaningful, distinct words within the emails. My hope is this will improve the models ability to spot words commonly assocaited with phishing like "Download", "quick" and other words that are common in phishing emails.

To implement:
```python
vectorizer = TfidfVectorizer(
    max_features=5000,          
    stop_words='english',     
    ngram_range=(1, 2)          
)

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)
```

This worked in intial testing with all 5 external phishing emails correctly classfied as phishing and all 5 non-phishing emails correclty classified as non-phishing

### Inital Model Performance Summary
The test data returned the following:
```
[[732   1]
 [  6 119]]
              precision    recall  f1-score   support

           0       0.99      1.00      1.00       733
           1       0.99      0.95      0.97       125

    accuracy                           0.99       858
   macro avg       0.99      0.98      0.98       858
weighted avg       0.99      0.99      0.99       858

Accuracy: 99.18%
```
- Evaluated the model on a holdout test set of 858 emails.
- Achieved **overall accuracy of 99.18%**.
- **Very high precision and recall** for both spam (label 1) and ham (label 0).
- Confusion matrix showed:
  - Only **6 false negatives** (spam misclassified as ham).
  - Just **1 false positive** (ham misclassified as spam).
- Indicates the model generalizes well and correctly classifies most emails.
- Next steps may include further testing on external datasets and real-world examples.

This will require further testing but it does hopefully address the core issue.

# Testing
In testing I want to use a varied source for emails, one which contains emails of many different subject matters and from various sources. I found a dataset on kaggle by the name of the "spam or not spam dataset" which contains 2873 emails, around 2300 of these are non-phishing (ham) and 500 phishing (spam).

### link
https://www.kaggle.com/datasets/ozlerhakan/spam-or-not-spam-dataset/data

## First round of testing
In my first round of testing for the model I achieved the following results:
|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **0**         | 0.96      | 0.83   | 0.89     | 2500    |
| **1**         | 0.50      | 0.84   | 0.63     | 500     |
|               |           |        |          |         |
| **Accuracy**  |           |        | 0.83     | 3000    |
| **Macro Avg** | 0.73      | 0.83   | 0.76     | 3000    |
| **Weighted Avg** | 0.89   | 0.83   | 0.85     | 3000    |

**Accuracy:** 83.40%

These intially seemed great but on closer examination I noticed the low recall for spam results, with almost 50% of emails labeled as spam actually being ham which is a large number of false positives and substantially decreases the models usefullness in a practical application.

I believe this could have been caused by the training data used consisting of mostly non-phishing emails causing it to have a bias towards non-phishing emails.

To improve this, I plan to experiment with the alpha parameter of the model, as well as the number and range of n-grams included in the feature extraction. While increasing these parameters can enhance the model’s ability to capture meaningful patterns, it also raises the risk of overfitting. To mitigate this, I intend to adjust them incrementally and re-evaluate the model’s performance after each change.

### Alpha
I initally changed the alpha to 0.5, this only made the model worse and decreased its precision to 0.29 for spam, decreasing it further only made the model worse. Increasing the alpha did improve it but the precision never exceeded 0.5. From this I concluded alpha was not an adequate way to improve the model's precision.

### Number of n-grams
Increasing the max number of features should increase the number of words the model captures meaning it has a larger vocabulary size, hopefully this will increase the model's precision. I initally had my ```max_features``` set to only 5000, increasing it to 10000 lead to much better results:
|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **0**         | 0.94      | 0.96   | 0.95     | 2500    |
| **1**         | 0.79      | 0.70   | 0.74     | 500     |
|               |           |        |          |         |
| **Accuracy**  |           |        | 0.83     | 3000    |
| **Macro Avg** | 0.86      | 0.83   | 0.85     | 3000    |
| **Weighted Avg** | 0.92   | 0.92   | 0.92     | 3000    |

**Accuracy:** 91.83%

- decrease of spam recall from 0.84 to 0.70
- increase of spam precision from 0.5 to 0.79
- increase of macro avg of precion from 0.73 to 0.86
- increase of macro avg f1 from 0.76 to 0.85
- increase of weighted avg precision from 0.89 to 0.92
- increase of weighted avg recall from 0.83 to 0.92
- weighted f1 from 0.85 to 0.92
- increase in accuracy from 83.40% to 91.83%.

#### Evaluation of changes
Positives:
- Improvement of accuracy by 8%
- Precision for spam detection improved, leading to fewer false positives
- Weighted recall and F1 show that a higher proportion of correct predictions were made overall, accounting for the class imbalance

Negatives:
- Recall for spam detection decreased meaning model misses more actual spam, this is not great for the goal of the project is to create a ML model to detect spam, so it missing spam is a huge issue that needs addressing
- This does however mean a decrease in false positives which could be seen as a good thing

These results make sense considering the increase in number of max features, whilst the model now has a richer vocabulary it is more precise when it does detect spam, but is more hesitant to predict an email is spam.

### N-gram range
Another change I believe will help improve the model is increasing the models N-gram range to accept tri-grams, this will make the model slower but will hopefully allow it to pick up on phrases like "click here" whereas before it would only see "click" and "here" as two seperate words.

This could have a negative effect on the model as increasing the model to accept tri-grams could result in overfitting to the training data. But it is worth trying nonetheless.

```python 
vectorizer = TfidfVectorizer(
    max_features=10000,          
    stop_words='english',     
    ngram_range=(1, 3)          
)
```

**The results:**
|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **0**         | 0.94      | 0.96   | 0.95     | 2500    |
| **1**         | 0.80      | 0.68   | 0.74     | 500     |
|               |           |        |          |         |
| **Accuracy**  |           |        | 0.92     | 3000    |
| **Macro Avg** | 0.87      | 0.82   | 0.84     | 3000    |
| **Weighted Avg** | 0.91   | 0.92   | 0.92     | 3000    |

**Accuracy:** 91.80%

- Increase in class 1 precision from 0.79 to 0.80
- Decrease in class 1 recall from 0.70 to 0.68
- Increase in F1 accuracy from 0.83 to 0.92
- Decrease in weighted avg precision from 0.92 to 0.91
- Decrease in accuracy from 91.83% to 91.80%

#### Evaluation of changes
So comparing this with the previous model these improvements are not particularly drastic, whilst the model is more precise with fewer false positives, it did miss more spam which sort of undermines the models purpose, so I think I will revert to the model only using bi-grams as oppose to tri-grams.

### Further changes to N-gram
Decided to increase the max features to 15000 as the increase from 5000 to 10000 proved to cause the most drastic change out of any other previous tests:
```python
vectorizer = TfidfVectorizer(
    max_features=15000,          
    stop_words='english',     
    ngram_range=(1, 2)          
)
```

**Results:**
|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **0**         | 0.92      | 0.99   | 0.95     | 2500    |
| **1**         | 0.92      | 0.57   | 0.70     | 500     |
|               |           |        |          |         |
| **Accuracy**  |           |        | 0.92     | 3000    |
| **Macro Avg** | 0.92      | 0.78   | 0.83     | 3000    |
| **Weighted Avg** | 0.92   | 0.92   | 0.91     | 3000    |

**Accuracy:** 92.00%

This change had some interesting results, whilst the recall for class 1 did decrease most others were stable with the largest change being an increase in overall accuracy, and an increase in class 0 recall and class 1 precision, with both increase over 0.05%.

It seems the recall of class 1 is the models largest fault with this being the hardest feature to get working correctly. 

It seems the changes made in the first stage of testing have resulted in some increases in accuracy and recall but havent't really shown a significant improvement in decreaseing the number of false positives and false negatives (still saying around 217 emails in the test data were ham when actually spam).

To get around this I plan on increasing the amount of spam emails the model is tested on as I think the model has not seen enough spam to accurately predict what some look like, leading to many spam emails slipping through the cracks.

## Second round of testing
Now the dataset has been increased from emails 2001 to 34570 emails with 17959 of them being spam and 16611 emails being ham. This should make the model far more accurate and ideally improve its ability to label an email as phishing.

### Increased dataset size results
**Results**:
|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **0**         | 0.93      | 0.99   | 0.96     | 2500    |
| **1**         | 0.96      | 0.60   | 0.74     | 500     |
|               |           |        |          |         |
| **Accuracy**  |           |        | 0.93     | 3000    |
| **Macro Avg** | 0.94      | 0.80   | 0.85     | 3000    |
| **Weighted Avg** | 0.93   | 0.93   | 0.92     | 3000    |

**Accuracy:** 92.83%

- Increase in class 0 precision from 0.92 to 0.93
- Increase in F1 score for class 0 from 0.95 to 0.96
- Increase in F1 accuracy from 0.92 to 0.93
- Increase in macro avg precision from 0.92 to 0.94
- Increase in Macro avg recall from 0.78 to 0.80
- Increase in F1 Macro avg from 0.83 to 0.85
- Increase in Weighted avg precision from 0.92 to 0.93
- Increase in Weighted avg recall from 0.92 to 0.93
- Increase in Weighted avg F1 0.91 to 0.92

#### Evaluation of changes
Increasing the training data seems to have almost universally positive effects with all round increases in the performance of the model. 

# Final testing
I created a csv of around 50 emails to test the models performance on a new set of emails not found in the training set it achieved the following results:
**Results**:
|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **0**         | 0.91      | 0.80   | 0.85     | 25    |
| **1**         | 0.96      | 0.60   | 0.74     | 24     |
|               |           |        |          |         |
| **Accuracy**  |           |        | 0.86     | 49    |
| **Macro Avg** | 0.86      | 0.86   | 0.86     | 49    |
| **Weighted Avg** | 0.86   | 0.86   | 0.86     | 49    |

**Accuracy:** 85.71%

These results, whilst not as great as in testing show the model is capable of correctly classifying emaails as spam or ham with 85.71% accuracy, this lower score could be due to the smaller size of the emails used in the final round of testing making it harder for the model to correctly classify each email, however it only incorrectly classified 7 of the 50 emails included in the test.

Overall I see this project as a success so far and am now looking at ways to further develop the idea such as a better user interface.

# Final code for the model
**MNBmodel.py:**
```python
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
```

**DataCleaner.py:**
```python
import pandas as pd
import csv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_csv(file_path):
    df = pd.read_csv(file_path)

    if {'subject', 'body', 'label'}.issubset(df.columns):
        df = df[['subject', 'body', 'label']]
    elif {'sender','receiver','date','subject','body','urls','label'}.issubset(df.columns):
        df = df[['subject', 'body', 'label']]
    elif {'sender','receiver','date','subject','body','label','urls'}.issubset(df.columns):
        df = df[['sender', 'body', 'label']]
    elif {'body','label'}.issubset(df.columns):
        df = df[['sender', 'body', 'label']]
    else:
        raise ValueError(f"Unexpected format in file: {file_path}")
    return df

def combine_csvs(folder_path):
    combined_df = pd.DataFrame(columns=['text', 'label'])
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                df = load_csv(file_path)
                
                # Combine columns into one text column
                if 'subject' in df.columns:
                    df['text'] = df['subject'].fillna('') + " " + df['body'].fillna('')
                elif 'sender' in df.columns:
                    df['text'] = df['sender'].fillna('') + " " + df['body'].fillna('')
                else:
                    df['text'] = df['body'].fillna('')
                
                # Keep only text and label
                
                df = df[['text', 'label']]
                
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return combined_df



df = combine_csvs('data')

# print(df['label'].value_counts())
df = pd.DataFrame(df)


df['label'] = df['label'].astype(int)

#X = df['subject'] + " " + df['text']
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# print(f'Number of training emails: {len(y_train)}')
print(f'Training data split: {y_train.value_counts()}')

vectorizer = TfidfVectorizer(
    max_features=15000,          
    stop_words='english',     
    ngram_range=(1, 2)          
)

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

__all__ = ['X_train_vectors', 'X_test_vectors', 'y_train', 'y_test', 'vectorizer']
```

# Further developments
Since the model tests well I am happy with its performance and ability to classify emails. That being said it is undeniable that the model can be improved upon massively, Multinomial naive bayes was the model type I chose for this model, however the model could see better results using some other model types such as:
- Logistic regression
- Random forest
- XGboost

The model could also benefit from utilizing NLP (Natural Language Processing) techniques to give better representation and understanding of the text data.

Beyond these, additional directions for improvement include:
## Deep Learning Approaches
Exploring neural networks such as Recurrent Neural Networks (RNNs) or Long Short-Term Memory (LSTM) networks can help capture the sequential nature of email text. These models learn context over longer spans and can detect more complex patterns of phishing language.

## Ensemble Methods
Combining multiple classifiers into an ensemble (e.g., voting classifiers or stacking) can leverage the strengths of each model and improve generalization across different types of phishing attempts.

## Metadata and Behavioral Features
Incorporating metadata such as sender domain reputation, frequency of contact, or the presence of suspicious attachments can provide richer signals beyond text content alone.

## Incremental Learning / Online Learning
Implementing models that can continuously learn from new emails over time (rather than retraining from scratch) can help keep detection up to date with evolving phishing tactics.

## Explainability Tools
Adding explainable AI methods (like SHAP or LIME) to show why the model flagged an email.

## Adversarial Robustness Testing
Evaluating how the model performs against adversarial examples or obfuscated phishing text can help you harden it against attacks designed to evade detection.

## Deployment Pipeline
Developing an API or real-time processing pipeline that integrates with email systems or security gateways to classify emails live.

# Flask

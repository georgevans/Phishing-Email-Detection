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

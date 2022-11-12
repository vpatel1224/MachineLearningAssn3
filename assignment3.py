#!/usr/bin/env python3
from pydoc import doc
import sys
import time
import math
import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
import nltk
from nltk.corpus import stopwords
import string
import nltk

''' '''

__author__ = 'Jacob Hajjar, Michael-Ken Okolo'
__email__ = 'hajjarj@csu.fullerton.edu, michaelken.okolo1@csu.fullerton.edu'
__maintainer__ = 'jacobhajjar, michaelkenokolo'


# Description: This program detects if the email is spam or not spam


# nltk.download('stopwords')
df = pd.read_csv('emails.csv')
# print(df.head(5))
# print(df.columns)
df.drop_duplicates(inplace=True)

def txt_process(text):
    remove_punc = [character for character in text if character not in string.punctuation]
    remove_punc = ''.join(remove_punc)

    cl_words = [word for word in remove_punc.split() if word.lower() not in stopwords.words('english')]

    return cl_words



print(df['text'].head().apply(txt_process))


from sklearn.feature_extraction.text import CountVectorizer

e_messages = CountVectorizer(analyzer=txt_process).fit_transform((df['text']))
y=df['spam']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(e_messages, y, test_size = 0.2)
print(e_messages.shape)


from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB().fit(x_train, y_train)
NB_predict = NB_classifier.predict(x_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred = NB_classifier.predict(x_train)
print("Classification Report",classification_report(y_train, pred))
print("\n Confusion Matrix: \n",confusion_matrix(y_train, pred) )

def main():
    """the main function"""
    

if __name__ == '__main__':
    main()

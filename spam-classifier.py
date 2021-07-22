# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:33:42 2021

@author: hp
"""

import pandas as pd

dataset = pd.read_csv('spamdata/SMSSpamCollection',sep='\t', names=['label', 'messages'])

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
lemma = WordNetLemmatizer()
corpus = []
for i in range(len(dataset)):
    a = re.sub('[^a-zA-z]', ' ', dataset['messages'][i])
    a = a.lower()
    a = a.split()
    a = [ps.stem(b) for b in a if b not in set(stopwords.words('english'))]
    a = ' '.join(a)
    corpus.append(a)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
c = cv.fit_transform(corpus).toarray()

d = pd.get_dummies(dataset['label'])
d = d.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(c,d, test_size=0.30, random_state=0)


from sklearn.naive_bayes import MultinomialNB
nbc = MultinomialNB().fit(x_train, y_train)

e = nbc.predict(x_test)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, e, labels=nbc.classes_)
dis = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=nbc.classes_)
dis.plot()

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, e)

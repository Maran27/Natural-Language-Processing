# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 18:57:29 2021

@author: hp
"""

import pandas as pd

path ="F:\\Natural Language Processing\\NLP\\Data.csv"
df = pd.read_csv(path, encoding='ISO-8859-1')

df1 = df.head()

train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

data = train.iloc[:, 2:27]
data.replace("[^A-Za-z]", " ", regex=True, inplace=True)

list1 = [i for i in range(25)]
newi = [str(i) for i in list1]
data.columns = newi
df2 = data.head()

for index in newi:
    data[index] = data[index].str.lower()
df3 = data.head()

p1 = ' '.join(str(x) for x in data.iloc[1, 0:25])

headlines = []
for row in range(0, len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row, 0:25]))
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

cv = CountVectorizer(ngram_range=(2,2))
traindata = cv.fit_transform(headlines)

rfc = RandomForestClassifier(n_estimators=200, criterion='entropy')
rfc.fit(traindata, train['Label'])

test_transform = []
for row in range(0, len(test.index)):
    test_transform.append(' '.join(str(x) for x in data.iloc[row, 2:27]))
testdata = cv.transform(test_transform)
results = rfc.predict(testdata)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
cm = confusion_matrix(test['Label'], results)
print(cm)
ac = accuracy_score(test['Label'], results)
print(ac)
report = classification_report(test['Label'], results)
print(report)


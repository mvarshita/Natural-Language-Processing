#@use: Spam classifier for SMS.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:31:06 2020

@author: varshita
"""

import pandas as pd

messages = pd.read_csv('smsSpamCollection/SMSSpamCollection', sep = "\t",
                       names = ["label","message"])

# Data Cleaning and pre-processing
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

ps = PorterStemmer()
corpus = []

for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

Y=pd.get_dummies(messages['label'])
Y=Y.iloc[:,1].values

# Train Test Split
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(xtrain, ytrain)
ypred=spam_detect_model.predict(xtest)

# Calculating the accuracy score
from sklearn.metrics import accuracy_score
accuracyScore=accuracy_score(ytest,ypred)
print("The accuracy of the Spam Classifier model is %.2f "%accuracyScore)

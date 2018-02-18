print("Training the model..... \n")

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

stemmer = SnowballStemmer("english", ignore_stopwords=True)
tfidf_transformer = TfidfVectorizer(stop_words='english', analyzer='char_wb', ngram_range=(1, 1))

text_clf_svm = Pipeline([('vect', stemmed_count_vect),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                         alpha=1e-3, max_iter=5, random_state=42)), ])

data = pd.read_csv('train_set.txt', sep="\n", header=None)
labels = pd.read_csv('train_set_labels.txt', sep="\n", header=None)


text_clf_svm.fit(data[0], labels[0])
print("Model Trained \n ")
print("Now we can make our predictions , Please make sure the test file is in the same folder as this script")
test_file = input("Enter file name:")
data_test = pd.read_csv(test_file, sep="\n", header=None)

test_results_final = text_clf_svm.predict(data_test[0])

submission = open('results.txt', 'w')
for item in test_results_final:
  submission.write("%s\n" % item)

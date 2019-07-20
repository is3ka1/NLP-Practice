import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('SPAM text message 20170820 - Data.csv')

clf = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                ('clf', None)])

parameters = {
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'clf': [LogisticRegressionCV(cv=4 , solver='newton-cg', scoring='roc_auc'),
            SVC(gamma='auto'), BernoulliNB()],
}

clf = GridSearchCV(clf, parameters, cv=4, scoring='roc_auc')


X = df['Message']
Y = pd.get_dummies(df['Category'])['spam']

x_train, x_test, y_train, y_test = train_test_split(X, Y)

clf.fit(x_train, y_train)
print(clf.best_params_)
print(f'score: {clf.score(x_test, y_test)}')


y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

samples = df.loc[y_test.index].copy()
samples['true_result'] = y_test
samples['model_perdict'] = y_pred
print(samples.sample(n=10))

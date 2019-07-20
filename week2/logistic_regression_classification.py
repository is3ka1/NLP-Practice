import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import seaborn as sns


df = pd.read_csv('SPAM text message 20170820 - Data.csv')

clf = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    LogisticRegressionCV(cv=5 , solver='liblinear', scoring='roc_auc'))

X = df['Message']
Y = pd.get_dummies(df['Category'])['spam']

x_train, x_test, y_train, y_test = train_test_split(X, Y)

clf.fit(x_train, y_train)


y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')

accuracy = clf.score(x_test, y_test)
print(accuracy)

samples = df.loc[y_test.index].copy()
samples['true_result'] = y_test
samples['model_perdict'] = y_pred
print(samples.sample(n=10))

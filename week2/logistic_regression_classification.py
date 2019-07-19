import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


df = pd.read_csv('SPAM text message 20170820 - Data.csv')

tfid_vectorizer = TfidfVectorizer(stop_words='english')
X = tfid_vectorizer.fit_transform(df['Message'])

Y = pd.get_dummies(df['Category'])['spam']

x_train, x_test, y_train, y_test = train_test_split(X, Y)

classifier = LogisticRegressionCV(cv=5 , solver='liblinear')
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = classifier.score(x_test, y_test)

samples = df.loc[y_test.index].copy()
samples['true_result'] = y_test
samples['model_perdict'] = y_pred

print(cm, accuracy, sep='\n', end='\n\n')

print(samples.sample(n=10))

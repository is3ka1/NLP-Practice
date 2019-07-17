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

classifier = LogisticRegressionCV()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = classifier.score(x_test, y_test)

print(cm, accuracy, sep='\n')


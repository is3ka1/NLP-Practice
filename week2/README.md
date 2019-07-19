

```python
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

```

    [[1202    2]
     [  21  168]]
    0.9834888729361091
    
         Category                                            Message  true_result  \
    127       ham  Just so that you know,yetunde hasn't sent mone...            0   
    1260      ham  We have sent JD for Customer Service cum Accou...            0   
    3404      ham       Good night my dear.. Sleepwell&amp;Take care            0   
    1961     spam  Guess what! Somebody you know secretly fancies...            1   
    3240      ham           Am okay. Will soon be over. All the best            0   
    4141      ham  Leave it wif me lar... Ü wan to carry meh so h...            0   
    2525     spam  FREE entry into our £250 weekly comp just send...            1   
    1966      ham  Thanks. It was only from tescos but quite nice...            0   
    2440      ham  Rightio. 11.48 it is then. Well arent we all u...            0   
    3713      ham                                 Wat u doing there?            0   
    
          model_perdict  
    127               0  
    1260              0  
    3404              0  
    1961              1  
    3240              0  
    4141              0  
    2525              1  
    1966              0  
    2440              0  
    3713              0  


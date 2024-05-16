import numpy as np
from AdaBoost import AdaBoost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000,n_features=10,n_informative=3,n_redundant=5,random_state=9)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=1)
adaboost = AdaBoost(iterations=10)
adaboost.fit(X_train, y_train)
y_pred_1 = adaboost.predict(X_test)


jj = AdaBoostClassifier()
jj.fit(X_train, y_train)
y_pred = jj.predict(X_test)

ass = accuracy_score(y_pred, y_test)
print('sk',ass*100)
ass1 = accuracy_score(y_pred_1, y_test)
print('my:',ass1*100)
from sklearn.metrics import accuracy_score
from sklearn import linear_model
import numpy as np
from Logistic_regression import LogisticRegression
import Regularization_class
# Number of samples
num_samples = 1000

# Generate features (X) - two-dimensional random data
X = np.random.randn(num_samples, 2)

# Generate labels (y) - binary classification (0 or 1)
y = np.random.randint(0, 2, size=num_samples)

logreg = LogisticRegression(0.01, 1000, Regularization_class.l1_reg(0.1))
logreg.fit(X, y)
log_pred = logreg.predict(X)
log_score = accuracy_score(y, log_pred)
print("The accuracy_score of the trained model (logistic): ", log_score)

#sklearn algo

print(100*'#')

lr = linear_model.LogisticRegression()
lr.fit(X,y)
lp = lr.predict(X)
ls = accuracy_score(y, lp)
print("The accuracy_score of the trained model (logistic): ", ls)



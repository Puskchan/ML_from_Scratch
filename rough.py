from Regression import Linear_regression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=2, noise=15, random_state=21)
lr = Linear_regression.LinearRegression(alpha=0.1, iter=100)
lr.fit(X,y)
pred = lr.predict((3,4))
print(pred)
lr.show_line(X[:,1],y,lr.y_pred)
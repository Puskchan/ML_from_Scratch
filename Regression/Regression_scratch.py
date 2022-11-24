from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype = np.float64)
ys = np.array([5,4,6,5,6,7], dtype = np.float64)

# plt.scatter(xs,ys)
# plt.show()

def best_fit_slope(xs,ys):
    m = ( (mean(xs) * mean(ys) - mean(xs*ys)) /
            ((mean(xs)**2) - mean(xs**2)))

    b = mean(ys) - m*mean(xs)
    return m, b


def sq_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

def coef_deter(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    sq_error_reg = sq_error(ys_orig, ys_line)
    sq_error_y_mean = sq_error(ys_orig, y_mean_line)
    return 1-(sq_error_reg/sq_error_y_mean)


m, b = best_fit_slope(xs,ys)

regression_line = [(m*x)+b for x in xs]

predict_x = 8
predict_y = (m*predict_x) + b

r_sq = coef_deter(ys, regression_line)
print(r_sq)

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()

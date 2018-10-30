# Linear Regression for any number of parameters in dataset.

"""
    Assumptions :
        1) Each parameter is independent of each other.
        2) The last variable is target dataset[-1].
        3) No none/null/empty values in dataset
        4) No outliers in dataframe
"""

from seaborn import pairplot
from pandas import DataFrame
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from numpy import multiply as mul_matrix
#from sklearn.metrics import confusion_matrix

class Model:
    def __init__(self, reg_coeff):
        self.reg_coeff = reg_coeff

    def predict(self, test_values):
        """
            Expects : list of x
            Returns : list of predictions
            one extra parameter in reg_coeff list of linear shift
        """

        predictions = []
        for x in test_values:
            predictions.append(mul_matrix(self.reg_coeff, [x, 1]).sum())

        return predictions

    def accuracy(self, predictions, expectations):
        n = min([len(predictions), len(expectations)])
        return 1-sum([abs(p[i]-e[i])/e[i] for i in range(n) if e[i] != 0])/n



class LinearRegressor:
    def __init__(self, data_set, test_size=0.2):
        self.data_set = DataFrame(
            {
                'x': data_set[0],  # represents x axes points
                'y': data_set[1]
            }
        )
        self.train, self.test = train_test_split(self.data_set, test_size=test_size)

    def plot(self, data=None):
        if data is None:
            "default: plotting the given dataframe"
            pairplot(self.data_set)
        else:
            "Plot each variable against every other variable."
            pairplot(data)
        plt.show()

    def fit(self):
        # X = mX + c

        # getting variables ready :
        X = self.train['x']    # capital indicates : train data(larger one)
        Y = self.train['y']
        n = len(X)
        sum_X = X.sum()
        sum_Y = Y.sum()
        sum_XX = mul_matrix(X, X).sum()
        sum_YY = mul_matrix(Y, Y).sum()
        sum_XY = mul_matrix(X, Y).sum()

        # y = mx + c is the regression line.
        m = (n*sum_XY - sum_X*sum_Y)/(n*sum_XX - sum_X**2)
        c = (sum_Y - m*sum_X)/n

        self.model = Model([m, c])
        return self.model


regressor = LinearRegressor([list(range(7)), [1, 0, 2, 3, 4, 6, 5]])

model = regressor.fit()
d = list(range(3))
p = model.predict(d)
e = d
print("Accuracy of model : ", model.accuracy(p, e))
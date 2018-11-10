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
# from sklearn.metrics import confusion_matrix


class Model:
    def __init__(self, l_regressor, reg_coeff):
        self.regressor = l_regressor
        self.reg_coeff = reg_coeff

    def predict(self, test_values):
        """
            Expects : list of x
            Returns : list of predictions
            one extra parameter in reg_coeff list of linear shift
        """
        predictions = []
        for ele in test_values:
            predictions.append(mul_matrix(self.reg_coeff, [ele, 1]).sum())

        return predictions

    def accuracy(self, predictions=None, expectations=None):

        if not predictions or not expectations:
            # user wants to get accuracy based on test data
            train_data = self.regressor.train
            predictions = self.predict(train_data["x"])
            expectations = list(train_data["y"])

        n = len(min([predictions, expectations], key=len))   # if predictions and expectations are different in length
        p, e = predictions, expectations
        return 1-sum(
            [
                abs(p[i]-e[i])/e[i]
                for i in range(n)
                if e[i] != 0]
        )/n


class LinearRegressor:
    def __init__(self, data_set, test_size=0.2):
        self.data_set = DataFrame(
            {
                'x': data_set[0],  # represents x axes points
                'y': data_set[1]
            }
        )
        self.train, self.test = train_test_split(self.data_set, test_size=test_size, shuffle=False)

    def plot(self, data=None):
        if data is None:
            "default: plotting the given dataframe"
            pairplot(self.data_set)
        else:
            pairplot(data)
        plt.show()


    def fit(self):
        # Y = mX + c

        # getting variables ready :
        X = self.train['x']    # capital indicates : train data(larger one)
        Y = self.train['y']
        n = len(X)
        sum_X = X.sum()
        sum_Y = Y.sum()
        sum_XX = mul_matrix(X, X).sum()
        sum_YY = mul_matrix(Y, Y).sum()
        sum_XY = mul_matrix(X, Y).sum()

        m = (n*sum_XY - sum_X*sum_Y)/(n*sum_XX - sum_X**2)
        c = (sum_Y - m*sum_X)/n

        self.model = Model(self, [m, c])    # model has reg_coeff in decreasing order of power.
        return self.model


if __name__ == "__main__":
    x = [0, 1, 3, 3, 4, 5, 6, 7, 8, 9]          # line y=x with some jagginess.
    y = list(range(10))
    dataset = [x, y]
    regressor = LinearRegressor(dataset)

    model = regressor.fit()

    test_data = range(10)

    prediction = model.predict(test_data)

    print("Accuracy of model : ", model.accuracy(predictions=prediction, expectations=test_data))

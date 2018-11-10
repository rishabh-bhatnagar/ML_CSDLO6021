from sklearn.model_selection import train_test_split
from numpy import array


class LinearRegressor:
    def __init__(self, dataset, error_limit=10**(-10)):
        """
        Assumptions :
            1) last row is target
        """
        self.dataset = dataset
        self.error_limit = error_limit
        self.train_data, self.test_data = train_test_split(dataset, shuffle=False)

        self.x = [[1] + row[:-1] for row in self.train_data]  # [1] is for  bias.
        self.y = [row[-1] for row in self.train_data]

        self.test_x = [[1] + row[:-1] for row in self.test_data]  # [1] is for  bias.
        self.test_y = [row[-1] for row in self.test_data]

        self.n = len(self.x[0])  # n is # features.
        self.m = len(self.x)

        self.θ = array([0 for i in range(self.n)])

    def hθ(self, x):
        return self.θ.dot(x)

    def Jθ(self):  # input should be regression coefficient
        return (1 / (2 * self.m)) * \
               sum(
                   [(self.hθ(self.x[i]) - self.y[i]) ** 2
                    for i in range(self.m)]
               )

    def Jθ_derivative(self, j):
        return (1 / (self.m)) * \
               sum(
                   [
                       (self.hθ(self.x[i]) - self.y[i]) * self.x[i][j]
                       for i in range(self.m)
                   ]
               )

    def is_convergent(self):
        cost = self.Jθ()
        if cost >= 0:
            if cost < self.error_limit:
                return True

    def fit(self, α=0.1, max_epochs=1000):
        # α : learning rate
        epoch_number = 0
        while True:
            if self.is_convergent(): break
            if epoch_number and epoch_number % max_epochs == 0:
                print(epoch_number, "epochs elapsed")
                print(self.accuracy(), "is the current accuracy.")
                stop = 1 if input("Do you want to stop training (y/*)??").lower() == 'y' else 0
                if stop:
                    print("Training aborted.")
                    break
            temp = []
            for j in range(len(self.θ)):
                temp.append(self.θ[j] - α * self.Jθ_derivative(j))
            self.θ = array(temp)
            epoch_number += 1
        print("Training completed successfully. Required {} epochs".format(epoch_number))

    def predict(self, x=None):
        error = False
        if x is None:
            x = self.test_x
        try:
            x[0][0]
        except:
            "Expected 2d array."
            error = True
        try:
            if len(x[0]) != len(self.test_x[0]) - 1:
                "Incorrect number of feature variables."
                error = True
        except:
            error = True
        if error:
            "Error occured, Predicting with test data."
            x = self.test_x
        result = []
        for x in x:
            if not error:
                x = [1]+x  # [1] is for  bias.
            result.append(self.hθ(x))
        return result


    def accuracy(self, predictions=None, expectations=None):

        if not predictions or not expectations:
            # user wants to get accuracy based on test data
            train_data = self.test_x
            predictions = self.predict(self.test_x)
            expectations = list(self.test_y)

        n = len(min([predictions, expectations], key=len))   # if predictions and expectations are different in length
        p, e = predictions, expectations
        return 1-sum(
            [
                abs(p[i]-e[i])/e[i]
                for i in range(n)
                if e[i] != 0]
        )/n


if __name__ == '__main__':
    l = LinearRegressor([
        [2, 1, 2],                # table of two
        [2, 2, 4],
        [2, 3, 6],
        [2, 4, 8],
        [2, 5, 10]
    ])
    l.fit(max_epochs=200)

    test_data = [[2, 11], [2, 12], [2, 13], [2, 14]]
    target_data = [22, 24, 26, 28]
    predictions = l.predict(test_data)
    print(l.accuracy(predictions, target_data), "is the accuracy")
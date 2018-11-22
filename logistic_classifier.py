from numpy import array
from matplotlib import pyplot as plt

class LogisticRegressor:
    def __init__(self, dataset):
        self.x = array([row[:-1] for row in dataset])
        self.y = array([row[-1] for row in dataset])
        '''plt.scatter(
            [self.x[i][0] for i in range(len(dataset)) if self.y[i]==0], 
            [self.x[i][1] for i in range(len(dataset)) if self.y[i]==0]
        )
        plt.scatter(
            [self.x[i][0] for i in range(len(dataset)) if self.y[i]==1], 
            [self.x[i][1] for i in range(len(dataset)) if self.y[i]==1]
        )
        plt.plot([i for i in range(10)], [i for i in range(10)])
        plt.show()
        input()'''
    def cost(self, )
dataset = [
              [2, 3, 0],
              [2, 4, 0],
              [3, 5, 0],
              [5, 7, 0],
              [4, 5, 0],
              [1, 4, 0],
              [3, 2, 1],
              [4, 2, 1],
              [3, 2, 1],
              [5, 3, 1],
              [6, 4, 1],
              [2, 1, 1]
          ]
l = LogisticRegressor(dataset)

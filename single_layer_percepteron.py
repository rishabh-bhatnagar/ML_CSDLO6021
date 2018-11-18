from numpy import array
class Percepteron:
    def __init__(self, dataset, learning_rate = 0.01):
        """
        dataset is combination of features(X) and target(Y)
            dataset = [ X0 X1 X2  ... Xn Y]
        """
        self.x = array(dataset[:-1])
        self.y = array(dataset[-1])
        self.n =  
        w = [0 for i in range(n)]
Percepteron(
    [
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1]
    ]
)

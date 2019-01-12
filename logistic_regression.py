def cost(self, ):
    total_cost = 0
    for i in range(self.m):
        total_cost += 
def loss(y_hat, y):
    return -(y*log(y_hat)+(1-y)*log(1-y_hat))

"Wrong code."

# from pandas import DataFrame
# from sklearn.model_selection import train_test_split
#
# from linear_regression import LinearRegressor
# """
# Logistic function == sigmoid function
# Some assumptions of logistic regression :
#     1) All predictors are independent of each other.
#     2) Each predictor must atleast have 50 rows each for better output.
# """
#
# """
#     S curve is linear in logit
#     Logit  = log of odds"
#     Logit(p) = log(p/(1-p))
# """
#
# class Model:
#     def __init__(self, regressor, reg_coeff):
#         self.regressor = regressor
#         self.b1, self.b0 = reg_coeff
#         b1, b0 = self.b1, self.b0
#         print("b1, b0 : ", b1, b0)
#         e_b1_b0x = lambda x : __import__("math").exp(b1*x + b0)
#
#         self.coeff = lambda x : e_b1_b0x(x)/\
#                                 (e_b1_b0x(x)+1)
#     def predict(self, test_values):
#         predictions = []
#         for x in test_values :
#             predictions.append(self.coeff(x))
#         return predictions
#
# class LogisticRegressor:
#     """
#         Data will have last column as target variable.
#         For simplicity, considering only one feature.
#     """
#
#     def __init__(self, data_set, test_size=0.2):
#         self.data_set = DataFrame(
#             {
#                 'x': data_set[0],  # represents x axes points
#                 'y': data_set[1]
#             }
#         )
#         self.train, self.test = train_test_split(self.data_set, test_size=test_size)
#
#     def fit(self):
#         data_set = [list(self.train["x"]), list(self.train["y"])]
#         linear_regressor = LinearRegressor(data_set)
#         linear_model = linear_regressor.fit()
#         b1, b0 = linear_model.reg_coeff        #b is beta.
#         e = __import__("math").e
#
#         #logit(p) = b1*x+b0
#         self.model = Model(self, [b1, b0])
#         return self.model
#
# data_set = [list(range(10)), [1 if i%5 == 0 else 0 for i in range(10)]]
# print(data_set[1])
# regressor = LogisticRegressor(data_set)
# model = regressor.fit()
# print(model.predict([5, 10, 12]))
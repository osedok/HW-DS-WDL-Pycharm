from sklearn.linear_model import LinearRegression
from pandas import DataFrame


class LinearRegressionModelAnalysis:
    # parameterized constructor
    def __init__(self, model: LinearRegression, data: DataFrame, independent_variable_name, target_variable_name):
        self.model = model
        self.data = data
        self.x_name = independent_variable_name
        self.y_name = target_variable_name

        self.x_mean = self.data[self.x_name].mean()
        print(self.x_name, "mean: ", self.x_mean)
        self.y_mean = self.data[self.y_name].mean()
        print(self.y_name, "mean: ", self.y_mean)

    def show_options_for_target(self, target_variable_value):
        a = self.model.coef_[0]
        b = self.model.intercept_
        y = target_variable_value
        print("Target mean value for dependent variable is: ", y)

        # Target mean value for independent variable
        # if y = a*x + b then:
        x = (y - b) / a
        print("Target mean value for independent variable is: ", x)

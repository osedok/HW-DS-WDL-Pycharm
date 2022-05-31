from sklearn.linear_model import LinearRegression
from pandas import DataFrame


class LinearRegressionModelAnalysis:

    # parameterized constructor
    def __init__(self, model: LinearRegression, data: DataFrame, independent_variable_name, target_variable_name):

        self.x_sum_diff = 0.00
        self.model = model
        self.data = data
        self.x_name = independent_variable_name
        self.y_name = target_variable_name

        self.records_count = self.data[self.x_name].count()
        print("\n\nRecords count is: ", self.records_count)

        self.x_mean = self.data[self.x_name].mean()
        print("\n\nCurrent", self.x_name, "mean:", self.x_mean)
        self.x_sum = self.data[self.x_name].sum()
        print("Current", self.x_name, "sum:", self.x_sum)

        self.y_mean = self.data[self.y_name].mean()
        print("\nCurrent", self.y_name, "mean:", self.y_mean)

    def show_options_for_target(self, target_variable_value):
        a = self.model.coef_[0]
        b = self.model.intercept_
        y = target_variable_value
        print("\n\nTarget mean value for dependent variable ", self.y_name, "is: ", y)

        # Target mean value for independent variable
        # if y = a*x + b then:
        x = (y - b) / a
        print("Target mean value for independent variable", self.x_name, "is: ", x)

        # To reach target value for dependant variable we need to apply required changes
        # to the independent variable, so the "new" mean value target is met

        # To achieve above:
        # 1. Calculate the new sum of "x" values required to reach the target
        # X' = sum(x)/count(x) => sum(x) = X' * count(x)

        x_sum_target = x * self.records_count

        # 2. Now we need to find the difference between original sum of x and target sum of x
        self.x_sum_diff = self.x_sum - x_sum_target

        # 3. Show possible options for reaching the target y value
        # For flexibility in terms of setting the values by user this will be called from the main.py

    def reach_target_option_1(self, target_variable_value):
        # This option aim is to provide how much we need to change the value of independent variable in every row (z)
        # to reach the target mean value
        z = self.x_sum_diff / self.records_count
        action = ""
        if z < 0:
            action = "raise"
        else:
            action = "reduce"

        print("\n\nSolution 1: To reach target value of", self.y_name, "=", target_variable_value,
              " you need to", action, "value of", self.x_name, "in every row by", abs(z))

    def reach_target_option_2(self, target_variable_value, target_top_deprived_count,
                              target_top_deprived_reduction_percent):
        # This option aim is to change the value of independent variable by 25% in 50 rows containing the highest value
        # of the CIF and distribute evenly the remaining difference if needed.

        top = self.data.sort_values(by=[self.x_name], ascending=[False]).head(target_top_deprived_count)
        top_x_sum = top[self.x_name].sum()

        top_x_sum_reduced = top_x_sum * target_top_deprived_reduction_percent * 0.01

        if self.records_count - target_top_deprived_count > 0:
            # calculate the difference
            z = (self.x_sum_diff - top_x_sum_reduced) / (self.records_count - target_top_deprived_count)

            action = ""
            if z < 0:
                action = "raise"
            else:
                action = "reduce"

            print("\n\nSolution 2: To reach target value of", self.y_name, "=", target_variable_value,
                  " you could", action, "value of", self.x_name, "in", target_top_deprived_count, "most "
                                                                                                  "deprived areas by",
                  target_top_deprived_reduction_percent, "%. All remaining areas would "
                                                         "also need to", action, self.x_name, "by the fixed value of",
                  abs(z),
                  "per row.")

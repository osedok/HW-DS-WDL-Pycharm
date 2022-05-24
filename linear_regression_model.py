import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from yellowbrick.regressor import ResidualsPlot


class LinearRegressionModel:

    # parameterized constructor
    def __init__(self, data: DataFrame, independent_variable_name, target_variable_name):
        self.RMSE2 = None
        self.RMSE = None
        self.MSE = None
        self.MAE = None

        # Preparing train and test data
        # Independent Features
        self.X = data[[independent_variable_name]]
        # Dependent or Target variable
        self.y = data[target_variable_name]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25,
                                                                                random_state=0)
        # summarize the shape of the training dataset
        print("Trained dataset shape: ", self.X_train.shape, self.y_train.shape)

        self.process_outliers()
        self.linear_regression_model = self.initialise_model()
        # Predict the values given test set
        self.predictions = self.linear_regression_model.predict(self.X_test)
        self.calculate_model_errors()
        self.generate_model_visualisation(self.linear_regression_model)
        self.plot_residuals()

    def process_outliers(self):
        # Dealing with outliers
        # Initiate Local Outlier Factor
        lof = LocalOutlierFactor()
        outliers = lof.fit_predict(self.X_train)
        # select all rows that are not outliers
        mask = outliers != -1
        self.X_train = self.X_train[mask]
        self.y_train = self.y_train[mask]
        # summarize the shape of the updated training dataset
        print("Trained dataset shape (outliers removed): ", self.X_train.shape, self.y_train.shape)

    def initialise_model(self):
        # Create linear regression model
        linear_regression = LinearRegression()

        # Fit the linear regression model
        linear_regression.fit(self.X_train, self.y_train)
        return linear_regression

    def calculate_model_errors(self):
        # Folowing Errors were calculated for the model
        self.MAE = mean_absolute_error(self.y_test, self.predictions)
        self.MSE = mean_squared_error(self.y_test, self.predictions)
        self.RMSE = np.sqrt(self.MSE)
        self.RMSE2 = 2 * self.RMSE  # 2*sigma ~ 95% confidence region

        # Evaluate mean absolute error
        print("Mean Absolute Error(MAE): {0}".format(self.MAE))

        # Evaluate mean squared error
        print("Mean Squared Error(MSE): {0}".format(self.MSE))

        # Evaluate root mean squared error
        print("Root Mean Squared Error(RMSE): {0}".format(self.RMSE))

        # Evaluate R2-square
        print("RMSE2 - 2 sigma ~ 95% confidence region : {0}".format(self.RMSE2))

    def generate_model_visualisation(self, model: LinearRegression):
        # Print the intercept and coefficients
        print('LM Intercept:' + str(model.intercept_))
        print('LM Coefficient:' + str(model.coef_[0]))
        # Plotting a linear regression line on a scatter plot
        # Displays a scatter plot of data along with a line of best fit for the data.
        fig, axs = plt.subplots(1, 1, figsize=(16, 12))
        plt.title("CIF and Employment Rate - Linear Regretion Model Visualisation")
        plt.xlabel("CIF")
        plt.ylabel("Deprived Employment Rate")
        plt.plot(self.X_train, self.y_train, 'o')

        # Plot regression line
        plt.plot(self.X_train, model.coef_ * self.X_train + model.intercept_, color='red',
                 alpha=0.8)

        equasion = "y={0}x{1}".format(model.coef_[0], model.intercept_)
        plt.annotate(equasion, (5, 0.41), fontsize=14)

        # Two Errors ranges have been added to the plot utilising continues method described here:
        # https://jakevdp.github.io/PythonDataScienceHandbook/04.03-errorbars.html

        xfit = np.linspace(0, 350, 1000)  # https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        yfit = model.predict((xfit[:,
                              np.newaxis]))  # https://numpy.org/doc/stable/reference/constants.html?highlight=newaxis#numpy.newaxis

        plt.plot(xfit, yfit, '-', color='gray')
        plt.fill_between(xfit, yfit - self.RMSE2, yfit + self.RMSE2,
                         color='gray', alpha=0.2, label="RMSE 2 - 95% confidence region ({0})".format(self.RMSE2))
        plt.fill_between(xfit, yfit - self.MAE, yfit + self.MAE,
                         color='red', alpha=0.1, label="MAE - Mean Absolute Error ({0})".format(self.MAE))
        plt.legend()
        plt.xlim(0, 300)
        plt.show()
        fig.savefig("output_graphs/lr_model_visualisation.png")

    def plot_residuals(self):
        # plotting the residuals, in the context of regression models,
        # are the difference between the observed value of the target variable (y)
        # and the predicted value (ŷ), i.e. the error of the prediction.
        # The residuals plot shows the difference between residuals on the vertical
        # axis and the dependent variable on the horizontal axis, allowing you to detect
        # regions within the target that may be susceptible to more or less error.
        # - Source: https://www.scikit-yb.org/en/latest/api/regressor/residuals.html

        # A common use of the residuals plot is to analyze the variance of the error of the regressor.
        # If the points are randomly dispersed around the horizontal axis, a linear regression model
        # is usually appropriate for the data; otherwise, a non-linear model is more appropriate.
        # In the case above, we see a fairly random, uniform distribution of the residuals against
        # the target in two dimensions. This seems to indicate that our linear model is performing well.
        # We can also see from the histogram that our error is normally distributed around zero,
        # which also generally indicates a well-fitted model.

        fig, axs = plt.subplots(1, 1, figsize=(16, 10))
        sns.histplot(self.y_test - self.predictions, kde=True, stat="density", linewidth=1)
        # normal distribution indicates that the model is doing well
        plt.show()
        fig.savefig("output_graphs/lr_residuals.png")

        visualizer = ResidualsPlot(self.linear_regression_model, size=(1080, 720),
                                   title='Residuals for SIMD (CIF and Employment Rate) Linear Regression Model')
        visualizer.fit(self.X_train, self.y_train)  # Fit the training data to the visualizer
        visualizer.score(self.X_test, self.y_test)  # Evaluate the model on the test data
        visualizer.show()

        fig = visualizer.fig
        fig.savefig("output_graphs/lr_residuals_1.png")




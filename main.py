import pandas as pd
from models import ConfigInfo
from functions import Functions
from linear_regression_model import LinearRegressionModel
from linear_regression_model_analysies import LinearRegressionModelAnalysis

# https://cnvrg.io/pycharm-machine-learning/
# https://towardsdatascience.com/jupyter-notebook-vs-pycharm-7301743a378
# https://datasciencenerd.com/pycharm-vs-jupyter-which-is-better-for-data-science/

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Read data from MS Excel Format, parameter sheet_name was provided to access the specific task within an MS
    # Excel book.
    simd = pd.read_excel(ConfigInfo.datafile, sheet_name=ConfigInfo.datafile_tab_name)
    simd = Functions.clean_simd(simd)
    Functions.generate_correlation_heatmap(simd)

    x = simd.iloc[:, 3:]
    Functions.calc_vif(x)
    simd.drop(['Income_rate', 'EMERG', 'DEPRESS', 'SMR'], axis=1, inplace=True)
    x = simd.iloc[:, 4:]
    Functions.calc_vif(x)

    # Let's take a copy of the data just for the NLC extent.
    nlc = simd.loc[(simd.Council_area == 'North Lanarkshire')].copy()
    Functions.plot_histograms(simd, nlc)

    # test if variables matches normal distribution
    Functions.test_normal_distribution(simd, nlc)

    lr = LinearRegressionModel(simd, "CIF", "Employment_rate")
    lra = LinearRegressionModelAnalysis(lr.linear_regression_model, nlc, "CIF", "Employment_rate")
    lra.show_options_for_target(0.10)







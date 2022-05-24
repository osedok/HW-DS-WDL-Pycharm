import pandas as pd
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor



class Functions:

    @staticmethod
    def clean_simd(data: DataFrame):
        # Remove not relevant data attributes leaving only Health and income related fields.
        data.drop(data.columns[list(range(19, 29))], axis=1, inplace=True)
        data.drop(['Total_population', 'Working_age_population', 'Income_count', 'Employment_count', 'Attendance',
                   'Attainment', 'no_qualifications', 'crime_count', 'crime_rate', 'overcrowded_count',
                   'overcrowded_rate',
                   'nocentralheat_count', 'nocentralheat_rate', 'Broadband', 'PT_retail', 'LBWT'], axis=1, inplace=True)

        # Number of attributes are of data type 'object' - considering the above we will replace '*' with 0 and
        # convert the column to numeric type. Some values are converted to numeric (float) and some to integer form
        # as appropriate for the attribute e.g. percentages values will be represented as float and standardised or
        # cases per 10K like crime will be integer.
        data.loc[(data.Income_rate == '*'), 'Income_rate'] = '0'
        data.Income_rate = pd.to_numeric(data.Income_rate)
        data.loc[(data.Employment_rate == '*'), 'Employment_rate'] = '0'
        data.Employment_rate = pd.to_numeric(data.Employment_rate)
        data.loc[(data.CIF == '*'), 'CIF'] = '0'
        data.CIF = data['CIF'].astype(int)
        data.loc[(data.ALCOHOL == '*'), 'ALCOHOL'] = '0'
        data.ALCOHOL = data['ALCOHOL'].astype(int)
        data.loc[(data.DRUG == '*'), 'DRUG'] = '0'
        data.DRUG = data['DRUG'].astype(int)
        data.loc[(data.SMR == '*'), 'SMR'] = '0'
        data.SMR = data['SMR'].astype(int)
        data.loc[(data.DEPRESS == '*'), 'DEPRESS'] = '0'
        data.DEPRESS = pd.to_numeric(data.DEPRESS)
        data.loc[(data.EMERG == '*'), 'EMERG'] = '0'
        data.EMERG = data['EMERG'].astype(int)

        print("Cleaning SIMD finished...")
        print("Checking dataset for NULL values within numerical attributes:")
        print(data.select_dtypes(['float64', 'int64']).isnull().any())
        print("Checking dataset finished - all values above should read 'False'")
        return data

    @staticmethod
    def generate_correlation_heatmap(data: DataFrame):
        correlation = pd.DataFrame(data[data.columns[list(range(3, 11))]])
        corr_matrix = correlation.corr()
        plt.subplots(figsize=(10, 10))
        heatmap = sns.heatmap(corr_matrix, annot=True)
        plt.show()
        fig = heatmap.get_figure()
        fig.savefig("output_graphs/simd_correlation_heatmap.png")

    # Import library for VIF Source code:
    # https://gist.githubusercontent.com/aniruddha27/579153e1c77773d7a7038b548f929b57/raw
    # /9ac37b7120770ca6e2a005bf7beb944a00f1bfc0/Multicollinearity_VIF.py
    @staticmethod
    def calc_vif(x):
        # Calculating VIF
        vif = pd.DataFrame()
        vif["variables"] = x.columns
        vif["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
        # print(vif)
        return vif

    @staticmethod
    def plot_histograms(simd: DataFrame, nlc: DataFrame):
        fig, axs = plt.subplots(2, 3, figsize=(16, 12))
        # Plot Scotland row
        sns.histplot(data=simd, x="CIF", kde=True, bins=20, alpha = .5, color="skyblue",  ax=axs[0, 0])
        axs[0, 0].set(xlabel='CIF', title='Scotland')
        sns.histplot(data=simd, x="ALCOHOL", kde=True, bins=20, alpha = .5, color="red",  ax=axs[0, 1])
        axs[0, 1].set(xlabel='ALCOHOL', title='Scotland')
        sns.histplot(data=simd, x="DRUG", kde=True, bins=20, alpha = .5, color="red",  ax=axs[0, 2])
        axs[0, 2].set(xlabel='DRUG', title='Scotland')
        # Plot NLC row
        sns.histplot(data=nlc, x="CIF", kde=True, bins=20, alpha = .5, color="skyblue",  ax=axs[1, 0])
        axs[1, 0].set(xlabel='CIF', title='North Lanarkshire')
        sns.histplot(data=nlc, x="ALCOHOL", kde=True, bins=20, alpha = .5, color="red",  ax=axs[1, 1])
        axs[1, 1].set(xlabel='ALCOHOL', title='North Lanarkshire')
        sns.histplot(data=nlc, x="ALCOHOL", kde=True, bins=20, alpha = .5, color="green",  ax=axs[1,2])
        axs[1, 2].set(xlabel='DRUG', title='North Lanarkshire')
        plt.show()
        fig.savefig("output_graphs/simd_histograms.png")

    @staticmethod
    def test_normal_distribution(simd: DataFrame, nlc: DataFrame):
        fig, axs = plt.subplots(2, 3, figsize=(16, 12))

        sm.qqplot(data=simd.CIF, line='s', marker='o', ax=axs[0, 0])
        axs[0, 0].set(xlabel='CIF', title='Scotland')

        sm.qqplot(simd.ALCOHOL, line='s', ax=axs[0, 1])
        axs[0, 1].set(xlabel='ALCOHOL', title='Scotland')

        sm.qqplot(simd.DRUG, line='s', ax=axs[0, 2])
        axs[0, 2].set(xlabel='DRUG', title='Scotland')

        sm.qqplot(nlc.CIF, line='s', ax=axs[1, 0])
        axs[1, 0].set(xlabel='CIF', title='North Lanarkshire')

        sm.qqplot(nlc.ALCOHOL, line='s', ax=axs[1, 1])
        axs[1, 1].set(xlabel='ALCOHOL', title='North Lanarkshire')

        sm.qqplot(nlc.DRUG, line='s', ax=axs[1, 2])
        axs[1, 2].set(xlabel='DRUG', title='North Lanarkshire')

        plt.show()
        fig.savefig("output_graphs/simd_normal_distribution_test.png")
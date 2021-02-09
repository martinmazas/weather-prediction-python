import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.naive_bayes import GaussianNB
# import sklearn as skl
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# import pydotplus
# pd.set_option("display.max_rows", None, "display.max_columns", None)


def categorical_feature_to_numerical(df, feature):
    # not_null_values = np.where(df[feature].isnull() == False)
    # print(not_null_values.unique())
    df[f'{feature}_num'] = pd.Categorical(df[feature], categories=df[feature].unique()).codes


def main():
    # Create data set
    weather_df = pd.read_csv('weather3.csv')
    # print(weather_df.info())
    # columns = weather_df.columns
    # print(columns)
    # print(weather_df.shape)
    # print(weather_df.head())
    # print(weather_df.describe(include='all'))

    """Check how many values are in each column. 
    There are some columns with less than 60% of the data, so ignore them"""
    # print(weather_df.count().sort_values())
    weather_df = weather_df.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am'], axis=1)
    # print(weather_df.shape)

    """There are some columns that still have nan values, so ignore them to"""
    weather_df = weather_df.dropna()
    print(weather_df.count().sort_values())

    # categorical_features = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
    # weather_df.dropna(inplace=True, subset=categorical_features)
    #
    # # shape (145460, 23)
    # # print(weather_df.shape)
    #
    # # 23 features
    # # features = weather_df.columns
    # # print(features)
    #
    # # To see which feature has null values
    # # print(weather_df.describe(include='all'))
    #
    # # Fill Nan values on numerical features with mean column value
    # weather_df.fillna(weather_df.mean(), inplace=True)
    # # print(weather_df.info())
    #
    # # Create column with each categorical data in numeric values
    # # categorical_feature_to_numerical(weather_df, 'Location')
    # # categorical_feature_to_numerical(weather_df, 'WindGustDir')
    #
    # # Check which of the categorical features has nan values(Only Location doesnt have), they are all nominal
    # categorical_features = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
    # for cat in categorical_features:
    #     categorical_feature_to_numerical(weather_df, cat)
    #
    # # sns.violinplot(x='RainTomorrow_num', data=weather_df)
    # # plt.show()
    #
    # sns.violinplot(x='WindGustDir_num', data=weather_df)
    # plt.show()
    #
    # # weather_df['location_num'] = pd.Categorical(weather_df['Location'], categories=weather_df['Location'].unique()).codes
    # # categorical_features = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
    # # test = weather_df['WindGustDir'].isnull()
    # # m = np.where(test == False)
    # # print(np.where(test == True))
    # # print(weather_df['WindGustDir'])
    # print(weather_df.describe(include='all'))


if __name__ == '__main__':
    main()

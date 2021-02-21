import pandas as pd
import numpy as np
from io import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz


import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.naive_bayes import GaussianNB
# import sklearn as skl
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# import pydotplus
# pd.set_option("display.max_rows", None, "display.max_columns", None)

from sklearn.inspection import permutation_importance


rain = pd.read_csv('weather3.csv', index_col='Date')
rain.dropna(inplace=True)

rain['RainToday'].replace({'No': 0.0, 'Yes': 1.0}, inplace=True)
rain['RainTomorrow'].replace({'No': 0.0, 'Yes': 1.0}, inplace=True)

rain['Location'] = pd.Categorical(rain['Location'], categories=rain['Location'].unique()).codes
rain['WindGustDir'] = pd.Categorical(rain['WindGustDir'], categories=rain['WindGustDir'].unique()).codes
rain['WindDir9am'] = pd.Categorical(rain['WindDir9am'], categories=rain['WindDir9am'].unique()).codes
rain['WindDir3pm'] = pd.Categorical(rain['WindDir3pm'], categories=rain['WindDir3pm'].unique()).codes


X = rain.drop(["RainTomorrow"],axis=1)
y = rain['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)  # 70% training and 30% test
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

result = permutation_importance(clf, X, y, n_repeats=10,random_state=0)
importance = zip(X.columns,result['importances_mean'])
# summarize feature importance
for i,v in importance:
    print('Feature: %s, Score: %.5f' % (i,v))
# plot feature importance
print(len(X.columns),[x[1] for x in importance])
plt.bar(range(len(X.columns)), result['importances_mean'])
plt.xticks(ticks=range(len(X.columns)),labels=X.columns, rotation=90)
plt.show()

y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))

dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,feature_names=X.columns,class_names=clf.classes_)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Rain.png')
Image(graph.create_png())


def categorical_feature_to_numerical(df, feature):
    # not_null_values = np.where(df[feature].isnull() == False)
    # print(not_null_values.unique())
    df[f'{feature}_num'] = pd.Categorical(df[feature], categories=df[feature].unique()).codes


def main():
    # Create data set
    weather_df = pd.read_csv('weather3.csv')
    weather_df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
    weather_df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)
    sns.pairplot(weather_df[['Humidity3pm', 'Humidity9am', 'RainToday', 'RainTomorrow']], hue='RainTomorrow', height=2.5)
    sns.pairplot(weather_df[['Pressure9am', 'Pressure3pm', 'Temp3pm', 'RainTomorrow']], hue='RainTomorrow',
                 height=2.5)
    plt.show()
    # print(weather_df.info())
    # columns = weather_df.columns
    # print(columns)
    # print(weather_df.shape)
    # print(weather_df.head())
    # print(weather_df.describe(include='all'))

    """Check how many values are in each column. 
    There are some columns with less than 60% of the data, so ignore them"""
    # print(weather_df.count().sort_values())
    # weather_df = weather_df.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am'], axis=1)
    # # print(weather_df.shape)
    #
    # """There are some columns that still have nan values, so ignore them to"""
    # weather_df = weather_df.dropna()
    # print(weather_df.count().sort_values())


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

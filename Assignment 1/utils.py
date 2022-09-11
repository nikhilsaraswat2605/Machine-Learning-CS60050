import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt


class DecisionTree:
    pass


def entropy(lables):
    total = len(lables)
    diff_values = lables.value_counts().tolist()
    entropys = [-1 * (i/total) * math.log2(i/total) for i in diff_values]
    return sum(entropys)


def information_gain(data, lables, attribute, split_value):
    # calculate the information gain
    if attribute == 'Age' or attribute == 'Family_Size' or attribute == 'Work_Experience':
        filtered = data[attribute] < split_value
    else:
        filtered = data[attribute] == split_value
    left = lables[filtered]
    right = lables[~filtered]
    gain = entropy(lables) - (len(left)/len(lables)) * \
        entropy(left) - (len(right)/len(lables)) * entropy(right)
    return gain


def split_data(df, train_ratio=0.8, test_ratio=0.2):
    # split the data into train and test set
    if abs(train_ratio + test_ratio - 1) > 1e-6:  # check if the sum of ratios is 1
        raise ValueError("Train and test ratio must sum to 1")
    # number of rows in train set
    train_size = int(train_ratio * len(df.index))
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle the data
    df1 = df.iloc[:train_size, :].reset_index(drop=True)  # train set
    df2 = df.iloc[train_size:, :].reset_index(drop=True)  # test set
    return df1, df2


def find_best_splitting(data, lables, attributes):
    # find the best splitting attribute and split value
    best_gain = 0
    best_split_value = None
    if attributes == 'Age' or attributes == 'Family_Size' or attributes == 'Work_Experience':
        vals = np.sort(data[attributes].unique())
        for i in range(len(vals)-1):
            split_value = (vals[i] + vals[i+1])/2
            gain = information_gain(data, lables, attributes, split_value)
            if gain > best_gain:
                best_gain = gain
                best_split_value = split_value
    else:
        vals = data[attributes].unique()
        for i in range(len(vals)):
            split_value = vals[i]
            gain = information_gain(data, lables, attributes, split_value)
            if gain > best_gain:
                best_gain = gain
                best_split_value = split_value
    return best_gain, best_split_value


def split_df_column(df):
    # split the dataframe into data and lables
    data = df.iloc[:, :-1]
    lables = df.iloc[:, -1]
    return data, lables


def get_prediction_accuracy(tree, test_df):
    # get the prediction accuracy
    test_data, test_lables = split_df_column(test_df)
    predictions = tree.predict(test_data)
    # print(predictions)
    # print(test_lables)
    accuracy = np.mean(predictions == test_lables)*100
    return (predictions, accuracy)

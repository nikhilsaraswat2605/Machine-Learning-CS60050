from tkinter.tix import Tree
import pandas as pd
import argparse
from utils import get_prediction_accuracy, split_data, split_df_column
from decision_tree import DecisionTree


def Calculate_Impurity_Measure(train, test, max_depth = 5):
    # create a decision tree
    tree = DecisionTree(max_depth=max_depth)

    tree.train(train)
    _, train_accuracy = get_prediction_accuracy(tree, train)
    preds, test_accuracy = get_prediction_accuracy(tree, test)
    _, lables = split_df_column(test)
    # f1 = F1_score(preds, lables)
    print('------------ RESULTS ------------')
    print('Train Accuracy: ', train_accuracy)
    print('Test Accuracy: ', test_accuracy)
    # print('F1 Score: ', f1)


def select_best_tree(train, test, max_depth, num_splits=2):
    average_train_accuracy = 0
    average_test_accuracy = 0
    best_test_accuracy = 0
    best_train = None
    best_tree = None
    best_valid = None
    for i in range(num_splits):
        train_df, valid = split_data(train, 0.8, 0.2)
        tree = DecisionTree(max_depth=max_depth)
        tree.train(train_df)
        _, train_accuracy = get_prediction_accuracy(tree, train_df)
        _, test_accuracy = get_prediction_accuracy(tree, test)
        print(
            f'Split {i+1}: Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}')
        average_train_accuracy += train_accuracy
        average_test_accuracy += test_accuracy
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_train = train_df
            best_tree = tree
            best_valid = valid
    average_train_accuracy /= num_splits
    average_test_accuracy /= num_splits
    return best_train, best_tree, best_valid, average_train_accuracy, average_test_accuracy, best_test_accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Decision Tree Classifier")  # create an argument parser
    parser.add_argument('--file', type=str, default='Dataset_A.csv',
                        help='Data file name')  # add an argument file
    parser.add_argument('--depth', type=int, default=5,
                        help='Maximum depth of the tree')  # add an argument depth
    args = parser.parse_args()  # parse the arguments and store them in args
    max_depth = args.depth  # get the value of the argument depth
    if max_depth <= 0:  # check if the value of depth is valid
        raise ValueError("Maximum depth must be positive")
    file = args.file  # get the value of the argument file
    df = pd.read_csv(file)  # read the data file
    print('\n------------ LOADED DATA ------------')
    df.dropna(inplace=True) # drop rows with missing values
    df.drop('ID', axis = 1, inplace=True) # drop the id column
    # split the data into train and test set
    train, test = split_data(df, 0.8, 0.2)
    # print(test.head())  # print the first 5 rows of the data
    print('\n------------ Calculating Impurity Measure ------------')
    # create a decision tree
    Calculate_Impurity_Measure(train, test, max_depth)
    print('\n-------------- Determine Accuracy over 2 random splits --------------')
    best_train, best_tree, best_valid, average_train_accuracy, average_test_accuracy, best_test_accuracy = select_best_tree(
        train, test, max_depth)
    print(f'Average Test Accuracy: {average_test_accuracy}')
    print(f'Best Test Accuracy: {best_test_accuracy}')
    print('\n-------------- print before pruning --------------')
    best_tree.print_tree('before_pruning.gv')

    _, valid_accuracy = get_prediction_accuracy(best_tree, best_valid)
    best_tree.root.prune(best_tree, valid_accuracy, best_valid)
    print('\n-------------- print after pruning --------------')
    best_tree.print_tree('after_pruning.gv')

if __name__ == "__main__":
    main()

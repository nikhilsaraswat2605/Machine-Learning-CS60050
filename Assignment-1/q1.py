import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import graphviz


def Calculate_Entropy(df):
    """
        Calculates the entropy of the given dataset.
        Args:
            df (DataFrame): The dataset for which the entropy is to be calculated.
    """
    label = df.keys()[-1]
    values = df[label].unique()
    entropy = 0
    for value in values:
        fraction = df[label].value_counts()[value]/len(df[label])
        entropy -= fraction*np.log2(fraction)
    return entropy


def Calculate_Feature_Entropy(df, feature):
    """
        Calculates the entropy of the given feature in the given dataset.
        Args:
            df (DataFrame): The dataset for which the entropy is to be calculated.
            feature (str): The feature for which the entropy is to be calculated.
    """
    epsilon = np.finfo(float).eps
    label = df.keys()[-1]
    variables = df[feature].unique()
    feature_entropy = 0
    target_variables = df[label].unique()
    for variable in variables:
        den = len(df[feature][df[feature] == variable])
        entropy = 0
        for target_variable in target_variables:
            num = len(df[feature][df[feature] == variable]
                      [df[label] == target_variable])
            fraction = num/(den + epsilon)
            entropy -= fraction*np.log(fraction + epsilon)
        fraction = den/len(df)
        feature_entropy -= fraction * entropy
    return abs(feature_entropy)


def Information_Gain(df):
    """
        Calculates the information gain of the given dataset.
        Args:
            df (DataFrame): The dataset for which the information gain is to be calculated.
    """
    Info_Gain = []
    for key in df.keys()[:-1]:
        Info_Gain.append(Calculate_Entropy(
            df) - Calculate_Feature_Entropy(df, key))
    return Info_Gain


def get_best_attribute(df):
    """
        Returns the attribute with the highest information gain.
        Args:
            df (DataFrame): The dataset for which the attribute is to be selected.
    """
    IG = Information_Gain(df)
    return df.keys()[:-1][np.argmax(IG)]


def Shuffled_Split_Data(df, train_ratio=0.8):
    """
        Randomly splits the given dataset into train and test sets.
        Args:
            df (DataFrame): The dataset to be split.
            train_ratio (float): The ratio of the dataset to be used as the train set.
            test_ratio (float): The ratio of the dataset to be used as the test set.
    """
    # split the data into train and test set
    train_size = int(train_ratio * len(df.index))
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle the data
    df2 = df.iloc[train_size:, :].reset_index(drop=True)  # test set
    df1 = df.iloc[:train_size, :].reset_index(drop=True)  # train set
    return df1, df2


def split_df_column(df):
    """
        Splits the given dataframe into data and labels.
        Args:
            df (DataFrame): The dataframe to be split.
    """
    # split the dataframe into data and lables
    lables = df.iloc[:, -1]
    data = df.iloc[:, :-1]
    return data, lables


class Node:
    def __init__(self, attribute, prob_lable=None):
        """
            Constructor for the Node class
            Args:
                attribute (str): The attribute of the node
                prob_lable (str): The most probable lable of the node
        """
        self.children = []
        self.values = []
        self.prob_lable = prob_lable
        self.values = None
        self.attribute = attribute

    def is_leaf(self):
        """
            Return True if the node is a leaf node, False otherwise
        """
        if self.children == []:
            return True
        return False

    def CountNodes(self):
        """
            Return the number of nodes in the subtree rooted at the node
        """
        if self.is_leaf():
            return 1
        else:
            count = 1
            for child in self.children:
                if child is not None:
                    count += child.CountNodes()
            return count

    def StringFormat(self):
        """
            Return a string representation of the node
        """
        if self.is_leaf():
            outcome = self.prob_lable
            return f"{self.attribute}\n{outcome}"
        else:
            return f"{self.attribute}\n"

    def reduced_error_pruning(self, tree, accuracy, valid):
        """
            Prunes the tree using reduced error pruning.
            Args:
                tree (Node): The root of the tree to be pruned.
                accuracy (float): The accuracy of the tree.
                valid (DataFrame): The validation set.
        """
        new_accuracy = 0
        if self.children == []:
            return accuracy
        for child in self.children:
            if child == None:
                continue
            new_accuracy = child.reduced_error_pruning(tree, accuracy, valid)
        children = self.children
        self.children = []
        _, temp_accuracy = get_prediction_accuracy(tree, valid)
        # decide if we will prune this node or node
        if temp_accuracy < new_accuracy or tree.root.CountNodes() <= 5:
            self.children = children
        else:
            new_accuracy = temp_accuracy
            self.attr = 'Outcome'
        return new_accuracy

    def print_tree_dfs(self, file, line_gap=""):
        """
            Prints the tree to the given file.
            Args:
                file (str): The file to which the tree is to be printed.
                line_gap (str): The line gap to be used to just give intendation.
        """
        if self.is_leaf():
            print(f"{line_gap}OutCome = {self.prob_lable}", file=file)
            return
        print(line_gap, end="", file=file)
        print(self.attribute, end=" ", file=file)
        print(self.values, file=file)
        line_gap += "\t\t"
        for child in self.children:
            if child is None:
                continue
            child.print_tree_dfs(file, line_gap)
        return


def get_attribute_subset(df, attribute, attrValue):
    """
        Returns the subset of the given dataset with the given attribute value.
        Args:
            df (DataFrame): The dataset to be subsetted.
            attribute (str): The attribute to be used for subsetting.
            attrValue (str): The attribute value to be used for subsetting.
    """
    return df[df[attribute] == attrValue].reset_index(drop=True)


class DecisionTree:

    def __init__(self, max_depth=10, min_samples=1):
        """
            Constructor for the DecisionTree class
            Args:
                max_depth (int): The maximum depth of the tree.
                min_samples (int): The minimum number of samples in a node.
        """
        self.tree_depth = 0
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.root = None

    def Tree_Build(self, df, depth=0):
        """
            Builds the decision tree.
            Args:
                df (DataFrame): The dataset to be used for building the tree.
                depth (int): The depth of the current node.
        """
        label = df.columns[-1]
        _, train_lables = split_df_column(df)
        if len(train_lables) == 0:
            return None
        if ((depth == self.max_depth) or (len(train_lables) < self.min_samples) or (len(train_lables.unique()) == 1)):
            return self.create_leaf(train_lables)
        best_attribute = get_best_attribute(df)
        values = df[best_attribute].unique()
        if best_attribute == 'Age' or best_attribute == 'Family_Size' or best_attribute == 'Work_Experience':
            values = np.sort(values)
        node = Node(best_attribute, train_lables.value_counts().idxmax())
        node.values = values
        for value in values:
            subset = get_attribute_subset(df, best_attribute, value)
            _, cnt = np.unique(subset[label], return_counts=True)
            if len(cnt) == 1:
                node.children.append(self.create_leaf(subset[label]))
            else:
                node.children.append(self.Tree_Build(subset, depth+1))
        return node

    def train(self, train_df):
        """
            Trains the decision tree.
            Args:
                train_df (DataFrame): The dataset to be used for training the tree.
        """
        self.root = self.Tree_Build(train_df)

    def create_leaf(self, lables):
        """
            Creates a leaf node.
            Args:
                lables (Series): The lables of the leaf node.
        """
        if len(lables) > 0:
            prob_lable = lables.value_counts().idxmax()
        else:
            prob_lable = None
        return Node('Outcome', prob_lable)

    def get_row_prediction(self, test_dict, root):
        """
            Predicts the lable of a single test instance.
            Args:
                test_dict (dict): The test instance to be predicted.
                root (Node): The root of the tree.
        """
        if root is None:
            return None
        if root.is_leaf():
            return root.prob_lable
        if test_dict[root.attribute] in root.values:
            index = np.where(root.values == test_dict[root.attribute])[0][0]
            return self.get_row_prediction(test_dict, root.children[index])
        return root.prob_lable

    def predict(self, test_data):
        """
            Predicts the lables of the test instances.
            Args:
                test_data (DataFrame): The test instances to be predicted.
        """
        predicitions = pd.Series([self.get_row_prediction(
            row, self.root) for row in test_data.to_dict(orient='records')])
        return predicitions

    def Tree_Print(self, file):
        """
            Prints the tree to the given file.
            Args:
                file (str): The file to which the tree is to be printed.
        """
        uid = 1
        root = self.root
        root.id = 0
        queue = []
        queue.append(root)
        tree = graphviz.Digraph(
            filename=file, format='png', node_attr={'shape': 'box'})
        while len(queue) > 0:
            node = queue.pop(0)
            tree.node(str(node.id), label=node.StringFormat())
            for i, child in enumerate(node.children):
                if child is not None:
                    child.id = uid
                    uid += 1
                    queue.append(child)
                    tree.edge(str(node.id), str(child.id),
                              label=str(node.values[i]))
        tree.render(file)


def select_best_tree(train, test, max_depth=5, num_splits=2):
    best_tree = None
    best_valid = None
    average_train_accuracy = 0
    best_test_accuracy = 0
    best_train = None
    average_test_accuracy = 0
    for i in range(num_splits):
        train_df, valid = Shuffled_Split_Data(train, 0.8)
        tree = DecisionTree(max_depth=max_depth)
        tree.train(train_df)
        _, test_accuracy = get_prediction_accuracy(tree, test)
        _, train_accuracy = get_prediction_accuracy(tree, train_df)
        print(
            f'Split {i+1}: Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}', file=OutputFile)
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


def get_prediction_accuracy(tree, test_df):
    # get the prediction accuracy
    test_data, test_lables = split_df_column(test_df)
    predictions = tree.predict(test_data)
    accuracy = np.mean(predictions == test_lables)*100
    return (predictions, accuracy)


def save_plot(x, y, param):
    """
    Creates a plot of y v/s x and saves it. 
    Args:
        param (str): A parameter describing the type of the plot, 'depth' for the accuracy v/s depth 
                plot, or 'nodes' for the accuracy v/s  no. of nodes plot.
        y (List): The corresponding values on the vertical axis.
        x (List): The values on the horizontal axis.
    """
    label = ('Depth' if param == 'depth' else 'No. of Nodes')
    plt.title(f'Test Accuracy v/s {label}')
    plt.xlabel(label)
    plt.ylabel('Test Accuracy')
    plt.plot(x, y)
    plt.savefig(f'{param}_accuracy.png')


def Vary_Node_Depth(df):
    """
        Varies the depth of the tree and plots the accuracy v/s depth plot.
        Args:
            df (DataFrame): The dataset to be used for training the tree.
    """
    depths = []
    d_lim = 7
    max_accuracy = [0] * (d_lim+1)
    trees = [None] * (d_lim+1)
    accuracy = [0] * (d_lim+1)
    num_iterations = 5
    node_dict = dict()
    for iter in range(num_iterations):
        train, test = Shuffled_Split_Data(df, 0.8)
        print(f'Iteration {iter+1} :', file=OutputFile)
        for i in range(1, d_lim+1):
            tree = DecisionTree(max_depth=i)
            tree.train(train)
            _, test_accuracy = get_prediction_accuracy(tree, test)
            _, train_accuracy = get_prediction_accuracy(tree, train)
            if test_accuracy > max_accuracy[i]:
                trees[i] = tree
                max_accuracy[i] = test_accuracy
            count = tree.root.CountNodes()
            node_dict[count] = test_accuracy
            accuracy[i] += test_accuracy
    accuracy = [x/num_iterations for x in accuracy]
    for i in range(1, d_lim+1):
        print(f'Depth: {i}, Accuracy: {accuracy[i]}', file=OutputFile)
        depths.append(i)
    best_depth = accuracy.index(max(accuracy))

    trees[best_depth].Tree_Print('best_depth.gv')

    lists = sorted(node_dict.items())
    x, y = zip(*lists)
    save_plot(x, y, 'nodes')
    save_plot(depths, accuracy[1:], 'depth')
    print(f'Best Depth: {best_depth}', file=OutputFile)
    print(f'Best Accuracy: {max(accuracy)}', file=OutputFile)


def Calculate_Impurity_Measure(train, test, max_depth=5):
    """
        Calculates the impurity measure for the given dataset.
        Args:
            train (DataFrame): The training dataset.
            test (DataFrame): The test dataset.
            max_depth (int): The maximum depth of the tree.
    """
    # create a decision tree
    tree = DecisionTree(max_depth=max_depth)

    tree.train(train)
    _, train_accuracy = get_prediction_accuracy(tree, train)
    preds, test_accuracy = get_prediction_accuracy(tree, test)
    tree.Tree_Print('tree.gv')
    print('------------ RESULTS ------------', file=OutputFile)
    print('Train Accuracy: ', train_accuracy, file=OutputFile)
    print('Test Accuracy: ', test_accuracy, file=OutputFile)


OutputFile = open('outputQ1.txt', 'w') # Output file to store the results of the program.


def main():
    print('---------------------Running Program---------------------\n')
    print('--------------------------- DECISION TREE ------------------------------', file=OutputFile)
    df = pd.read_csv('Dataset_A.csv')  # read the data file
    print('\n-------------------------- LOADED DATA -------------------------------', file=OutputFile)
    print(df, file=OutputFile)  # print
    df.drop('ID', axis=1, inplace=True)  # drop the id column
    df['Age'] = df['Age'].apply(lambda x: x//10 + 1)
    df['Work_Experience'] = df['Work_Experience'].apply(
        lambda x: (x*5)//10 + 1)
    df['Age'].fillna(df['Age'].median(), inplace=True)

    df['Work_Experience'].fillna(df['Work_Experience'].mode()[0], inplace=True)
    df['Family_Size'].fillna(df['Family_Size'].mode()[0], inplace=True)
    df['Var_1'].fillna(df['Var_1'].mode()[0], inplace=True)
    df['Ever_Married'].fillna(df['Ever_Married'].mode()[0], inplace=True)
    df['Spending_Score'].fillna(df['Spending_Score'].mode()[0], inplace=True)
    df['Graduated'].fillna(df['Graduated'].mode()[0], inplace=True)
    df['Profession'].fillna(df['Profession'].mode()[0], inplace=True)
    df['Segmentation'].fillna(df['Segmentation'].mode()[0], inplace=True)

    train, test = Shuffled_Split_Data(df, 0.8)
    print('\n----------------------- Calculating Impurity Measure ----------------------', file=OutputFile)
    # create a decision tree
    Calculate_Impurity_Measure(train, test, max_depth=10)
    print('\n-------------- Determine Accuracy over 10 random splits --------------', file=OutputFile)
    best_train, best_tree, best_valid, average_train_accuracy, average_test_accuracy, best_test_accuracy = select_best_tree(
        train, test, 7, 10)
    print(f'Average Test Accuracy: {average_test_accuracy}', file=OutputFile)
    print(f'Best Test Accuracy: {best_test_accuracy}', file=OutputFile)

    print('\n-------------- Varying Depth --------------')
    Vary_Node_Depth(df)

    print('\n-------------- print before pruning --------------', file=OutputFile)
    train, test = Shuffled_Split_Data(df, 0.8)
    _, train_accuracy = get_prediction_accuracy(best_tree, test)
    best_tree.Tree_Print('before_pruning.gv')
    print("\n-------------printing tree before pruning------------")
    with open('D_Tree_before_pruning.txt', 'w') as f:
        best_tree.root.print_tree_dfs(f)
    _, valid_accuracy = get_prediction_accuracy(best_tree, best_valid)
    print(f'Best Accuracy before pruning: {valid_accuracy}', file=OutputFile)
    print('\n-------------- print after pruning --------------', file=OutputFile)
    best_tree.root.reduced_error_pruning(best_tree, valid_accuracy, best_valid)
    print("\n-------------printing tree after pruning------------")
    with open('D_Tree_after_pruning.txt', 'w') as f:
        best_tree.root.print_tree_dfs(f)
    _, valid_accuracy = get_prediction_accuracy(best_tree, best_valid)
    print(f'Best Accuracy after pruning: {valid_accuracy}', file=OutputFile)

    best_tree.Tree_Print('after_pruning.gv')
    print('---------------------Program Finished---------------------\n')


if __name__ == "__main__":
    main()

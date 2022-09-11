import pandas as pd
import graphviz
from utils import find_best_splitting, split_df_column, get_prediction_accuracy


class DecisionTree:
    pass


class Node:
    def __init__(self, attribute, split_value, prob_lable):
        self.attribute = attribute
        self.split_value = split_value
        self.prob_lable = prob_lable
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.right is None

    def node_count(self):
        if self.is_leaf():
            return 1
        if self.left is None:
            left_cnt = self.right.node_count()
        else:
            left_cnt = 0
        if self.right is None:
            right_cnt = self.left.node_count()
        else:
            right_cnt = 0
        return 1 + left_cnt + right_cnt

    def format_string(self):
        if self.is_leaf():
            outcome = self.prob_lable
            return f"{self.attribute}\n{outcome}"
        else:
            return f"{self.attribute}\n{self.split_value}"

    def prune(self, tree, accuracy, valid):
        new_accuracy = 0
        if self.left is None and self.right is None:
            return accuracy
        if self.left is not None:
            new_accuracy = self.left.prune(tree, accuracy, valid)
        if self.right is not None:
            new_accuracy = self.right.prune(tree, accuracy, valid)
        left, right = self.left, self.right
        self.left, self.right = None, None
        _, temp_accuracy = get_prediction_accuracy(tree, valid)
        if temp_accuracy < new_accuracy or tree.root.node_count() <= 5:
            self.left, self.right = left, right
        else:
            new_accuracy = temp_accuracy
            self.attribute = 'Outcome'
        return new_accuracy



class DecisionTree:
    def __init__(self, max_depth=10, min_samples=1):
        self.root = None
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree_depth = 0

    def build_tree(self, train_data, train_lables, depth=0):
        if len(train_lables) == 0:
            # print("gadbad HAI")
            return None
        if((depth == self.max_depth) or (len(train_lables) < self.min_samples) or (len(train_lables.unique()) == 1)):
            return self.create_leaf(train_lables)
        best_attribute, best_split_value = self.get_best_attribute(
            train_data, train_lables)
        # print(best_attribute, best_split_value)
        node = Node(best_attribute, best_split_value,
                    train_lables.value_counts().idxmax())
        if best_attribute == 'Age' or best_attribute == 'Family_Size' or best_attribute == 'Work_Experience':
            filtered = train_data[best_attribute] < best_split_value
        else:
            filtered = train_data[best_attribute] == best_split_value
        node.left = self.build_tree(
            train_data[filtered], train_lables[filtered], depth+1)

        node.right = self.build_tree(
            train_data[~filtered], train_lables[~filtered], depth+1)

        self.tree_depth = max(self.tree_depth, depth)
        return node

    def train(self, train_df):
        train_data, train_lables = split_df_column(train_df)
        self.root = self.build_tree(train_data, train_lables)

    def create_leaf(self, lables):
        # print(lables.head())
        if len(lables) > 0:
            prob_lable = lables.value_counts().idxmax()
        else:
            print("DIKKAT HAI")
            prob_lable = None
        return Node('Outcome', None, prob_lable)

    def predict_one(self, test_dict, root):
        if root is None:
            return None
        if root.is_leaf():
            return root.prob_lable
        # print(test_dict[root.attribute], root.split_value)
        if root.attribute == 'Age' or root.attribute == 'Family_Size' or root.attribute == 'Work_Experience':
            if test_dict[root.attribute] < root.split_value:
                return self.predict_one(test_dict, root.left)
            else:
                return self.predict_one(test_dict, root.right)
        else:
            if test_dict[root.attribute] == root.split_value:
                return self.predict_one(test_dict, root.left)
            else:
                return self.predict_one(test_dict, root.right)

    def predict(self, test_data):
        predicitions = pd.Series([self.predict_one(row, self.root)
                                 for row in test_data.to_dict(orient='records')])
        return predicitions

    def get_best_attribute(self, train_data, train_lables):
        attributes = train_data.columns
        max_gain = -10**18
        best_attribute = None
        best_split_value = None

        for attribute in attributes:
            gain, split_value = find_best_splitting(
                train_data, train_lables, attribute)
            if gain > max_gain:
                max_gain = gain
                best_attribute = attribute
                best_split_value = split_value
        return best_attribute, best_split_value

    def print_tree(self, file):
        tree = graphviz.Digraph(filename=file, format='png', node_attr={'shape': 'box'})
        root = self.root
        queue = []
        queue.append(root)
        root.id = 0
        tree.node(str(root.id), label=root.format_string())
        uid = 1
        edge_label = ['True', 'False']
        while len(queue) > 0:
            node = queue.pop(0)
            for i, child in enumerate([node.left, node.right]):
                if child is not None:
                    child.id = uid
                    uid += 1
                    queue.append(child)
                    tree.node(str(child.id), label=child.format_string())
                    tree.edge(str(node.id), str(child.id), label=edge_label[i])

        tree.render(file)

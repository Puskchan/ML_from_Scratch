import math
import numpy as np

class TreeNode(object):
    def __init__(self, parents=None):
        self.children = []
        self.split_feature = None
        self.split_feature_value = None
        self.parents = parents
        self.label = None
      
class DecisionTreeClassifier():
    def __init__(self, feature_proportion=.66):
        self.fit_tree = None
        self.feature_proportion = feature_proportion
    
    def data_to_distribution(self, y_train):
            types = set(y_train)
            distribution = []
            for i in types:
                distribution.append(list(y_train).count(i)/float(len(y_train)))
            return distribution


    def entropy(self, distribution):
        return -sum([p * math.log(p,2) for p in distribution])


    def split_data(self, x_train, y_train, feature_index):
        attribute_values = x_train[:,feature_index]
        for attribute in set(attribute_values):
            data_subset = []
            for index, point in enumerate(x_train):
                if point[feature_index] == attribute:
                    data_subset.append([point, y_train[index]])
            yield data_subset


    def gain(self, x_train, y_train, feature_index):
        entropy_gain = self.entropy(self.data_to_distribution(y_train))
        for data_subset in self.split_data(x_train, y_train, feature_index):
            entropy_gain -= \
                self.entropy(self.data_to_distribution([label for (_, label) in data_subset]))
        return entropy_gain


    def homogeneous(self, y_train):
        return len(set(y_train)) <= 1

    def majority_vote(self, y_train, node):
        labels = y_train
        choice = max(set(labels), key=list(labels).count)
        node.label = choice
        return node


    def build_decision_tree(self, x_train, y_train, root, remaining_features):
        remaining_features = np.array(list(remaining_features))
        if self.homogeneous(y_train):
            root.label = y_train[0]
            return root

        if len(remaining_features) == 0:
            return self.majority_vote(y_train, root)

        indices = np.random.choice(int(remaining_features.shape[0]),
                   int(self.feature_proportion * remaining_features.shape[0]), replace = False)

        best_feature = max(remaining_features[indices], key=lambda index: 
                           self.gain(x_train, y_train, index))
        remaining_features = set(remaining_features)

        if self.gain(x_train, y_train, best_feature) == 0:
            return self.majority_vote(y_train, root)

        root.split_feature = best_feature

        for data_subset in self.split_data(x_train, y_train, best_feature):
            child = TreeNode(parents = root)
            child.split_feature_value = data_subset[0][0][best_feature]
            root.children.append(child)

            new_x = np.array([point for (point, label) in data_subset])
            new_y = np.array([label for (point, label) in data_subset])

            self.build_decision_tree(new_x, new_y, child, remaining_features - set([best_feature]))

        return root
    
    def find_nearest(self, array, value):
        nearest = (np.abs(array-value)).argmin()
        return array[nearest]


    def classify(self, tree, point):
        if tree.children == []:
            return tree.label
        else:
            try:
                matching_children = [child for child in tree.children
                    if child.split_feature_value == point[tree.split_feature]]
                return self.classify(matching_children[0], point)
            except:
                array = [child.split_feature_value for child in tree.children]
                point[tree.split_feature] = self.find_nearest(array, point[tree.split_feature])
                matching_children = [child for child in tree.children
                    if child.split_feature_value == point[tree.split_feature]]
                return self.classify(matching_children[0], point)


    def text_classification(self, x_train, tree):
        predicted_labels = [self.classify(tree, point) for point in x_train]
        return predicted_labels

    def fit(self, x_train, y_train):
        tree = self.build_decision_tree(x_train, y_train, TreeNode(), 
                                   set(range(len(x_train[0]))))
        self.fit_tree = tree
        
    def predict(self, x_test):
        predictions = self.text_classification(x_test, self.fit_tree)
        return predictions
import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None,
                 left=None, right=None, info_gain=None, value=None):

        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value


##to reuse this decision tree classifier in random forest, we add max_feature subsampling in the tree

class DecisionTreeClassifier:

    def __init__(self, min_samples_split=2, max_depth=2,criterion="gini",max_features = None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion.lower()
        self.is_trained =False
        self.max_features = max_features

    ##building the tree
    def _build_tree(self, X, y, curr_depth):

        num_samples, num_features = X.shape

        # stopping conditions - total-samples less than min_samples, depth more than max depth or pure node 
        if (
            num_samples < self.min_samples_split
            or curr_depth >= self.max_depth
            or len(np.unique(y)) == 1
        ):
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        best_split = self._get_best_split(X, y, num_samples, num_features)

        if best_split["info_gain"] <= 0:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        left_subtree = self._build_tree(
            best_split["X_left"], best_split["y_left"], curr_depth + 1
        )
        right_subtree = self._build_tree(
            best_split["X_right"], best_split["y_right"], curr_depth + 1
        )

        return Node(
            feature_index=best_split["feature_index"],
            threshold=best_split["threshold"],
            left=left_subtree,
            right=right_subtree,
            info_gain=best_split["info_gain"],
        )

    
    ## here we implement the max feature subsampling 
    def _get_best_split(self, X, y, num_samples, num_features):

        best_split = {}
        max_info_gain = -float("inf")

        feature_indices = np.arange(num_features)
        if self.max_features is not None:
            feature_indices = np.random.choice(num_features,self.max_features,replace=False)

        for feature_index in feature_indices:
            feature_values = X[:, feature_index]
            values = np.sort(np.unique(feature_values))
            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:

                X_left, y_left, X_right, y_right = self._split(
                    X, y, feature_index, threshold
                )

                if len(y_left) > 0 and len(y_right) > 0:

                    curr_info_gain = self._information_gain(y, y_left, y_right)

                    if curr_info_gain > max_info_gain:
                        best_split = {
                            "feature_index": feature_index,
                            "threshold": threshold,
                            "X_left": X_left,
                            "y_left": y_left,
                            "X_right": X_right,
                            "y_right": y_right,
                            "info_gain": curr_info_gain,
                        }
                        max_info_gain = curr_info_gain

        if len(best_split) == 0:
            best_split["info_gain"] = -1

        return best_split

    # --------------------------------------------------
    # SPLIT DATA
    # --------------------------------------------------

    def _split(self, X, y, feature_index, threshold):

        left_idx = X[:, feature_index] <= threshold
        right_idx = X[:, feature_index] > threshold

        return X[left_idx], y[left_idx], X[right_idx], y[right_idx]

    # --------------------------------------------------
    # INFORMATION GAIN
    # --------------------------------------------------

    def _information_gain(self, parent, l_child, r_child):

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)

        if self.criterion == "gini":
            gain = self._gini(parent) - (
                weight_l * self._gini(l_child) + weight_r * self._gini(r_child)
            )
        else:
            gain = self._entropy(parent) - (
                weight_l * self._entropy(l_child) + weight_r * self._entropy(r_child)
            )

        return gain

    # --------------------------------------------------
    # IMPURITY METRICS
    # --------------------------------------------------

    def _gini(self, y):

        classes = np.unique(y)
        gini = 0.0

        for cls in classes:
            p = np.sum(y == cls) / len(y)
            gini += p ** 2

        return 1 - gini

    def _entropy(self, y):

        classes = np.unique(y)
        entropy = 0.0

        for cls in classes:
            p = np.sum(y == cls) / len(y)
            entropy += -p * np.log2(p + 1e-9)

        return entropy

    # --------------------------------------------------
    # LEAF VALUE
    # --------------------------------------------------

    def _calculate_leaf_value(self, y):
        return np.bincount(y).argmax()

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        self.root = self._build_tree(X, y, curr_depth=0)
        self.is_trained = True
        return self

    def predict(self, X):

        X = np.asarray(X)
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, x, tree):

        if tree.value is not None:
            return tree.value

        if x[tree.feature_index] <= tree.threshold:
            return self._predict_sample(x, tree.left)
        else:
            return self._predict_sample(x, tree.right)

    # --------------------------------------------------
    # PRINT TREE
    # --------------------------------------------------

    def print_tree(self, tree:Node=None, indent="  "):

        if tree is None:
            tree = self.root

        if tree.value is not None:
            print(tree.value)
        else:
            print(
                f"X_{tree.feature_index} <= {tree.threshold} "
                f"(gain={tree.info_gain:.4f})"
            )
            print(f"{indent}left:", end="")
            self.print_tree(tree.left, indent + indent)
            print(f"{indent}right:", end="")
            self.print_tree(tree.right, indent + indent)

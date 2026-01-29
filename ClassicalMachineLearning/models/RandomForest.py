import numpy as np
from models.DecisionTree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self,n_trees=10,max_depth=6,min_samples_split=2,max_features=None,criterion='gini',verbose=False):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.is_trained = False
        self.verbose=verbose
    
    def fit(self,X,y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]

        if self.max_features is None: ## if user doesn't specify it we use the heuristics used in sklearn
            self.max_features = int(np.sqrt(n_features))

        self.trees = []

        for idx in range(self.n_trees):
            X_bts,y_bts = self._bootstrap(X,y) ##heee hee hee

            tree = DecisionTreeClassifier(
                min_samples_split= self.min_samples_split,
                max_depth= self.max_depth,
                criterion=self.criterion,
                max_features=self.max_features
            )

            tree.fit(X_bts,y_bts)

            self.trees.append(tree)

            if self.verbose and idx%(self.n_trees//10) == 0:
                print('Tree-{idx} trained'.format(idx=idx))

        self.is_trained = True
        return self

    def _bootstrap(self,X:np.ndarray,y:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        n = X.shape[0]
        idx = np.random.choice(n,n,replace=True)
        return X[idx],y[idx]

    def predict(self,X):

        assert self.is_trained, "Train model first."

        X = np.asarray(X)

        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # shape: (n_trees, n_samples)

        final_preds = []

        for i in range(X.shape[0]):
            vals, counts = np.unique(tree_preds[:, i], return_counts=True)
            final_preds.append(vals[np.argmax(counts)])

        return np.array(final_preds)
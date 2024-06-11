import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
import time

class SimpleDecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree_ = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if len(set(map(tuple, y))) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return np.mean(y, axis=0)
        feature, threshold = self._best_split(X, y)
        if feature is None or threshold is None:  
            return np.mean(y, axis=0)
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return (feature, threshold, left_tree, right_tree)

    def _best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_loss = float('inf')
        loop_counter = 0
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                loop_counter += 1
                if loop_counter % 1000 == 0:
                    print(f'Loop iteration: {loop_counter}')
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold
                if len(left_mask) == 0 or len(right_mask) == 0:
                    continue
                loss = self._compute_loss(y[left_mask], y[right_mask])
                if loss < best_loss:
                    best_loss = loss
                    best_feature, best_threshold = feature, threshold
        return best_feature, best_threshold

    def _compute_loss(self, left_y, right_y):
        if len(left_y) == 0 or len(right_y) == 0:
            return float('inf')
        left_loss = np.var(left_y, axis=0).sum() * len(left_y)
        right_loss = np.var(right_y, axis=0).sum() * len(right_y)
        return left_loss + right_loss

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def _predict_one(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feature, threshold, left_tree, right_tree = tree
        if x[feature] <= threshold:
            return self._predict_one(x, left_tree)
        else:
            return self._predict_one(x, right_tree)

class SimpleRandomForest:
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        for i in range(self.n_estimators):
            if i % 1000 == 0:
                print(f'Fitting estimator number: {i}')
            bootstrap_X, bootstrap_y = self._bootstrap_sample(X, y)
            tree = SimpleDecisionTree(max_depth=self.max_depth)
            selected_features = self._select_features(X.shape[1])
            tree.fit(bootstrap_X[:, selected_features], bootstrap_y)
            self.trees.append((tree, selected_features))

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def _select_features(self, n_features):
        if self.max_features == 'sqrt':
            n_selected = int(np.sqrt(n_features))
        else:
            n_selected = n_features
        return np.random.choice(n_features, n_selected, replace=False)

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X[:, features]) for tree, features in self.trees])
        return np.mean(tree_predictions, axis=0)

def main():
    data = pd.read_csv("database.csv")

    print("start dataframe")
    data = pd.DataFrame(data, columns=['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude'])
    timestamp = []
    for d, t in zip(data['Date'], data['Time']):
      try:
          ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')
          timestamp.append(time.mktime(ts.timetuple()))
      except ValueError:
          timestamp.append('ValueError')
    timeStamp = pd.Series(timestamp)
    data['Timestamp'] = timeStamp.values
    data = data[data.Timestamp != 'ValueError']

    print("start numpy")
    X = data[['Timestamp', 'Latitude', 'Longitude']].to_numpy()
    y = data[['Depth', 'Magnitude']].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("start decision tree")
    decision_tree = SimpleDecisionTree(max_depth=8)
    decision_tree.fit(X_train, y_train)
    dt_predictions = decision_tree.predict(X_test)
    dt_mse_depth = mean_squared_error(y_test[:, 0], dt_predictions[:, 0])
    dt_r2_depth = r2_score(y_test[:, 0], dt_predictions[:, 0])
    dt_mse_magnitude = mean_squared_error(y_test[:, 1], dt_predictions[:, 1])
    dt_r2_magnitude = r2_score(y_test[:, 1], dt_predictions[:, 1])

    logger.info(f'Decision Tree - Depth MSE: {dt_mse_depth}, Depth R²: {dt_r2_depth}')
    logger.info(f'Decision Tree - Magnitude MSE: {dt_mse_magnitude}, Magnitude R²: {dt_r2_magnitude}')

    print("start random forest")
    random_forest = SimpleRandomForest(n_estimators=8, max_depth=8)
    random_forest.fit(X_train, y_train)
    rf_predictions = random_forest.predict(X_test)
    rf_mse_depth = mean_squared_error(y_test[:, 0], rf_predictions[:, 0])
    rf_r2_depth = r2_score(y_test[:, 0], rf_predictions[:, 0])
    rf_mse_magnitude = mean_squared_error(y_test[:, 1], rf_predictions[:, 1])
    rf_r2_magnitude = r2_score(y_test[:, 1], rf_predictions[:, 1])

    logger.info(f'Random Forest - Depth MSE: {rf_mse_depth}, Depth R²: {rf_r2_depth}')
    logger.info(f'Random Forest - Magnitude MSE: {rf_mse_magnitude}, Magnitude R²: {rf_r2_magnitude}')

    print(f'Decision Tree - Depth MSE: {dt_mse_depth}, Depth R²: {dt_r2_depth}')
    print(f'Decision Tree - Magnitude MSE: {dt_mse_magnitude}, Magnitude R²: {dt_r2_magnitude}')
    print(f'Random Forest - Depth MSE: {rf_mse_depth}, Depth R²: {rf_r2_depth}')
    print(f'Random Forest - Magnitude MSE: {rf_mse_magnitude}, Magnitude R²: {rf_r2_magnitude}')

    plt.figure(figsize=(14, 7))

    plt.subplot(2, 2, 1)
    plt.scatter(y_test[:, 0], dt_predictions[:, 0], color='blue', label='Predictions')
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'k--', lw=2)
    plt.xlabel('Actual Depth')
    plt.ylabel('Predicted Depth')
    plt.title('Decision Tree - Depth Predictions')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.scatter(y_test[:, 1], dt_predictions[:, 1], color='blue', label='Predictions')
    plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], [y_test[:, 1].min(), y_test[:, 1].max()], 'k--', lw=2)
    plt.xlabel('Actual Magnitude')
    plt.ylabel('Predicted Magnitude')
    plt.title('Decision Tree - Magnitude Predictions')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.scatter(y_test[:, 0], rf_predictions[:, 0], color='green', label='Predictions')
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'k--', lw=2)
    plt.xlabel('Actual Depth')
    plt.ylabel('Predicted Depth')
    plt.title('Random Forest - Depth Predictions')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.scatter(y_test[:, 1], rf_predictions[:, 1], color='green', label='Predictions')
    plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], [y_test[:, 1].min(), y_test[:, 1].max()], 'k--', lw=2)
    plt.xlabel('Actual Magnitude')
    plt.ylabel('Predicted Magnitude')
    plt.title('Random Forest - Magnitude Predictions')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    main()

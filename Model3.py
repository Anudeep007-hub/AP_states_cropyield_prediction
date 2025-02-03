import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Enhanced Decision Tree Regressor
class EnhancedDecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        np.random.seed(7)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return np.mean(y)

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return np.mean(y)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices

        if not np.any(left_indices) or not np.any(right_indices):
            return np.mean(y)

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_tree,
            "right": right_tree,
        }

    def _find_best_split(self, X, y):
        best_feature, best_threshold, best_mse = None, None, float("inf")
        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices

                if not np.any(left_indices) or not np.any(right_indices):
                    continue

                mse = self._mse_split(y[left_indices], y[right_indices])
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _mse_split(self, left_y, right_y):
        def mse(y):
            if len(y) == 0:
                return 0
            mean_y = np.mean(y)
            return np.mean((y - mean_y) ** 2)

        n = len(left_y) + len(right_y)
        return (len(left_y) / n) * mse(left_y) + (len(right_y) / n) * mse(right_y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node is None:
            return None
        if not isinstance(node, dict):
            return node
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        else:
            return self._traverse_tree(x, node["right"])


# Enhanced Random Forest Regressor
class EnhancedRandomForestRegressor:
    def __init__(self, n_trees=50, max_depth=10, min_samples_split=5, max_features='sqrt', bootstrap_ratio=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap_ratio = bootstrap_ratio
        self.trees = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_features = X.shape[1]
        n_samples = X.shape[0]

        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif isinstance(self.max_features, float):
            max_features = int(self.max_features * n_features)
        else:
            max_features = min(self.max_features, n_features)

        self.trees = []
        for _ in range(self.n_trees):
            sample_indices = np.random.choice(
                n_samples, int(n_samples * self.bootstrap_ratio), replace=True
            )
            X_sample, y_sample = X[sample_indices], y[sample_indices]

            feature_indices = np.random.choice(
                n_features, max_features, replace=False
            )

            tree = EnhancedDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sample[:, feature_indices], y_sample)

            self.trees.append((tree, feature_indices))

    def predict(self, X):
        X = np.array(X)
        predictions = np.array([tree.predict(X[:, features]) for tree, features in self.trees])
        return np.mean(predictions, axis=0)


# Preprocessing Function
def preprocess_data(data):
    data.rename(
        columns={
            'Area(Hectare)': 'Area',
            'Yield(Tonne/Hectare)': 'Yield',
            'ANNUAL_RAINFALL(Millimeters)': 'Annual_Rainfall',
            'Fertilizer(KG_per_hectare)': 'Fertilizer',
        },
        inplace=True,
    )

    label_encoders = {}
    for col in ['Crop', 'District', 'Season']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)

    data['Rainfall_Fertilizer'] = data['Annual_Rainfall'] * data['Fertilizer']

    return data, label_encoders


# Plotting Metrics
def plot_metrics(y_test, y_pred):
    residuals = y_test - y_pred

    plt.figure(figsize=(12, 6))

    # Plot Predicted vs Actual
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title("Predicted vs Actual")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")

    # Plot Residuals
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, color='purple', edgecolor='black', alpha=0.7)
    plt.title("Residuals Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# Main function
def main():
    data = pd.read_csv("final_data.csv")
    processed_data, label_encoders = preprocess_data(data)

    X = processed_data.drop(columns=['Yield', 'Year'])
    y = processed_data['Yield'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    while True:
        print("\nSelect an option:")
        print("1. Train a new model")
        print("2. Evaluate existing model")
        print("3. Predict crop yield")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            rf = EnhancedRandomForestRegressor()
            rf.fit(X_train, y_train)
            with open('enhanced_random_forest_regressor.pkl', 'wb') as model_file:
                pickle.dump(rf, model_file)
            print("Model trained and saved successfully!")

        elif choice == '2':
            try:
                with open('enhanced_random_forest_regressor.pkl', 'rb') as model_file:
                    loaded_rf = pickle.load(model_file)

                y_pred = loaded_rf.predict(X_test)

                print("\nModel Evaluation Metrics:")
                print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")
                print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
                print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
                print(f"R^2 Score: {r2_score(y_test, y_pred):.4f}")

                # Plot metrics
                print("\nPlotting evaluation metrics...")
                plot_metrics(y_test, y_pred)

            except FileNotFoundError:
                print("Model not found! Please train the model first.")

        elif choice == '3':
            pass  # Your existing code for prediction

        elif choice == '4':
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()

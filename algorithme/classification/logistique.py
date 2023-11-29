import numpy as np
from scipy.special import expit  # Sigmoid function


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the logistic regression model.

        Parameters:
        - learning_rate: The learning rate for gradient descent.
        - num_iterations: The number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the logistic regression model to the training data.

        Parameters:
        - X: The input features.
        - y: The true labels.

        Updates the weights and bias using gradient descent.
        """
        # Add a bias term to the input features
        X = np.c_[np.ones((X.shape[0], 1)), X]

        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.num_iterations):
            dw, db = self.gradient_of_loss(X, y)
            # Update weights and bias using gradient descent
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Make predictions using the trained logistic regression model.

        Parameters:
        - X: The input features.

        Returns:
        - predictions: Binary predictions (0 or 1).
        """
        # Add a bias term to the input features
        X = np.c_[np.ones((X.shape[0], 1)), X]

        # Compute the predicted probabilities
        y_pred = self._sigmoid(np.dot(X, self.weights) + self.bias)

        # Convert probabilities to binary predictions
        predictions = np.where(y_pred >= 0.5, 1, 0)

        return predictions

    def _sigmoid(self, z):
        """
        Sigmoid activation function.

        Parameters:
        - z: The input.

        Returns:
        - sigmoid(z): The sigmoid of the input.
        """
        return expit(z)

    def gradient_of_loss(self, X, y):
        """
        Compute the gradient of the logistic loss function with respect to weights and bias.

        Parameters:
        - X: The input features.
        - y: The true labels.

        Returns:
        - dw: The gradient of the loss with respect to weights.
        - db: The gradient of the loss with respect to bias.
        """

        # Compute the predicted probabilities
        y_pred = self._sigmoid(np.dot(X, self.weights) + self.bias)

        # Compute the gradient of the loss with respect to weights and bias
        dw = (1 / X.shape[0]) * np.dot(X.T, (y_pred - y))
        db = (1 / X.shape[0]) * np.sum(y_pred - y)

        return dw, db

    def score(self, X, y):
        """
        Compute the accuracy score of the model on the given data.

        Parameters:
        - X: The input features.
        - y: The true labels.

        Returns:
        - accuracy: The accuracy score.
        """
        # Compute the accuracy score of the model
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


if __name__ == "__main__":
    # Example usage:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression as logistiqueregression

    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, (iris.target == 0).astype(int)  # Binary classification for simplicity

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and fit the logistic regression model
    model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    model.fit(X_train, y_train)
    # Make predictions on the test set
    predictions = model.predict(X_test)

    model_ = logistiqueregression()
    model_.fit(X_train, y_train)
    # Compute the accuracy score
    accuracy = model.score(X_test, y_test)
    accuracy_ = model_.score(X_test, y_test)
    print(f"Accuracy for our model: {accuracy}")
    print(f"Accuracy for sklearn: {accuracy_}")
import numpy as np


class Knn:

    def __init__(self, x_train, y_train, k):

        self._X = x_train
        self._y = y_train
        self._k = k

    def euclidean_distance(self, x1, x2):
        """
        Calculate the Euclidean distance between two vectors.

        Parameters:
        - x1, x2 : Vectors between which the distance is calculated.

        Returns:
        - float : The Euclidean distance between x1 and x2.
        """
        return np.sqrt(np.sum((x1-x2)**2))

    def classify(self, xi):
        """
        Classify an example using the k-Nearest Neighbors algorithm.

        Parameters:
        - X_train : Matrix of training examples.
        - y_train : Vector of labels corresponding to the training examples.
        - example : Example to be classified.
        - k : Number of neighbors to consider.

        Returns:
        - The predicted class for the example.
        """
        distances = [self.euclidean_distance(xi, x) for x in self._X]
        nearest_indices = np.argsort(distances)[:self._k]
        nearest_labels = self._y[nearest_indices]
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        return predicted_label

    def predict(self, x):
        """
        Predict labels for multiple examples using the k-Nearest Neighbors algorithm.

        Parameters:
        - X_train : Matrix of training examples.
        - y_train : Vector of labels corresponding to the training examples.
        - X_test : Matrix of test examples.
        - k : Number of neighbors to consider.

        Returns:
        - np.array : Array of predicted labels for each example in X_test.
        """
        predictions = [self.classify(example) for example in X_test]
        return np.array(predictions)

    def accuracy(self, y_true, y_pred):
        """
        Calculate the accuracy of the model.

        Parameters:
        - y_true : Vector of true labels.
        - y_pred : Vector of predicted labels.

        Returns:
        - float : The accuracy of the model.
        """
        correct_predictions = np.sum(y_true == y_pred)
        total_examples = len(y_true)
        accuracy = correct_predictions / total_examples
        return accuracy


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Choose the value of k
    k = 3

    # Using scikit-learn's KNeighborsClassifier

    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    # Predict labels for the test set
    y_pred = knn_classifier.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    model = Knn(X_train, y_train, k)
    y_ = model.predict(y_test)
    accuracy_knn = model.accuracy(y_test, y_)
    print(f"Accuracy for library: {accuracy * 100:.2f}%")
    print(f"Accuracy for our model: {accuracy_knn * 100:.2f}%")

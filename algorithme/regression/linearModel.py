import numpy as np


class LinearRegressionCloseForm:
    def __init__(self):
        """
        Initialize a SimpleLinearRegression object.
        """
        self.coefficients_ = None

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data using the closed-form solution.

        Parameters:
        - X : array-like, shape (n_samples, n_features)
            Training data.
        - y : array-like, shape (n_samples,)
            Target values.

        Returns:
        - None
        """
        # Add a column of ones to X for the intercept term
        X_ = np.c_[np.ones(X.shape[0]), X]
        # Compute the closed-form solution
        X_inverse = np.linalg.inv(X_.T @ X_)
        self.coefficients_ = X_inverse @ X_.T @ y

    def predict(self, X):
        """
        Make predictions for new data.

        Parameters:
        - X : array-like, shape (n_samples, n_features)
            Data for which to make predictions.

        Returns:
        - array-like, shape (n_samples,)
            Predicted values.
        """
        if self.coefficients_ is None:
            raise ValueError("Model has not been fitted. Call the fit() method first.")

        # Add a column of ones to X for the intercept term
        X_ = np.c_[np.ones(X.shape[0]), X]

        return X_ @ self.coefficients_

    def score(self, X, y):
        """
        Compute the coefficient of determination R^2 of the prediction.

        Parameters:
        - X : array-like, shape (n_samples, n_features)
            Test samples.
        - y : array-like, shape (n_samples,)
            True values.

        Returns:
        - float
            R^2 score.
        """
        if self.coefficients_ is None:
            raise ValueError("Model has not been fitted. Call the fit() method first.")

        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum((y - y_pred)**2)

        r_squared = 1 - (ss_residual / ss_total)
        return r_squared


if __name__ == "__main__":

    from sklearn.datasets import  fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np

    # Load the Boston Housing dataset
    california = fetch_california_housing()
    X = california.data
    y = california.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Using scikit-learn's LinearRegression
    linear_regression_model = LinearRegression()
    linear_regression_model.fit(X_train, y_train)
    # Make predictions for the test set
    y_pred = linear_regression_model.predict(X_test)

    # Compute the R^2 score
    r_squared = r2_score(y_test, y_pred)
    print(f"R^2 Score: {r_squared}")
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    linear_regressionCloseForm_model = LinearRegressionCloseForm()
    linear_regressionCloseForm_model.fit(X_train, y_train)

    # Make predictions for the test data
    y_closeForm = linear_regressionCloseForm_model.predict(X_test)

    # Compute the R^2 score
    r_squared_ = r2_score(y_test, y_closeForm)
    print(f"R^2 Score: {r_squared_}")
    mse_ = mean_squared_error(y_test, y_closeForm)
    print(f"Mean Squared Error: {mse_}")



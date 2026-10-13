import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None

    def calculate_gradient(self, theta, X, y):
        m = y.size

        predictions = X @ theta
        error = predictions - y
        gradient = (X.T @ error) / m
        return gradient


    def fit(self, X, y):

        # Add bias/intercept column
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Initialize theta
        self.theta = np.zeros(X_b.shape[1])

        # Gradient descent
        for i in range(self.n_iterations):

            gradient = self.calculate_gradient(
                self.theta,
                X_b,
                y
            )

            self.theta -= self.learning_rate * gradient

        return self

    def predict(self, X):

        # Add bias/intercept column
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        return X_b @ self.theta


    def save_model(self, mean, std, filename='./csv/model.csv'):

        with open(filename, 'w') as f:

            f.write('theta00,theta01,mean,std\n')
            f.write(f'{self.theta[0]},{self.theta[1]},{mean},{std}')
            f.write('\n')
        return self

    def show_plot(self, X, y, predict):
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='steelblue', label='Data')
        plt.plot(X, predict, color='red', linewidth=2, label='Regression line')
        plt.xlabel('Mileage (km)')
        plt.ylabel('Price')
        plt.title('Mileage vs Price — Linear Regression')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
        plt.close()
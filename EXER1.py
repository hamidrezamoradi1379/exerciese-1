import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

sample_sizes = [5, 25, 100, 1000]
noise_std = 0.2
degrees = [1, 4, 16]

def generate_data(num_samples, slope, intercept, noise_std=0):
    X = np.random.rand(num_samples, 1)
    y = slope * X + intercept + np.random.normal(0, noise_std, size=(num_samples, 1))
    return X, y

def plot_results(X, y_true, y_pred, title):
    plt.scatter(X, y_true, color='black', label='Actual Data')
    plt.plot(X, y_pred, color='red', linewidth=3, label='Regression Line')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    
datasets = []
for size in sample_sizes:
    X, y = generate_data(size, slope=2, intercept=1, noise_std=noise_std)
    datasets.append((X, y))
    
for degree in degrees:
    for X, y in datasets:
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)

        title = f'Degree {degree} Regression - {len(X)} Samples'
        plot_results(X, y, y_pred, title)
plt.figure()

for degree in degrees:
    accuracies = []
    for X, y in datasets:
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)

        accuracy = mean_squared_error(y, y_pred)
        accuracies.append(accuracy)

    plt.plot(sample_sizes, accuracies, label=f'Degree {degree}')

plt.title('Model Accuracy vs. Sample Size')
plt.xlabel('Sample Size')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
        

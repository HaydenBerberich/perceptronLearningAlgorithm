import numpy as np

# Generate random data
def generate_data(n_points, linearly_separable=True):
    np.random.seed(42)
    X = np.random.randn(n_points, 2)
    if linearly_separable:
        y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
    else:
        y = np.where(X[:, 0]**2 + X[:, 1]**2 > 1, 1, -1)
    return X, y

# Augment data with bias term
def augment_data(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))

# Perceptron Learning Algorithm
def perceptron_learning_algorithm(X, y, eta=1.0, max_passes=1000):
    w = np.random.randn(X.shape[1])
    initial_w = w.copy()
    updates = 0
    for iteration in range(max_passes):
        updated = False
        for xi, yi in zip(X, y):
            if yi * np.dot(w, xi) <= 0:
                w += eta * yi * xi
                updates += 1
                updated = True
        if not updated:
            break
    return w, initial_w, updates, iteration + 1

# Misclassification error
def misclassification_error(X, y, w):
    predictions = np.sign(np.dot(X, w))
    errors = predictions != y
    return np.mean(errors) * 100

# Generate training and test data
X_train_ls, y_train_ls = generate_data(30, linearly_separable=True)
X_train_nls, y_train_nls = generate_data(30, linearly_separable=False)
X_test, y_test = generate_data(10, linearly_separable=True)

# Augment data
X_train_ls_aug = augment_data(X_train_ls)
X_train_nls_aug = augment_data(X_train_nls)
X_test_aug = augment_data(X_test)

# Train the perceptron
w_ls, initial_w_ls, updates_ls, iterations_ls = perceptron_learning_algorithm(X_train_ls_aug, y_train_ls)
w_nls, initial_w_nls, updates_nls, iterations_nls = perceptron_learning_algorithm(X_train_nls_aug, y_train_nls)

# Misclassification errors
train_error_ls = misclassification_error(X_train_ls_aug, y_train_ls, w_ls)
test_error_ls = misclassification_error(X_test_aug, y_test, w_ls)
train_error_nls = misclassification_error(X_train_nls_aug, y_train_nls, w_nls)
test_error_nls = misclassification_error(X_test_aug, y_test, w_nls)

# Print the results
print("Linearly Separable Training Data:")
print(f"Initial Weights: {initial_w_ls}")
print(f"Final Weights: {w_ls}")
print(f"Total Updates: {updates_ls}")
print(f"Total Iterations: {iterations_ls}")
print(f"Training Error: {train_error_ls}%")
print(f"Test Error: {test_error_ls}%")

print("\nNon-Linearly Separable Training Data:")
print(f"Initial Weights: {initial_w_nls}")
print(f"Final Weights: {w_nls}")
print(f"Total Updates: {updates_nls}")
print(f"Total Iterations: {iterations_nls}")
print(f"Training Error: {train_error_nls}%")
print(f"Test Error: {test_error_nls}%")

# Print the training and test data
print("\nLinearly Separable Training Data (Features):")
print(X_train_ls)
print("Linearly Separable Training Data (Labels):")
print(y_train_ls)

print("\nNon-Linearly Separable Training Data (Features):")
print(X_train_nls)
print("Non-Linearly Separable Training Data (Labels):")
print(y_train_nls)

print("\nTest Data (Features):")
print(X_test)
print("Test Data (Labels):")
print(y_test)
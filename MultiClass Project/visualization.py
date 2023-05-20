import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(X, y, W, b, prediction_function, additional_args = [], h = 0.01):

    """
    Plots the decision boundaries for a One-vs-All (OvA) classifier.

    Args:
        X: The input data, shape (n_samples, n_features).
        y: The target values, shape (n_samples,).
        W_ova: The weights for each classifier, shape (n_classes, n_features).
        b_ova: The bias for each classifier, shape (n_classes,).
    """


    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    X_pred = np.c_[xx.ravel(), yy.ravel()]
    
    args = [X_pred, W, b]

    if len(additional_args) > 0:
        args.extend(additional_args)
        

    # apply predict_one_versus_one to the mesh and color the points accordingly
    
    Z = prediction_function(*args)
    Z = Z.reshape(xx.shape)

    # plot the decision boundaries
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k', cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

import numpy as np
from tqdm import tqdm

def svm_with_smo(X, y, C, eps, max_iter, kernel):
    n_samples, n_features = X.shape

    # Initialize alpha to 0
    alpha = np.zeros(n_samples)

    # Initialize b to 0
    b = 0

    # Pre-compute the kernel matrix
    K = np.array([[kernel(X[i], X[j]) for j in range(n_samples)] for i in range(n_samples)])

    # Start iterations
    k = 0
    while k < max_iter:
        num_alpha_changed = 0
        for i in range(n_samples):
            Ei = b + np.sum(alpha * y * K[i]) - y[i]
            if (y[i] * Ei < -eps and alpha[i] < C) or (y[i] * Ei > eps and alpha[i] > 0):
                j = np.random.randint(0, n_samples)
                while j == i:
                    j = np.random.randint(0, n_samples)

                Ej = b + np.sum(alpha * y * K[j]) - y[j]
                alpha_old_i = alpha[i]
                alpha_old_j = alpha[j]

                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])

                if L == H:
                    continue

                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                alpha[j] -= y[j] * (Ei - Ej) / eta
                alpha[j] = np.clip(alpha[j], L, H)

                if abs(alpha[j] - alpha_old_j) < 0.00001:
                    continue

                alpha[i] += y[i] * y[j] * (alpha_old_j - alpha[j])

                b1_new_term = y[i] * (alpha[i] - alpha_old_i) * K[i, i]
                b2_new_term = y[j] * (alpha[j] - alpha_old_j) * K[j, j]
                b1 = b - Ei - b1_new_term - b2_new_term
                b2 = b - Ej - b1_new_term - b2_new_term

                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                num_alpha_changed += 1

        if num_alpha_changed == 0:
            k += 1
        else:
            k = 0

    # Compute the weights
    w = np.sum(alpha * y * X.T, axis=1)

    return w, b

def one_versus_all(X, y, C, eps, kernel, max_iter):
    """
    Implementation of the one-versus-all strategy for multi-class classification.

    Args:
        train_X: The training data, shape (n_samples, n_features).
        train_y: The training labels, shape (n_samples,).
        C: The regularization strength.
        eps: The tolerance for stopping criterion.
        max_iter: The maximum number of iterations.

    Returns:
        W: The weights, shape (n_classes, n_features).
        b: The bias terms, shape (n_classes,).
    """

    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    # Initialize W and b
    W = np.zeros((n_classes, n_features))
    b = np.zeros(n_classes)

    # Train a binary classifier for each class
    for i in tqdm(range(n_classes)):
        # Create a copy of the labels
        y_train = np.copy(y)
        # Set all the labels to -1
        y_train[y_train != i] = -1
        # Set the labels of the current class to 1
        y_train[y_train == i] = 1

        # Train the binary classifier
        W[i], b[i] = svm_with_smo(X, y_train, C, eps, max_iter, kernel)

    return W, b

def one_versus_one(X, y, C, eps, kernel, max_iter):
    """
    Implementation of the one-versus-one strategy for multi-class classification.

    Args:
        X: The training data, shape (n_samples, n_features).
        y: The training labels, shape (n_samples,).
        C: The regularization strength.
        eps: The tolerance for stopping criterion.
        max_iter: The maximum number of iterations.
        kernel: The kernel function.

    Returns:
        W: The weights, shape (n_classes * (n_classes - 1) / 2, n_features).
        b: The bias terms, shape (n_classes * (n_classes - 1) / 2,).
    """

    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    # Initialize W and b
    W = []
    b = []

    # Train a binary classifier for each pair of classes
    for i in tqdm(range(n_classes)):
        for j in range(i + 1, n_classes):
            # Create a binary label array
            binary_y = np.where((y == classes[i]) | (y == classes[j]), y, 0)
            binary_y = np.where(binary_y == classes[i], -1, binary_y)
            binary_y = np.where(binary_y == classes[j], 1, binary_y)
            
            # Exclude samples that don't belong to the i-th or j-th class
            binary_y_nonzero = binary_y[binary_y != 0]
            X_nonzero = X[binary_y != 0]

            # Train the binary classifier
            w, b_value = svm_with_smo(X_nonzero, binary_y_nonzero, C, eps, max_iter, kernel)

            W.append(w)
            b.append(b_value)

    return np.array(W), np.array(b)

def error_correcting_output_codes(X, y, C, eps, kernel, max_iter, encoding_function):
    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    # Use the encoding function to create the coding matrix
    coding_matrix = encoding_function(n_classes)

    # Initialize W and b
    W = []
    b = []

    # Create a binary label array for each binary problem
    binary_ys = []
    for i in range(n_classes):
        binary_y = np.zeros(n_samples)
        binary_y[y == classes[i]] = 1
        binary_y[y != classes[i]] = -1
        binary_ys.append(binary_y)

    # Train a binary classifier for each binary problem
    for i in range(n_classes):
        w, b_value = svm_with_smo(X, binary_ys[i], C, eps, max_iter, kernel)
        W.append(w)
        b.append(b_value)

    return np.array(W), np.array(b), coding_matrix
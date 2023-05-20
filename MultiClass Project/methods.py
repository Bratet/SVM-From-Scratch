import numpy as np

def predict_ova(X, W, b):
    """
    Make predictions with the one-versus-all strategy.

    Args:
        X: The data to make predictions for, shape (n_samples, n_features).
        W: The weights, shape (n_classes, n_features).
        b: The bias terms, shape (n_classes,).

    Returns:
        y_pred: The predicted labels, shape (n_samples,).
    """
    # calculate the output for each class
    scores = np.dot(X, W.T) + b
    # return the class with the highest output
    return np.argmax(scores, axis=1) + 1

def predict_ovo(X, W, b, classes):
        """
        Make predictions with the one-versus-one strategy.

        Args:
            X: The data to make predictions for, shape (n_samples, n_features).
            W: The weights, shape (n_classes * (n_classes - 1) / 2, n_features).
            b: The bias terms, shape (n_classes * (n_classes - 1) / 2,).
            classes: Unique classes in the dataset.

        Returns:
            y_pred: The predicted labels, shape (n_samples,).
        """
        n_samples, n_features = X.shape
        n_classes = len(classes)
        n_classifiers = len(W)

        # Initialize the vote matrix
        votes = np.zeros((n_samples, n_classes))

        # Compute the votes for each classifier
        classifier_index = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                y_pred = np.sign(np.dot(X, W[classifier_index]) + b[classifier_index])
                votes[y_pred == -1, i] += 1
                votes[y_pred == 1, j] += 1
                classifier_index += 1

        # Predict the class with the most votes
        y_pred = classes[votes.argmax(axis=1)]

        return y_pred
    
def predict_ecoc(X, W, b, coding_matrix, decoding_function):
    n_samples = X.shape[0]
    n_classes = coding_matrix.shape[0]

    # Get binary predictions for each classifier
    binary_preds = []
    for i in range(n_classes):
        binary_preds.append(np.sign(X @ W[i] + b[i]))

    binary_preds = np.column_stack(binary_preds)  # Shape: (n_samples, n_classes)

    # Use the decoding function to get class predictions
    preds = decoding_function(binary_preds, coding_matrix)
    
    return np.array(preds)

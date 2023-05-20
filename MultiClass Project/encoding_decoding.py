import numpy as np

def construct_tree(classes):
    """
    This function constructs a binary tree structure for the problem.

    Args:
        classes: List of classes.

    Returns:
        The binary tree structure.
    """

    # The base case: if there is only one class, return it as a leaf node.
    if len(classes) == 1:
        return classes[0]

    # If there are more than one classes, split them into two groups and recursively construct the subtrees.
    mid = len(classes) // 2
    left_subtree = construct_tree(classes[:mid])
    right_subtree = construct_tree(classes[mid:])

    # Return the current tree node, which contains the two subtrees.
    return [left_subtree, right_subtree]


def generate_encoding(tree, encoding, depth):
    """
    Recursively generates the encoding matrix from the binary tree.

    Args:
        tree: The binary tree structure.
        encoding: The current encoding as a list of 1s and -1s.
        depth: The current depth in the tree.

    Returns:
        The list of encoding columns generated from the tree.
    """

    if isinstance(tree, int):  # Leaf node
        return [encoding + [0]*(depth - len(encoding))]

    left_encoding = generate_encoding(tree[0], encoding + [1], depth)
    right_encoding = generate_encoding(tree[1], encoding + [-1], depth)

    return left_encoding + right_encoding

def decoc_encoding(n_classes):
    """
    This function implements the DECOC algorithm.

    Args:
        n_classes: The number of classes.

    Returns:
        The DECOC encoding matrix.
    """

    # Construct a binary tree structure for the problem.
    classes = list(range(n_classes))
    tree = construct_tree(classes)

    # Generate the encoding matrix from the binary tree.
    max_depth = int(np.ceil(np.log2(n_classes)))
    M = generate_encoding(tree, [], max_depth)

    # Convert the list of encoding columns to a NumPy array and transpose it.
    M = np.array(M).T

    return M


def inverse_hamming_decoding(binary_preds, coding_matrix):
    """
    This function implements the inverse Hamming decoding algorithm.

    Args:
        binary_preds: The binary predictions, shape (n_samples, n_classes - 1).
        coding_matrix: The coding matrix, shape (n_classes, n_classes - 1).

    Returns:
        The decoded labels, shape (n_samples,).
    """

    n_samples, _ = binary_preds.shape
    n_classes, _ = coding_matrix.shape
    
    print("binary_preds shape:", binary_preds.shape)
    print("coding_matrix shape:", coding_matrix.shape)

    # Initialize an array to store the decoded labels.
    decoded_labels = np.empty(n_samples, dtype=int)

    # Calculate the Hamming distance between the binary predictions and each row of the coding matrix.
    for i in range(n_samples):
        hamming_distances = np.sum(binary_preds[i] != coding_matrix, axis=1)
        # Assign the label corresponding to the row with the smallest Hamming distance.
        decoded_labels[i] = np.argmin(hamming_distances)

    return decoded_labels



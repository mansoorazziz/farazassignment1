def correlationCoefficient(x, y):
    """
    Computes the Pearson correlation coefficient between two vectors.

    Parameters:
        x (list or array-like): The first vector of numeric values.
        y (list or array-like): The second vector of numeric values.

    Returns:
        float: The correlation coefficient between the two vectors.
    """
    # Ensure the input vectors are not empty and have the same length
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Input vectors must not be empty.")
    if len(x) != len(y):
        raise ValueError("Input vectors must have the same length.")

    # Step 1: Compute the means of both vectors
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    # Step 2: Compute the numerator (covariance)
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))

    # Step 3: Compute the denominator (product of standard deviations)
    std_x = (sum((xi - mean_x) ** 2 for xi in x) / len(x)) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y) / len(y)) ** 0.5
    denominator = std_x * std_y

    # Step 4: Compute the correlation coefficient
    if denominator == 0:
        raise ValueError("Denominator is zero. Correlation is undefined.")
    corr_coefficient = numerator / denominator

    return corr_coefficient
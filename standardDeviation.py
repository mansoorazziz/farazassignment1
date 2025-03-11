
#  Function for standardDeviation

def standardDeviation(arr):
    """
    Computes the standard deviation of an array from scratch.

    Parameters:
        arr (list or array-like): The input array of numeric values.

    Returns:
        float: The standard deviation of the array.
    """
    # Ensure the array is not empty
    if len(arr) == 0:
        raise ValueError("The input array must not be empty.")

    # Step 1: Compute the mean
    mean = sum(arr) / len(arr)
    
    # Step 2: Compute the squared differences from the mean
    squared_diffs = [(x - mean) ** 2 for x in arr]

    # Step 3: Compute the mean of the squared differences (variance)
    variance = sum(squared_diffs) / len(arr)

    # Step 4: Compute the square root of the variance (standard deviation)
    std_dev = variance ** 0.5

    return std_dev
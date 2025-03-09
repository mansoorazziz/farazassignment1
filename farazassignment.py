import pandas as pd

# Define column names
column_names = [
    'mpg', 'cylinders', 'displacement', 'horsepower',
    'weight', 'acceleration', 'model_year', 'origin', 'car_name'
]


# ************************************************ PART- I ***********************************************************

# -------------------------------Load the dataset ----------------------------------------------
# 
# 
rawData = pd.read_csv(
    'cars_dataa.txt',
    sep=r'\s+',  # Handles irregular whitespace
    header=None,
    names=column_names,
    na_values='NA',  # Treat 'NA' as missing values
    on_bad_lines='skip'  # Skip problematic lines
)

# -------------------------------Print the last 10 rows of the dataset -------------------------------------------------

# 
# print(rawData.tail(10))

# ------------------------------- Check for missing values in each column ----------------------------------------------

# #
# missing_values = rawData.isnull().sum()
# print("Missing values in each column:")
# print(missing_values)





# ************************************************ PART- II ***********************************************************

# -------------------------------Handling Missing Values ------------------------------------------------------------

# Step 1: Handle missing values
# Fill numerical columns with their mean
numerical_columns = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']
rawData[numerical_columns] = rawData[numerical_columns].fillna(rawData[numerical_columns].mean())

# Step 2: Convert data types if necessary
# Convert 'cylinders', 'model_year', and 'origin' to integers
rawData['cylinders'] = rawData['cylinders'].astype(int)
rawData['model_year'] = rawData['model_year'].astype(int)
rawData['origin'] = rawData['origin'].astype(int)

# Step 3: Store processed data
preprocessedData = rawData.copy()

# Output processed data information
# print("Preprocessed Data (first 10 rows):")
# print(preprocessedData.head(10))





# ************************************************ PART- III ***********************************************************

# -------------------------------1. Calculate the average MPG ----------------------------------------------------------
# 
averageMPG = preprocessedData['mpg'].mean()

# Output the result
# print(f"The average MPG for the vehicles in the dataset is: {averageMPG}")


# -------------------------------2. Calculate Common vehicle type -------------------------------------------------------
# Define a function to categorize vehicle types based on car names
def categorize_vehicle_type(car_name):
    if 'wagon' in car_name.lower():
        return 'wagon'
    elif 'sedan' in car_name.lower():
        return 'sedan'
    elif 'convertible' in car_name.lower():
        return 'convertible'
    elif 'coupe' in car_name.lower():
        return 'coupe'
    elif 'hatchback' in car_name.lower():
        return 'hatchback'
    elif 'pickup' in car_name.lower():
        return 'pickup'
    else:
        return 'other'

# # Apply the function to compute the most common vehicle type
# commonVehicleType = preprocessedData['car_name'].apply(categorize_vehicle_type).value_counts().idxmax()

# # Output the result
# print(f"The most common vehicle type in the dataset is: {commonVehicleType}")



# -------------------------------3. Frequently occurring cylinder count -------------------------------------------------------
# Find the most frequently occurring cylinder count
commonCylinderCount = preprocessedData['cylinders'].value_counts().idxmax()

# Output the result
# print(f"The most frequently occurring cylinder count is: {commonCylinderCount}")


# -------------------------------4. Function for Standard deviation -------------------------------------------------------
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

# # Example array
# data = [10, 12, 23, 23, 16, 23, 21, 16]

# # Compute standard deviation
# result = standardDeviation(data)
# print(f"Standard Deviation: {result}")


# -------------------------------5. Function for correlationCoefficient -------------------------------------------------------
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

# # Example vectors
# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]

# # Compute the correlation coefficient
# result = correlationCoefficient(x, y)
# print(f"Correlation Coefficient: {result}")

# ---------------------------6. Computing correlationCoefficient for MPG, Horsepower and weight------------------------------------------
# Select the attributes of interest
attributes = preprocessedData[['mpg', 'horsepower', 'weight']]

# Compute the correlation matrix
attributeCorrelations = attributes.corr()

# Output the correlation matrix
# print("Correlation Matrix:")
# print(attributeCorrelations)

# ---------------------------7. Computing the correlation between displacement and MPG------------------------------------------

# 
correlationDispMPG = preprocessedData['displacement'].corr(preprocessedData['mpg'])

# Output the result
# print(f"The correlation between displacement and MPG is: {correlationDispMPG}")



# ---------------------------8.  Computing the most important attribute that influences the MPG------------------------------------------
# Select relevant numerical attributes
numerical_columns = ['horsepower', 'weight', 'displacement', 'acceleration', 'cylinders']

# Compute correlations with MPG
correlations_with_mpg = preprocessedData[numerical_columns].corrwith(preprocessedData['mpg'])

# Identify the attribute with the strongest correlation
most_influential_attribute = correlations_with_mpg.abs().idxmax()  # Use absolute value for strength
strongest_correlation = correlations_with_mpg[most_influential_attribute]

# Output the results
print(f"The most important attribute influencing MPG is: {most_influential_attribute}")
print(f"Correlation with MPG: {strongest_correlation}")

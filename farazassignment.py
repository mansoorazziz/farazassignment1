import pandas as pd

# Define column names
column_names = [
    'mpg', 'cylinders', 'displacement', 'horsepower',
    'weight', 'acceleration', 'model_year', 'origin', 'car_name'
]

# Load the dataset
rawData = pd.read_csv(
    'cars_dataa.txt',
    sep=r'\s+',  # Handles irregular whitespace
    header=None,
    names=column_names,
    na_values='NA',  # Treat 'NA' as missing values
    on_bad_lines='skip'  # Skip problematic lines
)

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
print("Preprocessed Data (first 10 rows):")
print(preprocessedData.head(10))

# Print the last 10 rows of the dataset
# print(rawData.tail(10))

# # Check for missing values in each column
# missing_values = rawData.isnull().sum()
# print("Missing values in each column:")
# print(missing_values)
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from categorize_vehicle_type import categorize_vehicle_type
from standardDeviation import standardDeviation
from correlationCoefficient import correlationCoefficient

# Define column names
column_names = [
    'mpg', 'cylinders', 'displacement', 'horsepower',
    'weight', 'acceleration', 'model_year', 'origin', 'car_name'
]

# ======================================================================================================================
# ************************************************ PART- I ***********************************************************
# ======================================================================================================================


# -------------------------------Load the dataset ----------------------------------------------
# 
# 
rawData = pd.read_csv(
    'auto-mpg.data',
    sep=r'\s+',  # Handles irregular whitespace
    header=None,
    names=column_names,
    na_values='NA',  # Treat 'NA' as missing values
    on_bad_lines='skip'  # Skip problematic lines
)

# -------------------------------Print the last 10 rows of the dataset -------------------------------------------------

# 
print(rawData.tail(10))

# ------------------------------- Check for missing values in each column ----------------------------------------------

#
missing_values = rawData.isnull().sum()
print("Missing values in each column:")
print(missing_values)





# ======================================================================================================================
# ************************************************ PART- II ***********************************************************
# ======================================================================================================================


# -------------------------------Handling Missing Values ------------------------------------------------------------

# Step 1: Handle missing values
# Fill numerical columns with their mean
numerical_columns = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']
# Convert non-numeric columns to numeric (if necessary)
for col in numerical_columns:
    rawData[col] = pd.to_numeric(rawData[col], errors='coerce')
rawData[numerical_columns] = rawData[numerical_columns].fillna(rawData[numerical_columns].mean())

# Step 2: Convert data types
# Convert 'cylinders', 'model_year', and 'origin' to integers
rawData['cylinders'] = rawData['cylinders'].astype(int)
rawData['model_year'] = rawData['model_year'].astype(int)
rawData['origin'] = rawData['origin'].astype(int)

# Step 3: Store processed data
preprocessedData = rawData.copy()

# Output processed data information
print("Preprocessed Data (first 10 rows):")
print(preprocessedData.head(10))





# ======================================================================================================================
# ************************************************ PART- III ***********************************************************
# ======================================================================================================================

# -------------------------------1. Calculate the average MPG ----------------------------------------------------------
# 
averageMPG = preprocessedData['mpg'].mean()

# Output the result
print(f"The average MPG for the vehicles in the dataset is: {averageMPG}")


# -------------------------------2. Calculate Common vehicle type -------------------------------------------------------


# # Apply the function to compute the most common vehicle type
commonVehicleType = preprocessedData['car_name'].apply(categorize_vehicle_type).value_counts().idxmax()

# Output the result
print(f"The most common vehicle type in the dataset is: {commonVehicleType}")



# -------------------------------3. Frequently occurring cylinder count -------------------------------------------------------
# Find the most frequently occurring cylinder count
commonCylinderCount = preprocessedData['cylinders'].value_counts().idxmax()

# Output the result
print(f"The most frequently occurring cylinder count is: {commonCylinderCount}")


# -------------------------------4. Function for Standard deviation -------------------------------------------------------


# Example array
data = [10, 12, 23, 23, 16, 23, 21, 16]

# Compute standard deviation
result = standardDeviation(data)
print(f"Standard Deviation: {result}")


# -------------------------------5. Function for correlationCoefficient -------------------------------------------------------

# Example vectors
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Compute the correlation coefficient
result = correlationCoefficient(x, y)
print(f"Correlation Coefficient: {result}")

# ---------------------------6. Computing correlationCoefficient for MPG, Horsepower and weight------------------------------------------
# Select the attributes of interest
attributes = preprocessedData[['mpg', 'horsepower', 'weight']]

# Compute the correlation matrix
attributeCorrelations = attributes.corr()

# Output the correlation matrix
print("Correlation Matrix:")
print(attributeCorrelations)

# ---------------------------7. Computing the correlation between displacement and MPG------------------------------------------

# 
correlationDispMPG = preprocessedData['displacement'].corr(preprocessedData['mpg'])

# Output the result
print(f"The correlation between displacement and MPG is: {correlationDispMPG}")



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






# ======================================================================================================================
# ************************************************ PART- IV    Feature Engineering**************************************
# ======================================================================================================================


# -------------------------------1. Calculate the Age of vehicle ----------------------------------------------------------
# 
# Step 1: Get the current year
current_year = datetime.now().year

# Step 2: Compute the vehicle's age
preprocessedData['vehicle_age'] = current_year - (1900 + preprocessedData['model_year'])

# Output the first few rows to verify
print(preprocessedData[['model_year', 'vehicle_age']].head())



# -------------------------------2. Standardized_dataset ----------------------------------------------------------

# Initialize the StandardScaler
scaler = StandardScaler()

# Standardize the data
standardized_data = scaler.fit_transform(preprocessedData[numerical_columns])

# Create a new DataFrame for standardized features
standardizedDF = pd.DataFrame(standardized_data, columns=numerical_columns)

# Output the standardized data
print("Standardized Data (First 5 rows):")
print(standardizedDF.head())


# -------------------------------3. Categorise the data ----------------------------------------------------------
# Function to extract the make (brand) from the car_name
def extract_make(car_name):
    return car_name.split()[0].lower()  # Extract the first word and convert to lowercase for consistency

# Apply the function to the 'car_name' column and create a new label feature
preprocessedData['make_label'] = preprocessedData['car_name'].apply(extract_make)

# Display the first few rows with the new label feature
print(preprocessedData[['car_name', 'make_label']].head())






# ======================================================================================================================
# ************************************************ PART- V  Data Visualization******************************************
# ======================================================================================================================


# -------------------------------1. Histogram Niles per Gallon ----------------------------------------------------------
# 


# 
# Target variable: 'mpg'

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=preprocessedData, x='mpg', kde=False, bins=20, color='blue')
plt.title('Histogram of Miles Per Gallon (MPG)')
plt.xlabel('MPG')
plt.ylabel('Frequency')
plt.show()



# -------------------------------2. Bar Plot ----------------------------------------------------------
# Step 1: Extract the 'make' from the 'car_name' column
preprocessedData['make'] = preprocessedData['car_name'].apply(lambda x: x.split(' ')[0])

# Step 2: Find the 10 most frequent makes
top_makes = preprocessedData['make'].value_counts().head(10)

# Step 3: Create a bar plot
plt.figure(figsize=(10, 6))
top_makes.plot(kind='bar', color='skyblue')
plt.title('Top 10 Most Frequent Car Makes')
plt.xlabel('Car Make')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# -------------------------------3. Box Plot for MPG ----------------------------------------------------------
top_makess = preprocessedData['make'].value_counts().head(10).index

# Step 3: Filter the dataset for only the top 10 makes
filtered_data = preprocessedData[preprocessedData['make'].isin(top_makess)]

# Step 4: Create the box plot
plt.figure(figsize=(12, 8))
sns.boxplot(data=filtered_data, x='make', y='mpg', palette='pastel')
plt.title('Box Plot of MPG for the Top 10 Most Frequent Makes')
plt.xlabel('Car Make')
plt.ylabel('MPG')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# -------------------------------4. Scatter Plot for MPG ----------------------------------------------------------
# Features to plot against MPG
featuresScatterplot = ['horsepower', 'weight', 'displacement']

# Generate scatter plots
for feature in featuresScatterplot:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=preprocessedData, x=feature, y='mpg', color='blue', alpha=0.6)
    plt.title(f'Relationship between {feature.capitalize()} and MPG')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Miles Per Gallon (MPG)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# ======================================================================================================================
# ************************************************ PART- VI   Model Building********************************************
# ======================================================================================================================


# -------------------------------1. Train Test Splits ----------------------------------------------------------
# 
# Step 1: Split the dataset into training and temporary sets
train_data, temp_data = train_test_split(preprocessedData, test_size=0.3, random_state=42)  # 70% training, 30% temporary

# Step 2: Split the temporary set into validation and test sets
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)  # 15% validation, 15% test

# Output the sizes of each set
print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(validation_data)}")
print(f"Test set size: {len(test_data)}")




# -------------------------------3. Train Linear Regression modal ----------------------------------------------------------

# Step 1: Define features (X) and target variable (y)
X = preprocessedData[['horsepower', 'weight', 'displacement', 'acceleration', 'cylinders']]  # Features
y = preprocessedData['mpg']  # Target variable

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 5: Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R-squared

# Output the evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

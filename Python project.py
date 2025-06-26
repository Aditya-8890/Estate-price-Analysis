import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("Housing.csv")

# Display first few records
print(df.head())

# Show info about the dataset
df.info()

# Summary statistics of numeric columns
print(df.describe())

# Summary statistics at different percentiles
print(df.describe(percentiles=[0.2, 0.4, 0.6, 0.8]))
print(df.describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))

# Extract numeric columns and show 25th percentile
numeric_cols = df.select_dtypes(include=[np.number])
print("25th Percentile:\n", numeric_cols.quantile(0.25))

# Check for missing values
print("Missing values per column:\n", df.isnull().sum())

# Display a random sample of 10 rows
print("Simple Random Sampling:")
print(df.sample(10))

# Outlier detection example using random normal data
data = np.random.normal(0, 1, 100)
Q1, Q3 = np.percentile(data, [25, 75])
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = [x for x in data if x < lower_bound or x > upper_bound]
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)
print("Outliers:", outliers)

# Boxplot of house prices
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['price'], color='m')
plt.title("Boxplot of House Prices")
plt.xlabel("Price")
plt.grid()
plt.show()

# Histogram of house prices with KDE
plt.figure(figsize=(10, 5))
sns.histplot(df['price'], bins=30, kde=True, color='blue')
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Scatter plot of area vs price
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['area'], y=df['price'], alpha=0.6)
plt.title("Price vs Area")
plt.xlabel("Area")
plt.ylabel("Price")
plt.grid()
plt.show()

# --- Linear Regression Section ---
# Reshape the data for sklearn
X = df[['area']]  # Independent variable
y = df['price']   # Dependent variable

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict prices using the model
predictions = model.predict(X)

# Plotting the regression line with the scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x='area', y='price', data=df, alpha=0.6, label='Actual')
plt.plot(df['area'], predictions, color='red', linewidth=2, label='Regression Line')
plt.title("Linear Regression: Price vs Area")
plt.xlabel("Area")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

# Print model coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient for area: {model.coef_[0]}")

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Compute correlation matrix using only numeric columns
numeric_df = df.select_dtypes(include='number')
corr_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data=corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Outlier detection in each numeric column
for col in numeric_cols.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"Column: {col}, Outliers: {len(outliers)}")

# Bar plot of price vs number of stories
plt.figure(figsize=(8, 5))
sns.barplot(x='stories', y='price', data=df, palette='viridis')
plt.title("Prices by Stories")
plt.xlabel("Stories")
plt.ylabel("Average Price")
plt.show()

# Boxplot of price vs furnishing status (before encoding for better readability)
plt.figure(figsize=(8, 5))
sns.boxplot(x='furnishingstatus', y='price', data=df, palette='pastel')
plt.title("Prices by Furnishing Status")
plt.xlabel("Furnishing Status")
plt.ylabel("Price")
plt.show()

# Histogram of house area
plt.figure(figsize=(10, 5))
sns.histplot(df['area'], bins=30, kde=True, color='green')
plt.title("Distribution of House Areas")
plt.xlabel("Area")
plt.ylabel("Frequency")
plt.show()

# Boxplot of price vs parking
plt.figure(figsize=(8, 5))
sns.boxplot(x='parking', y='price', data=df, palette='coolwarm')
plt.title("Prices by Parking")
plt.xlabel("Parking Spaces")
plt.ylabel("Price")
plt.show()



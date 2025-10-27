# Pandas Essentials: Data Manipulation in Python

This tutorial teaches you to work with tabular data using pandas, Python's most powerful data analysis library. You'll learn the fundamental operations that transform messy CSV files into clean, model-ready datasets through practical examples and clear explanations.

**Estimated time:** 60 minutes

## Why This Matters

**Problem statement:** Real-world data arrives in spreadsheets, databases, and CSV files—tables with hundreds of columns, thousands of rows, missing values, mixed types, and unclear patterns. Without pandas, loading and manipulating this data requires hundreds of lines of code, nested loops, and manual error checking. Simple tasks like "show me customers over age 25 from Tirana" become programming challenges.

**Practical benefits:** Pandas reduces complex data operations to single readable commands. What takes 50 lines in pure Python becomes one line in pandas. It handles the messy reality of real data—missing values, mixed types, date parsing, string operations—with built-in methods. Pandas makes data exploration fast enough to be interactive, cleaning operations repeatable, and feature engineering productive. You spend time discovering insights instead of debugging loops.

**Professional context:** Every data scientist uses pandas daily—it's as fundamental as knowing Git or SQL. Before neural networks or gradient boosting, data must be cleaned and shaped correctly. Pandas is the universal tool for this work. Job interviews test pandas proficiency because it's essential infrastructure. Open any data science notebook on Kaggle or GitHub, and you'll see pandas in the first cell. 

PLACEHOLDER

## Prerequisites & Learning Objectives

**Required knowledge:**
- Python basics (lists, dictionaries, loops, functions)
- Basic statistics (mean, median, sum, count)
- Understanding of tables and spreadsheets (rows, columns, cells)

**Required tools:**
- Python 3.x installed
- pandas library (`pip install pandas`)
- Jupyter Notebook recommended for interactive exploration

**Learning outcomes:**
- Understand DataFrames and Series as pandas data structures
- Load data from CSV files and inspect basic properties
- Select specific columns and filter rows based on conditions
- Handle missing values through removal or imputation
- Create new columns from existing data (feature engineering)
- Group data by categories and compute aggregate statistics
- Connect each operation to machine learning pipeline phases

**High-level approach:** Learn pandas through progressive examples, starting with loading data and building to complex grouping operations, always connecting concepts to practical ML workflows.

## Core Concepts

### What is Pandas?

Pandas is a Python library that provides high-performance data structures and analysis tools for working with structured (tabular) data. Think of it as "Excel but programmable, faster, and more powerful."

**Two core data structures:**

**DataFrame:** A 2D labeled table with rows and columns (like a spreadsheet or SQL table). Each column can have a different data type.

**Series:** A single column (or row) with an index. Essentially a 1D array with labels.

**Why "pandas"?** The name comes from "panel data," an econometrics term for multidimensional datasets, but it's commonly used for all tabular data.

### Why Pandas for Machine Learning?

**Speed:** Built on top of NumPy (C-optimized arrays), pandas inherits NumPy's performance while adding labels and intuitive syntax.

**Real-world data handling:** Deals with missing values, mixed types, string operations, date parsing, which are all common in production datasets.

**Readable syntax:** Operations read like English questions: "give me rows where age is greater than 25."

**Integration:** Works well with `scikit-learn`, `matplotlib`, and other ML libraries.

**Key insight:** While NumPy provides fast numerical arrays, pandas adds structure and labels that make data manipulation intuitive. You think in terms of "customers" and "purchases," not "row 47, column 3."

PLACEHOLDER

## Step-by-Step Instructions

###  Understanding DataFrames and Series

Before loading real data, understand pandas' two fundamental structures.

**DataFrame basics:**

```python
import pandas as pd

# Create a DataFrame from a dictionary
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'city': ['Tirana', 'Durrës', 'Tirana', 'Vlorë'],
    'salary': [45000, 52000, 61000, 48000]
}

df = pd.DataFrame(data)
print(df)
```

**Expected output:**
```
      name  age    city  salary
0    Alice   25  Tirana   45000
1      Bob   30  Durrës   52000
2  Charlie   35  Tirana   61000
3    Diana   28   Vlorë   48000
```

**Key observations:**
- Left column (0, 1, 2, 3) is the **index** (row labels)
- Top row contains **column names**
- Each column can have different types (strings, integers)
- Data organized in rows (samples) and columns (features)

**Basic properties:**

```python
# Shape: (rows, columns)
print(f"Shape: {df.shape}")  # (4, 4)

# Column names
print(f"Columns: {df.columns.tolist()}")

# Data types
print(df.dtypes)
```

**Series: Single column**

```python
# Extract one column (returns a Series)
ages = df['age']
print(type(ages))  # <class 'pandas.core.series.Series'>
print(ages)
```

**ML connection:** In machine learning, DataFrames represent datasets where rows are samples (observations, data points) and columns are features (variables, attributes). A customer dataset might have 10,000 rows (customers) and 20 columns (age, location, purchase history, etc.).

### Loading Data from Files

Real data comes from files, not dictionaries. Learn to import common formats.

**Loading CSV (most common):**

```python
import pandas as pd

# Load CSV file
df = pd.read_csv('data.csv')

# Common parameters
df = pd.read_csv('data.csv',
                 sep=',',           # delimiter (default is comma)
                 header=0,          # which row contains column names
                 index_col=None,    # which column to use as index
                 encoding='utf-8')  # character encoding
```

**First look at data:**

```python
# First 5 rows
print(df.head())

# Last 5 rows
print(df.tail())

# First N rows
print(df.head(10))
```

**Dataset overview:**

```python
# Concise summary: columns, types, non-null counts
print(df.info())

# Statistical summary of numerical columns
print(df.describe())

# Number of rows and columns
print(f"Shape: {df.shape}")
print(f"{df.shape[0]} rows, {df.shape[1]} columns")
```

**Example output from `.info()`:**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 244 entries, 0 to 243
Data columns (total 7 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   total_bill  244 non-null    float64
 1   tip         244 non-null    float64
 2   sex         244 non-null    object 
 3   smoker      244 non-null    object 
 4   day         244 non-null    object 
 5   time        244 non-null    object 
 6   size        244 non-null    int64  
dtypes: float64(2), int64(1), object(4)
```

**Reading other formats:**

```python
# Excel file
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# From URL
url = 'https://example.com/data.csv'
df = pd.read_csv(url)
```

**ML connection:** This is the **Data Collection** phase of the ML pipeline. Always start by loading data and inspecting it with `.head()`, `.info()`, and `.describe()` before any analysis or modeling.

### Selecting Columns

Extract specific columns to focus your analysis or prepare features for modeling.

**Single column (returns Series):**

```python
# Select 'age' column
ages = df['age']
print(type(ages))  # pandas.core.series.Series

# Also works with dot notation (if column name has no spaces)
ages = df.age
```

**Multiple columns (returns DataFrame):**

```python
# Select multiple columns - pass a list
subset = df[['name', 'age', 'city']]
print(subset.head())

# Order matters - columns appear in the order you specify
reordered = df[['city', 'name', 'age']]
```

**Column names:**

```python
# List all column names
print(df.columns)

# Convert to regular Python list
print(df.columns.tolist())

# Number of columns
print(len(df.columns))
```

**Dropping columns:**

```python
# Drop one column
df_smaller = df.drop('salary', axis=1)

# Drop multiple columns
df_smaller = df.drop(['age', 'salary'], axis=1)

# Drop columns in place (modifies original DataFrame)
df.drop('salary', axis=1, inplace=True)
```

**Why select columns?**

- Focus on relevant features
- Reduce memory usage
- Prepare specific features for modeling
- Create different views of the same data

**ML connection:** Feature selection is critical in machine learning. Not all columns are useful for predictions. Selecting relevant features improves model performance and reduces training time.

### Filtering Rows (Selection by Condition)

Select rows that meet specific criteria—one of pandas' most powerful features.

**Single condition:**

```python
# People older than 30
older = df[df['age'] > 30]

# People from Tirana
from_tirana = df[df['city'] == 'Tirana']

# Salaries greater than or equal to 50000
high_earners = df[df['salary'] >= 50000]
```

**Multiple conditions with AND (`&`):**

```python
# People from Tirana AND age > 25
result = df[(df['city'] == 'Tirana') & (df['age'] > 25)]

# IMPORTANT: Parentheses around each condition are required!
```

**Multiple conditions with OR (`|`):**

```python
# People from Tirana OR Durrës
result = df[(df['city'] == 'Tirana') | (df['city'] == 'Durrës')]

# Age less than 25 OR greater than 35
result = df[(df['age'] < 25) | (df['age'] > 35)]
```

**Using `.isin()` for multiple values:**

```python
# People from Tirana, Durrës, or Vlorë
cities = ['Tirana', 'Durrës', 'Vlorë']
result = df[df['city'].isin(cities)]
```

**Using `.query()` method (more readable):**

```python
# Same as above but cleaner syntax
result = df.query('age > 30')
result = df.query('city == "Tirana" and age > 25')
result = df.query('city in ["Tirana", "Durrës"]')
```

**Negation (NOT):**

```python
# People NOT from Tirana
result = df[df['city'] != 'Tirana']

# Using tilde (~) for negation
result = df[~(df['city'] == 'Tirana')]
```

**Counting filtered results:**

```python
# How many people are over 30?
count = len(df[df['age'] > 30])
print(f"{count} people are over 30")
```

**ML connection:** Filtering is essential for:

- Removing outliers before training
- Creating training/test splits by condition
- Analyzing subgroups (stratified analysis)
- Generating class-specific statistics

### Handling Missing Data

Real-world data contains missing values. Pandas provides tools to detect and handle them.

**Detecting missing values:**

```python
# Check for any missing values
print(df.isnull())  # Returns Boolean DataFrame

# Count missing values per column
print(df.isnull().sum())

# Count total missing values
print(df.isnull().sum().sum())

# Which rows have ANY missing value?
rows_with_missing = df[df.isnull().any(axis=1)]
```

**Example output:**
```
name      0
age       2
city      1
salary    0
dtype: int64
```

**Strategy 1: Drop missing values**

```python
# Drop rows with ANY missing value
df_clean = df.dropna()

# Drop rows where specific columns have missing values
df_clean = df.dropna(subset=['age', 'salary'])

# Drop columns with ANY missing value
df_clean = df.dropna(axis=1)

# Drop only if ALL values in row are missing
df_clean = df.dropna(how='all')
```

**Strategy 2: Fill missing values**

```python
# Fill with a specific value
df['age'].fillna(0, inplace=True)

# Fill with mean
df['age'].fillna(df['age'].mean(), inplace=True)

# Fill with median (better for skewed distributions)
df['salary'].fillna(df['salary'].median(), inplace=True)

# Fill with mode (most common value)
df['city'].fillna(df['city'].mode()[0], inplace=True)

# Forward fill (use previous valid value)
df['age'].fillna(method='ffill', inplace=True)

# Backward fill (use next valid value)
df['age'].fillna(method='bfill', inplace=True)
```

**Fill all columns at once:**

```python
# Fill all numeric columns with their means
numeric_cols = df.select_dtypes(include=['number']).columns
for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)
```

**ML connection:** Machine learning models cannot process missing values. During the 

**Data Cleaning** phase, you must either:

- Remove rows/columns with missing data (acceptable if < 5% of data)
- Impute (fill) missing values with statistical measures
- Use advanced imputation methods (KNN, iterative)

The choice affects model performance—test different strategies.

### Creating New Columns

Feature engineering creates new columns from existing data to improve model predictions.

**Simple calculations:**

```python
# Create age in months
df['age_months'] = df['age'] * 12

# Calculate salary per year of age
df['salary_per_age'] = df['salary'] / df['age']

# Boolean column
df['is_tirana'] = df['city'] == 'Tirana'
```

**Using functions with `.apply()`:**

```python
# Define a function
def categorize_age(age):
    if age < 25:
        return 'Young'
    elif age < 35:
        return 'Middle'
    else:
        return 'Senior'

# Apply to create new column
df['age_category'] = df['age'].apply(categorize_age)

# Using lambda (inline function)
df['age_category'] = df['age'].apply(
    lambda x: 'Young' if x < 25 else 'Middle' if x < 35 else 'Senior'
)
```

**Combining multiple columns:**

```python
# Create full description
df['description'] = df['name'] + ' from ' + df['city']

# Conditional based on multiple columns
df['high_earner_from_capital'] = (
    (df['salary'] > 50000) & (df['city'] == 'Tirana')
)
```

**Using `.map()` for replacements:**

```python
# Map cities to regions
city_to_region = {
    'Tirana': 'Central',
    'Durrës': 'Coastal',
    'Vlorë': 'Coastal',
    'Shkodër': 'North'
}
df['region'] = df['city'].map(city_to_region)
```

**Renaming columns:**

```python
# Rename specific columns
df.rename(columns={'salary': 'annual_salary', 'age': 'years_old'}, inplace=True)

# Rename all columns (useful for cleaning names)
df.columns = [col.lower().replace(' ', '_') for col in df.columns]
```

**ML connection:** Feature engineering is often the most impactful way to improve model performance. Good features can make a simple model outperform a complex one with raw features. Common transformations:

- Ratios (revenue per customer)
- Categories from continuous values (age groups)
- Combinations (interactions between features)
- Time-based features (day of week, month, season)

### Sorting and Ranking

Order data to find patterns, extremes, or prepare for specific analyses.

**Sort by single column:**

```python
# Sort by age (ascending - default)
df_sorted = df.sort_values('age')

# Sort by age (descending)
df_sorted = df.sort_values('age', ascending=False)

# Sort in place
df.sort_values('age', inplace=True)
```

**Sort by multiple columns:**

```python
# Sort by city, then by age within each city
df_sorted = df.sort_values(['city', 'age'])

# Different order for each column
df_sorted = df.sort_values(['city', 'age'], 
                           ascending=[True, False])
```

**Reset index after sorting:**

```python
# After sorting, index might be out of order
df_sorted = df.sort_values('age').reset_index(drop=True)
```

**Finding top/bottom values:**

```python
# Top 5 earners
top_5 = df.nlargest(5, 'salary')

# Bottom 5 by age
youngest_5 = df.nsmallest(5, 'age')
```

**Ranking:**

```python
# Rank salaries (1 = lowest)
df['salary_rank'] = df['salary'].rank()

# Rank in descending order (1 = highest)
df['salary_rank'] = df['salary'].rank(ascending=False)

# Handle ties with average ranks
df['salary_rank'] = df['salary'].rank(method='average')
```

**ML connection:** Sorting is useful for:

- Finding outliers (highest/lowest values)
- Time-series data (ensuring chronological order)
- Ranking customers by value
- Identifying top performers for analysis

### Grouping and Aggregating

Group data by categories and compute summary statistics—one of pandas' most powerful features.

**Basic grouping:**

```python
# Average salary by city
avg_salary = df.groupby('city')['salary'].mean()
print(avg_salary)

# Count people per city
counts = df.groupby('city').size()
# or
counts = df.groupby('city')['name'].count()
```

**Expected output:**
```
city
Durrës    52000.0
Tirana    53000.0
Vlorë     48000.0
Name: salary, dtype: float64
```

**Multiple aggregations:**

```python
# Multiple statistics for salary
summary = df.groupby('city')['salary'].agg(['mean', 'median', 'min', 'max', 'std'])
print(summary)

# Different aggregations for different columns
summary = df.groupby('city').agg({
    'salary': ['mean', 'sum'],
    'age': ['mean', 'min', 'max'],
    'name': 'count'  # count of names = count of people
})
```

**Grouping by multiple columns:**

```python
# Average salary by city AND age category
avg_by_city_age = df.groupby(['city', 'age_category'])['salary'].mean()
print(avg_by_city_age)

# Convert to DataFrame with unstack()
result = df.groupby(['city', 'age_category'])['salary'].mean().unstack()
```

**Filtering groups:**

```python
# Only cities with more than 2 people
large_cities = df.groupby('city').filter(lambda x: len(x) > 2)

# Cities where average salary exceeds 50000
high_salary_cities = df.groupby('city').filter(
    lambda x: x['salary'].mean() > 50000
)
```

**Applying custom functions:**

```python
def salary_range(group):
    return group['salary'].max() - group['salary'].min()

# Apply custom function to each group
ranges = df.groupby('city').apply(salary_range)
```

**ML connection:** Grouping creates aggregate features that are highly predictive:

- Customer lifetime value (sum of purchases per customer)
- Average session duration (mean time per user)
- Purchase frequency (count of orders per month)
- Conversion rate by segment (ratio of conversions to visits)

These aggregated features often perform better than raw transaction data.

### Value Counts and Unique Values

Understand the distribution of categorical variables.

**Value counts:**

```python
# Count occurrences of each city
city_counts = df['city'].value_counts()
print(city_counts)

# With percentages
city_pcts = df['city'].value_counts(normalize=True) * 100
print(city_pcts)

# Include missing values in count
counts_with_na = df['city'].value_counts(dropna=False)
```

**Expected output:**
```
city
Tirana    2
Durrës    1
Vlorë     1
Name: count, dtype: int64
```

**Unique values:**

```python
# List all unique cities
cities = df['city'].unique()
print(cities)  # array(['Tirana', 'Durrës', 'Vlorë'], dtype=object)

# Count of unique values
n_cities = df['city'].nunique()
print(f"Number of unique cities: {n_cities}")
```

**Cross-tabulation:**

```python
# Count combinations of two categorical variables
crosstab = pd.crosstab(df['city'], df['age_category'])
print(crosstab)
```

**ML connection:** Understanding category distributions helps with:

- **Class imbalance detection:** If 95% of samples are one class, model will be biased
- **Feature encoding decisions:** Categories with few samples might need special handling
- **Stratification:** Ensuring train/test splits represent all categories proportionally

### Data Types and Conversion

Proper data types improve performance, enable appropriate operations, and reduce memory usage.

**Check current types:**

```python
# View data types of all columns
print(df.dtypes)

# Check type of specific column
print(df['age'].dtype)
```

**Convert types:**

```python
# Convert to integer
df['age'] = df['age'].astype('int')

# Convert to float
df['salary'] = df['salary'].astype('float')

# Convert to string
df['id'] = df['id'].astype('str')

# Convert to category (saves memory for repeated values)
df['city'] = df['city'].astype('category')
```

**Numeric conversions with error handling:**

```python
# Convert to numeric, coercing errors to NaN
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# Convert to numeric, raising error if impossible
df['age'] = pd.to_numeric(df['age'], errors='raise')
```

**Categorical optimization:**

```python
# Regular string column
df['city'] = df['city'].astype('object')  # default
print(df.memory_usage())

# Convert to categorical (more efficient for repeated values)
df['city'] = df['city'].astype('category')
print(df.memory_usage())  # Typically much smaller
```

**Why it matters:**

- **Memory:** Categorical types use less memory than strings
- **Speed:** Operations on proper types are faster
- **Correctness:** Some operations require specific types

**ML connection:** Proper data types are essential for:

- **Encoding:** Categorical variables need categorical type before one-hot encoding
- **Numerical operations:** Age stored as string can't be used in calculations
- **Memory efficiency:** Large datasets require careful type management

### Exporting Data

Save processed data for later use or sharing.

**Save to CSV:**

```python
# Basic export
df.to_csv('output.csv')

# Without row index
df.to_csv('output.csv', index=False)

# With specific encoding
df.to_csv('output.csv', index=False, encoding='utf-8')

# Only specific columns
df[['name', 'age']].to_csv('subset.csv', index=False)
```

**Save to Excel:**

```python
# Single sheet
df.to_excel('output.xlsx', index=False, sheet_name='Data')

# Multiple sheets
with pd.ExcelWriter('output.xlsx') as writer:
    df.to_excel(writer, sheet_name='Raw Data', index=False)
    summary.to_excel(writer, sheet_name='Summary')
```

**Other formats:**

```python
# Parquet (efficient for large datasets)
df.to_parquet('output.parquet')

# JSON
df.to_json('output.json', orient='records')

# SQL database
# df.to_sql('table_name', connection, if_exists='replace')
```

**ML connection:** Saving processed data is part of the **Storage** phase:

- Save cleaned data separately from raw data
- Export train/test splits for reproducibility
- Store engineered features for model training
- Version datasets alongside code

## Quick Reference

| Task | Code | Purpose |
|------|------|---------|
| Load CSV | `pd.read_csv('file.csv')` | Import data |
| First rows | `.head()`, `.tail()` | Preview data |
| Info | `.info()`, `.describe()` | Understand structure |
| Select column | `df['column']` | Extract feature |
| Select columns | `df[['col1', 'col2']]` | Multiple features |
| Filter rows | `df[df['age'] > 25]` | Conditional selection |
| Missing values | `.isnull().sum()` | Detect nulls |
| Drop nulls | `.dropna()` | Remove missing |
| Fill nulls | `.fillna(value)` | Impute missing |
| New column | `df['new'] = calculation` | Feature engineering |
| Apply function | `.apply(func)` | Transform values |
| Sort | `.sort_values('col')` | Order data |
| Group | `.groupby('col').mean()` | Aggregate statistics |
| Value counts | `.value_counts()` | Count categories |
| Unique values | `.unique()`, `.nunique()` | Distinct values |
| Data types | `.dtypes`, `.astype()` | Type information/conversion |
| Save CSV | `.to_csv('file.csv')` | Export data |

## Common Workflows

**Workflow 1: Load → Explore → Clean**

```python
# Load data
df = pd.read_csv('data.csv')

# Explore
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Clean
df = df.dropna()  # Remove missing
df = df[df['age'] > 0]  # Remove invalid values
```

**Workflow 2: Filter → Group → Analyze**

```python
# Filter relevant subset
active_users = df[df['status'] == 'active']

# Group and compute statistics
summary = active_users.groupby('region')['revenue'].agg(['sum', 'mean', 'count'])

# Sort results
summary = summary.sort_values('sum', ascending=False)
```

**Workflow 3: Feature Engineering → Export**

```python
# Create derived features
df['revenue_per_customer'] = df['revenue'] / df['customers']
df['month'] = pd.to_datetime(df['date']).dt.month

# Select relevant columns
df_model = df[['revenue_per_customer', 'month', 'region', 'target']]

# Export for modeling
df_model.to_csv('model_data.csv', index=False)
```

## Best Practices

**Always explore first:** Before making changes, use `.head()`, `.info()`, `.describe()`, and `.value_counts()` to understand your data.

**Use `.copy()` for safety:** When creating modified DataFrames, use `df_new = df.copy()` to avoid unintended changes to the original.

**Chain methods carefully:** `df.groupby('city').mean().sort_values('salary')` is readable, but break complex chains into steps for debugging.

**Check after transformations:** After each operation, verify it worked—check shape, null counts, and sample values.

**Keep raw data unchanged:** Never modify original files. Apply all transformations in code for reproducibility.

**Document transformations:** Comment your code explaining why you made each transformation decision.

**Test with small samples:** Before processing millions of rows, test your workflow on the first 1000 rows to catch errors quickly.

## ML Pipeline Integration

Pandas operations map directly to machine learning pipeline phases:

| Pipeline Phase | Pandas Operations | Purpose |
|---------------|------------------|---------|
| Collection | `pd.read_csv()`, `pd.read_excel()`, `pd.read_sql()` | Load data from sources |
| Exploration | `.head()`, `.info()`, `.describe()`, `.value_counts()` | Understand structure and distributions |
| Cleaning | `.dropna()`, `.fillna()`, filtering, deduplication | Remove errors and handle missing values |
| Transformation | New columns, `.apply()`, `.map()`, `.astype()` | Engineer features and prepare for modeling |
| Splitting | Boolean indexing, `.sample()` | Create train/validation/test sets |
| Storage | `.to_csv()`, `.to_parquet()` | Save processed data |

## Summary & Next Steps

**Key accomplishments:** You've learned to load data from CSV files, explore structure and distributions, select and filter subsets, handle missing values systematically, create new features through calculations and functions, group data by categories and compute aggregates, and connect every operation to machine learning workflows.

**Critical insights:**

- **DataFrames are the universal data structure** for tabular data in Python ML ecosystem
- **80% of ML work is data preparation:** pandas is your primary tool for this phase
- **Operations are composable:** chain methods to build powerful data transformations
- **Always validate:** check results after each transformation to catch errors early
- **Type matters:** proper data types enable correct operations and reduce memory

**Real-world readiness:** You can now:

- Load messy CSV files and make sense of their structure
- Clean data by detecting and handling missing values
- Filter datasets to analyze specific subgroups
- Engineer features that improve model predictions
- Group data to discover patterns and generate insights
- Prepare datasets in the exact format ML models expect

**Connections to previous topics:**

- **NumPy:** Pandas is built on NumPy—DataFrames are collections of NumPy arrays with labels
- **Statistics:** Use `.describe()`, `.groupby()`, and aggregations to compute statistical summaries
- **Linear Algebra:** Each DataFrame column is a vector; operations mirror vector/matrix operations

**External resources:**

- [Pandas Official Documentation](https://pandas.pydata.org/docs/) - comprehensive reference with examples
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf) - quick command reference (2 pages)
- [Kaggle Learn: Pandas](https://www.kaggle.com/learn/pandas) - interactive micro-lessons with immediate feedback
- [Python for Data Analysis (Book)](https://wesmckinney.com/book/) - written by pandas creator Wes McKinney

**Next tutorial:** Visualization with Matplotlib and Seaborn transform the insights you discovered in pandas into clear, compelling charts that communicate patterns and support decisions.

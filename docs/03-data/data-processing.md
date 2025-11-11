# Data Preprocessing: Preparing Data for Analysis

This tutorial covers the systematic process of preparing raw data for machine learning. You'll learn to identify data quality issues, apply appropriate corrections, and structure datasets so models can learn effectively from clean, consistent input.

**Estimated time:** 50 minutes

## Why This Matters

**Problem statement:** 

> Raw data is never ready for models.

**Real-world data arrives messy**; missing values scatter throughout columns, categories use inconsistent spellings, numeric ranges vary wildly, and formats contradict each other. 

**Models require clean, numeric matrices with no gaps or errors**. 

The work between receiving data and training a model is data preprocessing, and it determines whether your model succeeds or fails.

**Practical benefits:** Preprocessing skills let you spot problems early, fix them systematically, and deliver data that models can actually use. 

**Professional context:** Data scientists report spending 60-80% of their time on data preparation. Companies with rigorous preprocessing workflows build reliable systems; those that skip preprocessing steps face model failures in production. 

> Preprocessing isn't optional. it's the foundation. Every successful data 

![Data Preprocessing](https://i.imgur.com/MzZN8kT.png)

## Core Concepts

### Understanding the Terminology

Data work involves several related but distinct activities. Understanding what each term means helps you communicate clearly with colleagues and organize your work.

**Data Engineering:** Building infrastructure to collect, move, store, and version data at scale. Data engineers create pipelines that deliver data from databases, APIs, and files to where analysts need it. This tutorial assumes data is already accessible.

**Data Preprocessing:** The systematic process of cleaning errors, handling missing values, standardizing formats, and converting data into numeric forms that models require. This is the focus of this tutorial.

**Feature Engineering:** Creating new variables from existing data to improve model predictions. This is a subset of preprocessing focused specifically on building better inputs.

**Data Cleansing:** Fixing quality problems; correcting errors, removing duplicates, standardizing inconsistent formats, and handling missing values.

**Data Transformation:** Converting data types and scaling numeric values so all features contribute appropriately to model training.

**Data Munging/Wrangling:** Reshaping data structures; pivoting, melting, merging tables—to get data into the right format.

**Exploratory Data Analysis (EDA):** Examining data visually and statistically to understand distributions, spot outliers, identify patterns, and determine which preprocessing steps are needed.

![Phases](https://i.imgur.com/MUYlm92.jpeg)

### The Preprocessing Workflow

Preprocessing follows a systematic sequence, though you'll often loop back to earlier steps as you discover new issues.

**1. Explore → 2. Clean → 3. Transform → 4. Validate → 5. Split → 6. Apply**

Each phase has specific goals and standard techniques. Skip a phase, and you'll likely face problems during training or deployment.

## Step-by-Step Guide

### Phase 1: Exploratory Analysis Guides Preprocessing

> Discover what preprocessing your data needs before making changes.

Start by examining data visually and numerically to identify issues. Each pattern you find suggests specific preprocessing steps.

**Quick EDA workflow:**

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('raw_data.csv')

# Structure check
print(f"Shape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst rows:\n{df.head()}")

# Missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Numeric distributions
print(f"\nNumeric summary:\n{df.describe()}")

# Categorical distributions
for col in df.select_dtypes(include='object').columns:
    print(f"\n{col} value counts:\n{df[col].value_counts()}")
```

**Visual checks:**

```python
# Distribution of numeric features
df.hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()

# Box plots reveal outliers
for col in df.select_dtypes(include='number').columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f'{col} Distribution')
    plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlations')
plt.show()
```

**Pattern → Preprocessing action mapping:**

| What EDA Shows | Preprocessing Needed |
|----------------|---------------------|
| Skewed distribution (histogram) | Log or power transformation |
| Extreme outliers (box plot) | Capping, removal, or investigation |
| Missing values >5% | Imputation strategy needed |
| Inconsistent categories | Standardization required |
| Wide numeric ranges | Scaling required |
| High correlation (>0.9) | Consider removing redundant features |

**For comprehensive EDA techniques, see the [Visualization tutorial](./visualization.md).**

### Phase 2: Data Cleansing

**Purpose:** Fix quality problems that cause model errors or introduce bias.

#### 2.1 Handling Missing Values

Missing data is common in real datasets. Your strategy depends on how much is missing and why.

**Detection:**

```python
# Missing value summary
missing_summary = pd.DataFrame({
    'column': df.columns,
    'missing_count': df.isnull().sum(),
    'missing_pct': (df.isnull().sum() / len(df)) * 100
})
missing_summary = missing_summary[missing_summary['missing_count'] > 0]
print(missing_summary.sort_values('missing_pct', ascending=False))
```

**Decision framework:**

- **Less than 5% missing:** Usually safe to drop those rows
- **5-30% missing:** Impute with statistical measures
- **More than 30% missing:** Consider dropping the column or using advanced methods
- **Missing has pattern:** Investigate why values are missing

**Strategies:**

```python
# Strategy 1: Drop rows (only if missing is minimal)
df_clean = df.dropna(subset=['critical_column'])

# Strategy 2: Fill numeric with median (robust to outliers)
df['age'].fillna(df['age'].median(), inplace=True)

# Strategy 3: Fill categorical with mode
df['category'].fillna(df['category'].mode(), inplace=True)

# Strategy 4: Fill based on groups
df['salary'] = df.groupby('department')['salary'].transform(
    lambda x: x.fillna(x.median())
)

# Strategy 5: Create missing indicator
df['age_was_missing'] = df['age'].isnull().astype(int)
df['age'].fillna(df['age'].median(), inplace=True)
```

**For pandas mechanics, see the [Pandas tutorial - Handling Missing Data](./pandas.md).**

#### 2.2 Removing Duplicates

**Detection:**

```python
# Exact duplicates
duplicates = df.duplicated()
print(f"Exact duplicates: {duplicates.sum()}")

# Duplicates on specific columns
duplicates_subset = df.duplicated(subset=['customer_id', 'date'])
print(f"Duplicate transactions: {duplicates_subset.sum()}")

# View duplicates
df[df.duplicated(keep=False)].sort_values('customer_id')
```

**Removal:**

```python
# Remove exact duplicates, keep first occurrence
df_clean = df.drop_duplicates()

# Remove based on subset of columns
df_clean = df.drop_duplicates(subset=['customer_id', 'date'], keep='first')
```

**When duplicates are valid:** Sometimes duplicate rows represent real repeated events. Check with domain experts before removing.

#### 2.3 Fixing Inconsistencies

**Standardizing text:**

```python
# Lowercase and strip whitespace
df['city'] = df['city'].str.lower().str.strip()

# Fix common typos
df['city'] = df['city'].replace({
    'new york': 'new_york',
    'ny': 'new_york',
    'newyork': 'new_york'
})

# Standardize date formats
df['date'] = pd.to_datetime(df['date'], errors='coerce')
```

**Resolving contradictions:**

```python
# Example: End date before start date
invalid_dates = df[df['end_date'] < df['start_date']]
print(f"Invalid date ranges: {len(invalid_dates)}")

# Fix by investigation or removal
df = df[df['end_date'] >= df['start_date']]
```

#### 2.4 Handling Outliers

**Detection methods:**

```python
# Method 1: Z-score (assumes normal distribution)
from scipy import stats
z_scores = np.abs(stats.zscore(df['price']))
outliers_z = df[z_scores > 3]

# Method 2: IQR (more robust)
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = df[(df['price'] < Q1 - 1.5*IQR) | (df['price'] > Q3 + 1.5*IQR)]

print(f"Outliers found: {len(outliers_iqr)}")
```

**Treatment options:**

```python
# Option 1: Remove outliers
df_no_outliers = df[(df['price'] >= Q1 - 1.5*IQR) & (df['price'] <= Q3 + 1.5*IQR)]

# Option 2: Cap outliers
df['price_capped'] = df['price'].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)

# Option 3: Keep outliers but flag them
df['is_outlier'] = ((df['price'] < Q1 - 1.5*IQR) | (df['price'] > Q3 + 1.5*IQR)).astype(int)
```

**When to keep outliers:** Domain knowledge is critical. A $10M house sale might be valid in certain markets. Always investigate before removing.

### Phase 3: Data Transformation

>  Convert clean data into numeric formats with appropriate scales.

#### 3.1 Scaling Numeric Features

Different features often have vastly different ranges. Scaling puts them on comparable scales.

**Why scaling matters:** Features with large values (e.g., income: $20,000-$150,000) can dominate features with small values (e.g., number of children: 0-5) during model training. Scaling prevents this.

**Scaling techniques:**

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Min-Max Scaling: Scales to  range
scaler = MinMaxScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])

# Standardization: Mean=0, Std=1
scaler = StandardScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])

# Robust Scaling: Uses median and IQR (good with outliers)
scaler = RobustScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])
```

**Before and after comparison:**

```python
# Visualize scaling effect
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Before scaling
axes.hist(df_original['income'], bins=30)
axes.set_title('Before Scaling')
axes.set_xlabel('Income')

# After scaling
axes.hist(df['income'], bins=30)
axes.set_title('After Scaling')
axes.set_xlabel('Scaled Income')
plt.tight_layout()
plt.show()
```

**Critical warning:** Fit scalers only on training data, then apply to validation and test sets. Fitting on all data causes data leakage.

```python
# CORRECT approach
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on training only
X_test_scaled = scaler.transform(X_test)        # Apply to test

# WRONG approach
# scaler.fit_transform(X)  # Don't fit on entire dataset!
```

#### 3.2 Encoding Categorical Variables

Models require numeric input, but many features are categorical. Encoding converts categories to numbers.

**One-Hot Encoding (most common):**

Creates binary columns for each category.

```python
# Using pandas
df_encoded = pd.get_dummies(df, columns=['city', 'color'], drop_first=True)

# Using sklearn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, drop='first')
encoded = encoder.fit_transform(df[['city', 'color']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
```

**Example:**

| Color | → | Color_Blue | Color_Green | Color_Red |
|-------|---|-----------|------------|----------|
| Red   |   | 0         | 0          | 1        |
| Blue  |   | 1         | 0          | 0        |
| Green |   | 0         | 1          | 0        |

**Warning:** One-hot encoding creates many columns with high-cardinality features (e.g., 50 unique cities → 50 new columns). This causes the "curse of dimensionality."

**Label Encoding:**

Assigns integers to categories. Use only for ordinal data (categories with natural order).

```python
# Ordinal example: education level has order
from sklearn.preprocessing import LabelEncoder

# Define order
education_order = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
df['education_encoded'] = df['education'].map(education_order)

# Or use LabelEncoder (but it assigns arbitrary order)
encoder = LabelEncoder()
df['city_encoded'] = encoder.fit_transform(df['city'])
```

> **Danger:** Using label encoding on nominal categories (no natural order) misleads models into thinking higher numbers mean "more" of something.

#### 3.3 Feature Engineering Basics

Creating new features from existing data often improves model performance more than choosing better models.

**Common transformations:**

```python
# Ratios and rates
df['price_per_sqft'] = df['price'] / df['square_feet']
df['debt_to_income'] = df['debt'] / df['income']

# Binning continuous variables
df['age_group'] = pd.cut(df['age'], bins=,[2][11][12][13][14]
                          labels=['<18', '18-35', '36-50', '51-65', '65+'])

# Date/time features
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin().astype(int)

# Interaction features
df['price_x_quality'] = df['price'] * df['quality_score']

# Polynomial features
df['age_squared'] = df['age'] ** 2
```

**For more on creating columns, see the [Pandas tutorial - Creating New Columns](./pandas.md).**

### Phase 4: Avoiding Bias in Preprocessing

> Recognize and prevent bias introduced during data preparation.

Preprocessing choices can inadvertently introduce bias that degrades model performance or creates unfair outcomes. Three main sources require attention:

#### 4.1 Data Leakage

**Problem:** Using information that won't be available when making predictions.

**Common causes:**

```python
# WRONG: Scaling before splitting
scaler.fit(X)  # Fit on entire dataset
X_train, X_test = train_test_split(X, test_size=0.2)

# CORRECT: Split first, then scale
X_train, X_test = train_test_split(X, test_size=0.2)
scaler.fit(X_train)  # Fit only on training
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Target leakage example:**

```python
# WRONG: Creating feature that contains target information
df['has_churned'] = (df['last_activity'] == df['churn_date']).astype(int)
# Problem: 'churn_date' is only known after customer churns!

# CORRECT: Use information available before prediction
df['days_since_last_activity'] = (df['today'] - df['last_activity']).dt.days
```

#### 4.2 Sampling Bias

**Problem:** Training and test sets don't represent the same population.

**Solutions:**

```python
# Use stratified splitting for imbalanced classes
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# For time-series: Use temporal splitting (never random)
train_cutoff = df['date'].quantile(0.8)
train = df[df['date'] <= train_cutoff]
test = df[df['date'] > train_cutoff]
```

#### 4.3 Preprocessing-Induced Bias

**Problem:** Preprocessing choices that favor majority groups or create unfair patterns.

**Example: Imputation bias**

```python
# Problem: Filling missing income with overall median
# If group A has missing values but different income distribution,
# this biases predictions for group A
df['income'].fillna(df['income'].median(), inplace=True)

# Better: Fill based on relevant groups
df['income'] = df.groupby('region')['income'].transform(
    lambda x: x.fillna(x.median())
)
```

**Proxy variables:** Some features indirectly encode protected attributes (e.g., ZIP code strongly correlates with race). Be aware of these relationships.

### Phase 5: The Complete Preprocessing Pipeline

**Purpose:** Apply all steps in the correct order.

Here's the standard sequence for a complete preprocessing workflow:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# 1. Load data
df = pd.read_csv('raw_data.csv')

# 2. Initial exploration
print(df.info())
print(df.describe())
print(df.isnull().sum())

# 3. Remove duplicates
df = df.drop_duplicates()

# 4. Handle missing values (before splitting)
# Drop columns with >50% missing
high_missing = df.columns[df.isnull().mean() > 0.5]
df = df.drop(columns=high_missing)

# 5. Fix data types
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')

# 6. Remove obvious outliers (domain-specific)
df = df[df['age'] > 0]
df = df[df['price'] > 0]

# 7. Feature engineering (before splitting)
df['price_per_unit'] = df['price'] / df['units']
df['year'] = df['date'].dt.year

# 8. Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# 9. CRITICAL: Split data BEFORE fitting any transformations
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 10. Impute missing values (fit on training only)
imputer = SimpleImputer(strategy='median')
numeric_cols = X_train.select_dtypes(include='number').columns
X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

# 11. Encode categorical variables (fit on training only)
categorical_cols = X_train.select_dtypes(include='object').columns
encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
X_train_cat = encoder.fit_transform(X_train[categorical_cols])
X_test_cat = encoder.transform(X_test[categorical_cols])

# 12. Scale numeric features (fit on training only)
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# 13. Combine encoded and scaled features
X_train_final = np.hstack([X_train[numeric_cols].values, X_train_cat])
X_test_final = np.hstack([X_test[numeric_cols].values, X_test_cat])

# 14. Verify shapes
print(f"Training shape: {X_train_final.shape}")
print(f"Test shape: {X_test_final.shape}")

# 15. Save preprocessing objects for production use
import joblib
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

**Key sequence points:**

1. Explore before changing anything
2. Remove duplicates early
3. Basic cleaning before feature engineering
4. Split data BEFORE fitting transformations
5. Fit all transformations on training data only
6. Apply transformations to test data
7. Save fitted transformers for production

## Common Preprocessing Challenges

### Imbalanced Classes

**Problem:** One class has far more samples (e.g., 95% class A, 5% class B).

**Impact:** Models predict majority class for everything, achieving high accuracy but missing minority class entirely.

**Solutions:**

```python
# Check balance
print(y.value_counts(normalize=True))

# Option 1: Oversample minority class
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Option 2: Undersample majority class
from imblearn.under_sampling import RandomUnderSampler
undersampler = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
```

### High-Dimensional Data

**Problem:** Too many features relative to samples.

**Impact:** Models require more data, training slows, and performance suffers.

**Solutions:**

```python
# Option 1: Remove low-variance features
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_reduced = selector.fit_transform(X)

# Option 2: Remove highly correlated features
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_reduced = X.drop(columns=to_drop)

# Option 3: Use feature importance from a model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
top_features = importances.nlargest(20).index
X_reduced = X[top_features]
```

### Memory Issues with Large Datasets

**Problem:** Dataset doesn't fit in memory.

**Solutions:**

```python
# Option 1: Load in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process each chunk
    processed = preprocess_chunk(chunk)
    # Save or aggregate results

# Option 2: Use data types to reduce memory
df['age'] = df['age'].astype('int8')  # Instead of int64
df['category'] = df['category'].astype('category')  # Instead of object

# Option 3: Select relevant columns only
relevant_cols = ['col1', 'col2', 'col3']
df = pd.read_csv('large_file.csv', usecols=relevant_cols)
```

## Best Practices

**Always explore before preprocessing:** Spend 30 minutes examining data before making any changes. Understanding what you have prevents mistakes.

**Document every decision:** Write comments explaining why you chose specific thresholds, imputation methods, or transformations. Future you will need this.

**Split data early:** Separate training and test sets immediately after loading. Never fit transformations on test data.

**Validate after each step:** Check data shape, null counts, and sample values after every preprocessing operation to catch errors immediately.

**Keep raw data untouched:** Never modify original files. Apply all changes in code for reproducibility.

**Version preprocessing code with models:** A model is worthless without knowing how its input was prepared. Track them together.

**Test pipeline with small samples:** Run your full workflow on 1,000 rows before processing millions. Catch bugs fast.

**Save fitted transformers:** Production systems need the exact scalers, encoders, and imputers used during training. Save them as files.

## Quick Reference

### Preprocessing Decision Guide

| Issue | When to Use | Method |
|-------|-------------|--------|
| Missing <5% | Small, random gaps | Drop rows |
| Missing 5-30% | Moderate gaps | Impute with median/mode |
| Missing >30% | Extensive gaps | Drop column or advanced imputation |
| Skewed distribution | Long right tail | Log transformation |
| Outliers present | Extreme values visible | IQR capping or removal |
| Wide numeric ranges | Features on different scales | StandardScaler or MinMaxScaler |
| Categorical features | Any non-numeric categories | One-hot encoding |
| High cardinality (>50 categories) | Too many unique values | Group rare categories or target encoding |
| Imbalanced classes | One class <10% of data | SMOTE or class weights |

### Preprocessing Sequence Checklist

- Load and explore data (`.info()`, `.describe()`, `.head()`)
- Check for missing values (`.isnull().sum()`)
- Remove duplicates (`.drop_duplicates()`)
- Fix data types (`.astype()`, `pd.to_datetime()`)
- Handle outliers (based on domain knowledge)
- Engineer features (ratios, date parts, interactions)
- Split into train/test sets
- Impute missing values (fit on train only)
- Encode categorical variables (fit on train only)
- Scale numeric features (fit on train only)
- Validate final data (check shapes, null counts)
- Save preprocessing objects (`.pkl` files)

## Summary & Next Steps

**Key accomplishments:** You've learned to identify data quality issues through exploration, apply systematic cleaning techniques for missing values, duplicates, and outliers, encode categorical variables and scale numeric features appropriately, recognize and prevent common sources of bias, and build complete preprocessing pipelines with correct sequencing.

**Critical insights:**

- **Preprocessing determines model success:** Clean, properly scaled data matters more than choosing the fanciest model
- **Split data before fitting:** Fit all transformations on training data only to prevent data leakage
- **Document decisions:** Your preprocessing choices need justification and reproducibility
- **Iteration is normal:** Expect to revisit earlier steps as you discover new issues

**Connections to previous tutorials:**

- **Data Foundations:** This tutorial implements the cleaning and transformation phases described there
- **Pandas:** Your primary tool for executing every preprocessing operation
- **Visualization:** Essential for discovering preprocessing needs and validating results

**What's next:**

With clean, preprocessed data, you're ready to train models. The next tutorials cover model selection, training workflows, and performance evaluation.

**External resources:**

- [scikit-learn Preprocessing Documentation](https://scikit-learn.org/stable/modules/preprocessing.html) - comprehensive guide to transformation methods
- [Kaggle Learn: Data Cleaning](https://www.kaggle.com/learn/data-cleaning) - interactive exercises with real datasets
- [Feature Engineering Book](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) - advanced feature creation techniques

> **Remember:** Great models start with great preprocessing. Master these techniques, and you'll build reliable systems that work in production, not just in notebooks.

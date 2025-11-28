
# Regression: Predicting Continuous Values

This tutorial introduces regression, the **supervised learning technique** for predicting numeric values. You'll learn how different regression models work, when to use each one, and how to evaluate predictions using a real estate rent prediction example that connects every concept to practical decisions.

**Estimated time:** 45 minutes

## Why This Matters

**Problem statement:** 

> Business questions demand numbers, not categories. "How much will it cost? When will it arrive? How many will we sell?"

**Companies need numeric predictions to make decisions.** A retailer planning inventory needs to predict next month's sales in units, not "high/medium/low." A bank approving loans needs to estimate default probability as a percentage, not just "risky/safe." A delivery service needs estimated arrival times in minutes, not "soon/later." Regression provides these continuous predictions that drive operational and financial decisions.

**Practical benefits:** Regression models help you forecast revenue, estimate project timelines, predict customer lifetime value, optimize pricing, and plan resource allocation. Every prediction comes with an error estimate, telling you how confident to be. You'll understand which features matter most, enabling better data collection and feature engineering.

**Professional context:** 

> Regression is the workhorse of predictive analytics. 

Most business metrics are continuous numbers (revenue, time, quantity, price), making regression applicable across industries. Data scientists spend more time building regression models than any other model type. Master regression fundamentals, and you'll handle 60% of real-world ML projects. The principles you learn here apply to all supervised learning.

![Regression](https://i.imgur.com/LO0SMv1.jpeg)

## Running Example: Predicting Apartment Rent

Throughout this tutorial, we'll follow a real estate platform building a rent prediction system. They have data on 10,000 apartments across a city; each with features like `size`, `location`, `number of bedrooms`, `building age`, and `distance to transit`. 

**Their goal is predicting monthly rent for new listings** so property owners can price competitively and renters can identify fair deals.

**Success metric:** Predictions within `$50` of actual rent (roughly `10%` error for typical `$500/month` apartments).

This example anchors every concept with concrete decisions and results you can evaluate.

## Core Concepts

### What Is Regression?

**Regression predicts a continuous numeric value based on input features.** The model learns relationships between features (apartment size, location, age) and a target (monthly rent), then uses those relationships to estimate the target for new, unseen examples.

**Key characteristics:**

- **Output is numeric and continuous:** $850/month, not "expensive" or "affordable"
- **Learns from labeled examples:** Historical data with known rents trains the model
- **Finds patterns in features:** Discovers how size, location, and age affect rent
- **Generalizes to new data:** Predicts rent for apartments not in training set

**Regression vs Classification:**

| Regression | Classification |
|-----------|---------------|
| Predicts numbers | Predicts categories |
| "How much?" → $850 | "Which class?" → Expensive/Affordable |
| Infinite possible outputs | Fixed number of classes |
| MAE, RMSE, R² metrics | Accuracy, precision, recall metrics |

**When to use regression:**

- Target variable is numeric and continuous
- Need specific value predictions (not just categories)
- Understand how much each feature contributes
- Compare predictions across a range

### Real-World Applications

![Regression Applications](https://i.imgur.com/18tSKNP.jpeg)

**Finance:**

- Stock price forecasting
- Credit limit determination
- Risk assessment (default probability as percentage)

**E-commerce:**

- Dynamic pricing optimization
- Customer lifetime value prediction
- Demand forecasting for inventory

**Healthcare:**

- Hospital length-of-stay estimation
- Disease progression timeline prediction
- Dosage optimization

**Operations:**

- Delivery time estimation
- Resource allocation planning
- Maintenance cost forecasting

Each application shares the same structure: historical examples train a model that predicts numeric outcomes for new cases.


### Simple Linear Regression

**Goal:** Find the straight line that best describes the relationship between one feature and the target.

**A straight line equation:**

$$ y = \beta_0 + \beta_1 x $$

- **y** = predicted rent (what we want to know)
- **x** = apartment size in square feet (what we measure)
- **$\beta_1$** (slope) = how much rent changes per additional square foot
- **$\beta_0$** (intercept) = base rent when size = 0 (theoretical, not realistic)

![Linear Regression](https://i.imgur.com/KBnFhr5.jpeg)

**Learning means finding the best $\beta_0$ and $\beta_1$** so the line passes as close as possible to all training points.


### Apartment Example: Size Predicts Rent

You have `10,000` apartments with known size and rent. Plot them as scatter points (`x = m2`, `y = rent`). Draw a line through them. That line lets you predict rent for any size.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Prepare data: one feature (size), one target (rent)
X_train = apartments[['size_m2']]  # Must be 2D array
y_train = apartments['rent']    # 1D array of rents

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# The learned parameters
print(f"Slope: ${model.coef_[0]:.2f} per size_m2")
print(f"Intercept: ${model.intercept_:.2f}")

# Example: What's the rent for an 800 size_m2 apartment?
predicted_rent = model.predict([[800]])
print(f"Predicted rent: ${predicted_rent[0]:.0f}/month")
```

**Interpreting results:**

If $\beta_1 = €6/\text{m}^2$ and $\beta_0 = €150$, the model equation is:

$$ \text{Rent} = €150 + €6 \times \text{m}^2 $$

**Rent = €150 + €6 × Area**

- 60 $\text{m}^2$ apartment (typical 1+1) → €150 + €6(60) = **€510/month**
- 90 $\text{m}^2$ apartment (typical 2+1) → €150 + €6(90) = **€690/month**
- Each additional square meter adds **€6** to the monthly rent

> **Limitation:** Simple linear regression uses only one feature. Real rent depends on location, age, amenities; not just size. That's where multiple regression helps.

## Multiple Linear Regression

**Goal:** Use multiple features simultaneously to predict more accurately.

### The Core Idea

**Extended equation:**

$$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + ... $$

Each feature gets its own coefficient showing its individual contribution to the prediction.

![Multiple Linear Regression](https://i.imgur.com/Ll9OtNr.jpeg)

### Apartment Example: Four Features

Now use size, bedrooms, building age, and distance to subway:

$$ \text{Rent} = \beta_0 + \beta_1(\text{size_m2}) + \beta_2(\text{bedrooms}) + \beta_3(\text{age}) + \beta_4(\text{distance}) $$


```python
# Multiple features
features = ['size_m2', 'bedrooms', 'age', 'distance_to_subway']
X_train = apartments[features]
y_train = apartments['rent']

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Show each feature's effect
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: ${coef:.2f}")
```

**Sample output:**

```
size_m2: €5.50 per m2
bedrooms: €80.00 per bedroom
age: -€5.00 per year
distance_to_center: -€40.00 per km
```

**Interpretation:**

- Every square meter adds **€5.50** (holding other features constant)
- Each bedroom adds **€80** (independent of size)
- Each year older reduces rent by **€5** (older = cheaper)
- Each km farther from center reduces rent by **€40** (location matters)


**Why multiple features matter:**

- Size alone explains `55%` of rent variation ($R^2 = 0.55$)
- Adding bedrooms improves to 68%
- Adding age and location improves to 82%
- Real predictions need multiple factors

> **Key insight:** Coefficients show **independent** effects. 

The bedroom coefficient (`€80`) answers: "If two apartments are the same size, same age, same location, but one has an extra bedroom, it rents for `€80` more."

## Polynomial Regression

**Goal:** Capture non-linear relationships when straight lines don't fit.

### The Core Idea

Some relationships aren't linear. Rent might increase slowly for small apartments, then rapidly for large ones. Polynomial regression adds squared or cubed terms to capture curves.

**Equation:**

$$ y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + ... $$

For apartment size:

$$ \text{Rent} = \beta_0 + \beta_1(\text{size\_m2}) + \beta_2(\text{size\_m2}^2) $$

The squared term bends the line into a curve.

![Polynomial Regression](https://i.imgur.com/gnNhSiv.jpeg)


### When Relationships Curve

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Create polynomial features (degree=2 adds squared terms)
model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

model.fit(X_train[['size_m2']], y_train)
predictions = model.predict(X_test[['size_m2']])
```

**Apartment example:** 

Small apartments (40-60 m²) rent for **€350-€550**. Mid-size (70-100 m²) rent for **€600-€900**. Large luxury apartments (120-200 m²) rent for **€1,200-€2,500**. The increase accelerates—a polynomial curve fits better than a straight line.

**Warning:** Higher-degree polynomials (3rd, 4th degree) can overfit, fitting noise instead of true patterns. Start with degree 2, check if it helps, rarely go beyond degree 3.

**When to use polynomial regression:**

- Scatter plot shows clear curve (not straight line)
- Residuals from linear regression show pattern (not random)
- Domain knowledge suggests non-linearity (diminishing returns, exponential growth)


## Tree-Based Regression

**Goal:** Capture complex non-linear patterns using decision rules.

### Decision Tree Regression

**Core idea:** Split data into groups based on feature values, predict the average rent within each group.

**How it works:**

1. Find the feature and split point that best separates high-rent from low-rent apartments
2. Example: Split at `"size < 80 m2"`
3. For each resulting group, repeat the splitting process
4. Continue until groups are small or pure
5. Predict average rent within each final group

![Decision Tree Regressor](https://i.imgur.com/JUakD0z.jpeg)

```python
from sklearn.tree import DecisionTreeRegressor

# Train decision tree
model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=20)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Apartment example decision path:**

```
Is size < 80 size_m2?
├─ Yes: Is distance < 2 km?
│  ├─ Yes: Predict €550 (small, central)
│  └─ No: Predict €350 (small, far)
└─ No: Is size < 120 size_m2?
   ├─ Yes: Predict €850 (medium)
   └─ No: Predict €1,200 (large)
```

**Advantages:**

- Handles non-linear relationships naturally (no need for polynomial features)
- Captures interactions automatically (size matters more in central locations)
- Interpretable rules (easy to explain decisions)

**Disadvantages:**

- Overfits easily if tree is too deep
- Small changes in data can drastically change tree structure
- Less smooth predictions (jumps at split boundaries)

### Random Forest Regression

**Core idea:** Build many decision trees on random subsets of data, average their predictions.

**How it works:**

1. Create 100 different training sets by randomly sampling with replacement (bootstrapping)
2. Train a decision tree on each set, using random subsets of features at each split
3. For new apartment, get prediction from all 100 trees
4. Average those 100 predictions for final estimate

![Random Forest Regressor](https://i.imgur.com/Nwm70ZW.png)

```python
from sklearn.ensemble import RandomForestRegressor

# Train random forest (100 trees)
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Feature importance
importances = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importances)
```

**Apartment example output:**

```
feature          importance
size_m2          0.42
distance         0.23
bedrooms         0.15
age              0.11
bathrooms        0.06
parking          0.03
```

Square footage explains `42%` of rent variation, distance to transit `23%`, bedrooms `15%`.

**Why Random Forest beats single trees:**

- **Reduces overfitting:** Averaging many trees smooths out individual tree mistakes
- **More accurate:** Ensemble of 100 trees outperforms any single tree
- **Robust:** Small data changes don't drastically affect predictions
- **Feature importance:** Shows which features matter most

**Trade-offs:**

- Slower to train (100 trees vs 1 tree)
- Less interpretable (can't easily show one decision path)
- Requires more memory (stores all trees)

**When to use tree-based models:**

- Non-linear relationships exist
- Feature interactions matter (size affects rent differently in different neighborhoods)
- You need feature importance scores
- Interpretability is secondary to accuracy


## How Regression Learns: Loss Functions

**Goal:** Measure how wrong predictions are so the model can improve.

### Mean Squared Error (MSE)

**Most common regression loss.** For each apartment, calculate `the square of (predicted rent - actual rent)`, then average across all apartments.

**Formula:**

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

**Why square errors?**

- Large errors penalized heavily (off by `€500` is 25 times worse than off by `€100`)
- Always positive (errors don't cancel out)
- Math works nicely for finding optimal parameters

**Apartment example (Tirana):**

- Apt 1: Predict €500, actual €550 → Error = €50 → Squared = 2,500
- Apt 2: Predict €700, actual €800 → Error = €100 → Squared = 10,000
- Apt 3: Predict €450, actual €430 → Error = -€20 → Squared = 400

$$ \text{MSE} = \frac{2,500 + 10,000 + 400}{3} = 4,300 $$

**Training goal:** Adjust coefficients to minimize MSE across all training examples.

### Other Loss Functions

**Mean Absolute Error (MAE):** Average of absolute errors (not squared).

- Less sensitive to outliers
- Apartment example: $\text{MAE} = \frac{50 + 100 + 20}{3} \approx \text{€56.7}$

**Huber Loss:** Combines MSE and MAE benefits. Squared for small errors (smooth gradient), linear for large errors (robust to outliers).

```python
from sklearn.metrics import mean_absolute_error

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Average prediction error: €{mae:.0f}")
```

### R² Score (Coefficient of Determination)

**What it means:** Percentage of rent variation explained by your model (0 to 1).

**Formula:**

$$ R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} $$

**Interpretation:**

- R² = 0.85 → Model explains 85% of why rents differ
- R² = 0.50 → Model explains only 50% → mediocre
- R² = 0.20 → Model explains 20% → poor, most variation unexplained

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, predictions)
print(f"Variance explained: {r2:.2%}")
```

**Apartment example:** `$R^2 = 0.82$` means the four features (size, bedrooms, age, distance) explain `82%` of rent differences. The remaining `18%` comes from factors not in your data (views, renovation quality, landlord reputation).



### Which Metric to Use?

| Metric | Use When | Apartment Context |
|--------|---------|------------------|
| **MAE** | Easy interpretation needed | "Off by €85 on average" |
| **RMSE** | Penalize large errors | Critical to avoid big mistakes |
| **R²** | Compare models | "Model A explains 82%, Model B only 65%" |

**Decision framework for apartment rent:**

- MAE < €50 → Deploy confidently
- MAE €50-€150 → Deploy with warnings
- MAE > €150 → Don't deploy, collect better data or features


### Visualizing Predictions

**Predicted vs Actual plot:**

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', linewidth=2)
plt.xlabel('Actual Rent ($)')
plt.ylabel('Predicted Rent ($)')
plt.title('Prediction Quality')
plt.tight_layout()
plt.show()
```

**Perfect model:** All points lie on red diagonal line.
**Real model:** Points scatter around line. Tighter scatter = better predictions.

![Results](https://i.imgur.com/QS8C4eW.png)



## Common Regression Challenges

### Overfitting

**Problem:** Model memorizes training data noise instead of learning true patterns.

**Detection:** Training $R^2 = 0.95$, test $R^2 = 0.65$ (huge gap).

**Solutions:** Use regularization (stay tuned), collect more data, simplify model (fewer features, shallower trees).

### Outliers

**Problem:** Extreme values skew predictions.

**Apartment example:** One €3,000/month penthouse pulls model's predictions up for all large apartments.

**Solutions:** Remove outliers after investigation, use robust regression methods, cap extreme values, use tree-based models (naturally robust).

### Feature Scaling

**Problem:** Features with large ranges dominate distance calculations.

**Apartment example:** Size (50-500) overwhelms bedrooms (1-4) in unscaled models.

**Solution:** Standardize features before training (except for tree-based models).

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Complete Workflow Example

Here's end-to-end apartment rent prediction:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load data
df = pd.read_csv('apartments.csv')

# 2. Select features and target
features = ['size_m2', 'bedrooms', 'bathrooms', 'age', 'distance_to_subway']
X = df[features]
y = df['rent']

# 3. Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate on test set
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Test MAE: ${mae:.0f}")
print(f"Test R²: {r2:.2%}")

# 6. Predict for new apartment
new_apartment = [[850, 2, 1, 5, 0.3]]  # 850size_m2, 2bed, 1bath, 5yrs, 0.3km
predicted_rent = model.predict(new_apartment)
print(f"\nPredicted rent: ${predicted_rent[0]:.0f}/month")

# 7. Feature importance
for feature, importance in zip(features, model.feature_importances_):
    print(f"{feature}: {importance:.2%}")
```

![Success](https://i.imgur.com/F3KZhfS.png)

## Best Practices

**Start simple, add complexity only when needed.** Begin with linear regression. If $R^2 < 0.70$, try polynomial or tree-based models.

**Check assumptions before modeling.** Plot features vs target. If relationships look linear, linear regression works. If curved, try polynomial or trees.

**Always split data.** Train on `80%`, test on `20%`. Never evaluate on training data; overfitting hides there.

**Interpret coefficients.** Linear models tell you exactly how each feature affects the target. Use this to validate against domain knowledge.

**Visualize residuals.** Patterns in residuals reveal problems metrics miss (e.g., underpredicting expensive apartments).

**Set realistic expectations.** Perfect predictions are impossible. Aim for "useful" not "perfect." MAE of €80 for €900 apartments (`9%` error) is good.

**Monitor in production.** Track MAE monthly. If it increases from €80 to €130, retrain with recent data (market changed).

## Quick Reference

| Model Type | When to Use | Pros | Cons |
|-----------|------------|------|------|
| Linear Regression | Simple, linear relationships | Fast, interpretable, stable | Can't capture curves |
| Polynomial Regression | Non-linear relationships | Fits curves, still interpretable | Overfits easily |
| Ridge/Lasso | Many features, overfitting risk | Prevents overfitting, automatic feature selection (Lasso) | Less interpretable |
| Decision Tree | Complex interactions, non-linear | Handles interactions, interpretable rules | Overfits, unstable |
| Random Forest | Best accuracy needed | High accuracy, robust, feature importance | Slow, less interpretable |

| Metric | Meaning | Good Value |
|--------|---------|-----------|
| MAE | Average error in dollars | < 10% of typical value |
| RMSE | Error penalizing outliers | < 15% of typical value |
| R² | % variance explained | > 0.70 |

## Summary & Next Steps

**Key accomplishments:** You understand regression predicts continuous numeric values from features, know how linear regression finds coefficients minimizing prediction errors, can use multiple features to improve accuracy, recognize when to use polynomial, regularized, or tree-based models, and evaluate predictions using MAE, RMSE, and R² with clear interpretation.

**Connections to previous tutorials:**

- **ML Lifecycle:** Regression lives in the Model Training phase. You'll iterate between data preparation and model selection based on evaluation results.
- **Data preprocessing:** Feature scaling, handling missing values, and outlier treatment directly affect regression accuracy.

**What's next:**

- **Classification tutorial:** Predicting categories instead of numbers (will this customer churn? yes/no)
- **Model improvement:** Cross-validation, hyperparameter tuning, feature engineering strategies
- **Advanced regression:** Gradient boosting, neural networks for regression, time-series forecasting

**External resources:**

- [Scikit-learn Regression Guide](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning) - Complete API documentation
- [StatQuest Regression Playlist](https://www.youtube.com/c/joshstarmer) - Visual explanations of regression concepts
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) - Interactive regression exercises

> **Remember:** Regression isn't about perfect predictions. It's about understanding relationships between features and outcomes, then using those patterns to make useful estimates that improve decisions.

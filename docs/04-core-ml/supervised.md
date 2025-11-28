# Classification: Predicting Categories with Simple Algorithms

This tutorial introduces **classification, the supervised learning technique for predicting categories**. You'll learn four foundational algorithms, understand when to use each one, and master evaluation metrics using an email spam detection example that connects every concept to business decisions.

**Estimated time:** 50 minutes

## Why This Matters

**Problem statement:**

> Business decisions often require categorizing: "Will this customer churn? Is this transaction fraudulent? Should we approve this loan?"

**Categories drive yes/no decisions.** A bank needs to classify loan applications as approved or rejected. A hospital needs to diagnose patients as healthy or sick. An email provider needs to separate spam from legitimate messages. Unlike regression that predicts numbers, classification assigns discrete labels that trigger specific actions: block the email, approve the loan, schedule the appointment.

**Practical benefits:** Classification automates repetitive decisions, prioritizes limited resources, and reduces manual review costs. You can process thousands of cases instantly, focus human attention on borderline cases, and maintain consistent decision criteria. Every prediction comes with a confidence score, letting you set custom thresholds based on business costs.

**Professional context:** Classification powers recommendation systems, fraud detection, medical diagnosis, customer segmentation, and quality control across industries. Most business rules ("if condition X, then action Y") can be learned as classification models from historical examples. Master these foundational algorithms, and you'll recognize when simple interpretable models outperform complex black boxes.

![Classification](https://i.imgur.com/xOGm3AT.png)

## Running Example: Email Spam Detection

Throughout this tutorial, we'll follow a small business email platform building a spam filter. They have `10,000` labeled emails with features extracted from content and metadata: word count, presence of urgent language, number of links, whether sender is known, attachment count, and ratio of UPPERCASE text.

**Business need:** Reduce inbox clutter without blocking important business communications. False positives (blocking real emails) anger users more than false negatives (letting some spam through).

**Success metric:** 
- `Precision > 90%` (minimize false positives; don't block real emails)
- `Recall > 80%` (catch most spam)

This example anchors every algorithm in concrete, relatable decisions you can evaluate.

## Core Concepts

### What Is Classification?

**Classification predicts discrete categories based on input features.** The model learns patterns that distinguish classes (spam vs legitimate) from labeled training examples, then applies those patterns to assign categories to new, unseen data.

**Key characteristics:**

- **Output is a category:** Spam or Legitimate, not a spam probability score
- **Learns from labeled examples:** Historical emails with known labels train the model
- **Finds decision boundaries:** Discovers rules that separate classes in feature space
- **Generalizes to new data:** Classifies emails not in the training set

**Classification vs Regression:**

| Classification | Regression |
|---------------|-----------|
| Predicts categories | Predicts numbers |
| "Which class?" → Spam | "How much?" → €850 |
| Finite outputs (2-100 classes) | Infinite possible outputs |
| Accuracy, Precision, Recall | MAE, RMSE, R² |

**When to use classification:**

- Target variable is categorical (spam/legitimate, approved/rejected)
- Need discrete decisions that trigger specific actions
- Understand which features separate classes
- Prioritize cases by confidence scores

### Types of Classification Problems

**Binary Classification:** Two classes only.

- Email: spam vs legitimate
- Loan: approved vs rejected  
- Medical: positive vs negative test

**Multi-class Classification:** More than two classes.

- Product category: electronics, clothing, food, books
- Customer segment: high-value, medium, low, churned
- Priority level: urgent, normal, low

**Multi-label Classification:** Multiple labels per item.

- Article tags: politics + economy + international
- Movie genres: action + comedy + drama
- Skills: Python + SQL + Statistics

This tutorial focuses on binary and multi-class; multi-label is covered in the advanced tutorial.

### Real-World Applications

![Classification Applications](https://i.imgur.com/4kXdIuS.png)

**Finance:**

- Loan approval (approve/reject)
- Credit card fraud detection (fraudulent/legitimate)
- Customer churn prediction (will churn/won't churn)

**Healthcare:**

- Disease diagnosis (positive/negative)
- Patient risk stratification (high/medium/low risk)
- Treatment recommendation (surgery/medication/observation)

**E-commerce:**

- Product categorization (automatic tagging)
- Customer segmentation (targeting campaigns)
- Review sentiment (positive/negative/neutral)

**Operations:**

- Quality control (defective/acceptable)
- Maintenance prediction (needs service/ok)
- Support ticket routing (billing/technical/account)

Each application shares the same structure: historical labeled examples train a model that assigns categories to new cases.

## The Classification Workflow

![Workflow](https://i.imgur.com/nxOYV9v.png)

**Standard process:**

1. Load labeled data (features + known categories)
2. Explore class distribution (balanced? imbalanced?)
3. Split train/test sets (stratified to preserve class ratios)
4. Train classifier on training set
5. Predict categories for test set
6. Evaluate with confusion matrix and metrics
7. Tune and iterate

**Critical point:** Always check class distribution before training. Imbalanced classes (95% legitimate, 5% spam) require special handling or your model will just predict the majority class for everything.

## Simple Classification Algorithms

### Algorithm 1: Logistic Regression

**Goal:** Find the probability that an example belongs to each class using a linear combination of features.

#### The Core Idea

Despite the name, logistic regression is for classification, not regression. It uses a sigmoid function to squeeze any linear combination of features into a probability between 0 and 1.

**Formula:**

$$ P(\text{spam}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + ...)}} $$

The sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ maps any value to (0,1).

![Logistic Regression](https://i.imgur.com/sw5DHDM.png)

**Why start here:**
- Simple and interpretable (coefficients show feature importance)
- Fast to train, works well on many problems
- Outputs probabilities (useful for ranking or setting custom thresholds)
- Similar to linear regression (easy to understand if you know that)

#### Email Spam Example

Features: `num_links`, `capslock_ratio`, `word_count`, `has_urgent`, `sender_known`

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Prepare data
X = emails[['num_links', 'capslock_ratio', 'word_count', 'has_urgent', 'sender_known']]
y = emails['is_spam']  # 0 = legitimate, 1 = spam

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)  # Get probabilities for each class
```

**Interpreting results:**

```python
# Show feature coefficients
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.3f}")
```

Sample output:
```
num_links: 0.850
capslock_ratio: 1.200
word_count: -0.002
has_urgent: 0.650
sender_known: -2.100
```

**Interpretation:**

- Positive coefficient → increases spam probability
  - `num_links` (+0.85): More links → more likely spam
  - `capslock_ratio` (+1.2): More UPPERCASE → more likely spam
  - `has_urgent` (+0.65): Urgent language → more likely spam
- Negative coefficient → decreases spam probability
  - `sender_known` (-2.1): Known sender → very unlikely spam
  - `word_count` (-0.002): Longer emails slightly less likely spam

**Decision boundary:** By default, `if P(spam) > 0.5`, classify as spam. You can adjust this threshold based on business costs (e.g., require `P(spam) > 0.7` to reduce false positives).

**Strengths:** Simple, fast, interpretable, works well for linearly separable data, outputs calibrated probabilities

**Limitations:** Assumes linear decision boundary (can't learn complex patterns like "spam if many links AND unknown sender")


---

### Algorithm 2: K-Nearest Neighbors (KNN)

**Goal:** Classify based on the majority vote of k closest training examples.

#### The Core Idea

"You are the average of your k closest neighbors." For a new email, find the k most similar emails in the training set (by feature similarity), then predict the majority class among those k neighbors.

**How it works:**

1. Choose k (e.g., k=5)
2. For new email, calculate distance to all training emails
3. Find 5 closest neighbors
4. Count their classes (e.g., 3 spam, 2 legitimate)
5. Predict majority class (spam)

![KNN](https://i.imgur.com/aiuy6IH.png)

**Distance metric:** Usually Euclidean distance. **Important:** Features must be scaled to the same range, or features with large values dominate distance calculations.

#### Email Spam Example

New email: `10 links, 30% CAPSLOCK, 150 words, urgent=yes, known=no`

Find 5 training emails with most similar feature values. If 4 are spam and 1 is legitimate, predict spam.

```python 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Scale features (critical for KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Predict
predictions = model.predict(X_test_scaled)
```

**Choosing k:**

- Small k (1-3): Sensitive to noise, complex boundaries, overfits
- Large k (10-20): Smoother boundaries, less sensitive to outliers, underfits
- Rule of thumb: $k = \sqrt{N}$ where $N$ is the number of training examples
- Best practice: Use cross-validation to find optimal k

**Example k values:**

```python 
from sklearn.model_selection import cross_val_score

for k in :
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"k={k}: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Strengths:** No training time (just stores data), simple concept, no assumptions about data distribution, naturally handles multi-class

**Limitations:** Slow predictions (must compare to all training points), sensitive to irrelevant features, requires feature scaling, memory-intensive for large datasets

---

### Algorithm 3: Naive Bayes

**Goal:** Use probability theory to calculate the likelihood of each class given the features.

#### The Core Idea

Uses Bayes' theorem to calculate the probability of each class given the observed features. Assumes features are independent (the "naive" assumption), which is often wrong but works surprisingly well in practice.

**Formula (Bayes' Theorem):**

$$ P(\text{spam} | \text{features}) = \frac{P(\text{features} | \text{spam}) \times P(\text{spam})}{P(\text{features})} $$

**Why "naive"?** Assumes features are independent. For email: assumes `num_links` doesn't affect `capslock_ratio`, which isn't true (spam often has both). Despite this unrealistic assumption, Naive Bayes performs well for text classification.

![Naive Bayes](https://i.imgur.com/p5UGTe5.png)


#### Email Spam Example

Given features (many links, high CAPSLOCK, unknown sender), calculate:

- `P(spam | features) = ?`
- `P(legitimate | features) = ?`

Predict the class with higher probability.

```python 
from sklearn.naive_bayes import GaussianNB

# Train (no scaling needed!)
model = GaussianNB()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

**Variants:**

- **GaussianNB:** Assumes features follow normal distribution (use for continuous features)
- **MultinomialNB:** For count data (word frequencies in text)
- **BernoulliNB:** For binary features (presence/absence of words)

**When to use:**

- Text classification (classic spam filtering)
- Many features, relatively small dataset
- Features are reasonably independent
- Need very fast training and prediction

**Strengths:** Very fast training and prediction, works well with high-dimensional data, handles multi-class naturally, no hyperparameters to tune

**Limitations:** Independence assumption often violated (but often works anyway), probability estimates can be poor, sensitive to how you transform features

---

### Algorithm 4: Decision Tree Classifier

**Goal:** Build a tree of yes/no questions to partition data into pure groups.

#### The Core Idea

Creates a flowchart of if/else rules that splits data recursively until each leaf node is mostly one class. Each split chooses the feature and threshold that best separates classes.

**How it learns:**

The algorithm finds the best question at each step. "Best" means maximizing information gain (making child nodes purer than parent).

**Example tree structure:**

```
Has > 5 links?
├─ Yes: capslock_ratio > 20%?
│  ├─ Yes: Predict SPAM (95% spam in this group)
│  └─ No: Predict LEGITIMATE (70% legitimate)
└─ No: sender_known?
   ├─ Yes: Predict LEGITIMATE (98% legitimate)
   └─ No: Predict SPAM (80% spam)
```

![Decision Tree](https://i.imgur.com/jygCzQe.png)

#### Email Spam Example

Tree learns: "If unknown sender AND many links AND high CAPSLOCK → spam"

```python
from sklearn.tree import DecisionTreeClassifier

# Train
model = DecisionTreeClassifier(max_depth=5, min_samples_split=50)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Visualize (optional)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=['Legitimate', 'Spam'], filled=True)
plt.show()
```

**Key hyperparameters:**

- `max_depth`: How many questions deep? (deeper = more complex, more overfitting)
  - Too shallow: underfits (can't capture patterns)
  - Too deep: overfits (memorizes training data)
  - Typical values: 3-10
- `min_samples_split`: Minimum examples in node to split further
  - Larger value = simpler tree, less overfitting
  - Typical values: 20-100
- `min_samples_leaf`: Minimum examples in leaf node
  - Prevents tiny, overfitted leaves

**Strengths:** Highly interpretable (visualize as flowchart), handles non-linear patterns, no feature scaling needed, automatic feature selection

**Limitations:** Prone to overfitting (memorizes training data), unstable (small data changes = different tree), biased toward features with many values

---

## Evaluating Classification Models

**Goal:** Know if predictions are good enough for real-world deployment.

### The Confusion Matrix

**Purpose:** Show all four outcomes—correct and incorrect predictions for both classes.

|                | **Predicted Spam** | **Predicted Legitimate** |
|----------------|----------------|----------------------|
| **Actual Spam**       | TP (True Positive)  | FN (False Negative)  |
| **Actual Legitimate** | FP (False Positive) | TN (True Negative)   |

**Email example:**

- **TP (True Positive):** Correctly caught spam → Good!
- **TN (True Negative):** Correctly identified legitimate email → Good!
- **FP (False Positive):** Flagged real email as spam → Bad! User misses important message.
- **FN (False Negative):** Let spam through → Annoying but less critical than FP.

**Code:**

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Get predictions
y_pred = model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Legitimate', 'Spam'], 
            yticklabels=['Legitimate', 'Spam'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
```

**Sample output:**

```bash
[[850  30]   <- 850 legitimate correctly identified, 30 misclassified as spam
 [ 45 175]]  <- 175 spam caught, 45 spam missed
```

![Confusion Matrix](https://i.imgur.com/GAEYwrs.png)

---

### Key Metrics (Derived from Confusion Matrix)

#### 1. Accuracy

**What it means:** Overall correctness—what percentage of predictions are right?

**Formula:**

$$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$

**Example:** (850 + 175) / 1100 = 93.2%

**When to use:** Classes are balanced (50/50 or close).

**Problem:** Misleading with imbalanced data. If 95% of emails are legitimate and you predict "all legitimate," accuracy is 95% but you catch zero spam!

---

#### 2. Precision

**What it means:** Of emails we flagged as spam, how many were actually spam?

**Formula:**

$$ \text{Precision} = \frac{TP}{TP + FP} $$

**Example:** 175 / (175 + 30) = 85.4%

**Interpretation:** When model says "spam," it's right 85.4% of the time.

**When to prioritize:** False positives are costly (blocking real emails).

**Email context:** High precision = few false alarms. Users trust the spam filter.

---

#### 3. Recall (Sensitivity, True Positive Rate)

**What it means:** Of all actual spam, how many did we catch?

**Formula:**

$$ \text{Recall} = \frac{TP}{TP + FN} $$

**Example:** 175 / (175 + 45) = 79.5%

**Interpretation:** Model catches 79.5% of spam; 20.5% slips through.

**When to prioritize:** False negatives are costly (missing disease, letting fraud through).

**Email context:** High recall = catch most spam, clean inbox.

---

#### 4. F1-Score

**What it means:** Harmonic mean of precision and recall—balances both.

**Formula:**

$$ F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

**Example:** 2 × (0.854 × 0.795) / (0.854 + 0.795) = 82.3%

**When to use:** You need balance between precision and recall, or classes are imbalanced.

**Why harmonic mean?** Penalizes extreme imbalance. If precision is 90% but recall is 10%, F1 is only 18% (not the 50% that arithmetic mean would give).

---

### Code: Calculate All Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

y_pred = model.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2%}")
print(f"Precision: {precision_score(y_test, y_pred):.2%}")
print(f"Recall:    {recall_score(y_test, y_pred):.2%}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.2%}")

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Spam']))
```

**Sample output:**

```
Accuracy:  93.18%
Precision: 85.37%
Recall:    79.55%
F1-Score:  82.35%

Classification Report:
              precision    recall  f1-score   support

 Legitimate       0.95      0.97      0.96       880
        Spam       0.85      0.80      0.82       220

    accuracy                           0.93      1100
   macro avg       0.90      0.88      0.89      1100
weighted avg       0.93      0.93      0.93      1100
```

---

### Which Metric to Use?

| Situation | Prioritize | Example |
|-----------|------------|---------|
| False positives costly | **Precision** | Email (don't block real emails) |
| False negatives costly | **Recall** | Medical diagnosis (don't miss disease) |
| Classes balanced | **Accuracy** | Coin flip prediction |
| Need balance | **F1-Score** | General-purpose classification |
| Classes imbalanced | **F1-Score** or **Precision + Recall** | Fraud detection (rare events) |

**Email spam decision:**

- **Prioritize precision** (90%+): Users hate missing real emails
- **Acceptable recall** (80%+): Catching most spam is good enough
- Trade-off: Set threshold to P(spam) > 0.7 instead of 0.5 to increase precision at cost of recall

---

## Handling Imbalanced Classes

**Problem:** Real-world data is often imbalanced. If 95% of emails are legitimate and only 5% spam, a model that predicts "all legitimate" achieves 95% accuracy but is completely useless.

### Solutions

#### 1. Stratified Train/Test Split

**Problem:** Random split might put all spam in training set, none in test set.

**Solution:** Preserve class ratios in both sets.

```python 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

Now both train and test have 95% legitimate, 5% spam.


#### 2. Class Weights

**Problem:** Model optimizes overall accuracy, ignoring minority class.

**Solution:** Penalize errors on minority class more heavily.

```python
# Automatically balance class weights
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)
```

Internally calculates weights as `n_samples / (n_classes * class_count)`, making minority class errors more costly.


#### 3. Resampling

**Oversample minority class:** Duplicate spam examples to balance classes.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

SMOTE creates synthetic examples by interpolating between existing minority examples.

**Undersample majority class:** Randomly remove legitimate examples to balance.

```python
from imblearn.under_sampling import RandomUnderSampler

undersampler = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
```

**Trade-offs:**
- Oversample: Retains all information but increases training time
- Undersample: Fast but discards potentially useful data


#### 4. Use Appropriate Metrics

**Never rely on accuracy for imbalanced data.** Always check:

- Precision and recall for each class
- F1-score
- Confusion matrix

---

## Comparing Algorithms

### Quick Reference Table

| Algorithm | Speed | Interpretability | Handles Non-Linear | Scales to Many Features | Requires Scaling |
|-----------|-------|------------------|-------------------|------------------------|------------------|
| **Logistic Regression** | Fast | High (coefficients) | No | Yes | Yes |
| **KNN** | Slow | Medium (see neighbors) | Yes | No | Yes |
| **Naive Bayes** | Very Fast | Medium (probabilities) | No | Yes | No |
| **Decision Tree** | Fast | Very High (visualize tree) | Yes | Medium | No |

### When to Use Each

| Situation | Algorithm | Why |
|-----------|-----------|-----|
| Need interpretability | Decision Tree or Logistic Regression | Visualize tree or inspect coefficients |
| Text/word features | Naive Bayes | Designed for high-dimensional sparse data |
| Small dataset (<10k samples) | KNN or Naive Bayes | No training needed (KNN) or works with little data |
| Need probabilities | Logistic Regression or Naive Bayes | Calibrated probability outputs |
| Non-linear patterns | Decision Tree or KNN | Handle complex boundaries |
| Large dataset (>100k samples) | Logistic Regression or Naive Bayes | Scale efficiently |
| Real-time predictions | Naive Bayes or Logistic Regression | Fastest prediction time |

### Email Spam Recommendation

**Start with:** Logistic Regression (simple baseline)  
**Try next:** Naive Bayes (classic for text/email)  
**If need interpretability:** Decision Tree (show users why email was flagged)  
**If small dataset:** KNN (simple, no training)

**Typical performance order for spam:**

1. Naive Bayes (80-85% F1) — designed for text
2. Logistic Regression (75-80% F1) — solid baseline
3. Decision Tree (70-75% F1) — interpretable but overfits
4. KNN (65-70% F1) — struggles with high dimensions

---

## Best Practices

**Always stratify splits** when classes are imbalanced to preserve ratios in train and test sets.

**Scale features** for Logistic Regression and KNN (they're sensitive to feature magnitude). Not needed for Naive Bayes or Decision Tree.

**Cross-validate** to get reliable performance estimates instead of single train/test split.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
print(f"F1-Score: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Check confusion matrix** before trusting accuracy, especially with imbalanced data.

**Start simple** (Logistic Regression) before trying complex models. Simple models often work well and are easier to debug and deploy.

**Interpret models** to understand what they learned. Check coefficients (Logistic Regression) or visualize trees (Decision Tree) to catch data leakage or spurious correlations.

**Test on realistic data** that matches production distribution. If production has 10% spam but test set has 50%, results are misleading.

**Set thresholds based on business costs.** Default threshold is 0.5, but if false positives cost 10× more than false negatives, increase threshold to 0.7 or 0.8.

---

## Common Pitfalls

**Class imbalance ignored:** High accuracy but minority class never predicted. Always check precision and recall for each class separately.

**No feature scaling:** KNN and Logistic Regression fail when features have different scales (e.g., word count 0-1000 vs capslock ratio 0-1).

**Overfitting:** Decision trees too deep memorize training data. Limit `max_depth` to 5-10 and set `min_samples_split` to 20-50.

**Wrong metric:** Optimizing accuracy when precision matters (or vice versa). Choose metric that matches business costs.

**Data leakage:** Including features that contain target information. Example: including "spam_score" feature when predicting spam label.

**Not stratifying:** Random split creates imbalanced train/test sets, making evaluation unreliable.

**Assuming calibrated probabilities:** predict_proba outputs aren't always true probabilities. Logistic Regression is well-calibrated; Decision Tree and Naive Bayes are not.

---

## Complete Example Workflow

```python
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load data
emails = pd.read_csv('emails.csv')
X = emails[['num_links', 'capslock_ratio', 'word_count', 'has_urgent', 'sender_known']]
y = emails['is_spam']

# 2. Check class distribution
print(y.value_counts(normalize=True))
# Output: 0 (legitimate): 85%, 1 (spam): 15%

# 3. Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train with class weights
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Cross-validate on training set
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
print(f"Cross-val F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 7. Predict on test set
y_pred = model.predict(X_test_scaled)

# 8. Evaluate
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Spam']))

# 9. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Spam'],
            yticklabels=['Legitimate', 'Spam'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# 10. Interpret coefficients
print("\nFeature Importance:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.3f}")

# 11. Adjust threshold if needed
y_proba = model.predict_proba(X_test_scaled)[:, 1]  # Spam probabilities
y_pred_custom = (y_proba > 0.7).astype(int)  # Custom threshold
print("\nWith threshold=0.7:")
print(classification_report(y_test, y_pred_custom, target_names=['Legitimate', 'Spam']))
```

![Classification Guru](https://imgur.com/t2LVpGX)

---

## Summary & Next Steps

**Key accomplishments:** You've learned the difference between classification and regression, mastered four foundational algorithms (Logistic Regression, KNN, Naive Bayes, Decision Tree), understood confusion matrix and metrics (Precision, Recall, F1), handled imbalanced classes with stratification and class weights, and compared algorithms to choose appropriate ones for problems.

**Critical insights:**

- **Start simple:** Logistic Regression often works as well as complex models
- **Metric matters:** Choose precision, recall, or F1 based on business costs, not just accuracy
- **Check the matrix:** Confusion matrix reveals what accuracy hides
- **Balance classes:** Stratify splits and use class weights for imbalanced data
- **Interpret models:** Understand what features drive predictions to catch errors

**Connections:**

- **Data Preprocessing tutorial:** Feature scaling essential for Logistic Regression and KNN
- **Regression tutorial:** Similar scikit-learn API but different output (categories vs numbers)
- **Visualization tutorial:** Plot confusion matrices, ROC curves, decision boundaries

**What's next:**

- **Advanced Classification tutorial:** Ensemble methods (Random Forest, Gradient Boosting), SVM, Neural Networks
- **Model tuning:** Hyperparameter optimization with GridSearchCV, cross-validation strategies
- **Feature engineering for classification:** Creating better predictive features, handling text data

**External resources:**

- [scikit-learn Classification Documentation](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- [Google's ML Crash Course: Classification](https://developers.google.com/machine-learning/crash-course/classification)
- [Kaggle Learn: Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)

> **Remember:** Start simple (Logistic Regression), check the confusion matrix, and choose metrics that match your business costs. Interpretable models often outperform complex ones when stakeholders need to trust and understand predictions.



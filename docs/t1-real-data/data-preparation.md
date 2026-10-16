# Data Preparation and Visualization

<span class="badge badge--time">85 min</span>
<span class="badge badge--level">Real data</span>
<span class="badge">Needs: Pandas, Plotting, Probability</span>

You have a file of 7043 telephone customers and a column saying which of them
left. This page is what you do between opening that file and handing a table to
a model.

## Why this matters

You already know the commands. `dropna`, `astype`, `groupby`, `histplot`. What
you have not done is decide **which ones to run, in what order, and why**, on a
dataset nobody has cleaned for you.

That decision sequence is the job. Two analysts given the same file will produce
different tables, and the difference is judgment rather than syntax. This page
is about the judgment, and the churn data is here to make it concrete.

## Exploratory data analysis

EDA is a name for the loop you run when you meet a dataset.

<svg viewBox="0 0 680 236" role="img" aria-labelledby="e1-title e1-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="e1-title">The exploratory loop</title>
<desc id="e1-desc">Look at the data, ask what could be wrong with it, check that suspicion, then fix it or write it down. Each answer produces the next question.</desc>
<defs><marker id="e1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<rect x="5" y="44" width="152" height="70" rx="10" fill="var(--h-cherry)" stroke="var(--h-cherry)"/>
<text x="81" y="74" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">look</text>
<text x="81" y="94" text-anchor="middle" font-size="10.5" fill="#ffffff" opacity="0.9">head, info, describe</text>
<line x1="163" y1="79" x2="175" y2="79" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#e1)"/>
<rect x="175" y="44" width="152" height="70" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="251" y="74" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">ask</text>
<text x="251" y="94" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)" opacity="0.9">what would break this?</text>
<line x1="333" y1="79" x2="345" y2="79" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#e1)"/>
<rect x="345" y="44" width="152" height="70" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="421" y="74" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">check</text>
<text x="421" y="94" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)" opacity="0.9">plot it, count it, test it</text>
<line x1="503" y1="79" x2="515" y2="79" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#e1)"/>
<rect x="515" y="44" width="152" height="70" rx="10" fill="var(--h-cherry)" stroke="var(--h-cherry)"/>
<text x="591" y="74" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">fix or record</text>
<text x="591" y="94" text-anchor="middle" font-size="10.5" fill="#ffffff" opacity="0.9">and write down why</text>
<polyline points="601,118 601,156 81,156 81,120" fill="none" stroke="var(--h-steel)" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#e1)"/>
<text x="340" y="176" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-graphite)">every answer produces the next question</text>
<text x="340" y="206" text-anchor="middle" font-size="12" fill="var(--h-graphite)">EDA is not a phase you complete before modelling. It is what you keep doing whenever a result surprises you.</text>
</svg>

It is not a checklist you complete. Every answer raises the next question: the
target is imbalanced, so which metric will you use; total charges is text, so
why; eleven values are missing, so what do those eleven rows have in common.

Start with the three commands you already know.

```python
import pandas as pd

df = pd.read_csv("Telco-Customer-Churn.csv")
print(df.shape)
df.info()
df.describe()
df.describe(include="object")
```

`describe` on its own only reports numeric columns, which on this file is three
of twenty-one. Pass `include="object"` and you get counts, unique values and the
most frequent value for the rest. Two commands, and you already know the shape
of the problem.

## Missing data, and what it means

Finding missing values is a line of code. Deciding what they mean is the work,
and this dataset has a perfect example.

```python
print(df.isna().sum().sum())
```

```
0
```

Zero missing values. Except `TotalCharges` has dtype `object`, which for a
column of money is suspicious. Eleven of its entries are a single space
character, which pandas read as text rather than as missing.

```python
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(df["TotalCharges"].isna().sum())
```

```
11
```

Now look at what those eleven rows have in common.

<svg viewBox="0 0 680 296" role="img" aria-labelledby="e2-title e2-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="e2-title">Missing values that are not missing at random</title>
<desc id="e2-desc">Every row with a blank total charge has a tenure of zero. These are brand new customers who have not been billed yet, so the blank means zero rather than unknown.</desc>
<text x="20" y="26" font-size="12" font-weight="700" fill="var(--h-graphite)">a few rows from the churn file</text>
<rect x="20" y="36" width="170" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="105" y="55" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--md-default-fg-color)">customerID</text>
<rect x="190" y="36" width="90" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="235" y="55" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--md-default-fg-color)">tenure</text>
<rect x="280" y="36" width="150" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="355" y="55" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--md-default-fg-color)">MonthlyCharges</text>
<rect x="430" y="36" width="150" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="505" y="55" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--md-default-fg-color)">TotalCharges</text>
<rect x="20" y="66" width="170" height="28" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="105" y="80" dy="0.36em" text-anchor="middle" font-size="11" font-weight="400" fill="var(--h-graphite)">7590-VHVEG</text>
<rect x="190" y="66" width="90" height="28" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="235" y="80" dy="0.36em" text-anchor="middle" font-size="11" font-weight="400" fill="var(--h-graphite)">1</text>
<rect x="280" y="66" width="150" height="28" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="355" y="80" dy="0.36em" text-anchor="middle" font-size="11" font-weight="400" fill="var(--h-graphite)">29.85</text>
<rect x="430" y="66" width="150" height="28" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="505" y="80" dy="0.36em" text-anchor="middle" font-size="11" font-weight="400" fill="var(--h-graphite)">29.85</text>
<rect x="20" y="94" width="170" height="28" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="105" y="108" dy="0.36em" text-anchor="middle" font-size="11" font-weight="400" fill="var(--h-graphite)">5575-GNVDE</text>
<rect x="190" y="94" width="90" height="28" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="235" y="108" dy="0.36em" text-anchor="middle" font-size="11" font-weight="400" fill="var(--h-graphite)">34</text>
<rect x="280" y="94" width="150" height="28" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="355" y="108" dy="0.36em" text-anchor="middle" font-size="11" font-weight="400" fill="var(--h-graphite)">56.95</text>
<rect x="430" y="94" width="150" height="28" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="505" y="108" dy="0.36em" text-anchor="middle" font-size="11" font-weight="400" fill="var(--h-graphite)">1889.5</text>
<rect x="20" y="122" width="170" height="28" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="105" y="136" dy="0.36em" text-anchor="middle" font-size="11" font-weight="400" fill="var(--h-graphite)">4472-LVYGI</text>
<rect x="190" y="122" width="90" height="28" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="235" y="136" dy="0.36em" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-cherry)">0</text>
<rect x="280" y="122" width="150" height="28" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="355" y="136" dy="0.36em" text-anchor="middle" font-size="11" font-weight="400" fill="var(--h-graphite)">52.55</text>
<rect x="430" y="122" width="150" height="28" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="505" y="136" dy="0.36em" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-cherry)">blank</text>
<rect x="20" y="150" width="170" height="28" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="105" y="164" dy="0.36em" text-anchor="middle" font-size="11" font-weight="400" fill="var(--h-graphite)">3115-CZMFE</text>
<rect x="190" y="150" width="90" height="28" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="235" y="164" dy="0.36em" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-cherry)">0</text>
<rect x="280" y="150" width="150" height="28" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="355" y="164" dy="0.36em" text-anchor="middle" font-size="11" font-weight="400" fill="var(--h-graphite)">20.25</text>
<rect x="430" y="150" width="150" height="28" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="505" y="164" dy="0.36em" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-cherry)">blank</text>
<rect x="20" y="178" width="170" height="28" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="105" y="192" dy="0.36em" text-anchor="middle" font-size="11" font-weight="400" fill="var(--h-graphite)">9237-HQITU</text>
<rect x="190" y="178" width="90" height="28" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="235" y="192" dy="0.36em" text-anchor="middle" font-size="11" font-weight="400" fill="var(--h-graphite)">2</text>
<rect x="280" y="178" width="150" height="28" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="355" y="192" dy="0.36em" text-anchor="middle" font-size="11" font-weight="400" fill="var(--h-graphite)">70.70</text>
<rect x="430" y="178" width="150" height="28" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="505" y="192" dy="0.36em" text-anchor="middle" font-size="11" font-weight="400" fill="var(--h-graphite)">151.65</text>
<rect x="20" y="206" width="170" height="28" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="105" y="220" dy="0.36em" text-anchor="middle" font-size="11" font-weight="400" fill="var(--h-graphite)">5709-LVOEQ</text>
<rect x="190" y="206" width="90" height="28" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="235" y="220" dy="0.36em" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-cherry)">0</text>
<rect x="280" y="206" width="150" height="28" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="355" y="220" dy="0.36em" text-anchor="middle" font-size="11" font-weight="400" fill="var(--h-graphite)">80.85</text>
<rect x="430" y="206" width="150" height="28" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="505" y="220" dy="0.36em" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-cherry)">blank</text>
<text x="596" y="120" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">every blank</text>
<text x="596" y="136" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">has tenure 0</text>
<text x="340" y="262" text-anchor="middle" font-size="12" fill="var(--h-graphite)">All 11 blanks belong to customers who joined this month and have not been billed yet.</text>
<text x="340" y="282" text-anchor="middle" font-size="12" fill="var(--h-graphite)">The value is not unknown. It is zero. Filling it with the median would invent a billing history.</text>
</svg>

Every one of them has `tenure` equal to zero. These are customers who signed up
this month and have not been billed yet. The value is not unknown, it is zero.

!!! warning "Always ask why before you ask how"

    Fill those eleven with the column median, around 1400 euro, and you have
    invented two years of billing history for customers who joined last week.
    Every model you build afterwards learns from that invention.

    The three options are drop, replace and impute, and they are not
    interchangeable:

    **Drop** the rows when they are few and you cannot justify a value. Eleven
    out of 7043 is nothing, so this is defensible here.

    **Replace** with a value you can justify from the meaning of the data.
    Filling these with 0 is correct, because zero is what a customer with no
    billing cycle has been charged.

    **Impute** with a statistic, or with a model, when the value genuinely
    exists but was not recorded. Median for skewed columns, mean for symmetric
    ones, and a flag column either way.

Here, replacing with 0 is right and imputing is wrong, and no amount of looking
at the `isna` count would have told you that.

## Types

`TotalCharges` was text because of eleven spaces. That is the common case: one
bad value drags a whole column to `object` and every numeric operation you try
afterwards either fails or silently does something strange.

`SeniorCitizen` has the opposite problem. It is stored as 0 and 1, so pandas
reports it as an integer and `describe` cheerfully tells you the mean senior
citizen is 0.16. It is a category wearing a number's clothes.

```python
df["SeniorCitizen"] = df["SeniorCitizen"].astype("category")
```

The test is not what the values look like. It is whether arithmetic on the
column means anything.

## Duplicates and columns to drop

```python
print(df.duplicated().sum())
print(df["customerID"].duplicated().sum())
```

Two different questions. The first asks whether an entire row repeats. The
second asks whether the identifier repeats, which would mean the same customer
appears twice with different details, and that is a data collection problem
rather than a copy and paste one.

Then drop what cannot help.

**Identifiers.** `customerID` is unique for every row. A model given it can
memorise the training set perfectly and learn nothing that transfers. Drop it.

**Constant columns.** If every row has the same value, the column carries no
information.

**Columns that leak the answer.** Anything recorded at the same time as the
outcome, or after it. There is no such column in this file, but in a real churn
dataset a field like `cancellation_reason` would be one, and it would give you a
model with 99 percent accuracy and no use whatsoever.

## Look at the target first

Before any feature, plot the thing you are predicting.

```python
import seaborn as sns

print(df["Churn"].value_counts(normalize=True))
sns.countplot(data=df, x="Churn")
```

```
No     0.735
Yes    0.265
```

About one customer in four left. That single number changes several later
decisions:

- A model that predicts "No" for everybody scores 73.5 percent accuracy. That is
  your baseline, and anything you build must beat it before it is interesting.
- Accuracy is the wrong headline metric here, for the reason you saw on the
  Probability page.
- Your splits need to be stratified, which is the last section of this page.

!!! ml "ML connection"

    You met this as class imbalance in Working with Data and as base rates in
    Probability. This is the same idea for the third time, now with a number
    attached to a real dataset. If a later page quotes you an accuracy for this
    problem, 73.5 percent is the number to compare it against.

## Distributions

Categorical features first, because two thirds of this file is categorical.

```python
categorical = df.select_dtypes(include=["object", "category"]).columns
fig, axes = plt.subplots(4, 4, figsize=(18, 14))
for ax, col in zip(axes.flatten(), categorical):
    sns.countplot(data=df, x=col, ax=ax)
    ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
```

What you are looking for: categories with almost no members, which will be
unreliable no matter what the model does, and values that should be the same
category but are spelled differently.

This dataset has a nice example. Several service columns contain `No internet
service` as well as `No`. Those are two different statements and it is your
choice whether they stay separate.

Then the continuous ones. There are only three: `tenure`, `MonthlyCharges`,
`TotalCharges`.

```python
for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
    sns.histplot(data=df, x=col, bins=30, ax=ax)
```

`tenure` is U-shaped: many customers are brand new, many have been there for
years, fewer in between. That shape is a finding, not noise, and it is the
argument for the binning section later.

## Relationships with the target

A distribution tells you what a column looks like. What you need is whether it
relates to churn.

```python
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", center=0)
```

With three numeric columns this is a small heatmap, and it says one useful thing:
`tenure` and `TotalCharges` are strongly correlated, because total charges is
roughly tenure times monthly charges. Two columns carrying the same information
is worth knowing before you feed both to a model.

For categorical features against churn, proportions beat counts.

```python
pd.crosstab(df["Contract"], df["Churn"], normalize="index")
```

```
Churn                 No       Yes
Contract
Month-to-month     0.573     0.427
One year           0.887     0.113
Two year           0.972     0.028
```

`normalize="index"` is what makes this readable. Raw counts would mostly tell
you that month-to-month is the largest group. Percentages tell you that its churn
rate is fifteen times the two year rate, which is the actual finding.

For numeric features, compare the distributions across the two groups.

```python
sns.boxplot(data=df, x="Churn", y="tenure")
```

Churners have visibly shorter tenure. You can see it. The next section is about
whether you can defend it.

## Is the pattern real?

Your eyes will find patterns in noise. A statistical test asks whether the
pattern you are looking at could plausibly have arisen by chance.

<svg viewBox="0 0 680 262" role="img" aria-labelledby="e3-title e3-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="e3-title">Choosing a test for whether a feature relates to churn</title>
<desc id="e3-desc">For a categorical feature against a categorical target use a chi-square test of independence. For a numeric feature compared across two groups use Welch t-test.</desc>
<defs><marker id="e3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<rect x="200" y="8" width="280" height="44" rx="10" fill="var(--h-cherry)"/>
<text x="340" y="30" dy="0.36em" text-anchor="middle" font-size="13.5" font-weight="700" fill="#ffffff">does this feature relate to churn?</text>
<polyline points="340,52 340,72 180,72 180,96" fill="none" stroke="var(--h-steel)" stroke-width="1.5" marker-end="url(#e3)"/>
<polyline points="340,52 340,72 500,72 500,96" fill="none" stroke="var(--h-steel)" stroke-width="1.5" marker-end="url(#e3)"/>
<text x="180" y="66" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-graphite)">the feature is categorical</text>
<text x="500" y="66" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-graphite)">the feature is numeric</text>
<rect x="50" y="100" width="260" height="96" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="180" y="128" text-anchor="middle" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">Chi-square</text>
<text x="180" y="150" text-anchor="middle" font-size="11" fill="var(--h-graphite)">build a contingency table of</text>
<text x="180" y="166" text-anchor="middle" font-size="11" fill="var(--h-graphite)">counts, compare it with what</text>
<text x="180" y="182" text-anchor="middle" font-size="11" fill="var(--h-graphite)">independence would predict</text>
<rect x="370" y="100" width="260" height="96" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="500" y="128" text-anchor="middle" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">Welch t-test</text>
<text x="500" y="150" text-anchor="middle" font-size="11" fill="var(--h-graphite)">compare the mean for churners</text>
<text x="500" y="166" text-anchor="middle" font-size="11" fill="var(--h-graphite)">with the mean for the rest,</text>
<text x="500" y="182" text-anchor="middle" font-size="11" fill="var(--h-graphite)">without assuming equal spread</text>
<text x="340" y="228" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Both answer the same question: could this pattern plausibly have appeared by chance alone?</text>
<text x="340" y="248" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Neither tells you the effect is large enough to care about. With 7043 rows, tiny differences come back significant.</text>
</svg>

### Chi-square, for two categorical variables

Build a table of counts and compare it with the counts you would expect if the
two variables were unrelated.

```python
from scipy.stats import chi2_contingency

table = pd.crosstab(df["Contract"], df["Churn"])
chi2, p, dof, expected = chi2_contingency(table)
print(chi2, p)
```

A small p-value means the observed counts would be unlikely if contract type and
churn were independent.

### Welch's t-test, for a numeric variable across two groups

```python
from scipy.stats import ttest_ind

churned = df.loc[df["Churn"] == "Yes", "tenure"]
stayed = df.loc[df["Churn"] == "No", "tenure"]
stat, p = ttest_ind(churned, stayed, equal_var=False)
print(stat, p)
```

`equal_var=False` is what makes it Welch's version rather than the standard one.
The standard test assumes both groups have the same spread, which here they do
not: there are far more stayers than churners and their tenure is spread
differently. Welch's does not assume it, costs nothing, and is the safer default
in almost every real comparison.

!!! warning "Significant does not mean important"

    With 7043 rows almost everything comes back significant, because a large
    sample can detect a tiny difference reliably. A p-value answers "is this
    difference real". It does not answer "is this difference big enough to care
    about".

    Always look at the effect alongside the p-value. A two month difference in
    average tenure with a p-value of 0.001 is real and probably useful. A
    difference of 0.3 months with the same p-value is real and useless.

## Making better features

Two moves, and both are about helping a model express something it otherwise
could not.

### Binning

<svg viewBox="0 0 680 246" role="img" aria-labelledby="e7-title e7-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="e7-title">Binning a continuous feature</title>
<desc id="e7-desc">Tenure in months is cut into four bands, new, growing, established and loyal, so the model can express a jump between groups rather than a straight line.</desc>
<line x1="50" y1="160" x2="640" y2="160" stroke="var(--h-steel)" stroke-width="1.5"/>
<rect x="52" y="82" width="26" height="78" fill="var(--h-cherry)" opacity="0.55"/>
<rect x="81" y="108" width="26" height="52" fill="var(--h-cherry)" opacity="0.55"/>
<rect x="110" y="120" width="26" height="40" fill="var(--h-cherry)" opacity="0.55"/>
<rect x="139" y="126" width="26" height="34" fill="var(--h-cherry)" opacity="0.55"/>
<rect x="168" y="130" width="26" height="30" fill="var(--h-cherry)" opacity="0.55"/>
<rect x="197" y="132" width="26" height="28" fill="var(--h-cherry)" opacity="0.55"/>
<rect x="226" y="134" width="26" height="26" fill="var(--h-cherry)" opacity="0.55"/>
<rect x="255" y="135" width="26" height="25" fill="var(--h-cherry)" opacity="0.55"/>
<rect x="284" y="136" width="26" height="24" fill="var(--h-cherry)" opacity="0.55"/>
<rect x="313" y="136" width="26" height="24" fill="var(--h-cherry)" opacity="0.55"/>
<rect x="342" y="135" width="26" height="25" fill="var(--h-cherry)" opacity="0.55"/>
<rect x="371" y="134" width="26" height="26" fill="var(--h-cherry)" opacity="0.55"/>
<rect x="400" y="132" width="26" height="28" fill="var(--h-cherry)" opacity="0.55"/>
<rect x="429" y="130" width="26" height="30" fill="var(--h-cherry)" opacity="0.55"/>
<rect x="458" y="127" width="26" height="33" fill="var(--h-cherry)" opacity="0.55"/>
<rect x="487" y="123" width="26" height="37" fill="var(--h-cherry)" opacity="0.55"/>
<rect x="516" y="118" width="26" height="42" fill="var(--h-cherry)" opacity="0.55"/>
<rect x="545" y="110" width="26" height="50" fill="var(--h-cherry)" opacity="0.55"/>
<rect x="574" y="100" width="26" height="60" fill="var(--h-cherry)" opacity="0.55"/>
<rect x="603" y="86" width="26" height="74" fill="var(--h-cherry)" opacity="0.55"/>
<line x1="196" y1="40" x2="196" y2="168" stroke="var(--md-default-fg-color)" stroke-width="2" stroke-dasharray="5 4"/>
<line x1="340" y1="40" x2="340" y2="168" stroke="var(--md-default-fg-color)" stroke-width="2" stroke-dasharray="5 4"/>
<line x1="484" y1="40" x2="484" y2="168" stroke="var(--md-default-fg-color)" stroke-width="2" stroke-dasharray="5 4"/>
<text x="124" y="188" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">new</text>
<text x="124" y="204" text-anchor="middle" font-size="10" fill="var(--h-space)">0 to 12 months</text>
<text x="268" y="188" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">growing</text>
<text x="268" y="204" text-anchor="middle" font-size="10" fill="var(--h-space)">13 to 24 months</text>
<text x="412" y="188" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">established</text>
<text x="412" y="204" text-anchor="middle" font-size="10" fill="var(--h-space)">25 to 48 months</text>
<text x="562" y="188" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">loyal</text>
<text x="562" y="204" text-anchor="middle" font-size="10" fill="var(--h-space)">49 plus months</text>
<text x="345" y="32" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-graphite)">tenure in months</text>
<text x="340" y="232" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Binning throws away detail on purpose. Do it when the effect is a jump between groups, not a steady slope.</text>
</svg>

```python
df["tenure_band"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 100],
    labels=["new", "growing", "established", "loyal"],
)
```

Binning throws away information deliberately. That is worth it when the effect
is a step rather than a slope: the difference between month 2 and month 14 is
enormous here, while the difference between month 50 and month 62 is small. A
linear model cannot say that about a raw tenure column, and it can about bands.

Use `pd.cut` for boundaries you choose and `pd.qcut` for equal sized groups.
Choosing your own is usually better, because the boundaries then mean something
to the business.

### New columns

```python
df["avg_monthly"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
df["services"] = (df[service_cols] == "Yes").sum(axis=1)
```

The second one is the interesting kind. Counting how many services a customer
has turns sixteen separate columns into one number with an obvious meaning, and
a model can use it directly.

Every feature you create should have a sentence attached explaining what it
means. If you cannot write the sentence, you are generating columns rather than
engineering features.

## Encoding

Models take numbers. Most of this dataset is text.

<svg viewBox="0 0 680 300" role="img" aria-labelledby="e4-title e4-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="e4-title">Three ways to turn a category into numbers</title>
<desc id="e4-desc">Label encoding assigns 0, 1 and 2, which invents an order. Ordinal encoding does the same but is correct when the categories really are ordered. One-hot encoding creates one column per category and invents nothing.</desc>
<text x="76" y="26" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-graphite)">InternetService</text>
<rect x="16" y="40" width="120" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="76" y="55" dy="0.36em" text-anchor="middle" font-size="11.5" fill="var(--md-default-fg-color)">DSL</text>
<rect x="16" y="72" width="120" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="76" y="87" dy="0.36em" text-anchor="middle" font-size="11.5" fill="var(--md-default-fg-color)">Fiber</text>
<rect x="16" y="104" width="120" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="76" y="119" dy="0.36em" text-anchor="middle" font-size="11.5" fill="var(--md-default-fg-color)">None</text>
<text x="228" y="26" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-cherry)">label</text>
<rect x="178" y="40" width="100" height="30" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="228" y="55" dy="0.36em" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">0</text>
<rect x="178" y="72" width="100" height="30" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="228" y="87" dy="0.36em" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">1</text>
<rect x="178" y="104" width="100" height="30" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="228" y="119" dy="0.36em" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">2</text>
<text x="228" y="152" text-anchor="middle" font-size="10.5" fill="var(--h-cherry)">says None is twice Fiber</text>
<text x="228" y="168" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-cherry)">wrong here</text>
<text x="470" y="26" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">one-hot</text>
<rect x="330" y="36" width="96" height="0" fill="none"/>
<text x="378" y="34" text-anchor="middle" font-size="10" fill="var(--h-space)">DSL</text>
<rect x="430" y="36" width="96" height="0" fill="none"/>
<text x="478" y="34" text-anchor="middle" font-size="10" fill="var(--h-space)">Fiber</text>
<rect x="530" y="36" width="96" height="0" fill="none"/>
<text x="578" y="34" text-anchor="middle" font-size="10" fill="var(--h-space)">None</text>
<rect x="330" y="40" width="96" height="30" fill="var(--h-cherry)" stroke="var(--h-cherry-line)"/>
<text x="378" y="55" dy="0.36em" text-anchor="middle" font-size="11.5" font-weight="700" fill="#ffffff">1</text>
<rect x="430" y="40" width="96" height="30" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="478" y="55" dy="0.36em" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-space)">0</text>
<rect x="530" y="40" width="96" height="30" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="578" y="55" dy="0.36em" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-space)">0</text>
<rect x="330" y="72" width="96" height="30" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="378" y="87" dy="0.36em" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-space)">0</text>
<rect x="430" y="72" width="96" height="30" fill="var(--h-cherry)" stroke="var(--h-cherry-line)"/>
<text x="478" y="87" dy="0.36em" text-anchor="middle" font-size="11.5" font-weight="700" fill="#ffffff">1</text>
<rect x="530" y="72" width="96" height="30" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="578" y="87" dy="0.36em" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-space)">0</text>
<rect x="330" y="104" width="96" height="30" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="378" y="119" dy="0.36em" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-space)">0</text>
<rect x="430" y="104" width="96" height="30" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="478" y="119" dy="0.36em" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-space)">0</text>
<rect x="530" y="104" width="96" height="30" fill="var(--h-cherry)" stroke="var(--h-cherry-line)"/>
<text x="578" y="119" dy="0.36em" text-anchor="middle" font-size="11.5" font-weight="700" fill="#ffffff">1</text>
<text x="470" y="152" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">no order implied, one column each</text>
<text x="470" y="168" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-cherry)">correct here</text>
<text x="340" y="212" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Label encoding is only safe when the categories genuinely have an order, such as low, medium, high.</text>
<text x="340" y="232" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Used on unordered categories it tells a linear model that the gap from DSL to None is twice the gap from DSL to Fiber.</text>
<text x="340" y="252" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Tree models survive it. Linear models and anything distance based do not.</text>
</svg>

```python
df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"], drop_first=True)
df["Churn"] = (df["Churn"] == "Yes").astype(int)
```

For binary columns like `Partner`, a simple 0 and 1 is fine and there is no
ordering to invent. For unordered categories with more than two values, one-hot
is the default. For genuinely ordered categories, such as your `tenure_band`,
ordinal encoding preserves information that one-hot throws away.

`drop_first=True` removes one column per feature. The information is still there,
because "not DSL and not Fiber" means None, and it avoids a redundancy that
upsets linear models.

## Scaling

<svg viewBox="0 0 680 266" role="img" aria-labelledby="e5-title e5-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="e5-title">Two features before and after scaling</title>
<desc id="e5-desc">Before scaling, total charges run to thousands while tenure runs to seventy, so any distance calculation is dominated by total charges. After scaling both features sit on the same range.</desc>
<text x="170" y="26" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-graphite)">before</text>
<text x="510" y="26" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">after standardising</text>
<line x1="60" y1="186" x2="300" y2="186" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="60" y1="186" x2="60" y2="46" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="400" y1="186" x2="640" y2="186" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="400" y1="186" x2="400" y2="46" stroke="var(--h-steel)" stroke-width="1.5"/>
<circle cx="168.6" cy="143.3" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="194.3" cy="135.6" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="281.8" cy="55.7" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="171.8" cy="115.7" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="181.9" cy="101.7" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="201.0" cy="150.2" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="104.3" cy="155.6" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="182.9" cy="101.7" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="211.2" cy="141.1" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="250.3" cy="73.5" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="82.6" cy="178.2" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="132.8" cy="134.4" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="81.8" cy="178.5" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="254.3" cy="129.1" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="226.4" cy="86.5" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="70.1" cy="179.2" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="295.7" cy="105.1" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="291.5" cy="112.8" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="216.9" cy="137.4" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="207.7" cy="134.6" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="97.8" cy="157.4" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="63.6" cy="181.0" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="186.8" cy="141.4" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="74.3" cy="174.5" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="105.6" cy="168.6" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="118.1" cy="165.2" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="67.2" cy="180.4" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="171.3" cy="143.8" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="165.7" cy="153.3" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="262.2" cy="112.1" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="184.6" cy="133.2" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="213.7" cy="82.0" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="179.9" cy="122.8" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="219.0" cy="133.4" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="169.8" cy="112.2" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="126.8" cy="161.4" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="299.4" cy="46.0" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="299.0" cy="102.5" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="261.7" cy="56.4" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="229.9" cy="112.4" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="512.0" cy="131.2" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="526.7" cy="122.5" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="576.8" cy="33.0" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="513.8" cy="100.2" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="519.6" cy="84.5" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="530.5" cy="138.9" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="475.2" cy="145.0" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="520.1" cy="84.5" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="536.3" cy="128.8" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="558.7" cy="52.9" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="462.8" cy="170.3" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="491.5" cy="121.2" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="462.3" cy="170.6" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="561.0" cy="115.3" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="545.1" cy="67.5" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="455.6" cy="171.5" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="584.7" cy="88.3" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="582.3" cy="97.0" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="539.6" cy="124.6" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="534.4" cy="121.5" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="471.5" cy="147.0" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="451.9" cy="173.4" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="522.4" cy="129.0" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="458.0" cy="166.2" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="476.0" cy="159.6" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="483.1" cy="155.7" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="454.0" cy="172.8" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="513.6" cy="131.8" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="510.3" cy="142.4" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="565.5" cy="96.2" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="521.1" cy="119.8" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="537.8" cy="62.5" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="518.5" cy="108.2" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="540.8" cy="120.1" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="512.7" cy="96.3" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="488.0" cy="151.5" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="586.8" cy="22.1" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="586.6" cy="85.4" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="565.2" cy="33.8" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<circle cx="547.0" cy="96.6" r="3.2" fill="var(--h-cherry)" opacity="0.7"/>
<text x="180" y="206" text-anchor="middle" font-size="10.5" fill="var(--h-space)">tenure, 0 to 72</text>
<text x="50" y="40" text-anchor="middle" font-size="10.5" fill="var(--h-space)">8000</text>
<text x="520" y="206" text-anchor="middle" font-size="10.5" fill="var(--h-space)">tenure, about -2 to 2</text>
<text x="388" y="40" text-anchor="middle" font-size="10.5" fill="var(--h-space)">2</text>
<text x="340" y="238" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Distance based models treat one unit of tenure as one unit of euros. Scaling makes that comparison meaningful.</text>
<text x="340" y="258" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Tree models do not care at all, because they only ever ask whether a value is above a threshold.</text>
</svg>

`tenure` runs to 72 and `TotalCharges` runs past 8000. Any model that measures
distance, or that uses gradient descent, treats those numbers as if they were on
the same scale. They are not.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Standardisation** subtracts the mean and divides by the standard deviation,
giving a column centred on zero. **Min-max** squeezes into a 0 to 1 range, which
preserves shape but is more affected by outliers.

Which models care:

| Model | Needs scaling? |
|---|---|
| Linear and logistic regression | Yes, especially with regularisation |
| Anything distance based, such as k-nearest neighbours | Yes, critically |
| Neural networks | Yes, training is much slower without it |
| Decision trees and random forests | No, thresholds are unaffected |

!!! warning "Fit on the training data only"

    Look at the two lines above. `fit_transform` on the training set,
    `transform` on the test set. The scaler learns the mean and standard
    deviation from training data alone.

    Call `fit` on the whole dataset and the test set's statistics have leaked
    into your training. Your score comes out better than it should, and the
    model performs worse than promised on anything new. This is the single most
    common way a beginner produces a result that is not real, and the next page
    is largely about it.

## Splitting

<svg viewBox="0 0 680 286" role="img" aria-labelledby="e6-title e6-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="e6-title">A stratified split keeps the class balance</title>
<desc id="e6-desc">The full dataset is about 73 percent stayed and 27 percent churned. A stratified split keeps that ratio in the training, validation and test sets. A plain random split can distort it, especially in the smallest set.</desc>
<text x="20" y="26" font-size="12" font-weight="700" fill="var(--h-graphite)">the whole dataset, 7043 customers</text>
<rect x="20" y="36" width="640" height="34" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="20" y="36" width="467" height="34" rx="6" fill="var(--h-steel)" opacity="0.4"/>
<text x="253" y="53" dy="0.36em" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-graphite)">stayed, 73 percent</text>
<rect x="487" y="36" width="173" height="34" rx="6" fill="var(--h-cherry)"/>
<text x="573" y="53" dy="0.36em" text-anchor="middle" font-size="11.5" font-weight="700" fill="#ffffff">churned, 27 percent</text>
<text x="20" y="108" font-size="12" font-weight="700" fill="var(--h-cherry)">stratified: every part keeps 73 to 27</text>
<rect x="20" y="118" width="374" height="34" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="20" y="118" width="273" height="34" rx="6" fill="var(--h-steel)" opacity="0.4"/>
<rect x="293" y="118" width="101" height="34" rx="6" fill="var(--h-cherry)"/>
<text x="207" y="166" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">train, 60%</text>
<rect x="404" y="118" width="118" height="34" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="404" y="118" width="86" height="34" rx="6" fill="var(--h-steel)" opacity="0.4"/>
<rect x="490" y="118" width="32" height="34" rx="6" fill="var(--h-cherry)"/>
<text x="463" y="166" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">validation, 20%</text>
<rect x="532" y="118" width="118" height="34" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="532" y="118" width="86" height="34" rx="6" fill="var(--h-steel)" opacity="0.4"/>
<rect x="618" y="118" width="32" height="34" rx="6" fill="var(--h-cherry)"/>
<text x="591" y="166" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">test, 20%</text>
<text x="20" y="206" font-size="12" font-weight="700" fill="var(--h-graphite)">unstratified: the smallest set drifts</text>
<rect x="20" y="216" width="374" height="34" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="20" y="216" width="280" height="34" rx="6" fill="var(--h-steel)" opacity="0.4"/>
<rect x="300" y="216" width="94" height="34" rx="6" fill="var(--h-cherry)" opacity="0.75"/>
<text x="207" y="264" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">25 percent churned</text>
<rect x="404" y="216" width="118" height="34" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="404" y="216" width="84" height="34" rx="6" fill="var(--h-steel)" opacity="0.4"/>
<rect x="488" y="216" width="34" height="34" rx="6" fill="var(--h-cherry)" opacity="0.75"/>
<text x="463" y="264" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">29 percent churned</text>
<rect x="532" y="216" width="118" height="34" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="532" y="216" width="76" height="34" rx="6" fill="var(--h-steel)" opacity="0.4"/>
<rect x="608" y="216" width="42" height="34" rx="6" fill="var(--h-cherry)" opacity="0.75"/>
<text x="591" y="264" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">36 percent churned</text>
<text x="340" y="282" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">If your test set has a different balance from reality, its score answers a question nobody asked.</text>
</svg>

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Churn"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(y_train.mean(), y_test.mean())
```

`stratify=y` keeps the 73 to 27 balance in both halves. Without it the split is
random, and with a minority class the smaller set can drift several percentage
points, which makes your test score measure a slightly different problem from
the one you trained on.

`random_state` makes the split reproducible, so that a change in your score
comes from a change you made rather than from a different shuffle.

Split **before** you scale, before you impute, and before anything else that
learns from the data.

## The whole thing in one screen

```python
df = pd.read_csv("Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df["SeniorCitizen"] = df["SeniorCitizen"].astype("category")
df = df.drop_duplicates().drop(columns=["customerID"])
df["tenure_band"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 100], labels=["new", "growing", "established", "loyal"])
df["Churn"] = (df["Churn"] == "Yes").astype(int)
X = pd.get_dummies(df.drop(columns=["Churn"]), drop_first=True)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
```

Eleven lines. Every one of them is a decision somebody could disagree with, which
is why the last step of preparation is writing them down.

!!! tip "Keep a decisions file"

    One short note per choice: what you did, and why. Filled `TotalCharges`
    blanks with 0 because all eleven have tenure 0. Dropped `customerID` as an
    identifier. Binned tenure at 12, 24 and 48 months because those are contract
    boundaries.

    In three months you will not remember, and neither will the person reviewing
    your work. This is the same habit as the data statement from your first
    project.

## When it goes wrong

??? note "`describe()` shows only three columns"

    It reports numeric columns by default. Use `df.describe(include="object")`
    or `include="all"` for the rest.

??? note "A money column has dtype object"

    Something in it is not a number. `pd.to_numeric(col, errors="coerce")` then
    `isna()` shows you exactly which rows, and those rows usually explain
    themselves.

??? note "`isna().sum()` says zero but values are clearly missing"

    Empty strings, single spaces and the words unknown or n/a are not missing
    values as far as pandas is concerned. Pass `na_values` to `read_csv` or
    convert afterwards.

??? note "My test accuracy is suspiciously high"

    Check for an identifier column, for a feature recorded after the outcome,
    and for a scaler fitted before the split. All three produce excellent scores
    that mean nothing.

??? note "Every feature is statistically significant"

    With 7043 rows that is expected. Look at the size of the difference, not
    only the p-value.

??? note "`get_dummies` produced hundreds of columns"

    One of your categorical columns has very many distinct values, often an
    identifier that should have been dropped. Check `df[col].nunique()` before
    encoding.

## Check yourself

Use the churn file for all of these.

1. Report the shape, the dtype of every column, and which columns are
   categorical wearing numeric clothes.
2. Convert `TotalCharges` properly, show that exactly eleven values become
   missing, and demonstrate what those rows have in common.
3. Justify in one sentence why 0 is the right fill for those rows and the median
   is not.
4. Produce the churn rate by `Contract` as proportions, and state the baseline
   accuracy any model must beat.
5. Run a chi-square test for two categorical features and a Welch's t-test for
   `MonthlyCharges`. Report both the p-value and the size of the difference.
6. Bin tenure into four bands of your own choosing and defend the boundaries.
7. Encode, scale and split, in the correct order, and prove the class balance is
   preserved in both halves.
8. Write your decisions file. Aim for one line per decision.

## Quick reference

| Task | Code |
|---|---|
| Full summary | `df.describe(include="all")` |
| Force numeric | `pd.to_numeric(col, errors="coerce")` |
| Category from number | `df["SeniorCitizen"].astype("category")` |
| Missing map | `sns.heatmap(df.isna(), cbar=False)` |
| Duplicate rows, duplicate ids | `df.duplicated().sum()`, `df["id"].duplicated().sum()` |
| Target balance | `df["Churn"].value_counts(normalize=True)` |
| Rates by category | `pd.crosstab(df["a"], df["b"], normalize="index")` |
| Chi-square | `chi2_contingency(pd.crosstab(a, b))` |
| Welch t-test | `ttest_ind(g1, g2, equal_var=False)` |
| Bin by boundaries | `pd.cut(col, bins=[...], labels=[...])` |
| Bin by quantiles | `pd.qcut(col, q=4)` |
| One-hot | `pd.get_dummies(df, columns=[...], drop_first=True)` |
| Scale | `StandardScaler().fit(X_train)` then `.transform(X_test)` |
| Stratified split | `train_test_split(X, y, stratify=y, random_state=42)` |

## Questions to sit with

1. Eleven rows had a blank total charge and you decided it meant zero. What
   evidence supported that, and what would have changed your mind?
2. A model that predicts nobody churns is 73.5 percent accurate. Is that model
   useless, and does your answer depend on what the company does with the
   prediction?
3. Binning tenure throws away information on purpose. What exactly did you give
   up, and what did you buy with it?
4. Every choice on this page could have gone differently. If two analysts
   produce different churn rates from the same file, is one of them wrong?
5. Scaling must be fitted on training data only. Why is fitting on everything so
   tempting, and what does that tell you about how leakage usually happens?

## Next

You have a clean, encoded, scaled, split dataset. The next page is about the
ways that process quietly goes wrong, and how to build it so it cannot.

[Data Preprocessing](data-preprocessing.md){ .h-button }

Worth your time: the
[scikit-learn preprocessing guide](https://scikit-learn.org/stable/modules/preprocessing.html)
is the reference for the last three sections, and the
[Telco churn dataset page](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
has community notebooks worth reading after you have formed your own view.

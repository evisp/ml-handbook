# Data Preprocessing

<span class="badge badge--time">85 min</span>
<span class="badge badge--level">Real data</span>
<span class="badge">The capstone of this group</span>

Everything in this trimester so far has been one stage of a single process. This
page puts the stages in order, names them properly, and covers the two things
that turn a working notebook into a pipeline somebody else can trust.

## Why this matters

The next module is about models, and you will discover quickly that the model is
rarely the interesting part. Choosing between a random forest and a logistic
regression might move your score a couple of points. Fixing how your data was
prepared moves it far more, and it is the difference between a result that holds
up and one that quietly does not.

This page is also your project brief. The questions at the end are the ones you
answer about your own dataset, in writing, before you build anything.

## The whole pipeline

<svg viewBox="0 0 680 330" role="img" aria-labelledby="d1-title d1-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="d1-title">The whole data pipeline</title>
<desc id="d1-desc">Collect, store the raw data, clean it, explore it, engineer features, split it, transform it, and store the result. Splitting comes before transforming.</desc>
<defs><marker id="d1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<rect x="5" y="40" width="152" height="74" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="81" y="72" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">collect</text>
<text x="81" y="92" text-anchor="middle" font-size="10" fill="var(--h-graphite)" opacity="0.9">scraping, APIs, files</text>
<line x1="163" y1="77" x2="175" y2="77" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#d1)"/>
<rect x="175" y="40" width="152" height="74" rx="10" fill="var(--h-cherry)" stroke="var(--h-cherry)"/>
<text x="251" y="72" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">store raw</text>
<text x="251" y="92" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.9">never edited again</text>
<line x1="333" y1="77" x2="345" y2="77" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#d1)"/>
<rect x="345" y="40" width="152" height="74" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="421" y="72" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">clean</text>
<text x="421" y="92" text-anchor="middle" font-size="10" fill="var(--h-graphite)" opacity="0.9">types, missing, duplicates</text>
<line x1="503" y1="77" x2="515" y2="77" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#d1)"/>
<rect x="515" y="40" width="152" height="74" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="591" y="72" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">explore</text>
<text x="591" y="92" text-anchor="middle" font-size="10" fill="var(--h-graphite)" opacity="0.9">EDA, plots, tests</text>
<rect x="5" y="168" width="152" height="74" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="81" y="200" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">engineer</text>
<text x="81" y="220" text-anchor="middle" font-size="10" fill="var(--h-graphite)" opacity="0.9">new features, binning</text>
<line x1="163" y1="205" x2="175" y2="205" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#d1)"/>
<rect x="175" y="168" width="152" height="74" rx="10" fill="var(--h-cherry)" stroke="var(--h-cherry)"/>
<text x="251" y="200" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">split</text>
<text x="251" y="220" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.9">before you transform</text>
<line x1="333" y1="205" x2="345" y2="205" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#d1)"/>
<rect x="345" y="168" width="152" height="74" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="421" y="200" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">transform</text>
<text x="421" y="220" text-anchor="middle" font-size="10" fill="var(--h-graphite)" opacity="0.9">encode, scale, impute</text>
<line x1="503" y1="205" x2="515" y2="205" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#d1)"/>
<rect x="515" y="168" width="152" height="74" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="591" y="200" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">store ready</text>
<text x="591" y="220" text-anchor="middle" font-size="10" fill="var(--h-graphite)" opacity="0.9">versioned, reproducible</text>
<polyline points="641,114 660,114 660,128 20,128 20,164 5,164" fill="none" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#d1)"/>
<polyline points="641,242 660,242 660,268 300,268" fill="none" stroke="var(--h-steel)" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#d1)"/>
<rect x="120" y="252" width="180" height="32" rx="8" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="210" y="268" dy="0.36em" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">a model, next module</text>
<text x="340" y="26" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-graphite)">everything in this trimester, in the order it happens</text>
<text x="340" y="308" text-anchor="middle" font-size="12" fill="var(--h-graphite)">The two red boxes are the ones people get wrong. Keep the raw data untouched, and split before you transform.</text>
</svg>

Two boxes are marked because they are the ones people get wrong.

**Store the raw data and never edit it.** Your cleaning script reads the raw file
and writes a new one. When you discover in week three that a decision was wrong,
you change the script and rerun it. If you edited the original, you cannot.

**Split before you transform.** Everything that learns something from the data,
a scaler, an imputer, an encoder, must learn it from the training set alone. This
has a name and it gets its own section.

## The vocabulary, untangled

These words appear in job adverts as if they were interchangeable. They are not,
and knowing the difference is worth something in an interview.

<svg viewBox="0 0 680 300" role="img" aria-labelledby="d2-title d2-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="d2-title">How the words overlap</title>
<desc id="d2-desc">Data engineering is the widest term and covers moving and storing data. Wrangling covers cleansing and transformation. Feature engineering creates new columns. EDA runs alongside all of them rather than being a stage.</desc>
<rect x="20" y="34" width="500" height="200" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="38" y="56" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">data engineering</text>
<text x="38" y="72" font-size="10.5" fill="var(--h-space)">getting data from where it is to where you need it, and keeping it there</text>
<rect x="40" y="84" width="300" height="130" rx="10" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="56" y="104" font-size="12.5" font-weight="700" fill="var(--h-cherry)">wrangling, or munging</text>
<rect x="56" y="114" width="132" height="86" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="122" y="136" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--md-default-fg-color)">cleansing</text>
<text x="122" y="156" text-anchor="middle" font-size="10" fill="var(--h-graphite)">fix what is wrong:</text>
<text x="122" y="170" text-anchor="middle" font-size="10" fill="var(--h-graphite)">types, duplicates,</text>
<text x="122" y="184" text-anchor="middle" font-size="10" fill="var(--h-graphite)">missing, errors</text>
<rect x="196" y="114" width="132" height="86" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="262" y="136" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--md-default-fg-color)">transformation</text>
<text x="262" y="156" text-anchor="middle" font-size="10" fill="var(--h-graphite)">change the shape:</text>
<text x="262" y="170" text-anchor="middle" font-size="10" fill="var(--h-graphite)">encode, scale,</text>
<text x="262" y="184" text-anchor="middle" font-size="10" fill="var(--h-graphite)">reshape, aggregate</text>
<rect x="356" y="84" width="144" height="130" rx="10" fill="var(--h-cherry)"/>
<text x="428" y="108" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">feature</text>
<text x="428" y="126" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">engineering</text>
<text x="428" y="152" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.9">invent columns that</text>
<text x="428" y="166" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.9">were not there:</text>
<text x="428" y="180" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.9">ratios, counts, bands</text>
<rect x="540" y="34" width="120" height="200" rx="12" fill="var(--h-surface)" stroke="var(--h-cherry-line)" stroke-dasharray="6 4"/>
<text x="600" y="110" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">EDA</text>
<text x="600" y="134" text-anchor="middle" font-size="10" fill="var(--h-graphite)">not a stage.</text>
<text x="600" y="150" text-anchor="middle" font-size="10" fill="var(--h-graphite)">It runs beside</text>
<text x="600" y="166" text-anchor="middle" font-size="10" fill="var(--h-graphite)">all of them,</text>
<text x="600" y="182" text-anchor="middle" font-size="10" fill="var(--h-graphite)">the whole time</text>
<text x="340" y="266" text-anchor="middle" font-size="12" fill="var(--h-graphite)">These words are used loosely and interchangeably in job adverts. The distinctions that matter:</text>
<text x="340" y="286" text-anchor="middle" font-size="12" fill="var(--h-graphite)">cleansing fixes what is wrong, transformation changes the form, feature engineering adds what was not there.</text>
</svg>

**Data engineering** is the widest. Moving data from where it is to where you
need it, and keeping it there: pipelines, storage, scheduling, reliability. It is
a job title as well as an activity.

**Data wrangling**, also called munging, is getting a messy dataset into a usable
shape. It covers the next two.

**Data cleansing** fixes what is wrong. Wrong types, duplicates, impossible
values, missing entries. You did this on the churn file.

**Data transformation** changes the form without changing the meaning. Encoding,
scaling, reshaping, aggregating.

**Feature engineering** adds what was not there. A ratio, a count, a band, a
flag. This is the part with the most room for skill, because it is where domain
knowledge enters.

**EDA** is not a stage at all. It runs alongside every one of the others,
whenever you want to know what you are actually holding.

!!! tip "One sentence that separates them"

    Cleansing fixes errors. Transformation changes form. Feature engineering
    creates information that was implicit in the data and is now explicit.

## Where your data comes from

The first questions in your brief are about sources, and they matter more than
they look.

**One source or several?** Combining sources means joining them, and every join
is a chance to silently lose or duplicate rows. You saw this in the Pandas page:
check the row count before and after, every time.

**Do the sources agree?** Two systems will spell the same city three ways, store
dates in two formats, and disagree about the same customer's details. Decide
which one wins, and write that decision down.

**How fresh is each one?** Joining a table refreshed daily to one refreshed
quarterly means three months of your rows are wrong in a way nothing will warn
you about.

```python
merged = customers.merge(usage, on="customer_id", how="left", validate="one_to_one")
print(len(customers), len(merged), merged["usage_minutes"].isna().sum())
```

`validate` raises if the relationship is not what you claimed, which is far
better than discovering it three steps later.

## Formats and storage

Where you keep data is part of the design, not an afterthought.

| Format | Good for | Watch out for |
|---|---|---|
| CSV | Sharing, reading by eye, universal | Loses dtypes, slow, no compression |
| JSON | Nested structures, API responses | Awkward as a table, large |
| Parquet | Working storage for anything sizeable | Not human readable |
| SQL | Data that is queried and updated | Needs a server, needs SQL |

```python
df.to_parquet("data/processed/churn_clean.parquet")
df = pd.read_parquet("data/processed/churn_clean.parquet")
```

Parquet keeps your dtypes, compresses well and loads far faster. After the
`TotalCharges` episode, a format that remembers a column was numeric is worth
having.

A layout worth adopting now:

```
data/
  raw/           never edited, exactly as collected
  interim/       partly processed, safe to delete
  processed/     model ready, produced by a script
```

With `data/raw/` in `.gitignore` if the files are large, and a note in the README
saying where they came from.

## Hypotheses before you look

Your brief asks whether you have preexisting hypotheses. Write them down before
you start exploring, and the reason is not tidiness.

If you explore first and form hypotheses afterwards, you will find patterns,
because with enough columns some will look interesting by chance. You met this
on the Probability page. Twenty features tested at the usual threshold gives you
about one false finding for free.

```
H1: month-to-month customers churn more than contract customers
H2: churn is highest in the first year and falls after that
H3: customers with more services churn less
```

Three sentences, written first, and then tested. Anything else you find during
exploration is still worth having, but label it as exploratory rather than
confirmed, because it was not a prediction.

## Sparse, dense, missing and extreme

**Sparse** data is mostly zeros or mostly empty. One-hot encoding a column with
two hundred categories produces a sparse matrix, and that is normal. Text data is
extremely sparse. If your matrix is mostly zeros, some models handle it
gracefully and others fall over.

**Missing** you have covered twice. The rule that carries forward: ask why
before you ask how, and record what you chose.

**Outliers** are new, and the mistake is treating them as noise by default.

```python
q1, q3 = df["MonthlyCharges"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = df[(df["MonthlyCharges"] < low) | (df["MonthlyCharges"] > high)]
print(len(outliers))
```

That rule is the box plot whiskers you already know. What to do with what it
finds depends entirely on which of three things you are looking at.

**An error.** An age of 200, a negative price. Fix it or remove it.

**A rare but real value.** A genuinely enormous customer. Keep it. It is the kind
of case your model most needs to handle, and deleting it is choosing to be wrong
about the customers who matter most.

**A different population.** Business accounts mixed in with domestic ones. The
right answer is usually a separate model, or a column saying which is which.

!!! warning "Deleting outliers improves your metrics and may ruin your model"

    Removing the hard cases always makes the score go up. That is exactly why it
    is tempting, and exactly why it needs justifying in writing every time.

## Splitting, properly

You know stratified splitting. There are two more cases, and using the wrong one
is leakage that your test score will not reveal.

<svg viewBox="0 0 680 300" role="img" aria-labelledby="d4-title d4-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="d4-title">Four ways to split, and when each is right</title>
<desc id="d4-desc">Random splitting suits independent rows. Stratified splitting preserves class balance. Time based splitting trains on the past and tests on the future. Group splitting keeps all rows from one person on the same side.</desc>
<rect x="5" y="20" width="330" height="126" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="25" y="46" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">random</text>
<text x="25" y="64" font-size="10.5" fill="var(--h-graphite)">rows are independent of each other</text>
<rect x="25" y="78" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="49" y="78" width="20" height="26" rx="4" fill="var(--h-steel)" opacity="0.45"/>
<rect x="73" y="78" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="97" y="78" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="121" y="78" width="20" height="26" rx="4" fill="var(--h-steel)" opacity="0.45"/>
<rect x="145" y="78" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="169" y="78" width="20" height="26" rx="4" fill="var(--h-steel)" opacity="0.45"/>
<rect x="193" y="78" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="217" y="78" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="241" y="78" width="20" height="26" rx="4" fill="var(--h-steel)" opacity="0.45"/>
<rect x="265" y="78" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="289" y="78" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<text x="169" y="126" text-anchor="middle" font-size="10" fill="var(--h-graphite)">red is the minority class</text>
<rect x="345" y="20" width="330" height="126" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="365" y="46" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">stratified</text>
<text x="365" y="64" font-size="10.5" fill="var(--h-graphite)">one class is rare</text>
<rect x="365" y="78" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="389" y="78" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="413" y="78" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="437" y="78" width="20" height="26" rx="4" fill="var(--h-steel)" opacity="0.45"/>
<rect x="461" y="78" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="485" y="78" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="509" y="78" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="533" y="78" width="20" height="26" rx="4" fill="var(--h-steel)" opacity="0.45"/>
<rect x="557" y="78" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="581" y="78" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="605" y="78" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="629" y="78" width="20" height="26" rx="4" fill="var(--h-steel)" opacity="0.45"/>
<text x="509" y="126" text-anchor="middle" font-size="10" fill="var(--h-graphite)">red is the minority class</text>
<rect x="5" y="160" width="330" height="126" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="25" y="186" font-size="13" font-weight="700" fill="var(--h-cherry)">by time</text>
<text x="25" y="204" font-size="10.5" fill="var(--h-graphite)">you will predict the future</text>
<rect x="25" y="218" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="49" y="218" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="73" y="218" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="97" y="218" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="121" y="218" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="145" y="218" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="169" y="218" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="193" y="218" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="217" y="218" width="20" height="26" rx="4" fill="var(--h-steel)" opacity="0.45"/>
<rect x="241" y="218" width="20" height="26" rx="4" fill="var(--h-steel)" opacity="0.45"/>
<rect x="265" y="218" width="20" height="26" rx="4" fill="var(--h-steel)" opacity="0.45"/>
<rect x="289" y="218" width="20" height="26" rx="4" fill="var(--h-steel)" opacity="0.45"/>
<line x1="217" y1="212" x2="217" y2="250" stroke="var(--md-default-fg-color)" stroke-width="2"/>
<text x="121" y="266" text-anchor="middle" font-size="10" fill="var(--h-graphite)">past, train</text>
<text x="265" y="266" text-anchor="middle" font-size="10" fill="var(--h-graphite)">future, test</text>
<rect x="345" y="160" width="330" height="126" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="365" y="186" font-size="13" font-weight="700" fill="var(--h-cherry)">by group</text>
<text x="365" y="204" font-size="10.5" fill="var(--h-graphite)">several rows share a person</text>
<rect x="365" y="218" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="389" y="218" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="413" y="218" width="20" height="26" rx="4" fill="var(--h-steel)" opacity="0.45"/>
<rect x="437" y="218" width="20" height="26" rx="4" fill="var(--h-steel)" opacity="0.45"/>
<rect x="461" y="218" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="485" y="218" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="509" y="218" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="533" y="218" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="557" y="218" width="20" height="26" rx="4" fill="var(--h-steel)" opacity="0.45"/>
<rect x="581" y="218" width="20" height="26" rx="4" fill="var(--h-steel)" opacity="0.45"/>
<rect x="605" y="218" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="629" y="218" width="20" height="26" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<text x="389" y="264" text-anchor="middle" font-size="9.5" fill="var(--h-space)">A</text>
<text x="437" y="264" text-anchor="middle" font-size="9.5" fill="var(--h-space)">B</text>
<text x="509" y="264" text-anchor="middle" font-size="9.5" fill="var(--h-space)">C</text>
<text x="581" y="264" text-anchor="middle" font-size="9.5" fill="var(--h-space)">D</text>
<text x="629" y="264" text-anchor="middle" font-size="9.5" fill="var(--h-space)">E</text>
<text x="340" y="288" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Splitting randomly when time or grouping matters is leakage, and it is invisible in your score until deployment.</text>
</svg>

**Random** is fine when rows are independent of each other.

**Stratified** preserves class balance, which matters whenever one class is rare.

**By time** is required whenever you will predict the future. Train on the past,
test on what came after. Split a time series randomly and your model learns from
next month to predict this month, which it will never be able to do in
production.

```python
from sklearn.model_selection import TimeSeriesSplit

for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(X):
    pass
```

**By group** is required when several rows belong to the same entity. If one
customer has twelve monthly records and six land in train and six in test, your
model has effectively seen the test customers.

```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=df["customer_id"]))
```

Ask one question of every dataset: **could a row in my test set have been
influenced by a row in my training set?** If yes, a random split is wrong.

## Leakage

This is the most important idea on the page.

Leakage is any information reaching your training process that would not be
available when the model is used for real. It produces excellent scores and
useless models, and it is hard to spot precisely because nothing goes wrong until
it matters.

<svg viewBox="0 0 680 286" role="img" aria-labelledby="d3-title d3-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="d3-title">Where leakage comes from</title>
<desc id="d3-desc">Fitting a transformer on the whole dataset lets statistics from the test set reach the training data. Fitting on the training set alone keeps the test set genuinely unseen.</desc>
<defs><marker id="d3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<text x="170" y="26" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">fit on everything: leaked</text>
<rect x="20" y="38" width="300" height="34" rx="6" fill="var(--h-cherry)" opacity="0.35"/>
<text x="170" y="55" dy="0.36em" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-graphite)">the whole dataset</text>
<line x1="170" y1="76" x2="170" y2="100" stroke="var(--h-cherry)" stroke-width="2" marker-end="url(#d3)"/>
<rect x="60" y="104" width="220" height="30" rx="6" fill="var(--h-cherry)"/>
<text x="170" y="119" dy="0.36em" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">scaler learns mean and sd</text>
<line x1="170" y1="138" x2="170" y2="160" stroke="var(--h-cherry)" stroke-width="2" marker-end="url(#d3)"/>
<rect x="20" y="164" width="146" height="34" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="93" y="181" dy="0.36em" text-anchor="middle" font-size="11" fill="var(--md-default-fg-color)">train</text>
<rect x="174" y="164" width="146" height="34" rx="6" fill="var(--h-cherry-wash)" stroke="var(--h-cherry)" stroke-width="2"/>
<text x="247" y="181" dy="0.36em" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-cherry)">test, already seen</text>
<text x="170" y="222" text-anchor="middle" font-size="11" fill="var(--h-graphite)">the test set helped choose the numbers</text>
<text x="170" y="238" text-anchor="middle" font-size="11" fill="var(--h-graphite)">used to scale the training set</text>
<text x="510" y="26" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">split first: clean</text>
<rect x="360" y="38" width="300" height="34" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="510" y="55" dy="0.36em" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-graphite)">the whole dataset</text>
<line x1="435" y1="76" x2="435" y2="98" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#d3)"/>
<line x1="585" y1="76" x2="585" y2="158" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#d3)"/>
<rect x="362" y="102" width="146" height="30" rx="6" fill="var(--h-cherry)"/>
<text x="435" y="117" dy="0.36em" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">scaler fits on train</text>
<line x1="435" y1="136" x2="435" y2="160" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#d3)"/>
<rect x="362" y="164" width="146" height="34" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="435" y="181" dy="0.36em" text-anchor="middle" font-size="11" fill="var(--md-default-fg-color)">train</text>
<rect x="514" y="164" width="146" height="34" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="587" y="181" dy="0.36em" text-anchor="middle" font-size="11" fill="var(--md-default-fg-color)">test, untouched</text>
<text x="510" y="222" text-anchor="middle" font-size="11" fill="var(--h-graphite)">the test set is transformed using</text>
<text x="510" y="238" text-anchor="middle" font-size="11" fill="var(--h-graphite)">numbers it had no part in choosing</text>
<text x="340" y="272" text-anchor="middle" font-size="12" fill="var(--h-graphite)">The score on the left is better and means less. That is what makes leakage dangerous rather than merely wrong.</text>
</svg>

Four common sources:

**Fitting a transformer before splitting.** The scaler learns the mean of the
whole dataset, including the test set.

**A feature recorded after the outcome.** `cancellation_reason` predicts churn
perfectly and is only filled in for people who already left.

**A proxy for the answer.** Not the label itself, but something created because
of it. An internal risk flag set by the team who also decided the outcome.

**Time travel.** Any feature computed from data that did not exist yet at the
moment you are predicting.

!!! warning "The test that catches most of it"

    For every feature, ask: **would I have this value, with this content, at the
    moment I need to make the prediction?**

    If the answer is no, or if you have to think about it, it is a leak.

    The other signal is a model that is too good. If your accuracy jumps to 99
    percent on a hard problem, do not celebrate. Go and find out which column did
    it.

## Making leakage structurally impossible

You can remember to fit on the training set only. You will forget once, and once
is enough. The better answer is to build the preprocessing so that forgetting is
not possible.

<svg viewBox="0 0 680 300" role="img" aria-labelledby="d5-title d5-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="d5-title">A preprocessing pipeline as one object</title>
<desc id="d5-desc">A column transformer sends numeric columns through imputation and scaling, and categorical columns through imputation and one-hot encoding. The results join and feed the model. The whole thing is fitted once, on the training data.</desc>
<defs><marker id="d5" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<rect x="14" y="40" width="652" height="176" rx="12" fill="var(--h-surface)" stroke="var(--h-cherry)" stroke-width="2" stroke-dasharray="6 4"/>
<text x="30" y="32" font-size="12.5" font-weight="700" fill="var(--h-cherry)">one Pipeline object, fitted once on the training set</text>
<rect x="30" y="64" width="110" height="128" rx="9" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="85" y="122" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">raw</text>
<text x="85" y="140" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">columns</text>
<line x1="144" y1="98" x2="176" y2="98" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#d5)"/>
<line x1="144" y1="158" x2="176" y2="158" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#d5)"/>
<rect x="180" y="72" width="200" height="54" rx="8" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="280" y="92" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">numeric columns</text>
<text x="280" y="112" text-anchor="middle" font-size="10" fill="var(--h-graphite)">impute median, then scale</text>
<rect x="180" y="132" width="200" height="54" rx="8" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="280" y="152" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">categorical columns</text>
<text x="280" y="172" text-anchor="middle" font-size="10" fill="var(--h-graphite)">impute most frequent, one-hot</text>
<line x1="384" y1="98" x2="416" y2="120" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#d5)"/>
<line x1="384" y1="158" x2="416" y2="136" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#d5)"/>
<rect x="420" y="100" width="110" height="56" rx="9" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="475" y="122" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--md-default-fg-color)">joined</text>
<text x="475" y="140" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--md-default-fg-color)">matrix</text>
<line x1="534" y1="128" x2="560" y2="128" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#d5)"/>
<rect x="564" y="100" width="90" height="56" rx="9" fill="var(--h-cherry)"/>
<text x="609" y="128" dy="0.36em" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">model</text>
<text x="340" y="248" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Because the whole thing is one object, fitting it on the training fold and applying it to the test fold is automatic.</text>
<text x="340" y="268" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Cross-validation refits it on every fold. Leakage stops being something you have to remember to avoid.</text>
</svg>

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

numeric = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical = ["Contract", "InternetService", "PaymentMethod"]
numeric_steps = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical_steps = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer([
    ("num", numeric_steps, numeric),
    ("cat", categorical_steps, categorical),
])
model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

The whole thing is one object. `fit` learns every statistic from the training
data, and `score` applies those same statistics to the test data. There is no
line you could forget.

It also makes cross-validation correct for free, because the pipeline is refitted
inside every fold.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
print(scores.mean(), scores.std())
```

`handle_unknown="ignore"` matters. A category that appears only in the test set
would otherwise crash the encoder, and in production it would crash your service.

!!! ml "ML connection"

    In the next module you will compare several models. Doing that fairly means
    every one of them sees identically prepared data, which is exactly what a
    pipeline guarantees. Swap the last step and everything before it stays fixed.

## Bias

Your brief asks how you will ensure the dataset is unbiased. The honest answer is
that you cannot, so the real question is which biases you have and whether you
have looked.

**Sampling bias.** Who is in your data and who is not. A churn model built on
customers who answered a survey describes people who answer surveys.

**Historical bias.** The data faithfully records a world that was unfair. Nothing
is wrong with the collection and the model still learns the unfairness.

**Measurement bias.** Your proxy is not the thing you care about. Using support
tickets as a measure of dissatisfaction measures who complains.

**Aggregation bias.** One model for groups that behave differently. It ends up
fitting the largest group and serving the others badly.

The practical check is to stop looking at the average.

<svg viewBox="0 0 680 286" role="img" aria-labelledby="d6-title d6-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="d6-title">One overall score hiding four different ones</title>
<desc id="d6-desc">A model with 88 percent accuracy overall scores 93, 90, 78 and 61 percent on four subgroups. The average conceals the group it serves worst.</desc>
<text x="30" y="30" font-size="12.5" font-weight="700" fill="var(--h-graphite)">the headline number</text>
<rect x="30" y="42" width="600" height="38" rx="8" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="30" y="42" width="528" height="38" rx="8" fill="var(--h-steel)" opacity="0.4"/>
<text x="294" y="61" dy="0.36em" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-graphite)">88 percent accurate</text>
<text x="30" y="118" font-size="12.5" font-weight="700" fill="var(--h-cherry)">the same model, by group</text>
<rect x="30" y="130" width="600" height="26" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="30" y="130" width="558" height="26" rx="6" fill="var(--h-cherry)" opacity="0.4"/>
<text x="44" y="143" dy="0.36em" font-size="11" font-weight="700" fill="var(--h-graphite)">group A</text>
<text x="578" y="143" dy="0.36em" font-size="11" font-weight="700" fill="var(--h-graphite)">93</text>
<rect x="30" y="164" width="600" height="26" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="30" y="164" width="540" height="26" rx="6" fill="var(--h-cherry)" opacity="0.4"/>
<text x="44" y="177" dy="0.36em" font-size="11" font-weight="700" fill="var(--h-graphite)">group B</text>
<text x="560" y="177" dy="0.36em" font-size="11" font-weight="700" fill="var(--h-graphite)">90</text>
<rect x="30" y="198" width="600" height="26" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="30" y="198" width="468" height="26" rx="6" fill="var(--h-cherry)" opacity="0.9"/>
<text x="44" y="211" dy="0.36em" font-size="11" font-weight="700" fill="#ffffff">group C</text>
<text x="488" y="211" dy="0.36em" font-size="11" font-weight="700" fill="var(--h-cherry)">78</text>
<rect x="30" y="232" width="600" height="26" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="30" y="232" width="366" height="26" rx="6" fill="var(--h-cherry)" opacity="0.9"/>
<text x="44" y="245" dy="0.36em" font-size="11" font-weight="700" fill="#ffffff">group D</text>
<text x="386" y="245" dy="0.36em" font-size="11" font-weight="700" fill="var(--h-cherry)">61</text>
<text x="340" y="292" text-anchor="middle" font-size="0" fill="var(--h-graphite)"> </text>
<text x="340" y="278" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Nothing in the headline number tells you group D exists. You only find it if you go looking.</text>
</svg>

```python
for group, rows in df.groupby("Contract"):
    print(group, len(rows), rows["Churn"].mean())
```

Do this for every group you can identify, on the data before modelling and on the
predictions afterwards. You are looking for groups that are small, groups whose
rates are very different, and groups your model gets wrong more often.

This is the fourth time bias has appeared in this handbook. That is deliberate.
It is not a topic you cover once.

## Reproducibility

If you cannot rerun it and get the same answer, you do not have a result, you
have an anecdote.

**Seed everything.** `random_state=42` on every split and every model.

**Script it.** A notebook where cells were run out of order is not reproducible
even by you. When something works, move it into a `.py` file that runs top to
bottom.

**Version the data, not just the code.** At minimum, a filename with a date and a
line in your README saying where it came from and when.

**Write a data card.** Half a page: source, collection date, row and column
counts, known problems, every cleaning decision, who is represented and who is
not.

That last one is the data statement from your first project, grown up. It is also
the document that makes your portfolio project look professional rather than
academic, because it is what a real team would require.

## Your project brief

These are the questions to answer in writing, before you build. Most of them have
a section above them.

| Question | Where to look |
|---|---|
| What are my sources, and do I need to combine them? | Where your data comes from |
| What format is it in, and what will I store it as? | Formats and storage |
| What features exist, and how will I explore them? | EDA, and the previous page |
| What do I expect to find, and how will I test it? | Hypotheses before you look |
| Is it sparse or dense? What about missing values and outliers? | Sparse, dense, missing and extreme |
| How will I split it? | Splitting, properly |
| How will I check for bias? | Bias |
| Which features do I expect to use? | Feature engineering, previous page |
| What types am I handling, and how will I transform them? | Encoding and scaling, previous page |
| Where and how will I store the result? | Formats and storage, Reproducibility |

Answer them in a document, not in your head. The answers will change as you work,
and the changes are worth recording too.

## When it goes wrong

??? note "My cross-validation score is much better than my test score"

    Something leaked into the cross-validation. The usual cause is preprocessing
    done before the split rather than inside a pipeline.

??? note "Accuracy is 99 percent on a hard problem"

    Look for a leaking feature. Print the feature importances or coefficients,
    find the one doing all the work, and ask whether you would have it at
    prediction time.

??? note "`ValueError: Found unknown categories` at prediction time"

    A category appeared that was not in the training data. Use
    `handle_unknown="ignore"` in the encoder, and expect this to happen in
    production.

??? note "The model works in the notebook and fails on new data"

    Either the new data has a different shape, which a pipeline prevents, or you
    transformed the training data in a way you cannot reproduce, which a pipeline
    also prevents.

??? note "Rerunning gives a different answer"

    An unseeded split, an unseeded model, or a notebook run out of order. Set
    `random_state` everywhere and rerun from a clean kernel.

??? note "My joined table has more rows than either input"

    The key is not unique on one side. Check with `duplicated().sum()` before
    merging, and use `validate=` to make pandas complain.

## Check yourself

Use the churn data, or your own project data if you have it.

1. Write the three-folder data layout and move a dataset into it. Nothing should
   ever write to `raw/`.
2. Write three hypotheses about your data before exploring it. Then test them and
   report which survived.
3. Find the outliers in a numeric column with the IQR rule. Classify each one as
   an error, a rare real value, or a different population, and say what you did.
4. Build a full `ColumnTransformer` and `Pipeline` for your dataset and fit it in
   one call.
5. Deliberately introduce leakage by fitting a scaler before splitting. Record
   both scores and explain the gap.
6. Report your target rate for every subgroup you can identify. Name the smallest
   group and say what that means for any model you build.
7. Write the data card. Half a page, and no longer.

## Quick reference

| Task | Code |
|---|---|
| Safe join | `a.merge(b, on="k", how="left", validate="one_to_one")` |
| Fast typed storage | `df.to_parquet(path)`, `pd.read_parquet(path)` |
| Outlier bounds | `q1 - 1.5 * iqr`, `q3 + 1.5 * iqr` |
| Stratified split | `train_test_split(X, y, stratify=y, random_state=42)` |
| Time split | `TimeSeriesSplit(n_splits=5)` |
| Group split | `GroupShuffleSplit(test_size=0.2)` |
| Column routing | `ColumnTransformer([("num", steps, cols), ...])` |
| Whole pipeline | `Pipeline([("prep", preprocess), ("clf", model)])` |
| Honest validation | `cross_val_score(pipeline, X_train, y_train, cv=5)` |
| Unseen categories | `OneHotEncoder(handle_unknown="ignore")` |
| Subgroup rates | `df.groupby("col")["target"].mean()` |

## Questions to sit with

1. Leakage produces a better score and a worse model. What does that say about
   using the score as your only guide to whether the work is going well?
2. You can remember to fit on the training set only, or you can use a pipeline so
   that forgetting is impossible. Where else in your work would that distinction
   be worth applying?
3. Deleting outliers raises your accuracy. Under what circumstances is that an
   improvement, and how would you tell the difference?
4. You cannot make a dataset unbiased. Given that, what is the honest thing to
   put in a report, and what would you refuse to claim?
5. Somebody else has your code, your data and your seed. Can they reproduce your
   result exactly? If not, what is missing, and does it matter?

## Next

That completes Real data, and with it the data half of the trimester. You can
collect it, clean it, understand it, prepare it and store it so somebody else can
rerun your work.

The next module is where a model finally sees the table you built. Almost
everything that will go wrong there is something on this page.

[Back to the group overview](index.md){ .h-button }

Worth your time: the
[scikit-learn pipeline guide](https://scikit-learn.org/stable/modules/compose.html)
is the reference for the pipeline section, and
[Datasheets for Datasets](https://arxiv.org/abs/1803.09010) is the paper that
introduced the idea of the data card.

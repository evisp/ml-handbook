# Pandas

<span class="badge badge--time">70 min</span>
<span class="badge badge--level">Foundations</span>
<span class="badge">Needs: Linear Algebra with NumPy</span>

NumPy gives you fast grids of numbers. Pandas gives those grids names, mixed
types and an index, which is what turns an array into a dataset you can reason
about.

## Why this matters

Most of the work in machine learning happens before any model is trained. Data
arrives with missing values, inconsistent spellings, numbers stored as text and
rows that appear twice. Pandas is how you find those problems and fix them, and
it is the library you will spend more hours in than any other this year.

It is also what interviewers reach for. Being fluent here is closer to knowing
SQL than to knowing a framework, because it does not go out of fashion.

## A DataFrame is a labelled table

<svg viewBox="0 0 680 252" role="img" aria-labelledby="df-title df-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="df-title">The parts of a DataFrame</title>
<desc id="df-desc">A table with an index down the left, named columns across the top, and a data type for each column. A single column together with the index is a Series.</desc>
<text x="40" y="34" font-size="12" font-weight="700" fill="var(--h-graphite)">index</text>
<text x="100" y="34" font-size="12" font-weight="700" fill="var(--h-graphite)">column names</text>
<rect x="40" y="44" width="60" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="100" y="44" width="120" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="220" y="44" width="120" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="340" y="44" width="120" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="460" y="44" width="120" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="160" y="64" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">name</text>
<text x="280" y="64" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">city</text>
<text x="400" y="64" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">salary</text>
<text x="520" y="64" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">joined</text>
<rect x="40" y="74" width="60" height="30" fill="none" stroke="var(--h-surface-line)"/>
<rect x="100" y="74" width="480" height="30" fill="none" stroke="var(--h-surface-line)"/>
<text x="70" y="94" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-space)">0</text>
<text x="160" y="94" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Ana</text>
<text x="280" y="94" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Tirana</text>
<text x="400" y="94" text-anchor="middle" font-size="12" fill="var(--h-graphite)">1152000</text>
<text x="520" y="94" text-anchor="middle" font-size="12" fill="var(--h-graphite)">2021-03-01</text>
<rect x="40" y="104" width="60" height="30" fill="none" stroke="var(--h-surface-line)"/>
<rect x="100" y="104" width="480" height="30" fill="none" stroke="var(--h-surface-line)"/>
<text x="70" y="124" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-space)">1</text>
<text x="160" y="124" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Blerim</text>
<text x="280" y="124" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Durres</text>
<text x="400" y="124" text-anchor="middle" font-size="12" fill="var(--h-graphite)">94000</text>
<text x="520" y="124" text-anchor="middle" font-size="12" fill="var(--h-graphite)">2019-11-15</text>
<rect x="40" y="134" width="60" height="30" fill="none" stroke="var(--h-surface-line)"/>
<rect x="100" y="134" width="480" height="30" fill="none" stroke="var(--h-surface-line)"/>
<text x="70" y="154" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-space)">2</text>
<text x="160" y="154" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Eda</text>
<text x="280" y="154" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Vlore</text>
<text x="400" y="154" text-anchor="middle" font-size="12" fill="var(--h-graphite)">87000</text>
<text x="520" y="154" text-anchor="middle" font-size="12" fill="var(--h-graphite)">2022-06-30</text>
<rect x="340" y="44" width="120" height="120" fill="none" stroke="var(--h-cherry)" stroke-width="2.5"/>
<text x="160" y="184" text-anchor="middle" font-size="11" fill="var(--h-space)">object</text>
<text x="280" y="184" text-anchor="middle" font-size="11" fill="var(--h-space)">object</text>
<text x="400" y="184" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-cherry)">int64</text>
<text x="520" y="184" text-anchor="middle" font-size="11" fill="var(--h-space)">object</text>
<text x="600" y="110" font-size="11.5" font-weight="700" fill="var(--h-cherry)">one column</text>
<text x="600" y="126" font-size="11.5" font-weight="700" fill="var(--h-cherry)">is a Series</text>
<text x="40" y="216" font-size="12" fill="var(--h-graphite)">Each column has its own type. Notice that joined is object, meaning text, not a date. That will matter.</text>
<text x="40" y="236" font-size="12" fill="var(--h-graphite)">The index labels the rows. It is not a column, and it does not have to be 0, 1, 2.</text>
</svg>

Two objects, and you will use both constantly.

A **DataFrame** is the whole table. A **Series** is one column together with the
index. Every column you pull out of a DataFrame comes back as a Series, and most
of the methods you learn work on both.

## What the labels buy you

<svg viewBox="0 0 680 226" role="img" aria-labelledby="np-title np-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="np-title">A NumPy array compared with a DataFrame</title>
<desc id="np-desc">A NumPy array holds numbers of one type and is accessed by position. A DataFrame holds named columns of mixed types and is accessed by name.</desc>
<rect x="10" y="14" width="300" height="190" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="160" y="42" text-anchor="middle" font-size="13.5" font-weight="700" fill="var(--md-default-fg-color)">NumPy array</text>
<rect x="46" y="58" width="76" height="28" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<rect x="122" y="58" width="76" height="28" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<rect x="198" y="58" width="76" height="28" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<rect x="46" y="86" width="76" height="28" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<rect x="122" y="86" width="76" height="28" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<rect x="198" y="86" width="76" height="28" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="84" y="77" text-anchor="middle" font-size="12" fill="var(--h-graphite)">34</text>
<text x="160" y="77" text-anchor="middle" font-size="12" fill="var(--h-graphite)">1</text>
<text x="236" y="77" text-anchor="middle" font-size="12" fill="var(--h-graphite)">152000</text>
<text x="84" y="105" text-anchor="middle" font-size="12" fill="var(--h-graphite)">28</text>
<text x="160" y="105" text-anchor="middle" font-size="12" fill="var(--h-graphite)">2</text>
<text x="236" y="105" text-anchor="middle" font-size="12" fill="var(--h-graphite)">87000</text>
<text x="160" y="140" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">one type, no names</text>
<text x="160" y="170" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">arr[:, 2]</text>
<text x="160" y="188" text-anchor="middle" font-size="11" fill="var(--h-space)">which column was that again</text>
<rect x="370" y="14" width="300" height="190" rx="12" fill="var(--h-cherry)"/>
<text x="520" y="42" text-anchor="middle" font-size="13.5" font-weight="700" fill="#ffffff">DataFrame</text>
<rect x="406" y="58" width="76" height="28" fill="rgba(255,255,255,0.18)"/>
<rect x="482" y="58" width="76" height="28" fill="rgba(255,255,255,0.18)"/>
<rect x="558" y="58" width="76" height="28" fill="rgba(255,255,255,0.18)"/>
<text x="444" y="77" text-anchor="middle" font-size="11.5" font-weight="700" fill="#ffffff">age</text>
<text x="520" y="77" text-anchor="middle" font-size="11.5" font-weight="700" fill="#ffffff">city</text>
<text x="596" y="77" text-anchor="middle" font-size="11.5" font-weight="700" fill="#ffffff">salary</text>
<text x="444" y="105" text-anchor="middle" font-size="12" fill="#ffffff" opacity="0.9">34</text>
<text x="520" y="105" text-anchor="middle" font-size="12" fill="#ffffff" opacity="0.9">Tirana</text>
<text x="596" y="105" text-anchor="middle" font-size="12" fill="#ffffff" opacity="0.9">152000</text>
<text x="520" y="140" text-anchor="middle" font-size="11.5" fill="#ffffff" opacity="0.88">mixed types, named columns</text>
<text x="520" y="170" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">df["salary"]</text>
<text x="520" y="188" text-anchor="middle" font-size="11" fill="#ffffff" opacity="0.8">reads like what it means</text>
</svg>

Underneath, a DataFrame is still NumPy arrays. Everything you learned about
vectorised operations still applies. What pandas adds is names, an index, the
ability to mix text with numbers in one table, and a notion of missing data.

## Loading and looking

```python
import pandas as pd

df = pd.read_csv("employees.csv")
```

Useful arguments you will reach for: `sep=";"` for European exports,
`na_values=["", "NA", "unknown"]` to mark junk as missing on the way in, and
`parse_dates=["joined"]` so date columns arrive as dates.

Before doing anything else, look at the thing.

```python
df.head()          # first five rows
df.shape           # (rows, columns)
df.info()          # column names, types, non-null counts
df.describe()      # count, mean, std, min, quartiles, max for numeric columns
df["city"].value_counts()
```

`info()` is the one to internalise. In a single output it tells you how many
rows you have, which columns are the wrong type, and which have missing values.

!!! tip "Look before you transform"

    Every mistake in this page is cheaper to catch at this stage. Run `head`,
    `info` and `describe` before writing a single line of cleaning code, and
    again after each significant change.

## Selecting

This is where beginners lose the most time, because there are several ways to
select and they are not interchangeable.

<svg viewBox="0 0 680 236" role="img" aria-labelledby="sel-title sel-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="sel-title">Choosing the right selection syntax</title>
<desc id="sel-desc">Five ways to select. One column by name. Several columns with a list. Rows by label using loc. Rows by position using iloc. Rows by condition using a boolean mask.</desc>
<rect x="240" y="8" width="200" height="40" rx="10" fill="var(--h-cherry)"/>
<text x="340" y="28" dy="0.36em" text-anchor="middle" font-size="13.5" font-weight="700" fill="#ffffff">what do you want?</text>
<polyline points="340,48 340,68 68,68 68,92" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<polyline points="340,48 340,68 204,68 204,92" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<polyline points="340,48 340,92" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<polyline points="340,48 340,68 476,68 476,92" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<polyline points="340,48 340,68 612,68 612,92" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<rect x="5" y="92" width="126" height="84" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="68" y="118" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">one column</text>
<text x="68" y="146" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">df["city"]</text>
<text x="68" y="164" text-anchor="middle" font-size="10" fill="var(--h-space)">gives a Series</text>
<rect x="141" y="92" width="126" height="84" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="204" y="118" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">some columns</text>
<text x="204" y="146" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">df[["a","b"]]</text>
<text x="204" y="164" text-anchor="middle" font-size="10" fill="var(--h-space)">note the two brackets</text>
<rect x="277" y="92" width="126" height="84" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="340" y="118" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">rows by label</text>
<text x="340" y="146" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">df.loc[2]</text>
<text x="340" y="164" text-anchor="middle" font-size="10" fill="var(--h-space)">uses the index</text>
<rect x="413" y="92" width="126" height="84" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="476" y="118" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">rows by position</text>
<text x="476" y="146" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">df.iloc[2]</text>
<text x="476" y="164" text-anchor="middle" font-size="10" fill="var(--h-space)">counts from 0</text>
<rect x="549" y="92" width="126" height="84" rx="10" fill="var(--h-cherry)"/>
<text x="612" y="118" text-anchor="middle" font-size="11.5" fill="#ffffff" opacity="0.9">rows by condition</text>
<text x="612" y="146" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">df[mask]</text>
<text x="612" y="164" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.8">the one you use most</text>
<text x="340" y="212" text-anchor="middle" font-size="12" fill="var(--h-graphite)">loc uses labels, iloc uses positions. When the index is 0, 1, 2 they look the same, which is how people get caught.</text>
</svg>

### Columns

```python
df["salary"]                  # one column, a Series
df[["name", "salary"]]        # several columns, a DataFrame
df.drop(columns=["notes"])    # returns a new DataFrame without that column
```

The double brackets confuse people. The inner brackets are a Python list of
column names, and the outer ones are the selection. Ask for one name and you get
a Series, ask for a list and you get a table.

### Rows by label or position

```python
df.loc[2]                     # the row whose index label is 2
df.iloc[2]                    # the third row, whatever its label
df.loc[2, "salary"]           # one cell, by label
df.iloc[0:3, 0:2]             # first three rows, first two columns
```

On a freshly loaded file the index is 0, 1, 2 and the two behave identically.
Sort or filter the data and they diverge immediately, because `loc` follows the
label while `iloc` follows the position. When in doubt, ask yourself whether you
mean "the row called 5" or "the sixth row".

### Rows by condition

```python
df[df["salary"] > 50000]
df[df["city"] == "Tirana"]
df[df["city"].isin(["Tirana", "Durres"])]
df[~df["city"].isin(["Tirana"])]
```

This is the same boolean mask idea from NumPy, and you will use it more than
everything else on this page combined.

!!! warning "Brackets around every condition"

    ```python
    df[(df["age"] > 25) & (df["city"] == "Tirana")]
    ```

    Use `&` for and, `|` for or, and wrap each condition in its own brackets.
    Without them Python's operator precedence evaluates the `&` first and you
    get an error that explains nothing. Also use `&` rather than the word `and`,
    which cannot work on a whole column.

## The copy trap

You met this in NumPy, where a slice was a view into the same memory. Pandas has
its own version of the problem and it produces a warning that beginners learn to
ignore.

!!! warning "SettingWithCopyWarning"

    ```python
    subset = df[df["city"] == "Tirana"]
    subset["bonus"] = 1000
    ```

    Pandas cannot tell whether `subset` is a view into `df` or an independent
    table, so it warns you that your assignment might not do what you expect.

    Two fixes, depending on what you meant. If you wanted a separate table, say
    so with `subset = df[df["city"] == "Tirana"].copy()`. If you wanted to change
    the original, address it directly:

    ```python
    df.loc[df["city"] == "Tirana", "bonus"] = 1000
    ```

    Never chain two selections when assigning. `df[df["a"] > 1]["b"] = 0` looks
    reasonable and silently does nothing at all.

## Missing data

Finding it is easy. Deciding what to do is the actual skill.

```python
df.isna().sum()               # count of missing values per column
df.isna().sum().sum()         # total across the table
df[df["salary"].isna()]       # the rows that are missing a salary
```

<svg viewBox="0 0 680 262" role="img" aria-labelledby="miss-title miss-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="miss-title">Deciding what to do about missing values</title>
<desc id="miss-desc">If a column is mostly empty, drop the column. If only a few rows are affected, drop those rows. If the column matters, fill the values and add a flag column recording that they were filled.</desc>
<rect x="180" y="8" width="320" height="44" rx="10" fill="var(--h-cherry)"/>
<text x="340" y="30" dy="0.36em" text-anchor="middle" font-size="13.5" font-weight="700" fill="#ffffff">this column has missing values</text>
<polyline points="340,52 340,74 110,74 110,128" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<polyline points="340,52 340,128" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<polyline points="340,52 340,74 570,74 570,128" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<text x="110" y="100" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">mostly empty</text>
<text x="340" y="100" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">a few rows</text>
<text x="570" y="100" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">worth keeping</text>
<rect x="10" y="128" width="200" height="82" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="110" y="156" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">drop the column</text>
<text x="110" y="178" text-anchor="middle" font-size="11" fill="var(--h-graphite)">it carries almost</text>
<text x="110" y="194" text-anchor="middle" font-size="11" fill="var(--h-graphite)">no information</text>
<rect x="240" y="128" width="200" height="82" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="340" y="156" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">drop those rows</text>
<text x="340" y="178" text-anchor="middle" font-size="11" fill="var(--h-graphite)">check first that they</text>
<text x="340" y="194" text-anchor="middle" font-size="11" fill="var(--h-graphite)">are not all one group</text>
<rect x="470" y="128" width="200" height="82" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry)"/>
<text x="570" y="156" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">fill, and flag</text>
<text x="570" y="178" text-anchor="middle" font-size="11" fill="var(--h-graphite)">record that the value</text>
<text x="570" y="194" text-anchor="middle" font-size="11" fill="var(--h-graphite)">was filled, not measured</text>
<text x="340" y="242" text-anchor="middle" font-size="12" fill="var(--h-graphite)">before any of these, ask why the data is missing. That answer changes the right choice.</text>
</svg>

```python
df = df.dropna(subset=["salary"])              # drop rows missing a salary
df = df.drop(columns=["notes"])                # drop a column that is mostly empty
df["salary_missing"] = df["salary"].isna()     # flag before filling
df["salary"] = df["salary"].fillna(df["salary"].median())
```

Median rather than mean when the column is skewed, which salaries always are. A
handful of executives will drag a mean somewhere no real employee sits.

!!! ml "ML connection"

    Missing values are rarely missing at random. If income is blank more often
    for people who declined to answer, then filling those blanks with the
    average quietly invents a population that does not exist. The flag column
    costs nothing and lets a model learn that the absence itself carried
    information.

## Types and text

The `joined` column in the diagram was text, not a date. Fixing types is
unglamorous and it unlocks everything downstream.

```python
df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
df["joined"] = pd.to_datetime(df["joined"], errors="coerce")
df["city"] = df["city"].astype("category")
```

`errors="coerce"` turns anything unparseable into a missing value rather than
raising. Combine it with `isna()` afterwards to find exactly which rows were
junk.

Text needs its own cleaning, because `Tirana`, `tirana` and `Tirana ` are three
different cities as far as pandas is concerned.

```python
df["city"] = df["city"].str.strip().str.title()
df["email"] = df["email"].str.lower()
df = df.drop_duplicates()
df = df.drop_duplicates(subset=["email"], keep="last")
```

Once dates are real dates, they become useful:

```python
df["year_joined"] = df["joined"].dt.year
df["tenure_days"] = (pd.Timestamp.today() - df["joined"]).dt.days
```

## Creating columns

```python
df["monthly"] = df["salary"] / 12
df["senior"] = df["salary"] > 55000
df["band"] = np.where(df["salary"] > 55000, "high", "standard")
df["region"] = df["city"].map({"Tirana": "Central", "Durres": "Coast"})
```

`map` on a dictionary is the clean way to recode categories. Anything it does
not find becomes missing, which is a feature: it tells you about spellings you
did not anticipate.

When the logic will not fit in an expression, `apply` takes a function, but
reach for it last. It runs a Python loop underneath and gives up the speed you
came to pandas for.

## Grouping

Split the table into piles, compute something for each pile, stack the answers
back together.

<svg viewBox="0 0 680 264" role="img" aria-labelledby="gb-title gb-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="gb-title">Split, apply, combine</title>
<desc id="gb-desc">One table of salaries by city is split into one pile per city, the mean is applied to each pile, and the results are combined into a small table with one row per city.</desc>
<text x="100" y="28" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-graphite)">one table</text>
<rect x="20" y="40" width="160" height="26" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="20" y="66" width="160" height="26" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="20" y="92" width="160" height="26" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="20" y="118" width="160" height="26" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="20" y="144" width="160" height="26" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="20" y="170" width="160" height="26" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="45" y="58" font-size="11.5" fill="var(--md-default-fg-color)">Tirana</text>
<text x="150" y="58" text-anchor="end" font-size="11.5" fill="var(--h-graphite)">52</text>
<text x="45" y="84" font-size="11.5" fill="var(--md-default-fg-color)">Durres</text>
<text x="150" y="84" text-anchor="end" font-size="11.5" fill="var(--h-graphite)">41</text>
<text x="45" y="110" font-size="11.5" fill="var(--md-default-fg-color)">Tirana</text>
<text x="150" y="110" text-anchor="end" font-size="11.5" fill="var(--h-graphite)">60</text>
<text x="45" y="136" font-size="11.5" fill="var(--md-default-fg-color)">Vlore</text>
<text x="150" y="136" text-anchor="end" font-size="11.5" fill="var(--h-graphite)">38</text>
<text x="45" y="162" font-size="11.5" fill="var(--md-default-fg-color)">Durres</text>
<text x="150" y="162" text-anchor="end" font-size="11.5" fill="var(--h-graphite)">45</text>
<text x="45" y="188" font-size="11.5" fill="var(--md-default-fg-color)">Tirana</text>
<text x="150" y="188" text-anchor="end" font-size="11.5" fill="var(--h-graphite)">55</text>
<line x1="192" y1="118" x2="246" y2="118" stroke="var(--h-steel)" stroke-width="2"/>
<text x="219" y="108" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-cherry)">split</text>
<rect x="256" y="40" width="150" height="78" rx="8" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="272" y="58" font-size="11" font-weight="700" fill="var(--h-cherry)">Tirana</text>
<text x="272" y="78" font-size="11.5" fill="var(--h-graphite)">52</text>
<text x="322" y="78" font-size="11.5" fill="var(--h-graphite)">60</text>
<text x="372" y="78" font-size="11.5" fill="var(--h-graphite)">55</text>
<rect x="256" y="126" width="150" height="58" rx="8" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="272" y="144" font-size="11" font-weight="700" fill="var(--h-cherry)">Durres</text>
<text x="272" y="164" font-size="11.5" fill="var(--h-graphite)">41</text>
<text x="322" y="164" font-size="11.5" fill="var(--h-graphite)">45</text>
<rect x="256" y="192" width="150" height="46" rx="8" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="272" y="210" font-size="11" font-weight="700" fill="var(--h-cherry)">Vlore</text>
<text x="272" y="228" font-size="11.5" fill="var(--h-graphite)">38</text>
<line x1="418" y1="118" x2="472" y2="118" stroke="var(--h-steel)" stroke-width="2"/>
<text x="445" y="108" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-cherry)">apply mean</text>
<text x="570" y="28" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-graphite)">combine</text>
<rect x="482" y="40" width="180" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="482" y="70" width="180" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="482" y="100" width="180" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="502" y="60" font-size="11.5" font-weight="700" fill="var(--md-default-fg-color)">Tirana</text>
<text x="642" y="60" text-anchor="end" font-size="11.5" fill="var(--h-graphite)">55.7</text>
<text x="502" y="90" font-size="11.5" font-weight="700" fill="var(--md-default-fg-color)">Durres</text>
<text x="642" y="90" text-anchor="end" font-size="11.5" fill="var(--h-graphite)">43.0</text>
<text x="502" y="120" font-size="11.5" font-weight="700" fill="var(--md-default-fg-color)">Vlore</text>
<text x="642" y="120" text-anchor="end" font-size="11.5" fill="var(--h-graphite)">38.0</text>
<text x="572" y="164" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">one row per group,</text>
<text x="572" y="180" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">the group becomes</text>
<text x="572" y="196" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">the new index</text>
</svg>

```python
df.groupby("city")["salary"].mean()
df.groupby("city")["salary"].agg(["count", "mean", "median", "max"])
df.groupby(["city", "band"])["salary"].mean()
df.groupby("city").agg(headcount=("name", "count"), payroll=("salary", "sum"))
```

The grouping key becomes the index of the result, which is why the output looks
different from a normal table. `reset_index()` turns it back into an ordinary
column when you want one.

## Merging two tables

Real data arrives split across files: employees in one, departments in another.
`merge` joins them on a shared key.

<svg viewBox="0 0 680 320" role="img" aria-labelledby="mg-title mg-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="mg-title">Merging two tables on a shared key</title>
<desc id="mg-desc">An employees table and a departments table share a department id column. Merging matches rows with the same id and combines their columns into one table. An id present in only one table is dropped by an inner join.</desc>
<text x="150" y="26" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-graphite)">employees</text>
<rect x="30" y="36" width="70" height="28" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<rect x="100" y="36" width="140" height="28" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="65" y="55" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">dept_id</text>
<text x="170" y="55" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--md-default-fg-color)">name</text>
<rect x="30" y="64" width="70" height="26" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<rect x="100" y="64" width="140" height="26" fill="none" stroke="var(--h-surface-line)"/>
<text x="65" y="82" text-anchor="middle" font-size="11.5" fill="var(--h-cherry)">1</text>
<text x="170" y="82" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">Ana</text>
<rect x="30" y="90" width="70" height="26" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<rect x="100" y="90" width="140" height="26" fill="none" stroke="var(--h-surface-line)"/>
<text x="65" y="108" text-anchor="middle" font-size="11.5" fill="var(--h-cherry)">2</text>
<text x="170" y="108" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">Blerim</text>
<rect x="30" y="116" width="70" height="26" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<rect x="100" y="116" width="140" height="26" fill="none" stroke="var(--h-surface-line)"/>
<text x="65" y="134" text-anchor="middle" font-size="11.5" fill="var(--h-cherry)">9</text>
<text x="170" y="134" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">Eda</text>
<text x="530" y="26" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-graphite)">departments</text>
<rect x="420" y="36" width="70" height="28" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<rect x="490" y="36" width="150" height="28" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="455" y="55" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">dept_id</text>
<text x="565" y="55" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--md-default-fg-color)">dept_name</text>
<rect x="420" y="64" width="70" height="26" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<rect x="490" y="64" width="150" height="26" fill="none" stroke="var(--h-surface-line)"/>
<text x="455" y="82" text-anchor="middle" font-size="11.5" fill="var(--h-cherry)">1</text>
<text x="565" y="82" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">Engineering</text>
<rect x="420" y="90" width="70" height="26" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<rect x="490" y="90" width="150" height="26" fill="none" stroke="var(--h-surface-line)"/>
<text x="455" y="108" text-anchor="middle" font-size="11.5" fill="var(--h-cherry)">2</text>
<text x="565" y="108" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">Sales</text>
<text x="340" y="88" text-anchor="middle" font-size="20" fill="var(--h-graphite)">+</text>
<polyline points="150,148 150,168 340,168 340,196" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<polyline points="530,122 530,168 340,168" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<text x="340" y="188" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">match on dept_id</text>
<rect x="170" y="200" width="70" height="28" fill="var(--h-cherry)"/>
<rect x="240" y="200" width="130" height="28" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="370" y="200" width="140" height="28" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="205" y="219" text-anchor="middle" font-size="11.5" font-weight="700" fill="#ffffff">dept_id</text>
<text x="305" y="219" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--md-default-fg-color)">name</text>
<text x="440" y="219" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--md-default-fg-color)">dept_name</text>
<rect x="170" y="228" width="70" height="26" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<rect x="240" y="228" width="130" height="26" fill="none" stroke="var(--h-surface-line)"/>
<rect x="370" y="228" width="140" height="26" fill="none" stroke="var(--h-surface-line)"/>
<text x="205" y="246" text-anchor="middle" font-size="11.5" fill="var(--h-cherry)">1</text>
<text x="305" y="246" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">Ana</text>
<text x="440" y="246" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">Engineering</text>
<rect x="170" y="254" width="70" height="26" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<rect x="240" y="254" width="130" height="26" fill="none" stroke="var(--h-surface-line)"/>
<rect x="370" y="254" width="140" height="26" fill="none" stroke="var(--h-surface-line)"/>
<text x="205" y="272" text-anchor="middle" font-size="11.5" fill="var(--h-cherry)">2</text>
<text x="305" y="272" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">Blerim</text>
<text x="440" y="272" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">Sales</text>
<text x="340" y="302" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Eda had dept_id 9, which appears in no department. An inner join drops her silently.</text>
</svg>

```python
merged = employees.merge(departments, on="dept_id", how="inner")
merged = employees.merge(departments, on="dept_id", how="left")
merged = employees.merge(departments, left_on="dept", right_on="id", how="left")
```

| `how` | What you keep |
|---|---|
| `inner` | Only rows whose key appears in both tables |
| `left` | Every row of the left table, with blanks where there was no match |
| `right` | Every row of the right table |
| `outer` | Everything from both, with blanks on either side |

!!! warning "Always check the row count after a merge"

    ```python
    print(len(employees), len(merged))
    ```

    Rows vanishing means keys did not match, often because one side is text and
    the other is a number, or because of trailing spaces. Rows multiplying means
    the key is not unique on one side and you have accidentally created
    duplicates. Both are common and neither raises an error.

    `how="left"` plus `validate="many_to_one"` will make pandas complain loudly
    instead of silently producing a wrong table.

## Saving your work

```python
df.to_csv("clean.csv", index=False)
```

Pass `index=False` unless the index means something, otherwise you get a stray
unnamed column every time you reload the file. `to_excel`, `to_parquet` and
`to_json` work the same way, and parquet is worth knowing for anything large
because it keeps your data types.

## The whole thing on one screen

```python
import pandas as pd
import numpy as np

df = pd.read_csv("employees.csv", na_values=["", "NA", "unknown"])
df["city"] = df["city"].str.strip().str.title()
df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
df["joined"] = pd.to_datetime(df["joined"], errors="coerce")
df = df.drop_duplicates()
df["salary_missing"] = df["salary"].isna()
df["salary"] = df["salary"].fillna(df["salary"].median())
df["tenure_days"] = (pd.Timestamp.today() - df["joined"]).dt.days
df = df.merge(departments, on="dept_id", how="left")
summary = df.groupby("city").agg(headcount=("name", "count"), pay=("salary", "median"))
df.to_csv("clean.csv", index=False)
```

Twelve lines, and it is the shape of most real data preparation jobs. Load,
standardise text, fix types, deduplicate, handle missing values, derive what you
need, join what is elsewhere, summarise, save.

## Where this sits in the lifecycle

| Lifecycle stage | What you use |
|---|---|
| Collect | `read_csv`, `read_excel`, `read_sql`, `merge` |
| Prepare | `dropna`, `fillna`, `astype`, `str` methods, `drop_duplicates` |
| Prepare | new columns, `map`, `np.where`, `groupby` |
| Evaluate | `value_counts`, `describe`, `crosstab` to check what you built |

Collect and Prepare are the first two boxes of the loop from Fundamentals, and
they are where most of your time goes.

## When it goes wrong

??? note "`KeyError: 'salary'`"

    The column does not exist under that name. Print `df.columns.tolist()`.
    Usually it is a trailing space or a different capitalisation in the file
    header. `df.columns = df.columns.str.strip().str.lower()` right after
    loading prevents a whole category of this.

??? note "`SettingWithCopyWarning`"

    You assigned into something that may be a view. Use `.copy()` if you wanted
    a separate table, or `df.loc[condition, "column"] = value` if you wanted to
    change the original.

??? note "`ValueError: The truth value of a Series is ambiguous`"

    You used `and`, `or` or `not` on a whole column. Use `&`, `|` and `~`, with
    brackets around each condition.

??? note "My filter returned an empty table"

    Compare against the real values with `df["city"].unique()`. It is almost
    always whitespace or capitalisation, which is why cleaning text comes before
    filtering.

??? note "After merging I have more rows than I started with"

    The key is not unique in the right-hand table, so each left row matched
    several right rows. Check with `departments["dept_id"].duplicated().sum()`.

??? note "A column of numbers has dtype object"

    Something in it is not a number: a currency symbol, a comma, or the word
    unknown. `pd.to_numeric(col, errors="coerce")` then `isna()` shows you
    exactly which rows.

## Check yourself

Use any small CSV you have, or one column of the data from the project.

1. Load a file and, without looking at it in a spreadsheet, say how many rows it
   has, which columns are the wrong type, and which have missing values.
2. Select two columns, then the rows where a numeric column exceeds its own
   median. Do it in one expression.
3. Take a column with inconsistent text values and reduce it to a clean set.
   Prove it with `value_counts()` before and after.
4. Fill a column's missing values, and add a flag column recording which rows
   were filled. Explain why you chose mean or median.
5. Group by a category and produce count and median in one call, then reset the
   index so the result is an ordinary table.
6. Merge two tables on a key, then check that the row count is what you expected
   and explain any difference.

## Quick reference

| Task | Code |
|---|---|
| Load | `pd.read_csv("f.csv", na_values=["NA"])` |
| First look | `df.head()`, `df.info()`, `df.describe()` |
| Size | `df.shape` |
| One column | `df["col"]` |
| Several columns | `df[["a", "b"]]` |
| Row by label, by position | `df.loc[2]`, `df.iloc[2]` |
| Filter | `df[(df["a"] > 1) & (df["b"] == "x")]` |
| Missing counts | `df.isna().sum()` |
| Drop, fill | `df.dropna(subset=["a"])`, `df["a"].fillna(value)` |
| Fix types | `pd.to_numeric(...)`, `pd.to_datetime(...)`, `.astype("category")` |
| Clean text | `df["a"].str.strip().str.title()` |
| Duplicates | `df.drop_duplicates(subset=["id"])` |
| New column | `df["new"] = df["a"] / 12` |
| Recode | `df["a"].map({"x": 1})`, `np.where(cond, "y", "n")` |
| Group | `df.groupby("city")["salary"].agg(["count", "median"])` |
| Join | `a.merge(b, on="key", how="left")` |
| Save | `df.to_csv("out.csv", index=False)` |

## Questions to sit with

1. You fill missing salaries with the column median. What have you told the
   model, and what have you hidden from it?
2. An inner join quietly dropped a quarter of your rows. Nothing raised an
   error. What process would catch that before it reaches a model?
3. `df["city"].value_counts()` shows Tirana with 40 and Tirane with 12. Is
   merging them a data cleaning decision or a modelling decision, and does the
   distinction matter?
4. Pandas makes it easy to transform data until it gives the answer you wanted.
   What would you keep, in your code or your notes, so that somebody else could
   tell the difference between cleaning and tuning?

## Next

You can shape data. Now make it visible, which is how you find the problems that
`info()` cannot show you.

[Plotting](plotting.md){ .h-button }

Worth your time: the
[pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html) for
depth, and
[Python for Data Analysis](https://wesmckinney.com/book/) by the author of
pandas, free to read online.

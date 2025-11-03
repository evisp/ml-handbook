# Data Visualization: Seeing Patterns in Data

This tutorial introduces the essential role of visualization in machine learning workflows and guides you through choosing and creating effective visualizations. You'll understand why seeing your data visually is crucial for building successful models and learn the systematic approach to exploring data through charts and graphs.

**Estimated time:** 45 minutes

**Hands-on practice:** For practical code examples using the Albania House Prices dataset, see the [companion Jupyter notebook](../../notebooks/data_foundations/2.%20Data%20Visualization.ipynb).

## Why This Matters

**Problem statement:** 

> You can't understand what you can't see. 

**Data hidden in tables remains abstract**, making it nearly impossible to spot patterns, outliers, or relationships that directly impact model performance. A correlation of 0.85 between features means little until you see the actual scatter plot. An imbalanced dataset with 95% of one class isn't obvious from summary statistics alone, but a bar chart makes it immediately clear.


**Professional context:** Data scientists create dozens of exploratory plots before selecting model features. Companies that visualize data pipelines catch errors 3-5x faster than those relying on numerical validation alone. The ability to choose the right visualization and communicate insights visually separates junior practitioners from senior data scientists. Stakeholders rarely read statistical reports, but they always remember compelling visualizations.

![Data Visualization Overview](https://i.imgur.com/kRAJaHV.png)

## Core Concepts

### What is Data Visualization?

**Data visualization is the graphical representation of information**, transforming numerical values into visual forms like charts, graphs, and plots. In machine learning contexts, visualization serves three purposes:

- **Exploratory Analysis:** Discovering patterns, distributions, and relationships before modeling
- **Diagnostic Analysis:** Understanding why models succeed or fail through residual plots, confusion matrices, and learning curves
- **Communication:** Presenting findings to technical and non-technical audiences

### The Visualization Pipeline

**Data → Chart Type Selection → Encoding → Rendering → Interpretation**

![Data Visualization Pipeline](https://i.imgur.com/anfb8Ap.jpeg)


Each step requires decisions: What question am I answering? Which chart type reveals this pattern best? How should I map data to visual properties? What customizations improve clarity?

### Visual Encoding Channels

Visual encoding maps data values to visual properties:

- **Position:** Most accurate encoding (x-axis, y-axis placement)
- **Length:** Highly effective (bar heights, line lengths)
- **Angle:** Moderately effective (pie slices)
- **Area:** Less accurate (bubble sizes)
- **Color:** Good for categories, challenging for continuous values
- **Shape:** Effective for categorical distinctions

The hierarchy matters because position encodings (scatter plots) communicate information more accurately than area encodings (bubble charts), affecting how viewers interpret your data.

### Matplotlib vs Seaborn: Complementary Tools

**Matplotlib** provides complete control over every plot element, requiring explicit commands for each component. Think of it as the foundation layer.

**Seaborn** builds on Matplotlib, offering high-level interfaces for statistical visualizations with better default aesthetics and built-in statistical computations.

**When to use each:** Use Matplotlib for custom plots, precise control, and fundamental chart types. Use Seaborn for statistical analysis, quick exploratory plots, and when you need beautiful defaults with minimal code.

## Visualization Types and Their Purposes

Choosing the right visualization depends on your data structure and the question you're answering. Using the wrong chart type obscures insights or misleads viewers.

### Distribution Visualizations

> Understand how values are spread across a variable.

![Distribution Visualizations](https://i.imgur.com/Rr5ERuT.jpeg)

#### Histogram 

Shows frequency distribution by grouping continuous data into bins. Reveals shape (normal, skewed, bimodal), spread, and outliers.

Use it when exploring numerical features like age, price, or income to understand central tendency and variability.


#### Box Plot 
Displays five-number summary (min, Q1, median, Q3, max) plus outliers. Compact representation ideal for comparing distributions across categories.

Use when comparing numerical distributions across groups (price by zone, salary by department). It can help to quickly identifies outliers that might require handling. Reveals class imbalance when box plots for different target classes differ dramatically.

#### Violin Plot 
Combines box plot with kernel density estimation, showing distribution shape through width variations.

Use when you need both summary statistics and distribution shape, especially for multimodal distributions that box plots miss.

It helps to reveal complex distributional patterns that inform feature engineering and model selection decisions.

### Comparison Visualizations

![Comparison Visualizations](https://i.imgur.com/uEpM3cA.png)

> Compare values across categories or groups.

#### Bar Chart 

Uses rectangular bars to represent values across discrete categories. Bar heights encode magnitudes, making comparisons immediate.

Use when comparing metrics across categories (sales by region, model accuracy by algorithm).

Compares model performance metrics across different algorithms or hyperparameter settings. Horizontal bars work well for many categories or long labels.

#### Grouped Bar Chart 

Places multiple bars side-by-side for each category, enabling multi-variable comparisons.

Use when comparing multiple metrics simultaneously (precision and recall across classifiers).

Compares training vs validation performance across models, revealing overfitting when bars differ significantly.

### Relationship Visualizations

![Relationship Visualizations](blob:https://imgur.com/14172838-18eb-4dbb-8240-b450a31903d6)

> Reveal relationships, correlations, and patterns between variables.

#### Scatter Plot 

Displays individual data points using two numerical variables for x and y positions. Reveals correlations, clusters, and outliers.

Use when exploring relationships between continuous variables (size vs price, age vs salary).

It validates linear regression assumptions. Strong linear patterns suggest predictive relationships. Clusters indicate potential for classification. Outliers far from trends require investigation.

#### Line Plot 

Connects data points with lines, emphasizing trends and changes over sequences (often time).

Use when showing trends over time (stock prices, model performance across epochs, website traffic).

Tracks training and validation loss/accuracy across epochs. Diverging lines signal overfitting.

#### Heatmap 

Uses color intensity to represent values in a matrix format. Particularly effective for correlation matrices.

Use when visualizing relationships between many variables simultaneously (correlation matrix, confusion matrix).

Identifies highly correlated features that might cause multicollinearity. Confusion matrices reveal which classes models confuse most frequently.

#### Regression Plot 

Scatter plot with fitted regression line and confidence interval bands.

Use when assessing linear relationships and prediction confidence.

Visualizes regression model fit quality. Wide confidence bands indicate high uncertainty requiring more data or better features.

### Composition Visualizations

![Composition Visualizations](https://i.imgur.com/bjUk19b.png)

> Show how parts contribute to a whole.

#### Pie Chart

Circular chart divided into slices representing proportions of a whole. Most effective for 3-5 categories.

Use when showing simple proportions where exact values matter less than relative sizes (market share, budget allocation).

It visualizes class distribution in classification problems. Reveals severe imbalance requiring resampling techniques.

However, human perception struggles with angle comparisons. Use sparingly and only when showing parts of 100%.

#### Stacked Bar Chart 

Bars divided into segments representing component contributions.

Use when comparing both totals and component breakdowns across categories.

It shows how feature importance varies across different model versions or datasets.


## The Visualization Workflow

![Visualization Workflow](https://i.imgur.com/8uJTqZt.png)

### Phase 1: Define Your Question

> Clarify what you're trying to understand before creating visualizations.

**Key questions:**

- What pattern am I looking for? (distribution, relationship, comparison, trend)
- What decisions will this visualization inform?
- Who is the audience? (technical team vs business stakeholders)

**ML connection:** Exploratory analysis requires different visualizations than model diagnostics or stakeholder presentations. Asking "Why did this model fail on these samples?" guides you toward residual plots and confusion matrices, not histograms.

**Example questions:**

- "Are features correlated?" → Heatmap
- "Is the target class balanced?" → Bar chart or pie chart
- "Does feature X predict target Y?" → Scatter plot or regression plot
- "How are values distributed?" → Histogram or box plot

### Phase 2: Choose the Right Visualization

> Match visualization type to data structure and question.

**Decision framework:**

- **One numerical variable:** Histogram, box plot, violin plot
- **One categorical variable:** Bar chart, pie chart
- **Two numerical variables:** Scatter plot, line plot (if sequential), regression plot
- **One numerical + one categorical:** Box plot, violin plot, grouped bar chart
- **Two categorical variables:** Heatmap, stacked bar chart, count plot
- **Multiple numerical variables:** Correlation heatmap, pair plot, parallel coordinates

### Phase 3: Create the Visualization

> Implement the chosen visualization with appropriate libraries and parameters.

- **Matplotlib approach:** Build plots layer-by-layer with explicit control over every element.
- **Seaborn approach:** Use high-level functions that handle statistical computations and styling automatically.

**Key implementation considerations:**

- **Figure size:** Larger plots for presentations, smaller for reports
- **Color choices:** Colorblind-friendly palettes, meaningful color schemes
- **Labels and titles:** Clear, descriptive text without jargon
- **Axis scales:** Linear, logarithmic, or custom based on data range

### Phase 4: Customize for Clarity

> Improve readability and eliminate ambiguity.

**Essential customizations:**

- **Titles and labels:** Every plot needs descriptive title and labeled axes with units
- **Legends:** Required when multiple categories or series appear
- **Gridlines:** Subtle gridlines aid value reading without cluttering
- **Annotations:** Highlight specific points or regions of interest
- **Color schemes:** Use consistent, accessible palettes
- **Font sizes:** Readable at intended viewing distance

### Phase 5: Interpret and Act

> Extract insights that guide next steps in your ML workflow.

**Interpretation questions:**

- What patterns are visible? (clusters, trends, outliers)
- What surprises appear? (unexpected relationships, missing values)
- What actions should follow? (data cleaning, feature engineering, model selection)

**ML connection:** Visualization insights directly inform pipeline decisions:

- **Skewed distributions** → Apply transformations (log, square root)
- **Outliers** → Investigate causes, decide to keep/remove/cap
- **Missing value patterns** → Choose imputation strategy
- **Class imbalance** → Apply resampling or adjust class weights
- **Feature correlations** → Remove redundant features or apply PCA
- **Non-linear relationships** → Engineer polynomial features or choose non-linear models

## Common Visualization Challenges

### Overplotting

**Problem:** Too many data points overlap, obscuring density and patterns.

**Impact:** Cannot see true data distribution, clusters appear as uniform blobs.

**Solutions:**

- **Transparency (alpha):** Make points semi-transparent to show density
- **Sampling:** Plot random subset for initial exploration
- **Binning:** Use hexbin plots or 2D histograms
- **Jitter:** Add small random noise to separate overlapping points

### Choosing Colors

**Problem:** Color choices that aren't colorblind-safe or don't convey meaning appropriately.

**Impact:** ~8% of men and ~0.5% of women can't distinguish red-green, making many default palettes useless to them.

**Solutions:**

- **Use colorblind-safe palettes:** Viridis, Cividis, colorbrewer schemes
- **Don't rely solely on color:** Use shapes or line styles too
- **Sequential palettes:** For continuous values (light to dark)
- **Diverging palettes:** For values with meaningful zero (negative to positive)
- **Qualitative palettes:** For categorical data

Model comparison plots become meaningless if stakeholders can't distinguish the colors representing different algorithms.

### Scale and Range Issues

**Problem:** Inappropriate axis scales that hide patterns or mislead.

**Impact:** Linear scales compress high values, making trends invisible. Truncated axes exaggerate small differences.

**Solutions:**

- **Log scales:** For data spanning orders of magnitude
- **Start axes at zero:** For bar charts and comparisons
- **Full range:** Show complete data range unless justified otherwise
- **Breaks:** Indicate discontinuous axes explicitly

**ML connection:** Learning curves plotted on inappropriate scales can hide early overfitting or make converged training appear unstable.

### Too Much Information

**Problem:** Cramming multiple messages into one visualization, creating confusion.

**Impact:** Viewers miss key insights when overwhelmed by complexity.

**Solutions:**

- **One message per plot:** Focus each visualization on single insight
- **Progressive disclosure:** Start simple, add complexity gradually
- **Small multiples:** Create grid of related simple plots rather than one complex plot
- **Annotations:** Guide viewers to important patterns

Model diagnostic dashboards should separate concerns: one plot for learning curves, another for confusion matrix, another for feature importance. Combining them dilutes each message.

## Best Practices

**Start simple, add complexity only when necessary:** Begin with basic plot, add elements that enhance understanding. Remove anything that doesn't serve the message.

**Label everything explicitly:** Titles, axis labels, legends, and units should be clear without external context. Future you will forget what `variable_3` means.

**Choose appropriate chart types:** Bar charts for comparisons, scatter plots for relationships, histograms for distributions. Wrong chart type obscures rather than reveals.

**Use consistent styling:** Maintain color schemes, fonts, and layouts across related visualizations. Consistency reduces cognitive load.

**Consider colorblindness:** Use colorblind-safe palettes. Never encode information only through color.

**Size appropriately:** Match figure size to viewing context. Presentation slides need larger fonts than research papers.

**Annotate insights:** Highlight patterns, outliers, or thresholds directly on plots. Don't make viewers search for the message.

**Test with audience:** Show drafts to representative viewers. What's obvious to you may confuse others.

**Version and save source code:** Save visualization code, not just images. Reproducibility requires the ability to regenerate plots with updated data.

**Know when to use 3D:** Almost never. 3D charts look impressive but make reading values nearly impossible. Flat projections usually communicate better.


## Quick Reference: Visualization Selection Guide

| Data Structure | Question Type | Best Visualization | Library Recommendation |
|----------------|--------------|-------------------|----------------------|
| One numerical | Distribution shape | Histogram, Box plot | Matplotlib, Seaborn |
| One categorical | Category counts | Bar chart, Pie chart | Matplotlib, Seaborn |
| Two numerical | Relationship strength | Scatter plot, Regression plot | Matplotlib, Seaborn |
| Two numerical (time) | Trend over time | Line plot | Matplotlib |
| Numerical + Categorical | Distribution by group | Box plot, Violin plot | Seaborn |
| Many numerical | Correlations | Heatmap, Pair plot | Seaborn |
| Actual vs Predicted | Model accuracy | Scatter plot, Residual plot | Matplotlib |
| Classification errors | Confusion patterns | Confusion matrix heatmap | Seaborn |
| Training progress | Overfitting detection | Learning curves | Matplotlib |

## Summary & Next Steps

**Key accomplishments:** You've learned why visualization is fundamental to ML success, understood how to choose appropriate chart types for different data structures and questions, recognized common visualization pitfalls and solutions, and connected effective visualization practices to model development and communication.

**Critical insights:**

- **See before you model:** Visualization prevents costly modeling mistakes by revealing data issues early
- **Match chart to question:** Different analytical questions require different visualization types
- **Clarity over complexity:** Simple, well-labeled plots communicate better than elaborate designs
- **Iterate visually:** Create dozens of exploratory plots to understand data deeply

**Connections to previous topics:**

- **Data Foundations:** Visualization reveals the data quality issues identified in data pipelines
- **Statistics:** Distributions, correlations, and statistical tests become concrete through visualization
- **Pandas:** DataFrames integrate seamlessly with visualization libraries for rapid exploration


**External resources:**

- [Matplotlib Official Documentation](https://matplotlib.org/stable/contents.html) - comprehensive reference and tutorials
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html) - visual examples of all plot types
- [The Data Visualisation Catalogue](https://datavizcatalogue.com/) - guide to choosing chart types
- [ColorBrewer](https://colorbrewer2.org/) - colorblind-safe palette generator

> **Remember:** Data visualization isn't about making pretty pictures—it's about seeing patterns that guide better decisions. Master visualization, and you'll build better models because you'll understand your data deeply.

# Tree-Based Models

<span class="badge badge--time">85 min</span>
<span class="badge badge--level">Classic models</span>
<span class="badge">Needs: Classification, Linear Models</span>

A linear model adds everything up. A tree asks a series of yes or no questions.
That difference makes trees the easiest models to read and, once you combine
enough of them, the strongest thing you will run on a table all year.

## Why this matters

On tabular data like the churn file, a gradient boosted ensemble is very often
the best model available. Not a neural network, which comes into its own on
images, audio and text in trimester two. If somebody hands you a spreadsheet and
asks for the best possible prediction, this page is the answer.

Trees are also where you meet **ensembles**: the idea that many mediocre models
combined can beat one carefully built model. That idea turns up everywhere
afterwards.

## How a tree decides

A tree has one move, repeated. Pick a feature, pick a threshold, split the data
in two, and do it again on each side.

The only question is which split to make.

<svg viewBox="0 0 680 276" role="img" aria-labelledby="t1-title t1-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="t1-title">How a tree chooses where to split</title>
<desc id="t1-desc">The tree tries every threshold on every feature and keeps the one that produces the purest two groups. A cut that separates the classes well has a much lower impurity than one that does not.</desc>
<text x="30" y="26" font-size="12" font-weight="700" fill="var(--h-graphite)">one feature, tenure in months. Red customers churned.</text>
<text x="636" y="54" text-anchor="end" font-size="10.5" fill="var(--h-space)">impurity after</text>
<line x1="40" y1="78" x2="620" y2="78" stroke="var(--h-steel)" stroke-width="1.4"/>
<circle cx="64" cy="78" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="88" cy="78" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="112" cy="78" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="137" cy="78" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="161" cy="78" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="185" cy="78" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="217" cy="78" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="249" cy="78" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="282" cy="78" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="330" cy="78" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="378" cy="78" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="427" cy="78" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="475" cy="78" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="523" cy="78" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="572" cy="78" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="604" cy="78" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<line x1="620" y1="52" x2="620" y2="104" stroke="var(--md-default-fg-color)" stroke-width="2" stroke-dasharray="5 4"/>
<text x="30" y="82" text-anchor="end" font-size="11" font-weight="700" fill="var(--h-graphite)">no split</text>
<text x="636" y="82" text-anchor="end" font-size="11" font-weight="700" fill="var(--h-graphite)">0.47</text>
<line x1="40" y1="140" x2="620" y2="140" stroke="var(--h-steel)" stroke-width="1.4"/>
<circle cx="64" cy="140" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="88" cy="140" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="112" cy="140" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="137" cy="140" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="161" cy="140" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="185" cy="140" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="217" cy="140" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="249" cy="140" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="282" cy="140" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="330" cy="140" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="378" cy="140" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="427" cy="140" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="475" cy="140" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="523" cy="140" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="572" cy="140" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="604" cy="140" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<line x1="362" y1="114" x2="362" y2="166" stroke="var(--md-default-fg-color)" stroke-width="2" stroke-dasharray="5 4"/>
<text x="30" y="144" text-anchor="end" font-size="11" font-weight="700" fill="var(--h-graphite)">cut at 40</text>
<text x="636" y="144" text-anchor="end" font-size="11" font-weight="700" fill="var(--h-graphite)">0.38</text>
<line x1="40" y1="202" x2="620" y2="202" stroke="var(--h-steel)" stroke-width="1.4"/>
<circle cx="64" cy="202" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="88" cy="202" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="112" cy="202" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="137" cy="202" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="161" cy="202" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="185" cy="202" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="217" cy="202" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="249" cy="202" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="282" cy="202" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="330" cy="202" r="5.5" fill="var(--h-cherry)" opacity="0.9"/>
<circle cx="378" cy="202" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="427" cy="202" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="475" cy="202" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="523" cy="202" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="572" cy="202" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="604" cy="202" r="5.5" fill="var(--h-steel)" opacity="0.5"/>
<line x1="233" y1="176" x2="233" y2="228" stroke="var(--h-cherry)" stroke-width="3" stroke-dasharray="none"/>
<text x="30" y="206" text-anchor="end" font-size="11" font-weight="700" fill="var(--h-cherry)">cut at 24</text>
<text x="636" y="206" text-anchor="end" font-size="11" font-weight="700" fill="var(--h-cherry)">0.19</text>
<text x="340" y="244" text-anchor="middle" font-size="12" fill="var(--h-graphite)">The tree tries every threshold on every feature and keeps the one that makes the two sides purest.</text>
<text x="340" y="264" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Then it does the same thing again inside each side. That is the entire algorithm.</text>
</svg>

The tree scores every possible split by how **pure** the resulting groups are. A
group that is all churners or all stayers is pure. A fifty-fifty group is as
impure as it gets.

Two ways to measure it, and they rarely disagree:

**Gini impurity** is the chance that two items drawn at random from the group
have different labels. Zero when the group is pure, 0.5 at its worst for two
classes.

**Entropy** is the same idea from information theory, and you have met it. It is
the average surprise of a label drawn from the group, measured with the same
logarithm that produced cross-entropy on the Probability page.

Gini is slightly faster, entropy is slightly more sensitive to rare classes, and
the trees they produce are usually near identical. Do not spend time on the
choice.

<svg viewBox="0 0 680 296" role="img" aria-labelledby="t2-title t2-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="t2-title">A tree and the regions it carves out</title>
<desc id="t2-desc">Each question in the tree is a straight cut in the feature space. Following the branches down gives a set of rectangles, one per leaf.</desc>
<defs><marker id="t2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<text x="170" y="22" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">the tree</text>
<rect x="100" y="34" width="140" height="34" rx="7" fill="var(--h-cherry)"/>
<text x="170" y="51" dy="0.36em" text-anchor="middle" font-size="11.5" font-weight="700" fill="#ffffff">tenure &lt; 24?</text>
<line x1="140" y1="68" x2="90" y2="96" stroke="var(--h-steel)" stroke-width="1.6" marker-end="url(#t2)"/>
<line x1="200" y1="68" x2="250" y2="96" stroke="var(--h-steel)" stroke-width="1.6" marker-end="url(#t2)"/>
<text x="104" y="88" font-size="10" fill="var(--h-space)">yes</text>
<text x="226" y="88" font-size="10" fill="var(--h-space)">no</text>
<rect x="20" y="100" width="130" height="34" rx="7" fill="var(--h-cherry)" opacity="0.75"/>
<text x="85" y="117" dy="0.36em" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">charges &gt; 70?</text>
<rect x="196" y="100" width="110" height="34" rx="7" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="251" y="117" dy="0.36em" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-graphite)">stay</text>
<line x1="56" y1="134" x2="40" y2="160" stroke="var(--h-steel)" stroke-width="1.6" marker-end="url(#t2)"/>
<line x1="114" y1="134" x2="132" y2="160" stroke="var(--h-steel)" stroke-width="1.6" marker-end="url(#t2)"/>
<rect x="0" y="164" width="86" height="32" rx="7" fill="var(--h-cherry)"/>
<text x="43" y="180" dy="0.36em" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">churn</text>
<rect x="96" y="164" width="86" height="32" rx="7" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="139" y="180" dy="0.36em" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-graphite)">stay</text>
<text x="500" y="22" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">the same thing, as regions</text>
<rect x="360" y="34" width="280" height="176" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<line x1="453" y1="34" x2="453" y2="210" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<line x1="360" y1="112" x2="453" y2="112" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<rect x="361" y="35" width="91" height="76" fill="var(--h-cherry)" opacity="0.28"/>
<rect x="361" y="113" width="91" height="96" fill="var(--h-steel)" opacity="0.18"/>
<rect x="454" y="35" width="185" height="174" fill="var(--h-steel)" opacity="0.18"/>
<text x="406" y="76" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-cherry)">churn</text>
<text x="406" y="164" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-graphite)">stay</text>
<text x="546" y="124" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-graphite)">stay</text>
<text x="500" y="228" text-anchor="middle" font-size="10.5" fill="var(--h-space)">tenure along the bottom, monthly charges up the side</text>
<text x="340" y="262" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Every question is one straight cut, always parallel to an axis. Leaves are rectangles.</text>
<text x="340" y="282" text-anchor="middle" font-size="12" fill="var(--h-graphite)">This is why a tree needs no scaling: only the order of the values matters, never their size.</text>
</svg>

Reading this picture both ways is the thing to take from it. Each question is a
straight cut parallel to an axis, so the leaves are rectangles. That is the
axis-aligned boundary from the Classification page, now explained.

It is also why trees need **no scaling**. A split asks whether a value is above a
threshold, and that answer does not change if you multiply the column by a
thousand.

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
print(tree.score(X_train, y_train), tree.score(X_test, y_test))
```

```
1.0  0.727
```

Perfect on training data and worse than the baseline on test data. A tree left
unconstrained will keep splitting until every leaf is pure, which means it has
memorised the training set. This is overfitting in its purest form, and fixing it
is the next two sections.

## Reading the rules

The reason to care about a single tree is that you can read it.

```python
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt

print(export_text(tree, feature_names=list(X_train.columns), max_depth=3))
fig, ax = plt.subplots(figsize=(16, 8))
plot_tree(tree, feature_names=X_train.columns, class_names=["stay", "churn"],
          max_depth=3, filled=True, ax=ax)
```

Set `max_depth` on the display, not just on the model. A full tree on this
dataset has thousands of nodes and the picture is unreadable.

Each node tells you the question, how many samples reached it, how they were
distributed, and the impurity. Following one path from root to leaf gives you a
rule in plain language: short tenure, month to month contract, fiber internet,
predict churn.

!!! ml "ML connection"

    This is the most explainable model in the whole programme. A doctor or a
    loan officer can read a path and agree or disagree with it.

    Keep that in mind when you reach the ensembles later on this page. Five
    hundred trees voting are far more accurate and cannot be read at all, and
    that trade is a real decision rather than an obvious one.

## Pre-pruning

The first fix is to stop the tree before it goes too far.

```python
tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=50,
    min_samples_split=20,
    random_state=42,
)
```

| Parameter | What it does |
|---|---|
| `max_depth` | How many questions deep the tree may go |
| `min_samples_leaf` | A leaf must contain at least this many samples |
| `min_samples_split` | A node needs at least this many before it may split |
| `max_leaf_nodes` | A hard cap on the number of leaves |
| `min_impurity_decrease` | A split must improve purity by at least this much |

`min_samples_leaf` is the one to reach for first. A leaf built from three
customers is noise dressed as a rule, and setting a floor removes a whole class
of nonsense.

These are guesses, though. You are deciding in advance how complex the answer
should be, without knowing.

## Post-pruning

The second fix grows the tree fully and then cuts branches back, keeping only
those that earn their place.

<svg viewBox="0 0 680 286" role="img" aria-labelledby="t4-title t4-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="t4-title">Pre-pruning compared with post-pruning</title>
<desc id="t4-desc">Pre-pruning stops the tree growing past a limit you set in advance. Post-pruning grows the tree fully and then cuts branches back based on how much they earn.</desc>
<rect x="10" y="30" width="320" height="180" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="170" y="22" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">pre-pruning: stop early</text>
<rect x="350" y="30" width="320" height="180" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="510" y="22" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">post-pruning: grow, then cut</text>
<rect x="140" y="46" width="60" height="26" rx="6" fill="var(--h-cherry)" stroke="var(--h-surface-line)" opacity="1.0" stroke-dasharray="none"/>
<text x="170" y="59" dy="0.36em" text-anchor="middle" font-size="10" font-weight="700" fill="#ffffff" opacity="1.0">root</text>
<line x1="150+0" y1="72" x2="150" y2="72" stroke="none"/>
<line x1="170" y1="72" x2="118" y2="88" stroke="var(--h-steel)" stroke-width="1.4"/>
<rect x="88" y="88" width="60" height="26" rx="6" fill="var(--h-cherry)" stroke="var(--h-surface-line)" opacity="0.8" stroke-dasharray="none"/>
<text x="118" y="101" dy="0.36em" text-anchor="middle" font-size="10" font-weight="700" fill="#ffffff" opacity="0.8">split</text>
<line x1="170" y1="72" x2="222" y2="88" stroke="var(--h-steel)" stroke-width="1.4"/>
<rect x="192" y="88" width="60" height="26" rx="6" fill="var(--h-cherry)" stroke="var(--h-surface-line)" opacity="0.8" stroke-dasharray="none"/>
<text x="222" y="101" dy="0.36em" text-anchor="middle" font-size="10" font-weight="700" fill="#ffffff" opacity="0.8">split</text>
<line x1="118" y1="114" x2="92" y2="130" stroke="var(--h-steel)" stroke-width="1.4" opacity="1.0"/>
<rect x="68" y="130" width="48" height="24" rx="6" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)" opacity="1.0" stroke-dasharray="none"/>
<text x="92" y="142" dy="0.36em" text-anchor="middle" font-size="10" font-weight="700" fill="var(--h-graphite)" opacity="1.0">leaf</text>
<line x1="118" y1="114" x2="144" y2="130" stroke="var(--h-steel)" stroke-width="1.4" opacity="1.0"/>
<rect x="120" y="130" width="48" height="24" rx="6" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)" opacity="1.0" stroke-dasharray="none"/>
<text x="144" y="142" dy="0.36em" text-anchor="middle" font-size="10" font-weight="700" fill="var(--h-graphite)" opacity="1.0">leaf</text>
<line x1="222" y1="114" x2="196" y2="130" stroke="var(--h-steel)" stroke-width="1.4" opacity="1.0"/>
<rect x="172" y="130" width="48" height="24" rx="6" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)" opacity="1.0" stroke-dasharray="none"/>
<text x="196" y="142" dy="0.36em" text-anchor="middle" font-size="10" font-weight="700" fill="var(--h-graphite)" opacity="1.0">leaf</text>
<line x1="222" y1="114" x2="248" y2="130" stroke="var(--h-steel)" stroke-width="1.4" opacity="1.0"/>
<rect x="224" y="130" width="48" height="24" rx="6" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)" opacity="1.0" stroke-dasharray="none"/>
<text x="248" y="142" dy="0.36em" text-anchor="middle" font-size="10" font-weight="700" fill="var(--h-graphite)" opacity="1.0">leaf</text>
<line x1="30" y1="122" x2="310" y2="122" stroke="var(--md-default-fg-color)" stroke-width="2" stroke-dasharray="6 4"/>
<text x="170" y="180" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">max_depth stops it here, before it can overfit</text>
<text x="170" y="196" text-anchor="middle" font-size="10.5" fill="var(--h-space)">cheap, but the limit is a guess</text>
<rect x="480" y="46" width="60" height="26" rx="6" fill="var(--h-cherry)" stroke="var(--h-surface-line)" opacity="1.0" stroke-dasharray="none"/>
<text x="510" y="59" dy="0.36em" text-anchor="middle" font-size="10" font-weight="700" fill="#ffffff" opacity="1.0">root</text>
<line x1="150+340" y1="72" x2="150" y2="72" stroke="none"/>
<line x1="510" y1="72" x2="458" y2="88" stroke="var(--h-steel)" stroke-width="1.4"/>
<rect x="428" y="88" width="60" height="26" rx="6" fill="var(--h-cherry)" stroke="var(--h-surface-line)" opacity="0.8" stroke-dasharray="none"/>
<text x="458" y="101" dy="0.36em" text-anchor="middle" font-size="10" font-weight="700" fill="#ffffff" opacity="0.8">split</text>
<line x1="510" y1="72" x2="562" y2="88" stroke="var(--h-steel)" stroke-width="1.4"/>
<rect x="532" y="88" width="60" height="26" rx="6" fill="var(--h-cherry)" stroke="var(--h-surface-line)" opacity="0.8" stroke-dasharray="none"/>
<text x="562" y="101" dy="0.36em" text-anchor="middle" font-size="10" font-weight="700" fill="#ffffff" opacity="0.8">split</text>
<line x1="458" y1="114" x2="432" y2="130" stroke="var(--h-steel)" stroke-width="1.4" opacity="0.28"/>
<rect x="408" y="130" width="48" height="24" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)" opacity="0.28" stroke-dasharray="4 3"/>
<text x="432" y="142" dy="0.36em" text-anchor="middle" font-size="10" font-weight="700" fill="var(--h-graphite)" opacity="0.28">leaf</text>
<line x1="458" y1="114" x2="484" y2="130" stroke="var(--h-steel)" stroke-width="1.4" opacity="0.28"/>
<rect x="460" y="130" width="48" height="24" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)" opacity="0.28" stroke-dasharray="4 3"/>
<text x="484" y="142" dy="0.36em" text-anchor="middle" font-size="10" font-weight="700" fill="var(--h-graphite)" opacity="0.28">leaf</text>
<line x1="562" y1="114" x2="536" y2="130" stroke="var(--h-steel)" stroke-width="1.4" opacity="0.28"/>
<rect x="512" y="130" width="48" height="24" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)" opacity="0.28" stroke-dasharray="4 3"/>
<text x="536" y="142" dy="0.36em" text-anchor="middle" font-size="10" font-weight="700" fill="var(--h-graphite)" opacity="0.28">leaf</text>
<line x1="562" y1="114" x2="588" y2="130" stroke="var(--h-steel)" stroke-width="1.4" opacity="0.28"/>
<rect x="564" y="130" width="48" height="24" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)" opacity="0.28" stroke-dasharray="4 3"/>
<text x="588" y="142" dy="0.36em" text-anchor="middle" font-size="10" font-weight="700" fill="var(--h-graphite)" opacity="0.28">leaf</text>
<text x="510" y="180" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">the faded branches were grown, then removed</text>
<text x="510" y="196" text-anchor="middle" font-size="10.5" fill="var(--h-space)">slower, and the cut is chosen from evidence</text>
<text x="340" y="242" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Pre-pruning decides in advance how complex the tree may become. Post-pruning lets the data decide.</text>
<text x="340" y="262" text-anchor="middle" font-size="12" fill="var(--h-graphite)">In practice people use both: a sane depth limit to keep training fast, then ccp_alpha chosen by cross-validation.</text>
</svg>

Cost complexity pruning scores the whole tree as its error plus a penalty per
leaf. The size of that penalty is `ccp_alpha`.

$$
\text{cost} = \text{error} + \alpha \times (\text{number of leaves})
$$

This is the same trade you met with Ridge and Lasso: fit against complexity, with
one knob controlling the exchange rate. There, the penalty was on the size of the
coefficients. Here it is on the number of leaves.

scikit-learn will give you every alpha at which the tree changes shape.

```python
path = tree.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas[:-1]
print(len(alphas), alphas.min(), alphas.max())
```

The last value is dropped because it prunes the tree down to a single leaf.

Then train one tree per alpha and see which does best on held back data.

```python
from sklearn.model_selection import cross_val_score
import numpy as np

scores = []
for alpha in alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    scores.append(cross_val_score(clf, X_train, y_train, cv=5).mean())
best_alpha = alphas[int(np.argmax(scores))]
print(best_alpha, max(scores))
```

<svg viewBox="0 0 680 274" role="img" aria-labelledby="t3-title t3-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="t3-title">Accuracy as the pruning penalty grows</title>
<desc id="t3-desc">With no pruning the tree is perfect on training data and mediocre on test data. As the penalty grows both fall, but test accuracy rises first and peaks before declining. The peak is the value to choose.</desc>
<line x1="60" y1="200" x2="640" y2="200" stroke="var(--h-steel)" stroke-width="1.4"/>
<line x1="60" y1="200" x2="60" y2="40" stroke="var(--h-steel)" stroke-width="1.4"/>
<text x="350" y="224" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">ccp_alpha, the cost of keeping a branch</text>
<text x="52" y="46" text-anchor="end" font-size="10" fill="var(--h-space)">1.0</text>
<text x="52" y="204" text-anchor="end" font-size="10" fill="var(--h-space)">0.5</text>
<polyline points="60.0,40.0 65.8,40.1 71.6,40.4 77.4,40.7 83.2,41.1 89.0,41.6 94.8,42.1 100.6,42.6 106.4,43.2 112.2,43.8 118.0,44.5 123.8,45.1 129.6,45.9 135.4,46.6 141.2,47.4 147.0,48.2 152.8,49.0 158.6,49.9 164.4,50.8 170.2,51.7 176.0,52.6 181.8,53.5 187.6,54.5 193.4,55.5 199.2,56.6 205.0,57.6 210.8,58.7 216.6,59.8 222.4,60.9 228.2,62.0 234.0,63.1 239.8,64.3 245.6,65.5 251.4,66.7 257.2,67.9 263.0,69.2 268.8,70.4 274.6,71.7 280.4,73.0 286.2,74.3 292.0,75.6 297.8,77.0 303.6,78.3 309.4,79.7 315.2,81.1 321.0,82.5 326.8,83.9 332.6,85.4 338.4,86.8 344.2,88.3 350.0,89.8 355.8,91.3 361.6,92.8 367.4,94.3 373.2,95.9 379.0,97.4 384.8,99.0 390.6,100.6 396.4,102.2 402.2,103.8 408.0,105.4 413.8,107.1 419.6,108.7 425.4,110.4 431.2,112.1 437.0,113.8 442.8,115.5 448.6,117.2 454.4,119.0 460.2,120.7 466.0,122.5 471.8,124.2 477.6,126.0 483.4,127.8 489.2,129.6 495.0,131.5 500.8,133.3 506.6,135.1 512.4,137.0 518.2,138.9 524.0,140.7 529.8,142.6 535.6,144.5 541.4,146.5 547.2,148.4 553.0,150.3 558.8,152.3 564.6,154.3 570.4,156.2 576.2,158.2 582.0,160.2 587.8,162.2 593.6,164.2 599.4,166.3 605.2,168.3 611.0,170.4 616.8,172.4 622.6,174.5 628.4,176.6 634.2,178.7 640.0,180.8" fill="none" stroke="var(--h-steel)" stroke-width="2.5"/>
<polyline points="60.0,109.4 65.8,107.2 71.6,104.7 77.4,101.9 83.2,99.0 89.0,95.7 94.8,92.3 100.6,88.7 106.4,84.9 112.2,80.9 118.0,76.8 123.8,72.6 129.6,68.3 135.4,64.0 141.2,59.8 147.0,55.6 152.8,51.5 158.6,47.6 164.4,44.0 170.2,40.6 176.0,37.4 181.8,34.7 187.6,32.4 193.4,30.4 199.2,29.0 205.0,28.1 210.8,27.7 216.6,27.8 222.4,28.5 228.2,29.7 234.0,31.5 239.8,33.8 245.6,36.7 251.4,40.0 257.2,43.9 263.0,48.1 268.8,52.8 274.6,57.8 280.4,63.2 286.2,68.8 292.0,74.6 297.8,80.6 303.6,86.7 309.4,92.8 315.2,99.0 321.0,105.2 326.8,111.3 332.6,117.4 338.4,123.3 344.2,129.0 350.0,134.6 355.8,139.9 361.6,145.0 367.4,149.9 373.2,154.6 379.0,159.0 384.8,163.2 390.6,167.1 396.4,170.7 402.2,174.2 408.0,177.4 413.8,180.4 419.6,183.2 425.4,185.8 431.2,188.2 437.0,190.5 442.8,192.6 448.6,194.5 454.4,196.4 460.2,198.1 466.0,199.7 471.8,201.2 477.6,202.7 483.4,204.0 489.2,205.3 495.0,206.6 500.8,207.8 506.6,209.0 512.4,210.1 518.2,211.2 524.0,212.3 529.8,213.4 535.6,214.4 541.4,215.4 547.2,216.4 553.0,217.4 558.8,218.4 564.6,219.4 570.4,220.4 576.2,221.4 582.0,222.4 587.8,223.3 593.6,224.3 599.4,225.3 605.2,226.2 611.0,227.2 616.8,228.2 622.6,229.1 628.4,230.1 634.2,231.0 640.0,232.0" fill="none" stroke="var(--h-cherry)" stroke-width="3"/>
<line x1="222" y1="28" x2="222" y2="200" stroke="var(--h-cherry)" stroke-width="2" stroke-dasharray="5 4"/>
<circle cx="222" cy="28" r="6" fill="var(--h-cherry)"/>
<text x="222" y="14" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">best ccp_alpha</text>
<text x="110" y="62" font-size="11.5" font-weight="700" fill="var(--h-graphite)">training</text>
<text x="110" y="148" font-size="11.5" font-weight="700" fill="var(--h-cherry)">test</text>
<text x="640" y="196" text-anchor="end" font-size="10.5" fill="var(--h-space)">a single leaf</text>
<text x="340" y="250" text-anchor="middle" font-size="12" fill="var(--h-graphite)">On the left the tree has memorised the training set. On the right it has been cut back to nothing.</text>
<text x="340" y="268" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Only the held back set shows you the peak, which is the whole reason you kept one.</text>
</svg>

Use cross-validation rather than a single split for this, and never the test set.
Choosing alpha by looking at test scores means the test set has helped build the
model, which is the leakage you read about two pages ago.

**In practice** people use both kinds of pruning: a sensible `max_depth` to keep
training fast, then `ccp_alpha` chosen by cross-validation to do the fine work.

## Many trees are better than one

A single tree has a real weakness beyond overfitting: it is **unstable**. Change
a handful of rows and the top split can change, and every branch below it moves.

That instability turns out to be useful. Build many trees that are each unstable
in a different direction, and the errors cancel out.

<svg viewBox="0 0 680 300" role="img" aria-labelledby="t5-title t5-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="t5-title">Bagging compared with boosting</title>
<desc id="t5-desc">Bagging trains many independent trees on different samples at the same time and averages their votes. Boosting trains trees one after another, each correcting the mistakes of the ones before it.</desc>
<defs><marker id="t5" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<rect x="10" y="30" width="320" height="196" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="170" y="22" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">bagging, as in a random forest</text>
<rect x="350" y="30" width="320" height="196" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="510" y="22" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">boosting</text>
<rect x="120" y="44" width="100" height="24" rx="5" fill="var(--md-default-fg-color)"/>
<text x="170" y="56" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--md-default-bg-color)">the data</text>
<line x1="170" y1="68" x2="74" y2="88" stroke="var(--h-steel)" stroke-width="1.4" marker-end="url(#t5)"/>
<text x="74" y="84" text-anchor="middle" font-size="9" fill="var(--h-space)">sample 1</text>
<polygon points="74.0,92 98,138 50,138" fill="var(--h-cherry)" opacity="0.85"/>
<line x1="74" y1="140" x2="170" y2="164" stroke="var(--h-steel)" stroke-width="1.4" marker-end="url(#t5)"/>
<line x1="170" y1="68" x2="138" y2="88" stroke="var(--h-steel)" stroke-width="1.4" marker-end="url(#t5)"/>
<text x="138" y="84" text-anchor="middle" font-size="9" fill="var(--h-space)">sample 2</text>
<polygon points="138.0,92 162,138 114,138" fill="var(--h-cherry)" opacity="0.85"/>
<line x1="138" y1="140" x2="170" y2="164" stroke="var(--h-steel)" stroke-width="1.4" marker-end="url(#t5)"/>
<line x1="170" y1="68" x2="202" y2="88" stroke="var(--h-steel)" stroke-width="1.4" marker-end="url(#t5)"/>
<text x="202" y="84" text-anchor="middle" font-size="9" fill="var(--h-space)">sample 3</text>
<polygon points="202.0,92 226,138 178,138" fill="var(--h-cherry)" opacity="0.85"/>
<line x1="202" y1="140" x2="170" y2="164" stroke="var(--h-steel)" stroke-width="1.4" marker-end="url(#t5)"/>
<line x1="170" y1="68" x2="266" y2="88" stroke="var(--h-steel)" stroke-width="1.4" marker-end="url(#t5)"/>
<text x="266" y="84" text-anchor="middle" font-size="9" fill="var(--h-space)">sample 4</text>
<polygon points="266.0,92 290,138 242,138" fill="var(--h-cherry)" opacity="0.85"/>
<line x1="266" y1="140" x2="170" y2="164" stroke="var(--h-steel)" stroke-width="1.4" marker-end="url(#t5)"/>
<rect x="108" y="168" width="124" height="28" rx="6" fill="var(--h-cherry)"/>
<text x="170" y="182" dy="0.36em" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">average the votes</text>
<text x="170" y="214" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">all at once, independently, then combined</text>
<polygon points="390.0,90 412,134 368,134" fill="var(--h-cherry)" opacity="0.55"/>
<text x="390" y="150" text-anchor="middle" font-size="9.5" fill="var(--h-space)">tree 1</text>
<line x1="414" y1="112" x2="436" y2="112" stroke="var(--h-steel)" stroke-width="1.6" marker-end="url(#t5)"/>
<text x="425" y="104" text-anchor="middle" font-size="8.5" fill="var(--h-cherry)">errors</text>
<polygon points="462.0,90 484,134 440,134" fill="var(--h-cherry)" opacity="0.7000000000000001"/>
<text x="462" y="150" text-anchor="middle" font-size="9.5" fill="var(--h-space)">tree 2</text>
<line x1="486" y1="112" x2="508" y2="112" stroke="var(--h-steel)" stroke-width="1.6" marker-end="url(#t5)"/>
<text x="497" y="104" text-anchor="middle" font-size="8.5" fill="var(--h-cherry)">errors</text>
<polygon points="534.0,90 556,134 512,134" fill="var(--h-cherry)" opacity="0.8500000000000001"/>
<text x="534" y="150" text-anchor="middle" font-size="9.5" fill="var(--h-space)">tree 3</text>
<line x1="558" y1="112" x2="580" y2="112" stroke="var(--h-steel)" stroke-width="1.6" marker-end="url(#t5)"/>
<text x="569" y="104" text-anchor="middle" font-size="8.5" fill="var(--h-cherry)">errors</text>
<polygon points="606.0,90 628,134 584,134" fill="var(--h-cherry)" opacity="1.0"/>
<text x="606" y="150" text-anchor="middle" font-size="9.5" fill="var(--h-space)">tree 4</text>
<rect x="396" y="168" width="228" height="28" rx="6" fill="var(--h-cherry)"/>
<text x="510" y="182" dy="0.36em" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">add them up, in order</text>
<text x="510" y="214" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">one after another, each fixing what is left</text>
<text x="340" y="252" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Bagging reduces variance: many unstable trees average out to something stable. It is hard to overfit.</text>
<text x="340" y="272" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Boosting reduces bias: it keeps attacking the remaining errors. It is easy to overfit, and usually more accurate.</text>
<text x="340" y="292" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-cherry)">Bagging is parallel and forgiving. Boosting is sequential and sharp.</text>
</svg>

**Bagging** trains many trees at the same time on different random samples of the
data, then averages their votes. Because the trees are independent, their
mistakes are independent, and averaging cancels them. Bagging reduces variance
and is very hard to overfit.

**Boosting** trains trees one after another, each one focused on what the
previous ones got wrong. It reduces bias, is usually more accurate, and will
overfit if you let it run too long.

| | Bagging | Boosting |
|---|---|---|
| Trees are built | In parallel, independently | In sequence, each depending on the last |
| Each tree is | Deep and overfitted | Shallow and weak |
| It reduces | Variance | Bias |
| Overfits | Rarely | Yes, if unchecked |
| Tuning | Forgiving | Fiddly, and worth it |

## Random forests

A random forest is bagging with one extra idea.

```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
forest.fit(X_train, y_train)
print(forest.score(X_test, y_test))
```

Two sources of randomness, and the second is what makes it a forest rather than
just bagged trees.

**Bootstrap sampling.** Each tree sees a random sample of the rows, drawn with
replacement, so each one sees a slightly different dataset.

**Random feature subsets.** At every split, each tree may only consider a random
subset of the features, `sqrt` of them by default for classification.

The second one matters more than it looks. Without it, if one feature is strongly
predictive, every tree puts it at the root and all the trees end up nearly
identical. Forcing them to ignore it sometimes produces genuinely different
trees, and different trees are the entire point.

`n_estimators` is the one parameter you do not need to tune carefully. More trees
never hurt accuracy, they only cost time. Three hundred is plenty, and the curve
is flat long before that.

`n_jobs=-1` uses every core on your machine. The trees are independent, so this
is free speed.

## Feature importance, and its traps

A forest can tell you which features it relied on.

```python
import pandas as pd

importances = pd.Series(forest.feature_importances_, index=X_train.columns)
print(importances.sort_values(ascending=False).head(10))
```

This is **impurity-based importance**: how much each feature reduced impurity,
totalled across every split in every tree. It is free, because it was computed
during training, and it has two serious biases.

!!! warning "Impurity importance inflates some features and splits others"

    **It favours features with many distinct values.** A continuous column offers
    hundreds of possible thresholds, so it gets more chances to look useful than
    a yes or no column. An ID column, if you forgot to drop it, will top the
    chart.

    **It splits credit between correlated features.** If `tenure` and
    `TotalCharges` carry the same information, the forest uses each about half
    the time and both appear half as important as either one really is. You saw
    that correlation on the heatmap during preparation, which is why you drew it.

**Permutation importance** avoids both problems. Shuffle one column, see how much
the score drops, put it back, repeat.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(forest, X_test, y_test, n_repeats=10, random_state=42)
perm = pd.Series(result.importances_mean, index=X_test.columns)
print(perm.sort_values(ascending=False).head(10))
```

It is slower, it measures importance on held back data rather than training data,
and it answers a more honest question: how much does the model's performance
actually depend on this column?

The correlated-features problem remains, in a different form. Shuffle one of two
correlated columns and the model leans on the other, so both look unimportant.
There is no way around this except knowing your data.

For explaining individual predictions rather than the model as a whole, SHAP from
the Linear Models page works on trees too, and has a fast implementation designed
for them.

## Boosting

<svg viewBox="0 0 680 254" role="img" aria-labelledby="t6-title t6-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="t6-title">What boosting does over successive rounds</title>
<desc id="t6-desc">After one weak tree the prediction is a crude step. After several rounds the combined prediction follows the data closely, because each round fits what the previous ones got wrong.</desc>
<rect x="5" y="26" width="214" height="150" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="112" y="18" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">after 1 tree</text>
<polyline points="23,71 112,71 112,73 201,73" fill="none" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<circle cx="23" cy="103" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="32" cy="96" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="41" cy="89" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="50" cy="83" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="59" cy="77" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="68" cy="71" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="76" cy="67" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="85" cy="63" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="94" cy="61" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="103" cy="59" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="112" cy="59" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="121" cy="60" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="130" cy="62" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="139" cy="65" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="148" cy="68" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="156" cy="73" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="165" cy="79" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="174" cy="85" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="183" cy="92" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="192" cy="99" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="201" cy="106" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<rect x="233" y="26" width="214" height="150" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="340" y="18" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">after 3 trees</text>
<polyline points="251,89 287,89 287,67 322,67 322,59 358,59 358,68 393,68 393,92 429,92" fill="none" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<circle cx="251" cy="103" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="260" cy="96" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="269" cy="89" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="278" cy="83" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="287" cy="77" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="296" cy="71" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="304" cy="67" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="313" cy="63" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="322" cy="61" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="331" cy="59" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="340" cy="59" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="349" cy="60" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="358" cy="62" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="367" cy="65" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="376" cy="68" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="384" cy="73" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="393" cy="79" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="402" cy="85" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="411" cy="92" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="420" cy="99" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="429" cy="106" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<rect x="461" y="26" width="214" height="150" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="568" y="18" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-cherry)">after 20 trees</text>
<polyline points="479,98 492,98 492,88 504,88 504,79 517,79 517,71 530,71 530,65 543,65 543,61 555,61 555,59 568,59 568,59 581,59 581,62 593,62 593,67 606,67 606,73 619,73 619,81 632,81 632,91 644,91 644,101 657,101" fill="none" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<circle cx="479" cy="103" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="488" cy="96" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="497" cy="89" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="506" cy="83" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="515" cy="77" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="524" cy="71" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="532" cy="67" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="541" cy="63" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="550" cy="61" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="559" cy="59" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="568" cy="59" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="577" cy="60" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="586" cy="62" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="595" cy="65" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="604" cy="68" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="612" cy="73" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="621" cy="79" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="630" cy="85" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="639" cy="92" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="648" cy="99" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="657" cy="106" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<text x="340" y="206" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Each tree is deliberately weak, often only a few levels deep. Alone it is barely better than guessing.</text>
<text x="340" y="226" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Hundreds of them, each cleaning up after the last, produce the strongest models available for tabular data.</text>
<text x="340" y="246" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-cherry)">Keep going too long and the steps start chasing the noise.</text>
</svg>

Each tree in a boosted ensemble is deliberately weak, often only two or three
levels deep. Alone it is barely better than guessing. In sequence, each one
attacking what is left over, they become the strongest tabular models available.

**AdaBoost** was the original. After each round it increases the weight of the
examples that were misclassified, so the next tree pays more attention to them.
Simple, and sensitive to noisy labels, because a mislabelled row gets weighted up
round after round.

**Gradient boosting** is the modern approach and the one to understand. Instead
of reweighting rows, each new tree is trained to predict the **residual**, which
is what the ensemble so far got wrong. Add its prediction to the running total
and the error shrinks. It is gradient descent, with a tree as the step.

```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

ada = AdaBoostClassifier(n_estimators=200, random_state=42)
gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)
```

Three parameters matter and they interact:

**`learning_rate`** shrinks each tree's contribution. Smaller means slower and
usually better, because no single tree can dominate.

**`n_estimators`** is how many rounds. Unlike a forest, more is not always
better: past a point the ensemble starts fitting noise.

**`max_depth`** controls how complex each weak learner is. Two to four is normal.
Deep trees in a boosted ensemble overfit quickly.

Low learning rate with many estimators is the reliable combination. Start at 0.05
and 300.

### XGBoost and LightGBM

Both are gradient boosting, engineered hard.

**XGBoost** added regularisation to the tree-building objective, handles missing
values natively, and was fast enough to win most tabular competitions for years.

**LightGBM** grows trees leaf by leaf instead of level by level and bins
continuous features before training. Much faster on large data, and slightly more
prone to overfitting small datasets.

Both live outside scikit-learn and follow the same interface, so swapping them in
is one line.

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=42)
```

scikit-learn's own `HistGradientBoostingClassifier` uses the same binning idea,
is fast, needs no extra install, and is a perfectly good default.

!!! tip "The order worth trying on a new table"

    A dummy baseline. Then logistic regression, for the explanation. Then a
    random forest, which usually works with no tuning at all. Then gradient
    boosting, which usually wins and needs some care.

    Stop at the first one that is good enough for what you need. Every step
    beyond costs you interpretability and time.

## Evaluating them

Everything from the Classification page applies without change: confusion matrix,
precision, recall, F1, ROC AUC, the metric chosen before you see the scores, and
always against the dummy baseline.

```python
from sklearn.metrics import classification_report, roc_auc_score

pred = forest.predict(X_test)
proba = forest.predict_proba(X_test)[:, 1]
print(classification_report(y_test, pred))
print(roc_auc_score(y_test, proba))
```

Two things specific to trees.

**Class weights.** Tree models accept `class_weight="balanced"`, which makes the
minority class count for more during training. On imbalanced data like churn it
is often a bigger improvement than any amount of tuning.

**Probabilities from a forest** are the proportion of trees that voted for the
class. They are reasonable but not well calibrated, so if you need the number
itself to mean something rather than just to rank, look at
`CalibratedClassifierCV`.

## When it goes wrong

??? note "Training accuracy is 1.0 and test accuracy is poor"

    An unpruned tree. Set `max_depth` and `min_samples_leaf`, then tune
    `ccp_alpha` with cross-validation.

??? note "`plot_tree` produces an unreadable mess"

    Pass `max_depth=3` to the plotting call and make the figure larger. For a
    full description use `export_text` instead.

??? note "The most important feature is an ID or a date"

    You forgot to drop an identifier, or you have a leaking feature. Both were
    covered on the Data Preprocessing page and both show up first in an
    importance chart.

??? note "Two features I know matter both look unimportant"

    They are correlated, so the forest and the permutation test both split the
    credit between them. Check the correlation heatmap, and consider dropping
    one.

??? note "My boosted model is worse than the forest"

    Usually the learning rate is too high, the trees are too deep, or you ran
    too many rounds. Try `learning_rate=0.05`, `max_depth=3`, and watch the
    validation score as rounds increase.

??? note "The forest takes forever"

    Set `n_jobs=-1`, lower `n_estimators` while experimenting, and raise
    `min_samples_leaf`. Accuracy barely changes and training time falls sharply.

## Check yourself

1. Train an unpruned tree and report both training and test accuracy. Explain the
   gap in one sentence.
2. Print the first three levels of the rules and translate one root-to-leaf path
   into plain language.
3. Pre-prune with three different `max_depth` values and record what happens to
   both scores.
4. Get the pruning path, train a tree for each alpha, and plot accuracy against
   alpha. Mark the best one and say how you chose it.
5. Train a random forest and compare it with your best single tree. Report the
   improvement and the cost.
6. Compare impurity importance with permutation importance on the same forest.
   Name a feature that moves, and explain why.
7. Train AdaBoost and gradient boosting. Then halve the learning rate and double
   the estimators, and describe what changed.
8. Compare every model with cross-validation, reporting the mean and the spread
   against the dummy baseline.

## Quick reference

| Task | Code |
|---|---|
| A tree | `DecisionTreeClassifier(max_depth=5, min_samples_leaf=50)` |
| Read the rules | `export_text(tree, feature_names=cols)` |
| Draw it | `plot_tree(tree, max_depth=3, filled=True)` |
| Pruning path | `tree.cost_complexity_pruning_path(X, y).ccp_alphas` |
| Post-prune | `DecisionTreeClassifier(ccp_alpha=best)` |
| Random forest | `RandomForestClassifier(n_estimators=300, n_jobs=-1)` |
| Impurity importance | `forest.feature_importances_` |
| Permutation importance | `permutation_importance(model, X_test, y_test)` |
| AdaBoost | `AdaBoostClassifier(n_estimators=200)` |
| Gradient boosting | `GradientBoostingClassifier(learning_rate=0.05, max_depth=3)` |
| Fast boosting | `HistGradientBoostingClassifier()` or `LGBMClassifier()` |
| Imbalance | `class_weight="balanced"` |
| Honest comparison | `cross_val_score(model, X_train, y_train, cv=5)` |

## Questions to sit with

1. A single tree can be read by the person it affects. Five hundred trees cannot,
   and score several points higher. Who should decide which to deploy, and what
   would you need to tell them?
2. Pre-pruning fixes the complexity in advance and post-pruning lets the data
   choose. Where else have you seen that distinction, and which do you instinctively
   prefer?
3. Impurity importance favours features with many distinct values. How many
   published "most important factor" claims do you think survive that fact?
4. Bagging works because the trees make independent mistakes. What would make
   them stop being independent, and what would that do to the forest?
5. Boosting keeps attacking the errors that remain. On a dataset with some
   mislabelled rows, what is it eventually attacking?

## Next

Everything so far has needed a target column. The next two projects ask what you
can learn when nobody gave you one.

[Unsupervised Learning](unsupervised-learning.md){ .h-button }

Worth your time: the
[scikit-learn ensembles guide](https://scikit-learn.org/stable/modules/ensemble.html)
covers every variant here, and the
[permutation importance page](https://scikit-learn.org/stable/modules/permutation_importance.html)
explains the bias in impurity importance with worked examples.

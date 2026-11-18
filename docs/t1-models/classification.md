# Classification

<span class="badge badge--time">70 min</span>
<span class="badge badge--level">Classic models</span>
<span class="badge">The map for this group</span>

This page teaches no algorithm. It is the map you will read the next four
projects against, so that each one feels like filling in a region rather than
arriving somewhere unfamiliar.

## Why this matters

You have spent a trimester producing a clean, encoded, split table. A model is
what finally reads it.

Newcomers assume that this is where the skill lives, and that choosing the right
algorithm is the job. It is not. Switching between reasonable models usually
moves your score by a couple of points. What moves it far more is the work you
already did, and what protects it is knowing how to tell a real improvement from
a flattering one.

So this page covers four things: what kind of question you are asking, what
kinds of model exist, how they all share one interface, and how you will know
whether any of them worked.

## Where models sit

<svg viewBox="0 0 680 262" role="img" aria-labelledby="m1-title m1-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="m1-title">The model half of the pipeline</title>
<desc id="m1-desc">The data half ends with a prepared table. The model half runs from a baseline through training, evaluation and tuning, to deployment, and sends you back to the data whenever the answer is disappointing.</desc>
<defs><marker id="m1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<rect x="5" y="52" width="150" height="62" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)" stroke-dasharray="5 4"/>
<text x="80" y="78" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-graphite)">a prepared table</text>
<text x="80" y="97" text-anchor="middle" font-size="10" fill="var(--h-space)">everything so far</text>
<line x1="159" y1="83" x2="175" y2="83" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#m1)"/>
<rect x="179" y="52" width="114" height="62" rx="10" fill="var(--h-cherry)" stroke="var(--h-cherry)"/>
<text x="236" y="78" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">baseline</text>
<text x="236" y="97" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.9">the score to beat</text>
<line x1="297" y1="83" x2="305" y2="83" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#m1)"/>
<rect x="305" y="52" width="114" height="62" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="362" y="78" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">train</text>
<text x="362" y="97" text-anchor="middle" font-size="10" fill="var(--h-graphite)" opacity="0.9">fit on train</text>
<line x1="423" y1="83" x2="431" y2="83" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#m1)"/>
<rect x="431" y="52" width="114" height="62" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="488" y="78" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">evaluate</text>
<text x="488" y="97" text-anchor="middle" font-size="10" fill="var(--h-graphite)" opacity="0.9">on held back data</text>
<line x1="549" y1="83" x2="557" y2="83" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#m1)"/>
<rect x="557" y="52" width="114" height="62" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="614" y="78" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">tune</text>
<text x="614" y="97" text-anchor="middle" font-size="10" fill="var(--h-graphite)" opacity="0.9">then evaluate again</text>
<polyline points="629,118 629,150 240,150 240,120" fill="none" stroke="var(--h-steel)" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#m1)"/>
<text x="434" y="168" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-graphite)">not good enough yet</text>
<polyline points="400,118 400,196 80,196 80,120" fill="none" stroke="var(--h-cherry)" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#m1)"/>
<text x="240" y="214" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">usually the data, not the model</text>
<text x="340" y="30" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-graphite)">the model half of the same pipeline</text>
<text x="340" y="244" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Beginners loop on the short arrow. Experienced people loop on the long one.</text>
</svg>

Notice the two loops. The short one sends you back to try another model or
another setting. The long one sends you back to the data.

Beginners spend most of their time on the short loop. People who have done this
for a while spend most of theirs on the long one, because that is where the
gains are.

## Classification and regression

Every supervised problem is one of these two, and the difference is what the
answer looks like.

<svg viewBox="0 0 680 286" role="img" aria-labelledby="m2-title m2-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="m2-title">Classification compared with regression</title>
<desc id="m2-desc">A classifier draws a boundary between groups and answers with a category. A regressor fits a line or surface through the points and answers with a number.</desc>
<rect x="10" y="30" width="320" height="180" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="170" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">classification</text>
<circle cx="198.2" cy="153.9" r="4" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="244.7" cy="181.9" r="4" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="229.8" cy="179.1" r="4" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="37.8" cy="115.2" r="4" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="284.7" cy="140.9" r="4" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="273.2" cy="65.8" r="4" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="156.6" cy="84.5" r="4" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="176.8" cy="130.4" r="4" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="33.5" cy="80.3" r="4" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="105.5" cy="178.3" r="4" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="236.7" cy="72.3" r="4" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="245.2" cy="69.4" r="4" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="196.7" cy="67.7" r="4" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="30.5" cy="172.0" r="4" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="86.6" cy="80.2" r="4" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="295.3" cy="172.1" r="4" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="108.1" cy="184.6" r="4" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="175.6" cy="144.9" r="4" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="85.3" cy="181.7" r="4" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="216.5" cy="185.3" r="4" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="271.3" cy="91.8" r="4" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="127.5" cy="73.2" r="4" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="69.3" cy="59.1" r="4" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="111.4" cy="134.4" r="4" fill="var(--h-cherry)" opacity="0.85"/>
<line x1="22" y1="198" x2="318" y2="36" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<text x="170" y="232" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-graphite)">a boundary between groups</text>
<text x="170" y="250" text-anchor="middle" font-size="11" fill="var(--h-space)">the answer is a category, with a probability behind it</text>
<rect x="350" y="30" width="320" height="180" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="510" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">regression</text>
<circle cx="370.9" cy="183.2" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="456.8" cy="140.5" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="599.2" cy="79.9" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="504.7" cy="135.1" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="567.3" cy="147.9" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="376.4" cy="196.0" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="579.9" cy="103.5" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="590.6" cy="94.8" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="472.5" cy="145.0" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="383.1" cy="183.5" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="420.7" cy="178.9" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="581.6" cy="98.2" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="630.3" cy="94.4" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="469.3" cy="143.0" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="516.9" cy="129.5" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="579.6" cy="94.4" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="593.2" cy="99.0" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="634.8" cy="75.4" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="395.5" cy="167.4" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="627.1" cy="100.5" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="465.2" cy="167.9" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="457.5" cy="144.0" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="458.7" cy="155.6" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="411.7" cy="178.3" r="4" fill="var(--h-cherry)" opacity="0.75"/>
<line x1="368" y1="191" x2="654" y2="71" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<text x="510" y="232" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-graphite)">a line through the points</text>
<text x="510" y="250" text-anchor="middle" font-size="11" fill="var(--h-space)">the answer is a number, with an error either side</text>
<text x="340" y="278" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Same data, same features, different question. The question decides the model family, the metric and the loss.</text>
</svg>

**Classification** predicts a category. Will this customer churn, is this email
spam, which of ten digits is this.

**Regression** predicts a number. How much will this flat sell for, how many
units will we ship next month.

The same features can serve either. From the churn data you could predict whether
somebody leaves, which is classification, or how many months they will stay,
which is regression. The question you ask determines the model family, the metric
and the loss function, so it is worth being explicit about it before you start.

### Classification comes in three shapes

| Shape | One example gets | Example |
|---|---|---|
| **Binary** | One of two labels | Churn or stay |
| **Multi-class** | One of many labels | Which of three internet plans |
| **Multi-label** | Any number of labels at once | Which services a customer holds |

You met these on the Framing a Problem page. The distinction matters here
because it changes which metrics are available to you and how the model's output
is shaped.

### A classifier gives you a probability, not an answer

This is the part that surprises people, and it is the most useful thing in this
section.

```python
model.predict(X_test)
model.predict_proba(X_test)
```

```
['No' 'Yes' 'No']
[[0.82 0.18]
 [0.31 0.69]
 [0.95 0.05]]
```

The model produces a probability for each class. `predict` then applies a
threshold, by default 0.5, and hands you a label.

That threshold is a choice, and it is yours. Lower it to 0.3 and you catch more
churners and raise more false alarms. Raise it to 0.7 and the opposite. Nothing
about the model changed. You moved a dial that trades one kind of mistake
against the other.

!!! ml "ML connection"

    The probability comes from the cross-entropy loss you met on the Probability
    page. The model is trained to make the probability it assigns to the correct
    answer as high as possible, which is why `predict_proba` exists at all.

    Where to put the threshold is a business decision. If missing a churner costs
    you a customer and a false alarm costs you a phone call, those are not
    equally bad and 0.5 is almost certainly the wrong place.

## The families of algorithms

Each of these gets its own project. What follows is what they are, so the map is
complete before you walk into the first one.

**Linear models.** Logistic regression for classification, linear regression for
numbers. They assume the effect of each feature adds up. Fast, and the
coefficients tell you what the model thinks, which is why they are still the
default in medicine and finance. Need scaling.

**Tree-based models.** A decision tree asks a series of yes or no questions. One
tree is easy to read and unstable. Many trees combined, as a random forest or a
gradient boosted ensemble, are the strongest thing you will use on tabular data
all year. No scaling needed, handles mixed types, less interpretable as the
ensemble grows.

**Distance-based models.** K-nearest neighbours answers by looking at the most
similar rows it has seen. No training as such, all the work happens at prediction
time. Scaling is critical here, because the distance is meaningless without it.

**Probabilistic models.** Naive Bayes applies Bayes' theorem while assuming the
features are independent. That assumption is almost never true and the model
often works anyway. Very fast, still competitive for text.

**Margin-based models.** Support vector machines find the boundary with the
widest gap between classes, and can bend it into curves. Strong on small and
medium datasets, slow on large ones, and the results are hard to explain.

| Family | Scaling | Categoricals | Interpretable | Good first choice for |
|---|---|---|---|---|
| Linear | Required | Encode first | Very | Anything you must explain |
| Trees, single | No | Encode first | Very | Understanding the data |
| Trees, ensemble | No | Encode first | Somewhat | Tabular data, usually the winner |
| Distance based | Critical | Encode first | Poorly | Small clean datasets |
| Probabilistic | No | Naturally | Somewhat | Text, and speed |
| Margin based | Required | Encode first | Poorly | Small, hard boundaries |

### Unsupervised learning is on this map too

!!! tip "Where clustering fits"

    Everything above is **supervised**: you have labels and the model learns to
    reproduce them.

    Two of the projects in this group have no labels at all. **Unsupervised
    Learning** and **Clustering** ask a different question: given these rows and
    no answers, what natural groups exist? There is no target column, so none of
    the metrics on this page apply, and success is judged differently.

    It is the same map, a different continent. Keep it in view, because when
    somebody hands you data with no labels, the supervised toolkit is not an
    option.

## What the families actually differ on

One picture explains more than the table above.

<svg viewBox="0 0 680 266" role="img" aria-labelledby="m3-title m3-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="m3-title">The shape of boundary each family can draw</title>
<desc id="m3-desc">A linear model draws one straight line. A tree draws rectangles aligned to the axes. A distance based model draws an irregular boundary that follows the individual points.</desc>
<rect x="5" y="30" width="214" height="150" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="112" y="22" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">linear</text>
<text x="112" y="202" text-anchor="middle" font-size="10.5" fill="var(--h-space)">one straight cut</text>
<rect x="233" y="30" width="214" height="150" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="340" y="22" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">tree</text>
<text x="340" y="202" text-anchor="middle" font-size="10.5" fill="var(--h-space)">boxes, axis aligned</text>
<rect x="461" y="30" width="214" height="150" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="568" y="22" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">distance based</text>
<text x="568" y="202" text-anchor="middle" font-size="10.5" fill="var(--h-space)">follows the points</text>
<line x1="18" y1="168" x2="208" y2="44" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<circle cx="47.0" cy="129.7" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="107.9" cy="68.1" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="171.7" cy="106.6" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="60.8" cy="40.4" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="128.6" cy="49.0" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="60.1" cy="70.3" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="158.3" cy="153.8" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="21.6" cy="82.8" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="37.5" cy="163.8" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="45.0" cy="146.3" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="195.0" cy="98.9" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="81.5" cy="103.0" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="22.2" cy="152.6" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="166.4" cy="67.1" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="188.4" cy="138.1" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="127.7" cy="134.9" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="107.4" cy="147.5" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="194.1" cy="155.4" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="85.9" cy="98.7" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="208.0" cy="90.3" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="154.3" cy="169.1" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="110.4" cy="111.6" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="58.1" cy="119.2" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="135.8" cy="63.5" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="82.6" cy="160.5" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="130.9" cy="79.9" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<polyline points="240,110 340,110 340,60 448,60" fill="none" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<polyline points="340,110 340,168" fill="none" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<circle cx="275.0" cy="129.7" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="335.9" cy="68.1" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="399.7" cy="106.6" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="288.8" cy="40.4" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="356.6" cy="49.0" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="288.1" cy="70.3" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="386.3" cy="153.8" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="249.6" cy="82.8" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="265.5" cy="163.8" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="273.0" cy="146.3" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="423.0" cy="98.9" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="309.5" cy="103.0" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="250.2" cy="152.6" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="394.4" cy="67.1" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="416.4" cy="138.1" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="355.7" cy="134.9" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="335.4" cy="147.5" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="422.1" cy="155.4" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="313.9" cy="98.7" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="436.0" cy="90.3" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="382.3" cy="169.1" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="338.4" cy="111.6" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="286.1" cy="119.2" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="363.8" cy="63.5" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="310.6" cy="160.5" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="358.9" cy="79.9" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<polyline points="472,172 494,140 512,150 530,118 552,128 570,96 592,104 612,72 636,80 660,52" fill="none" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<circle cx="503.0" cy="129.7" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="563.9" cy="68.1" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="627.7" cy="106.6" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="516.8" cy="40.4" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="584.6" cy="49.0" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="516.1" cy="70.3" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="614.3" cy="153.8" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="477.6" cy="82.8" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="493.5" cy="163.8" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="501.0" cy="146.3" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="651.0" cy="98.9" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="537.5" cy="103.0" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="478.2" cy="152.6" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="622.4" cy="67.1" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="644.4" cy="138.1" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="583.7" cy="134.9" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="563.4" cy="147.5" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="650.1" cy="155.4" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="541.9" cy="98.7" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="664.0" cy="90.3" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="610.3" cy="169.1" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="566.4" cy="111.6" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="514.1" cy="119.2" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="591.8" cy="63.5" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="538.6" cy="160.5" r="3.6" fill="var(--h-steel)" opacity="0.55"/>
<circle cx="586.9" cy="79.9" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<text x="340" y="220" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Same points every time. What differs is the shape of boundary the family is able to express.</text>
<text x="340" y="240" text-anchor="middle" font-size="12" fill="var(--h-graphite)">That shape is the model’s assumption about the world, made before it ever saw your data.</text>
</svg>

Every model family can only draw certain shapes. That constraint is the
assumption it makes about the world, fixed before it ever saw your data.

A linear model assumes one straight cut separates the classes. A tree assumes the
boundary can be built from rectangles aligned to the axes, which is why trees
cope so well with a feature that only matters above a threshold. A distance-based
model assumes similar rows have similar answers, and its boundary bends around
individual points.

When a model performs badly, this is the first thing to ask: **is the boundary it
needs a shape this family can draw?**

## One interface for all of them

The practical consequence of everything above is small and liberating.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

models = {
    "logistic": LogisticRegression(max_iter=1000),
    "forest": RandomForestClassifier(random_state=42),
    "knn": KNeighborsClassifier(),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, model.score(X_test, y_test))
```

Four methods cover almost everything: `fit` to learn, `predict` for labels,
`predict_proba` for probabilities, `score` for a quick number.

Swapping a model is one line. That is why the pipeline you built on the last page
matters so much: with the preprocessing fixed inside it, every model sees
identically prepared data and the comparison is fair.

```python
model = Pipeline([("prep", preprocess), ("clf", RandomForestClassifier())])
```

Change the final step and nothing else moves.

## Start with something stupid

Before any real model, build one that cannot possibly be good.

```python
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
print(dummy.score(X_test, y_test))
```

```
0.735
```

On the churn data, predicting "nobody leaves" is 73.5 percent accurate. That is
your baseline, and it is the number every result must be compared against.

<svg viewBox="0 0 680 266" role="img" aria-labelledby="m6-title m6-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="m6-title">Choosing a first model</title>
<desc id="m6-desc">Start with a dummy baseline. Then a linear model if you need to explain it. Then a tree ensemble for tabular data. Reach for anything more complicated only when the simpler options have been beaten.</desc>
<defs><marker id="m6" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<rect x="5" y="60" width="152" height="74" rx="10" fill="var(--h-cherry)" stroke="var(--h-cherry)" stroke-dasharray="none"/>
<text x="81" y="90" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">a dummy</text>
<text x="81" y="110" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.9">the score to beat</text>
<line x1="163" y1="97" x2="175" y2="97" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#m6)"/>
<rect x="175" y="60" width="152" height="74" rx="10" fill="var(--h-cherry)" stroke="var(--h-cherry)" stroke-dasharray="none"/>
<text x="251" y="90" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">a linear model</text>
<text x="251" y="110" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.9">explainable, fast</text>
<line x1="333" y1="97" x2="345" y2="97" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#m6)"/>
<rect x="345" y="60" width="152" height="74" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)" stroke-dasharray="none"/>
<text x="421" y="90" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">a tree ensemble</text>
<text x="421" y="110" text-anchor="middle" font-size="10" fill="var(--md-default-fg-color)" opacity="0.9">usually wins on tables</text>
<line x1="503" y1="97" x2="515" y2="97" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#m6)"/>
<rect x="515" y="60" width="152" height="74" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)" stroke-dasharray="5 4"/>
<text x="591" y="90" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">something fancier</text>
<text x="591" y="110" text-anchor="middle" font-size="10" fill="var(--md-default-fg-color)" opacity="0.9">only if it earns it</text>
<text x="340" y="42" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-graphite)">stop at the first one that is good enough</text>
<text x="340" y="172" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Every step costs you something: time to train, difficulty explaining it, and more ways to fool yourself.</text>
<text x="340" y="192" text-anchor="middle" font-size="12" fill="var(--h-graphite)">A logistic regression you can explain to the person who has to act on it often beats a better model they will not trust.</text>
<text x="340" y="228" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-cherry)">For tabular data like the churn file, a gradient boosted tree is usually the strongest thing you will try.</text>
<text x="340" y="248" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Deep learning is for images, audio and text, which is trimester two.</text>
</svg>

A model at 76 percent accuracy sounds respectable and is barely better than
answering the same thing every time. Without the baseline on the page you would
not know that, and neither would the person reading your report.

!!! warning "Simple first, complicated only if it earns it"

    Every step up in complexity costs you: longer training, harder explanations,
    more hyperparameters, and more opportunities to fool yourself.

    A logistic regression that the person acting on it can understand often
    beats a better model they will not trust. Reach for something heavier when
    the simpler thing has been tried and found wanting, not before.

## Too simple, too complex

<svg viewBox="0 0 680 262" role="img" aria-labelledby="m4-title m4-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="m4-title">Underfitting, a good fit, and overfitting</title>
<desc id="m4-desc">A model that is too simple misses the pattern. A model of the right complexity follows the trend. A model that is too complex passes through every point including the noise.</desc>
<rect x="5" y="30" width="214" height="150" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="112" y="22" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">too simple</text>
<text x="112" y="202" text-anchor="middle" font-size="10.5" fill="var(--h-space)">misses the pattern</text>
<polyline points="21,150 203,74" fill="none" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<circle cx="21.0" cy="114.2" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="36.2" cy="94.1" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="51.3" cy="75.3" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="66.5" cy="63.5" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="81.7" cy="57.8" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="96.8" cy="59.9" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="112.0" cy="55.6" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="127.2" cy="60.0" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="142.3" cy="70.8" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="157.5" cy="71.8" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="172.7" cy="71.6" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="187.8" cy="87.1" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="203.0" cy="103.0" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<rect x="233" y="30" width="214" height="150" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="340" y="22" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">about right</text>
<text x="340" y="202" text-anchor="middle" font-size="10.5" fill="var(--h-space)">follows the trend</text>
<polyline points="249,108 264,96 279,84 294,74 310,66 325,60 340,58 355,58 370,62 386,69 401,78 416,89 431,101" fill="none" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<circle cx="249.0" cy="114.2" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="264.2" cy="94.1" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="279.3" cy="75.3" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="294.5" cy="63.5" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="309.7" cy="57.8" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="324.8" cy="59.9" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="340.0" cy="55.6" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="355.2" cy="60.0" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="370.3" cy="70.8" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="385.5" cy="71.8" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="400.7" cy="71.6" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="415.8" cy="87.1" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="431.0" cy="103.0" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<rect x="461" y="30" width="214" height="150" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="568" y="22" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">too complex</text>
<text x="568" y="202" text-anchor="middle" font-size="10.5" fill="var(--h-space)">memorises the noise</text>
<polyline points="477,114 492,94 507,75 522,64 538,58 553,60 568,56 583,60 598,71 614,72 629,72 644,87 659,103" fill="none" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<circle cx="477.0" cy="114.2" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="492.2" cy="94.1" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="507.3" cy="75.3" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="522.5" cy="63.5" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="537.7" cy="57.8" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="552.8" cy="59.9" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="568.0" cy="55.6" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="583.2" cy="60.0" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="598.3" cy="70.8" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="613.5" cy="71.8" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="628.7" cy="71.6" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="643.8" cy="87.1" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="659.0" cy="103.0" r="3.6" fill="var(--h-cherry)" opacity="0.8"/>
<text x="340" y="232" text-anchor="middle" font-size="12" fill="var(--h-graphite)">The left one is wrong on the training data too. The right one is perfect on it and useless on anything else.</text>
<text x="340" y="252" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Only a held back set can tell the middle one from the right one.</text>
</svg>

**Underfitting** means the model is too simple to represent the pattern. It is
wrong on the training data as well as on new data, which makes it easy to spot.

**Overfitting** means the model has memorised the training data, including its
noise. Perfect where it trained and poor everywhere else.

The gap between the two scores is the diagnosis:

| Training score | Test score | What it means |
|---|---|---|
| Low | Low | Underfitting. More capacity, or better features |
| High | High | You are done |
| High | Low | Overfitting. Simpler model, more data, or regularisation |
| Low | High | Something is wrong with your split |

This is the same picture as the training and validation curves from the
Fundamentals page, now with a model attached instead of a neural network. It is
also why the held back set exists at all: without it, the third row and the
second row look identical.

## How you will be judged

Accuracy is the number everyone quotes and the one that hides most.

On the churn data, 73.5 percent accuracy is achievable by a model that never
predicts churn. Any serious evaluation starts with the full picture.

<svg viewBox="0 0 680 310" role="img" aria-labelledby="m5-title m5-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="m5-title">The confusion matrix, with precision and recall marked</title>
<desc id="m5-desc">Rows are the truth and columns are the prediction. Precision is the share of predicted churners who really churned. Recall is the share of real churners the model found.</desc>
<text x="330" y="30" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-graphite)">what the model predicted</text>
<text x="270" y="56" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">churn</text>
<text x="430" y="56" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">stay</text>
<text x="178" y="30" text-anchor="end" font-size="12.5" font-weight="700" fill="var(--h-graphite)">truth</text>
<text x="178" y="104" text-anchor="end" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">churned</text>
<text x="178" y="184" text-anchor="end" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">stayed</text>
<rect x="190" y="66" width="160" height="72" rx="6" fill="var(--h-cherry)"/>
<text x="270" y="98" text-anchor="middle" font-size="20" font-weight="700" fill="#ffffff">280</text>
<text x="270" y="120" text-anchor="middle" font-size="10.5" fill="#ffffff" opacity="0.9">true positives</text>
<rect x="350" y="66" width="160" height="72" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="430" y="98" text-anchor="middle" font-size="20" font-weight="700" fill="var(--h-graphite)">94</text>
<text x="430" y="120" text-anchor="middle" font-size="10.5" fill="var(--h-space)">missed churners</text>
<rect x="190" y="146" width="160" height="72" rx="6" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="270" y="178" text-anchor="middle" font-size="20" font-weight="700" fill="var(--h-cherry)">120</text>
<text x="270" y="200" text-anchor="middle" font-size="10.5" fill="var(--h-cherry)">false alarms</text>
<rect x="350" y="146" width="160" height="72" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="430" y="178" text-anchor="middle" font-size="20" font-weight="700" fill="var(--h-graphite)">915</text>
<text x="430" y="200" text-anchor="middle" font-size="10.5" fill="var(--h-space)">correctly left alone</text>
<rect x="186" y="62" width="168" height="160" rx="8" fill="none" stroke="var(--h-cherry)" stroke-width="2.5" stroke-dasharray="6 4"/>
<text x="270" y="240" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">precision, the column</text>
<text x="270" y="256" text-anchor="middle" font-size="11" fill="var(--h-graphite)">280 of 400, 70 percent</text>
<rect x="186" y="62" width="328" height="80" rx="8" fill="none" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<text x="548" y="98" font-size="11.5" font-weight="700" fill="var(--md-default-fg-color)">recall, the row</text>
<text x="548" y="116" font-size="11" fill="var(--h-graphite)">280 of 374,</text>
<text x="548" y="132" font-size="11" fill="var(--h-graphite)">75 percent</text>
<text x="340" y="286" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Accuracy here is 85 percent and says nothing about the 94 churners who walked out unnoticed.</text>
<text x="340" y="304" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Which of the two off-diagonal cells costs more is a business question, not a modelling one.</text>
</svg>

**Precision** answers: of everyone the model flagged, how many really were
churners? It is the column. Low precision means wasted effort on false alarms.

**Recall** answers: of everyone who really churned, how many did the model find?
It is the row. Low recall means customers walking out unnoticed.

The two trade against each other, and the dial is the threshold from earlier.
Lower it and recall rises while precision falls.

**F1** combines them into one number, the harmonic mean, which stays low unless
both are decent. It is the usual single figure to quote when the classes are
imbalanced.

```python
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))
```

**ROC AUC** measures how well the model ranks, across every possible threshold
at once. It is useful for comparing models before you have chosen a threshold,
and it can look flattering on very imbalanced data.

For regression the equivalents are **MAE**, the average size of the error in the
original units, and **RMSE**, which punishes large errors more because of the
squaring you met in Probability.

!!! warning "Choose the metric before you see the scores"

    If you pick the metric afterwards, you will pick the one that makes your
    model look best. Write down which number matters, and why, while you are
    still framing the problem.

    For churn, ask what the company does with a prediction. If they phone the
    customer, false alarms are cheap and recall matters. If they send a three
    month discount, false alarms are expensive and precision matters.

## What comes next in this group

| Project | Family | The question it answers |
|---|---|---|
| Linear Models | Linear | How do coefficients turn features into a prediction? |
| Tree-Based Models | Trees and ensembles | How do splits and ensembles beat single models? |
| Unsupervised Learning | No labels | What can you learn with no target at all? |
| Clustering | No labels | How do you find the groups, and how do you know they are real? |

Each one goes deep on a region of the map. Come back to this page whenever you
lose sight of where you are.

## Questions to sit with

1. A classifier outputs a probability and a threshold turns it into a decision.
   Who in an organisation should choose that threshold, and what would they need
   to know to choose it well?
2. Your model scores 76 percent where the baseline is 73.5. Is that a result?
   What would you need to say alongside the number for it to be honest?
3. Every model family can only draw certain shapes of boundary. What does that
   mean for the claim that a model learns from data with no assumptions?
4. Precision and recall trade against each other. Describe a problem where you
   would accept terrible precision for high recall, and one where you would do
   the reverse.
5. The long loop in the first diagram goes back to the data. Why do beginners
   prefer the short one, and what does that preference cost them?

## Quick reference

| Task | Code |
|---|---|
| Baseline | `DummyClassifier(strategy="most_frequent")` |
| Fit and score | `model.fit(X_train, y_train)`, `model.score(X_test, y_test)` |
| Labels | `model.predict(X_test)` |
| Probabilities | `model.predict_proba(X_test)` |
| Custom threshold | `(proba[:, 1] > 0.3).astype(int)` |
| Confusion matrix | `confusion_matrix(y_test, y_pred)` |
| All the metrics | `classification_report(y_test, y_pred)` |
| Ranking quality | `roc_auc_score(y_test, proba[:, 1])` |
| Honest comparison | `cross_val_score(pipeline, X_train, y_train, cv=5)` |
| Swap the model | change the last step of the `Pipeline` |

## Next

You have the map. The first region is the one whose coefficients you can read
straight off.

[Linear Models](linear-models.md){ .h-button }

Worth your time: the
[scikit-learn model selection guide](https://scikit-learn.org/stable/tutorial/machine_learning_map/)
is a decision chart for choosing an estimator, and the
[classification metrics page](https://scikit-learn.org/stable/modules/model_evaluation.html)
is the reference for everything in the evaluation section.

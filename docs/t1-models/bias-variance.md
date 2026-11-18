# Bias, Variance and Tuning

<span class="badge badge--time">70 min</span>
<span class="badge badge--level">Classic models</span>
<span class="badge">Reference, no tasks</span>

Every page in this group has had a knob that trades fit against stability: alpha,
C, gamma, max_depth, ccp_alpha, learning rate. They are all the same knob. This
page explains what it is turning and how to decide where to set it.

## Why this matters

Two words have been used repeatedly without being defined. The Tree-Based page
said bagging reduces variance and boosting reduces bias. That sentence is exactly
right and means nothing until you have this page.

More practically, this is where you learn to answer three questions that come up
in every project:

- Is my model too simple or too complex?
- Would collecting more data help, or am I wasting the budget?
- How do I choose the settings without fooling myself?

## Bias and variance

Imagine training your model many times on different samples from the same
population. Each time you get a slightly different model, and each makes
predictions.

<svg viewBox="0 0 680 320" role="img" aria-labelledby="b1-title b1-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="b1-title">Bias and variance as shots at a target</title>
<desc id="b1-desc">Low bias and low variance means tight shots on the centre. High bias low variance means tight shots in the wrong place. Low bias high variance means scattered shots averaging to the centre. High bias high variance means scattered shots in the wrong place.</desc>
<rect x="20" y="34" width="320" height="124" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<circle cx="90" cy="90" r="46" fill="var(--h-cherry)" opacity="0.1"/>
<circle cx="90" cy="90" r="31" fill="var(--h-cherry)" opacity="0.16"/>
<circle cx="90" cy="90" r="16" fill="var(--h-cherry)" opacity="0.26"/>
<circle cx="90" cy="90" r="3" fill="var(--md-default-fg-color)"/>
<circle cx="92.2" cy="105.2" r="3.4" fill="var(--h-cherry)"/>
<circle cx="96.6" cy="96.7" r="3.4" fill="var(--h-cherry)"/>
<circle cx="93.9" cy="92.3" r="3.4" fill="var(--h-cherry)"/>
<circle cx="94.1" cy="90.0" r="3.4" fill="var(--h-cherry)"/>
<circle cx="85.8" cy="84.8" r="3.4" fill="var(--h-cherry)"/>
<circle cx="83.3" cy="92.0" r="3.4" fill="var(--h-cherry)"/>
<circle cx="94.0" cy="100.6" r="3.4" fill="var(--h-cherry)"/>
<circle cx="93.7" cy="92.3" r="3.4" fill="var(--h-cherry)"/>
<circle cx="94.7" cy="90.6" r="3.4" fill="var(--h-cherry)"/>
<text x="170" y="82" font-size="12.5" font-weight="700" fill="var(--h-cherry)">low bias, low variance</text>
<text x="170" y="104" font-size="11" fill="var(--h-graphite)">what you want</text>
<rect x="370" y="34" width="320" height="124" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<circle cx="440" cy="90" r="46" fill="var(--h-cherry)" opacity="0.1"/>
<circle cx="440" cy="90" r="31" fill="var(--h-cherry)" opacity="0.16"/>
<circle cx="440" cy="90" r="16" fill="var(--h-cherry)" opacity="0.26"/>
<circle cx="440" cy="90" r="3" fill="var(--md-default-fg-color)"/>
<circle cx="463.2" cy="73.1" r="3.4" fill="var(--h-cherry)"/>
<circle cx="471.9" cy="68.0" r="3.4" fill="var(--h-cherry)"/>
<circle cx="473.0" cy="84.5" r="3.4" fill="var(--h-cherry)"/>
<circle cx="481.7" cy="69.2" r="3.4" fill="var(--h-cherry)"/>
<circle cx="467.7" cy="72.2" r="3.4" fill="var(--h-cherry)"/>
<circle cx="469.1" cy="73.8" r="3.4" fill="var(--h-cherry)"/>
<circle cx="465.3" cy="66.1" r="3.4" fill="var(--h-cherry)"/>
<circle cx="480.4" cy="61.4" r="3.4" fill="var(--h-cherry)"/>
<circle cx="475.4" cy="67.7" r="3.4" fill="var(--h-cherry)"/>
<text x="520" y="82" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">high bias, low variance</text>
<text x="520" y="104" font-size="11" fill="var(--h-graphite)">consistently wrong</text>
<rect x="20" y="174" width="320" height="124" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<circle cx="90" cy="230" r="46" fill="var(--h-cherry)" opacity="0.1"/>
<circle cx="90" cy="230" r="31" fill="var(--h-cherry)" opacity="0.16"/>
<circle cx="90" cy="230" r="16" fill="var(--h-cherry)" opacity="0.26"/>
<circle cx="90" cy="230" r="3" fill="var(--md-default-fg-color)"/>
<circle cx="118.4" cy="220.0" r="3.4" fill="var(--h-cherry)"/>
<circle cx="111.7" cy="289.0" r="3.4" fill="var(--h-cherry)"/>
<circle cx="53.1" cy="210.0" r="3.4" fill="var(--h-cherry)"/>
<circle cx="105.6" cy="215.5" r="3.4" fill="var(--h-cherry)"/>
<circle cx="56.9" cy="271.8" r="3.4" fill="var(--h-cherry)"/>
<circle cx="119.2" cy="263.2" r="3.4" fill="var(--h-cherry)"/>
<circle cx="122.5" cy="252.9" r="3.4" fill="var(--h-cherry)"/>
<circle cx="71.2" cy="182.0" r="3.4" fill="var(--h-cherry)"/>
<circle cx="107.1" cy="204.4" r="3.4" fill="var(--h-cherry)"/>
<text x="170" y="222" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">low bias, high variance</text>
<text x="170" y="244" font-size="11" fill="var(--h-graphite)">right on average, unreliable</text>
<rect x="370" y="174" width="320" height="124" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<circle cx="440" cy="230" r="46" fill="var(--h-cherry)" opacity="0.1"/>
<circle cx="440" cy="230" r="31" fill="var(--h-cherry)" opacity="0.16"/>
<circle cx="440" cy="230" r="16" fill="var(--h-cherry)" opacity="0.26"/>
<circle cx="440" cy="230" r="3" fill="var(--md-default-fg-color)"/>
<circle cx="476.8" cy="222.0" r="3.4" fill="var(--h-cherry)"/>
<circle cx="441.5" cy="202.4" r="3.4" fill="var(--h-cherry)"/>
<circle cx="498.0" cy="232.4" r="3.4" fill="var(--h-cherry)"/>
<circle cx="431.6" cy="206.1" r="3.4" fill="var(--h-cherry)"/>
<circle cx="475.6" cy="227.1" r="3.4" fill="var(--h-cherry)"/>
<circle cx="495.8" cy="209.8" r="3.4" fill="var(--h-cherry)"/>
<circle cx="469.1" cy="235.1" r="3.4" fill="var(--h-cherry)"/>
<circle cx="436.7" cy="242.4" r="3.4" fill="var(--h-cherry)"/>
<circle cx="484.5" cy="197.8" r="3.4" fill="var(--h-cherry)"/>
<text x="520" y="222" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">high bias, high variance</text>
<text x="520" y="244" font-size="11" fill="var(--h-graphite)">wrong and unstable</text>
<text x="340" y="304" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Bias is where the cluster sits. Variance is how wide it is. Only the model on the top left is useful.</text>
</svg>

**Bias** is how far the average prediction lands from the truth. A high bias
model is consistently wrong in the same direction, because it is too simple to
represent the pattern. Underfitting.

**Variance** is how much the predictions move when the training data changes. A
high variance model gives you a different answer every time, because it is
fitting the particular rows it saw rather than the pattern behind them.
Overfitting.

**Noise** is the part nobody can remove. Two identical customers do different
things.

$$
\text{total error} = \text{bias}^2 + \text{variance} + \text{noise}
$$

You can trade the first two against each other. You cannot touch the third.

### The trade

Making a model more flexible lowers bias and raises variance. Constraining it
does the reverse. There is a point in between where the sum is smallest, and
finding it is what tuning means.

| Model | Bias | Variance | Why |
|---|---|---|---|
| Linear regression | High | Low | It can only draw a straight line, but it draws the same one every time |
| Deep decision tree | Low | High | It can represent anything, including the noise |
| Random forest | Low | Lower | Many high variance trees, averaged |
| Boosted trees | Lower | Higher | Each round attacks the remaining bias |
| K-nearest neighbours, k=1 | Very low | Very high | The boundary follows individual points |
| K-nearest neighbours, k=50 | Higher | Lower | Each answer is an average of fifty |

This table is the sentence from the Tree-Based page, unpacked. **Bagging attacks
variance** by averaging independent mistakes. **Boosting attacks bias** by
repeatedly going after what is still wrong, which is why it can overfit and a
forest rarely does.

!!! tip "The one-line version"

    Bias is being wrong in a way more data will not fix. Variance is being wrong
    in a way more data will fix.

### The floor

Some error is irreducible. Two customers with identical features churn
differently, because the feature set does not contain whatever made the
difference.

That matters for a practical reason: it tells you when to stop. If the best
achievable accuracy on your problem is around 85 percent and you are at 83, the
remaining work is in getting better features or better data, not in tuning. A
model that reaches 99 percent on a genuinely noisy problem has not beaten the
floor, it has found a leak.

## Diagnosing which problem you have

Start with the two scores.

| Training | Validation | Diagnosis | Do this |
|---|---|---|---|
| Low | Low | High bias | More flexible model, better features |
| High | Much lower | High variance | Regularise, simplify, more data |
| High | High | You are done | Stop |
| Low | Higher | Something is broken | Check your split |

That last row is worth knowing. Validation scoring above training usually means
a leak, a badly shuffled split, or a validation set that happens to be easier.
It is a bug, not a success.

## Learning curves

The two-score table tells you which problem you have. A learning curve tells you
whether more data would fix it, which is often a budget question rather than a
technical one.

<svg viewBox="0 0 680 286" role="img" aria-labelledby="b2-title b2-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="b2-title">Three shapes of learning curve</title>
<desc id="b2-desc">If both curves meet at a low score, the model is too simple and more data will not help. If a wide gap remains, the model is overfitting and more data will help. If both meet at a high score, you are done.</desc>
<rect x="5" y="30" width="214" height="160" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="112" y="22" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">too simple</text>
<line x1="35" y1="170" x2="201" y2="170" stroke="var(--h-steel)" stroke-width="1.3"/>
<line x1="35" y1="170" x2="35" y2="46" stroke="var(--h-steel)" stroke-width="1.3"/>
<polyline points="35.0,146.9 39.0,146.4 43.0,146.0 47.0,145.6 51.0,145.2 55.0,144.8 59.0,144.5 63.0,144.2 67.0,143.9 71.0,143.7 75.0,143.4 79.0,143.2 83.0,143.0 87.0,142.8 91.0,142.6 95.0,142.4 99.0,142.3 103.0,142.1 107.0,142.0 111.0,141.9 115.0,141.8 119.0,141.7 123.0,141.6 127.0,141.5 131.0,141.4 135.0,141.3 139.0,141.2 143.0,141.2 147.0,141.1 151.0,141.0 155.0,141.0 159.0,140.9 163.0,140.9 167.0,140.9 171.0,140.8 175.0,140.8 179.0,140.7 183.0,140.7 187.0,140.7 191.0,140.7 195.0,140.6" fill="none" stroke="var(--h-steel)" stroke-width="2.2"/>
<polyline points="35.0,150.2 39.0,149.7 43.0,149.3 47.0,148.9 51.0,148.5 55.0,148.1 59.0,147.8 63.0,147.5 67.0,147.2 71.0,147.0 75.0,146.7 79.0,146.5 83.0,146.3 87.0,146.1 91.0,145.9 95.0,145.7 99.0,145.6 103.0,145.4 107.0,145.3 111.0,145.2 115.0,145.1 119.0,145.0 123.0,144.9 127.0,144.8 131.0,144.7 135.0,144.6 139.0,144.5 143.0,144.5 147.0,144.4 151.0,144.3 155.0,144.3 159.0,144.2 163.0,144.2 167.0,144.2 171.0,144.1 175.0,144.1 179.0,144.0 183.0,144.0 187.0,144.0 191.0,144.0 195.0,143.9" fill="none" stroke="var(--h-cherry)" stroke-width="2.6"/>
<text x="112" y="208" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-graphite)">more data will not help</text>
<rect x="233" y="30" width="214" height="160" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="340" y="22" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">overfitting</text>
<line x1="263" y1="170" x2="429" y2="170" stroke="var(--h-steel)" stroke-width="1.3"/>
<line x1="263" y1="170" x2="263" y2="46" stroke="var(--h-steel)" stroke-width="1.3"/>
<polyline points="263.0,54.5 267.0,55.0 271.0,55.5 275.0,56.0 279.0,56.5 283.0,57.0 287.0,57.5 291.0,58.0 295.0,58.5 299.0,59.0 303.0,59.5 307.0,59.9 311.0,60.4 315.0,60.9 319.0,61.4 323.0,61.9 327.0,62.4 331.0,62.9 335.0,63.4 339.0,63.9 343.0,64.4 347.0,64.9 351.0,65.4 355.0,65.9 359.0,66.4 363.0,66.9 367.0,67.4 371.0,67.9 375.0,68.4 379.0,68.9 383.0,69.4 387.0,69.8 391.0,70.3 395.0,70.8 399.0,71.3 403.0,71.8 407.0,72.3 411.0,72.8 415.0,73.3 419.0,73.8 423.0,74.3" fill="none" stroke="var(--h-steel)" stroke-width="2.2"/>
<polyline points="263.0,153.5 267.0,150.4 271.0,147.5 275.0,144.8 279.0,142.2 283.0,139.8 287.0,137.5 291.0,135.4 295.0,133.4 299.0,131.5 303.0,129.7 307.0,128.0 311.0,126.4 315.0,124.9 319.0,123.5 323.0,122.2 327.0,120.9 331.0,119.7 335.0,118.6 339.0,117.6 343.0,116.6 347.0,115.7 351.0,114.8 355.0,114.0 359.0,113.2 363.0,112.5 367.0,111.8 371.0,111.1 375.0,110.5 379.0,110.0 383.0,109.4 387.0,108.9 391.0,108.4 395.0,108.0 399.0,107.6 403.0,107.2 407.0,106.8 411.0,106.4 415.0,106.1 419.0,105.8 423.0,105.5" fill="none" stroke="var(--h-cherry)" stroke-width="2.6"/>
<text x="340" y="208" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-cherry)">more data will help</text>
<rect x="461" y="30" width="214" height="160" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="568" y="22" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">about right</text>
<line x1="491" y1="170" x2="657" y2="170" stroke="var(--h-steel)" stroke-width="1.3"/>
<line x1="491" y1="170" x2="491" y2="46" stroke="var(--h-steel)" stroke-width="1.3"/>
<polyline points="491.0,77.6 495.0,77.8 499.0,77.9 503.0,78.1 507.0,78.3 511.0,78.4 515.0,78.6 519.0,78.8 523.0,78.9 527.0,79.1 531.0,79.2 535.0,79.4 539.0,79.6 543.0,79.7 547.0,79.9 551.0,80.1 555.0,80.2 559.0,80.4 563.0,80.6 567.0,80.7 571.0,80.9 575.0,81.1 579.0,81.2 583.0,81.4 587.0,81.6 591.0,81.7 595.0,81.9 599.0,82.1 603.0,82.2 607.0,82.4 611.0,82.5 615.0,82.7 619.0,82.9 623.0,83.0 627.0,83.2 631.0,83.4 635.0,83.5 639.0,83.7 643.0,83.9 647.0,84.0 651.0,84.2" fill="none" stroke="var(--h-steel)" stroke-width="2.2"/>
<polyline points="491.0,120.5 495.0,118.1 499.0,115.9 503.0,113.9 507.0,111.9 511.0,110.2 515.0,108.5 519.0,107.0 523.0,105.6 527.0,104.3 531.0,103.1 535.0,102.0 539.0,100.9 543.0,99.9 547.0,99.0 551.0,98.2 555.0,97.4 559.0,96.7 563.0,96.1 567.0,95.4 571.0,94.9 575.0,94.3 579.0,93.8 583.0,93.4 587.0,93.0 591.0,92.6 595.0,92.2 599.0,91.9 603.0,91.5 607.0,91.2 611.0,91.0 615.0,90.7 619.0,90.5 623.0,90.3 627.0,90.1 631.0,89.9 635.0,89.7 639.0,89.6 643.0,89.4 647.0,89.3 651.0,89.1" fill="none" stroke="var(--h-cherry)" stroke-width="2.6"/>
<text x="568" y="208" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-graphite)">you are done</text>
<text x="112" y="46" font-size="9.5" font-weight="700" fill="var(--h-steel)">train</text>
<text x="112" y="150" font-size="9.5" font-weight="700" fill="var(--h-cherry)">validation</text>
<text x="340" y="240" text-anchor="middle" font-size="11" fill="var(--h-space)">horizontal axis: how many training rows you used</text>
<text x="340" y="266" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Before anyone spends money collecting more data, draw this. It answers the question directly.</text>
</svg>

```python
from sklearn.model_selection import learning_curve
import numpy as np

sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring="roc_auc",
)
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)
```

Read the right-hand end of the plot.

**Both curves have met, at a low score.** High bias. The model is as good as it
can be and it is not good. More data changes nothing. Change the model or the
features.

**A wide gap remains and the validation curve is still climbing.** High variance.
More data would help, and so would regularisation in the meantime.

**Both curves have met, at a high score.** You are finished. Spend your time
elsewhere.

!!! ml "ML connection"

    The Plotting page listed the learning curve as one of the five charts you
    would make in every project and promised it for trimester two. Here it is
    early, because it answers a question people ask in week one: should we
    collect more data?

    Draw this before anybody spends money on collection. It is a fifteen line
    answer to a expensive question.

## Validation curves

A learning curve varies the amount of data. A validation curve varies one
hyperparameter and holds everything else fixed.

<svg viewBox="0 0 680 280" role="img" aria-labelledby="b3-title b3-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="b3-title">The validation curve</title>
<desc id="b3-desc">As a model is allowed more flexibility, training error falls continuously while validation error falls, reaches a minimum, and then rises. The minimum is the setting to choose.</desc>
<line x1="70" y1="196" x2="640" y2="196" stroke="var(--h-steel)" stroke-width="1.4"/>
<line x1="70" y1="196" x2="70" y2="36" stroke="var(--h-steel)" stroke-width="1.4"/>
<text x="86" y="48" font-size="11.5" font-weight="700" fill="var(--h-graphite)">error</text>
<text x="90" y="220" font-size="11" fill="var(--h-space)">simpler model</text>
<text x="620" y="220" text-anchor="end" font-size="11" fill="var(--h-space)">more flexible model</text>
<polyline points="70.0,46.0 75.7,49.9 81.4,53.8 87.1,57.5 92.8,61.1 98.5,64.6 104.2,68.0 109.9,71.4 115.6,74.6 121.3,77.7 127.0,80.7 132.7,83.7 138.4,86.6 144.1,89.3 149.8,92.0 155.5,94.7 161.2,97.2 166.9,99.7 172.6,102.1 178.3,104.5 184.0,106.7 189.7,108.9 195.4,111.1 201.1,113.2 206.8,115.2 212.5,117.2 218.2,119.1 223.9,120.9 229.6,122.7 235.3,124.5 241.0,126.2 246.7,127.8 252.4,129.4 258.1,131.0 263.8,132.5 269.5,134.0 275.2,135.4 280.9,136.8 286.6,138.2 292.3,139.5 298.0,140.7 303.7,142.0 309.4,143.2 315.1,144.3 320.8,145.5 326.5,146.6 332.2,147.6 337.9,148.7 343.6,149.7 349.3,150.7 355.0,151.6 360.7,152.6 366.4,153.5 372.1,154.3 377.8,155.2 383.5,156.0 389.2,156.8 394.9,157.6 400.6,158.3 406.3,159.1 412.0,159.8 417.7,160.5 423.4,161.1 429.1,161.8 434.8,162.4 440.5,163.0 446.2,163.6 451.9,164.2 457.6,164.8 463.3,165.3 469.0,165.9 474.7,166.4 480.4,166.9 486.1,167.4 491.8,167.9 497.5,168.3 503.2,168.8 508.9,169.2 514.6,169.6 520.3,170.0 526.0,170.4 531.7,170.8 537.4,171.2 543.1,171.6 548.8,171.9 554.5,172.3 560.2,172.6 565.9,172.9 571.6,173.2 577.3,173.6 583.0,173.9 588.7,174.1 594.4,174.4 600.1,174.7 605.8,175.0 611.5,175.2 617.2,175.5 622.9,175.7 628.6,176.0 634.3,176.2 640.0,176.4" fill="none" stroke="var(--h-steel)" stroke-width="2.4"/>
<polyline points="70.0,76.0 75.7,79.0 81.4,81.9 87.1,84.7 92.8,87.4 98.5,90.0 104.2,92.5 109.9,94.9 115.6,97.2 121.3,99.4 127.0,101.5 132.7,103.6 138.4,105.5 144.1,107.4 149.8,109.2 155.5,110.9 161.2,112.5 166.9,114.1 172.6,115.5 178.3,116.9 184.0,118.3 189.7,119.5 195.4,120.7 201.1,121.8 206.8,122.9 212.5,123.9 218.2,124.8 223.9,125.7 229.6,126.5 235.3,127.2 241.0,127.9 246.7,128.5 252.4,129.1 258.1,129.6 263.8,130.0 269.5,130.4 275.2,130.8 280.9,131.0 286.6,131.3 292.3,131.5 298.0,131.6 303.7,131.7 309.4,131.7 315.1,131.7 320.8,131.6 326.5,131.5 332.2,131.3 337.9,131.1 343.6,130.9 349.3,130.6 355.0,130.2 360.7,129.8 366.4,129.4 372.1,128.9 377.8,128.4 383.5,127.8 389.2,127.2 394.9,126.6 400.6,125.9 406.3,125.1 412.0,124.4 417.7,123.5 423.4,122.7 429.1,121.8 434.8,120.8 440.5,119.9 446.2,118.8 451.9,117.8 457.6,116.7 463.3,115.5 469.0,114.3 474.7,113.1 480.4,111.9 486.1,110.6 491.8,109.2 497.5,107.8 503.2,106.4 508.9,105.0 514.6,103.5 520.3,101.9 526.0,100.4 531.7,98.8 537.4,97.1 543.1,95.4 548.8,93.7 554.5,91.9 560.2,90.1 565.9,88.3 571.6,86.4 577.3,84.5 583.0,82.6 588.7,80.6 594.4,78.6 600.1,76.5 605.8,74.4 611.5,72.3 617.2,70.1 622.9,67.9 628.6,65.6 634.3,63.3 640.0,61.0" fill="none" stroke="var(--h-cherry)" stroke-width="3"/>
<line x1="309" y1="132" x2="309" y2="196" stroke="var(--h-cherry)" stroke-width="2" stroke-dasharray="5 4"/>
<circle cx="309" cy="132" r="6" fill="var(--h-cherry)"/>
<text x="309" y="118" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">choose here</text>
<text x="120" y="80" font-size="11.5" font-weight="700" fill="var(--h-graphite)">underfitting</text>
<text x="120" y="96" font-size="10.5" fill="var(--h-space)">both errors high</text>
<text x="560" y="80" text-anchor="end" font-size="11.5" font-weight="700" fill="var(--h-cherry)">overfitting</text>
<text x="560" y="96" text-anchor="end" font-size="10.5" fill="var(--h-space)">training error keeps falling, validation rises</text>
<text x="440" y="172" font-size="10.5" font-weight="700" fill="var(--h-steel)">training</text>
<text x="470" y="92" font-size="10.5" font-weight="700" fill="var(--h-cherry)">validation</text>
<text x="340" y="252" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Every knob on the next diagram produces a curve of this shape. Only the axis label changes.</text>
<text x="340" y="272" text-anchor="middle" font-size="12" fill="var(--h-graphite)">The gap between the two lines at any point is your variance. The height of the lower line is your bias plus noise.</text>
</svg>

```python
from sklearn.model_selection import validation_curve

alphas = np.logspace(-3, 3, 13)
train_scores, val_scores = validation_curve(
    Ridge(), X_train, y_train, param_name="alpha",
    param_range=alphas, cv=5, scoring="r2",
)
```

Every knob in this group produces a curve of this shape. Only the axis label
changes. Left side is too constrained, right side is too flexible, and the
minimum of the validation curve is where you set it.

Two things to read off the picture. The **gap** between the lines is your
variance. The **height of the lower line** is your bias plus the noise floor.

## Every knob, on one page

<svg viewBox="0 0 680 330" role="img" aria-labelledby="b4-title b4-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="b4-title">Every knob you have met, and which way simplifies the model</title>
<desc id="b4-desc">A table of regularisation controls from the linear and tree families, with the direction that reduces variance for each.</desc>
<rect x="10" y="30" width="400" height="26" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="410" y="30" width="260" height="26" fill="var(--h-cherry)"/>
<text x="26" y="48" font-size="11.5" font-weight="700" fill="var(--md-default-fg-color)">the control</text>
<text x="426" y="48" font-size="11.5" font-weight="700" fill="#ffffff">to reduce variance, do this</text>
<rect x="10" y="56" width="400" height="30" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<rect x="410" y="56" width="260" height="30" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="26" y="71" dy="0.36em" font-size="11.5" fill="var(--md-default-fg-color)">alpha in Ridge and Lasso</text>
<text x="426" y="71" dy="0.36em" font-size="11.5" font-weight="700" fill="var(--h-cherry)">raise it</text>
<rect x="10" y="86" width="400" height="30" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<rect x="410" y="86" width="260" height="30" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="26" y="101" dy="0.36em" font-size="11.5" fill="var(--md-default-fg-color)">C in logistic regression and SVM</text>
<text x="426" y="101" dy="0.36em" font-size="11.5" font-weight="700" fill="var(--h-cherry)">lower it</text>
<rect x="10" y="116" width="400" height="30" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<rect x="410" y="116" width="260" height="30" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="26" y="131" dy="0.36em" font-size="11.5" fill="var(--md-default-fg-color)">gamma in an RBF kernel</text>
<text x="426" y="131" dy="0.36em" font-size="11.5" font-weight="700" fill="var(--h-cherry)">lower it</text>
<rect x="10" y="146" width="400" height="30" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<rect x="410" y="146" width="260" height="30" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="26" y="161" dy="0.36em" font-size="11.5" fill="var(--md-default-fg-color)">max_depth in a tree</text>
<text x="426" y="161" dy="0.36em" font-size="11.5" font-weight="700" fill="var(--h-cherry)">lower it</text>
<rect x="10" y="176" width="400" height="30" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<rect x="410" y="176" width="260" height="30" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="26" y="191" dy="0.36em" font-size="11.5" fill="var(--md-default-fg-color)">min_samples_leaf</text>
<text x="426" y="191" dy="0.36em" font-size="11.5" font-weight="700" fill="var(--h-cherry)">raise it</text>
<rect x="10" y="206" width="400" height="30" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<rect x="410" y="206" width="260" height="30" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="26" y="221" dy="0.36em" font-size="11.5" fill="var(--md-default-fg-color)">ccp_alpha</text>
<text x="426" y="221" dy="0.36em" font-size="11.5" font-weight="700" fill="var(--h-cherry)">raise it</text>
<rect x="10" y="236" width="400" height="30" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<rect x="410" y="236" width="260" height="30" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="26" y="251" dy="0.36em" font-size="11.5" fill="var(--md-default-fg-color)">n_estimators in boosting</text>
<text x="426" y="251" dy="0.36em" font-size="11.5" font-weight="700" fill="var(--h-cherry)">lower it</text>
<rect x="10" y="266" width="400" height="30" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<rect x="410" y="266" width="260" height="30" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="26" y="281" dy="0.36em" font-size="11.5" fill="var(--md-default-fg-color)">learning_rate in boosting</text>
<text x="426" y="281" dy="0.36em" font-size="11.5" font-weight="700" fill="var(--h-cherry)">lower it, add rounds</text>
<text x="340" y="318" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Different libraries, one idea. Every one of these trades fit on the training data for stability on new data.</text>
</svg>

This is the reference. Different libraries, different names, one idea: each of
these trades accuracy on the training data for stability on new data.

Three more that are not parameters at all:

**More training data** reduces variance and leaves bias untouched.

**Fewer features** reduces variance. This is what Lasso did for you
automatically.

**Ensembling** reduces variance for free, which is why a random forest with
default settings is such a reliable first model.

### Families of technique

When you meet a regularisation method that is not on this list, it will be one of
these six.

**Constrain the capacity.** Limit what the model is able to represent.
`max_depth`, the degree of a polynomial, the number of neighbours.

**Penalise size.** Add a cost for large parameters. Ridge, Lasso, `ccp_alpha`.

**Add randomness.** Force the model not to rely on any one thing. Bootstrap
samples, random feature subsets, and dropout when you meet neural networks in
trimester two.

**Stop early.** Halt training at the point where validation error turns upward.
Boosting rounds now, training epochs later.

**Get more data.** Including making more from what you have, which is data
augmentation in trimester two.

**Ensemble.** Average several models so their independent errors cancel.

## Searching for the settings

You have been changing one parameter at a time by hand. That works for one knob
and falls apart at two, because parameters interact. `C` and `gamma` in an SVM
are the standard example: the best `gamma` depends on the `C` you chose.

<svg viewBox="0 0 680 286" role="img" aria-labelledby="b5-title b5-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="b5-title">Grid search compared with random search</title>
<desc id="b5-desc">With the same budget of nine trials, a grid tries only three distinct values of each parameter. A random search tries nine distinct values of each, which matters when only one of the two parameters affects the result.</desc>
<rect x="10" y="46" width="320" height="170" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="170" y="36" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">grid search, 9 trials</text>
<circle cx="90" cy="86" r="5" fill="var(--h-cherry)"/>
<line x1="90" y1="216" x2="90" y2="222" stroke="var(--h-cherry)" stroke-width="2"/>
<circle cx="90" cy="136" r="5" fill="var(--h-cherry)"/>
<line x1="90" y1="216" x2="90" y2="222" stroke="var(--h-cherry)" stroke-width="2"/>
<circle cx="90" cy="186" r="5" fill="var(--h-cherry)"/>
<line x1="90" y1="216" x2="90" y2="222" stroke="var(--h-cherry)" stroke-width="2"/>
<circle cx="170" cy="86" r="5" fill="var(--h-cherry)"/>
<line x1="170" y1="216" x2="170" y2="222" stroke="var(--h-cherry)" stroke-width="2"/>
<circle cx="170" cy="136" r="5" fill="var(--h-cherry)"/>
<line x1="170" y1="216" x2="170" y2="222" stroke="var(--h-cherry)" stroke-width="2"/>
<circle cx="170" cy="186" r="5" fill="var(--h-cherry)"/>
<line x1="170" y1="216" x2="170" y2="222" stroke="var(--h-cherry)" stroke-width="2"/>
<circle cx="250" cy="86" r="5" fill="var(--h-cherry)"/>
<line x1="250" y1="216" x2="250" y2="222" stroke="var(--h-cherry)" stroke-width="2"/>
<circle cx="250" cy="136" r="5" fill="var(--h-cherry)"/>
<line x1="250" y1="216" x2="250" y2="222" stroke="var(--h-cherry)" stroke-width="2"/>
<circle cx="250" cy="186" r="5" fill="var(--h-cherry)"/>
<line x1="250" y1="216" x2="250" y2="222" stroke="var(--h-cherry)" stroke-width="2"/>
<text x="170" y="238" text-anchor="middle" font-size="10.5" fill="var(--h-space)">the parameter that actually matters</text>
<text x="170" y="258" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">3 distinct values tried</text>
<rect x="350" y="46" width="320" height="170" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="510" y="36" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">random search, 9 trials</text>
<circle cx="520" cy="173" r="5" fill="var(--h-cherry)"/>
<line x1="520" y1="216" x2="520" y2="222" stroke="var(--h-cherry)" stroke-width="2"/>
<circle cx="630" cy="101" r="5" fill="var(--h-cherry)"/>
<line x1="630" y1="216" x2="630" y2="222" stroke="var(--h-cherry)" stroke-width="2"/>
<circle cx="582" cy="159" r="5" fill="var(--h-cherry)"/>
<line x1="582" y1="216" x2="582" y2="222" stroke="var(--h-cherry)" stroke-width="2"/>
<circle cx="555" cy="75" r="5" fill="var(--h-cherry)"/>
<line x1="555" y1="216" x2="555" y2="222" stroke="var(--h-cherry)" stroke-width="2"/>
<circle cx="397" cy="114" r="5" fill="var(--h-cherry)"/>
<line x1="397" y1="216" x2="397" y2="222" stroke="var(--h-cherry)" stroke-width="2"/>
<circle cx="577" cy="95" r="5" fill="var(--h-cherry)"/>
<line x1="577" y1="216" x2="577" y2="222" stroke="var(--h-cherry)" stroke-width="2"/>
<circle cx="515" cy="104" r="5" fill="var(--h-cherry)"/>
<line x1="515" y1="216" x2="515" y2="222" stroke="var(--h-cherry)" stroke-width="2"/>
<circle cx="602" cy="192" r="5" fill="var(--h-cherry)"/>
<line x1="602" y1="216" x2="602" y2="222" stroke="var(--h-cherry)" stroke-width="2"/>
<circle cx="490" cy="200" r="5" fill="var(--h-cherry)"/>
<line x1="490" y1="216" x2="490" y2="222" stroke="var(--h-cherry)" stroke-width="2"/>
<text x="510" y="238" text-anchor="middle" font-size="10.5" fill="var(--h-space)">the parameter that actually matters</text>
<text x="510" y="258" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">9 distinct values tried</text>
<text x="340" y="280" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Usually one parameter dominates and you do not know which. Random search covers every axis for the same cost.</text>
</svg>

```python
from sklearn.model_selection import GridSearchCV

grid = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [3, 5, 8],
    "clf__min_samples_leaf": [1, 5, 20],
}
search = GridSearchCV(pipeline, grid, cv=5, scoring="roc_auc", n_jobs=-1)
search.fit(X_train, y_train)
print(search.best_params_, search.best_score_)
```

Grid search tries every combination. Eighteen here, each trained five times, so
ninety fits. The cost multiplies with every parameter you add.

Random search samples combinations instead, and with more than two or three
parameters it usually wins for the same budget. The reason is in the diagram: in
most problems one parameter dominates and the others barely matter. A grid spends
its budget on a handful of values of the important one. A random search tries a
different value of it every single time.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, loguniform

distributions = {
    "clf__n_estimators": randint(100, 800),
    "clf__max_depth": randint(2, 12),
    "clf__learning_rate": loguniform(0.01, 0.3),
}
search = RandomizedSearchCV(pipeline, distributions, n_iter=40, cv=5, n_jobs=-1)
```

Sample learning rates and alphas on a **log scale**. The difference between 0.01
and 0.1 matters enormously and the difference between 0.21 and 0.22 does not.

!!! warning "Search inside the pipeline, never outside it"

    Note `clf__n_estimators` with a double underscore. That is how you reach a
    step inside a pipeline.

    Putting the whole pipeline into the search means the scaler and the encoder
    are refitted on the training fold of every split. Search the model alone,
    after transforming the data once, and every fold has seen the statistics of
    every other fold. That is the leakage from the Data Preprocessing page,
    arriving through the back door.

## Tuning without cheating

Here is the subtle failure, and it catches people who have done everything else
right.

You try a hundred configurations and keep the one with the best validation score.
That score is now optimistic, because you selected it for being high. Some of its
advantage is real and some is luck, and you have no way to tell which.

The fix is that the **test set is touched once**, at the very end, after all
decisions are made.

```python
search.fit(X_train, y_train)
print("validation:", search.best_score_)
print("test:", search.score(X_test, y_test))
```

Expect the test score to be a little lower. That gap is the price of having
searched, and reporting the validation score as your result is how people end up
promising something they cannot deliver.

When the decisions themselves need an honest estimate, **nested
cross-validation** puts a search inside every outer fold. It is expensive and
worth knowing the name of, and for the projects in this group a single held back
test set is enough.


## Keeping what you built

A tuned model that only exists in a notebook kernel is not a result.

```python
import joblib

joblib.dump(search.best_estimator_, "models/churn_pipeline.joblib")
model = joblib.load("models/churn_pipeline.joblib")
print(model.predict(X_test[:5]))
```

Save `best_estimator_`, not the search object, and note that it is the whole
pipeline: scaler, encoder and model together. Load it somewhere else and it
transforms raw columns exactly the way it did during training, which is the
point of having built it as one object.

!!! warning "A saved model is a snapshot of three things"

    The code, the data it was trained on, and the library versions. Load a
    pipeline saved by a different scikit-learn version and it may warn, misbehave
    or refuse.

    Save the version alongside it, in a filename or a small text file. This is
    the same reproducibility problem from the Data Preprocessing page, now
    applied to the model rather than the data.

## When it goes wrong

??? note "My validation score is higher than my training score"

    A leak, an unshuffled split, or a validation set that happens to be easier.
    Investigate rather than celebrate.

??? note "The grid search takes hours"

    Too many combinations. Switch to `RandomizedSearchCV`, reduce the folds
    while exploring, set `n_jobs=-1`, and search on a subsample first to find
    the promising region.

??? note "The best parameter is at the edge of my range"

    Your range was too narrow. Extend it in that direction and search again.

??? note "The learning curve is still rising at the right edge"

    More data would help. That is a finding worth reporting, and it may be more
    valuable than any tuning you could do.

??? note "Tuning barely changed my score"

    Common and not a failure. The gains in machine learning come mostly from
    features and data. If tuning moves you half a point, that is normal, and it
    is a signal to go back to the long loop from the Classification page.

??? note "`best_score_` and the test score disagree a lot"

    You searched too hard on too little data. Reduce the number of
    configurations, or accept that your test set is small and report the
    uncertainty alongside the number.

## Check yourself

1. Train a very deep tree and a depth-two tree on the same data. Report both
   scores for each and label which suffers from bias and which from variance.
2. Draw a learning curve for one model. Say whether more data would help and how
   you know.
3. Draw a validation curve for `max_depth`. Mark the minimum and explain what is
   happening on each side of it.
4. Run a grid search over three parameters, then a random search with the same
   number of fits. Compare the best scores.
5. Report your best validation score and your test score for the same model.
   Explain the gap in one sentence.
6. Take one model and reduce its variance three different ways. Record what each
   did to both scores.

## Quick reference

| Task | Code |
|---|---|
| Learning curve | `learning_curve(model, X, y, train_sizes=np.linspace(0.1, 1, 10))` |
| Validation curve | `validation_curve(model, X, y, param_name="alpha", param_range=...)` |
| Grid search | `GridSearchCV(pipeline, grid, cv=5, n_jobs=-1)` |
| Random search | `RandomizedSearchCV(pipeline, dists, n_iter=40, cv=5)` |
| Reach into a pipeline | `"clf__max_depth"` |
| Log scale sampling | `loguniform(0.01, 0.3)` |
| Best settings | `search.best_params_`, `search.best_score_` |
| Final honest score | `search.score(X_test, y_test)` |

## Questions to sit with

1. Bias is being wrong in a way more data will not fix. Name a problem where you
   suspect the bias comes from the features available rather than the model.
2. Some error is irreducible. How would you estimate that floor for a real
   problem, and what would you do with the number?
3. You tried a hundred configurations and kept the best. In what sense is that
   score dishonest, and how much does it matter?
4. Random search beats grid search because one parameter usually dominates. What
   does that say about the effort people spend tuning the others?
5. Tuning typically moves a score by less than a point, while better features can
   move it by ten. Why do you think tuning gets so much more attention?

## Next

That is the model half of the trimester, for supervised learning. The remaining
projects ask a different question: what can you do when there is no target column
at all?

[Unsupervised Learning](unsupervised-learning.md){ .h-button }

Worth your time: the
[scikit-learn model evaluation guide](https://scikit-learn.org/stable/modules/learning_curve.html)
covers both curve types with worked examples, and the
[tuning guide](https://scikit-learn.org/stable/modules/grid_search.html)
is the reference for both search strategies.

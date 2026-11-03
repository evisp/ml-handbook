# Probability

<span class="badge badge--time">100 min</span>
<span class="badge badge--level">Foundations</span>
<span class="badge">Needs: Plotting</span>

Machine learning is the business of making statements you are not certain about.
Probability is the language for saying how uncertain, and statistics is how you
check whether the uncertainty you claimed matches reality.

## Why this matters

A classifier does not tell you an image is a cat. It tells you it is 85 percent
sure. Every number a model gives you is a claim about probability, and by the end
of this programme you will be training models whose entire objective is to make
those claims better.

This page is also where the loss functions of trimester two stop being magic.
The thing a neural network minimises is a probability statement, and the last
section of this page is that statement.

Along the way you will learn how to tell whether a pattern in your data is real
or just luck, which is the question behind every hypothesis test you will run
on a dataset.

## The basics, quickly

A probability is a number between 0 and 1. Zero is impossible, one is certain,
everything interesting is in between.

| Notation | Meaning |
|---|---|
| $P(A)$ | the probability that A happens |
| $P(A^c)$ | the probability that A does not happen, equal to $1 - P(A)$ |
| $P(A \cap B)$ | both A and B happen |
| $P(A \cup B)$ | at least one of A and B happens |
| $P(A \mid B)$ | A happens, given that B has happened |

A fair die gives $P(6) = 1/6$ and $P(\text{even}) = 1/2$. Nothing here is
difficult. The difficulty starts when two events interact.

## Disjoint is not independent

These two words get swapped constantly, and they mean close to opposite things.

<svg viewBox="0 0 680 248" role="img" aria-labelledby="di-title di-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="di-title">Disjoint events compared with independent events</title>
<desc id="di-desc">Disjoint events cannot both happen, so their circles do not overlap. Independent events can both happen, and the overlap equals the product of the two probabilities.</desc>
<rect x="10" y="30" width="320" height="150" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="170" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">disjoint</text>
<circle cx="112" cy="105" r="48" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)" stroke-width="2"/>
<circle cx="228" cy="105" r="48" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)" stroke-width="2"/>
<text x="112" y="105" dy="0.36em" text-anchor="middle" font-size="16" font-weight="700" fill="var(--h-cherry)">A</text>
<text x="228" y="105" dy="0.36em" text-anchor="middle" font-size="16" font-weight="700" fill="var(--h-cherry)">B</text>
<text x="170" y="170" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">no overlap at all</text>
<rect x="350" y="30" width="320" height="150" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="510" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">independent</text>
<circle cx="466" cy="105" r="48" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)" stroke-width="2"/>
<circle cx="554" cy="105" r="48" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)" stroke-width="2"/>
<text x="440" y="105" dy="0.36em" text-anchor="middle" font-size="16" font-weight="700" fill="var(--h-cherry)">A</text>
<text x="580" y="105" dy="0.36em" text-anchor="middle" font-size="16" font-weight="700" fill="var(--h-cherry)">B</text>
<text x="510" y="109" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-cherry)">both</text>
<text x="510" y="170" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">overlap is exactly P(A) times P(B)</text>
<text x="340" y="206" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Disjoint events are never independent. If A happening rules B out, then A tells you a great deal about B.</text>
<text x="340" y="226" text-anchor="middle" font-size="12" fill="var(--h-graphite)">That is the opposite of independence, and it is the confusion that costs marks in every exam.</text>
</svg>

**Disjoint** means the events cannot both happen. Rolling a 3 and rolling a 5 on
one throw.

**Independent** means one event tells you nothing about the other. Two separate
coin flips.

If A and B are disjoint and A happens, you now know B did not. That is a very
strong piece of information, which is exactly what independence rules out.

## The two rules

**Addition**, for at least one of them happening:

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

The subtraction is there because adding the two counts the overlap twice. When
the events are disjoint the overlap is zero and the rule simplifies.

**Multiplication**, for both of them happening:

$$
P(A \cap B) = P(A) \cdot P(B \mid A)
$$

When the events are independent, $P(B \mid A) = P(B)$ and this becomes a plain
product.

Drawing two aces from a deck without replacement: the first is $4/52$, and the
second is $3/51$, because the deck changed. Multiply for about 0.45 percent.

## Conditional probability and Bayes

Conditional probability is the probability of one thing once you know another.

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}
$$

Bayes' theorem lets you turn a conditional around, which is the operation almost
every classifier performs.

$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$

The formula is easy to apply and hard to feel. So here is the classic example
counted out in people rather than symbols.

A disease affects 1 percent of the population. The test catches 95 percent of
people who have it, and wrongly flags 5 percent of people who do not. You test
positive. What is the chance you are ill?

<svg viewBox="0 0 680 336" role="img" aria-labelledby="by-title by-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="by-title">Bayes theorem worked as a group of 1000 people</title>
<desc id="by-desc">Out of 1000 people, 10 have the disease and 990 do not. The test finds 9 of the 10, and wrongly flags about 50 of the 990. So 59 people test positive and only 9 of them are actually ill.</desc>
<rect x="265" y="8" width="150" height="38" rx="9" fill="var(--md-default-fg-color)"/>
<text x="340" y="27" dy="0.36em" text-anchor="middle" font-size="13.5" font-weight="700" fill="var(--md-default-bg-color)">1000 people</text>
<polyline points="340,46 340,62 180,62 180,84" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<polyline points="340,46 340,62 500,62 500,84" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<rect x="100" y="84" width="160" height="38" rx="9" fill="var(--h-cherry)"/>
<text x="180" y="103" dy="0.36em" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">10 are ill</text>
<rect x="420" y="84" width="160" height="38" rx="9" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="500" y="103" dy="0.36em" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">990 are not</text>
<polyline points="180,122 180,138 96,138 96,160" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<polyline points="180,122 180,138 246,138 246,160" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<polyline points="500,122 500,138 416,138 416,160" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<polyline points="500,122 500,138 584,138 584,160" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<rect x="36" y="160" width="120" height="40" rx="8" fill="var(--h-cherry)"/>
<text x="96" y="176" dy="0.36em" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">9 positive</text>
<text x="96" y="192" dy="0.36em" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.85">correctly</text>
<rect x="186" y="160" width="120" height="40" rx="8" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="246" y="176" dy="0.36em" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-graphite)">1 missed</text>
<text x="246" y="192" dy="0.36em" text-anchor="middle" font-size="10" fill="var(--h-space)">told they are fine</text>
<rect x="356" y="160" width="120" height="40" rx="8" fill="var(--h-cherry-wash)" stroke="var(--h-cherry)" stroke-width="2"/>
<text x="416" y="176" dy="0.36em" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">50 positive</text>
<text x="416" y="192" dy="0.36em" text-anchor="middle" font-size="10" fill="var(--h-graphite)">wrongly</text>
<rect x="524" y="160" width="120" height="40" rx="8" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="584" y="176" dy="0.36em" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-graphite)">940 negative</text>
<text x="584" y="192" dy="0.36em" text-anchor="middle" font-size="10" fill="var(--h-space)">correctly</text>
<text x="340" y="232" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">59 people are told they tested positive</text>
<rect x="40" y="244" width="600" height="30" rx="6" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<rect x="40" y="244" width="92" height="30" rx="6" fill="var(--h-cherry)"/>
<text x="86" y="259" dy="0.36em" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">9 ill</text>
<text x="390" y="259" dy="0.36em" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-cherry)">50 perfectly healthy</text>
<text x="340" y="300" text-anchor="middle" font-size="12" fill="var(--h-graphite)">A positive result means about a 15 percent chance of being ill, not 95 percent.</text>
<text x="340" y="320" text-anchor="middle" font-size="12" fill="var(--h-graphite)">The test is good. The disease is rare. Rarity wins.</text>
</svg>

Almost everyone says 95 percent. The answer is about 15.

The reason is the **base rate**. There are so many more healthy people that 5
percent of them outnumbers 95 percent of the sick ones, five to one. The test
did nothing wrong. The population did it.

!!! ml "ML connection"

    This is class imbalance, which you met on the Working with Data page, seen
    from the probability side. Your fraud detector will run into exactly this:
    even a very accurate model, applied to a rare event, produces more false
    alarms than real catches. When somebody quotes you an accuracy figure, the
    first question is always what the base rate was.

You can check the whole thing by simulation, which is often faster than
rearranging symbols.

```python
import numpy as np

rng = np.random.default_rng(42)
n = 1_000_000
ill = rng.random(n) < 0.01
positive = np.where(ill, rng.random(n) < 0.95, rng.random(n) < 0.05)
print(ill[positive].mean())
```

```
0.16...
```

If you can simulate it, you never have to trust your algebra alone.

## Random variables and their distributions

A **random variable** is a quantity whose value depends on chance. Discrete ones
take countable values, such as the number of clicks. Continuous ones take any
value in a range, such as a prediction error.

A **distribution** describes how probability is spread across the possible
values. There are three ways to write one down and they answer different
questions.

<svg viewBox="0 0 680 236" role="img" aria-labelledby="fn-title fn-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="fn-title">PMF, PDF and CDF</title>
<desc id="fn-desc">A probability mass function gives the probability of each separate value. A probability density function gives probability as area under a curve. A cumulative distribution function gives the probability of being below a value.</desc>
<rect x="5" y="14" width="214" height="176" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="112" y="36" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">PMF</text>
<text x="112" y="52" text-anchor="middle" font-size="10.5" fill="var(--h-space)">one bar per value</text>
<line x1="35" y1="152" x2="195" y2="152" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="35" y1="152" x2="35" y2="66" stroke="var(--h-steel)" stroke-width="1.5"/>
<text x="112" y="178" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-cherry)">each bar is a probability</text>
<rect x="233" y="14" width="214" height="176" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="340" y="36" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">PDF</text>
<text x="340" y="52" text-anchor="middle" font-size="10.5" fill="var(--h-space)">a smooth curve</text>
<line x1="263" y1="152" x2="423" y2="152" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="263" y1="152" x2="263" y2="66" stroke="var(--h-steel)" stroke-width="1.5"/>
<text x="340" y="178" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-cherry)">area is the probability</text>
<rect x="461" y="14" width="214" height="176" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="568" y="36" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">CDF</text>
<text x="568" y="52" text-anchor="middle" font-size="10.5" fill="var(--h-space)">runs from 0 up to 1</text>
<line x1="491" y1="152" x2="651" y2="152" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="491" y1="152" x2="491" y2="66" stroke="var(--h-steel)" stroke-width="1.5"/>
<text x="568" y="178" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-cherry)">height is P(X at most x)</text>
<rect x="45" y="138" width="18" height="14" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="54" cy="138" r="3" fill="var(--h-cherry)"/>
<rect x="70" y="118" width="18" height="34" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="79" cy="118" r="3" fill="var(--h-cherry)"/>
<rect x="95" y="94" width="18" height="58" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="104" cy="94" r="3" fill="var(--h-cherry)"/>
<rect x="120" y="94" width="18" height="58" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="129" cy="94" r="3" fill="var(--h-cherry)"/>
<rect x="145" y="118" width="18" height="34" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="154" cy="118" r="3" fill="var(--h-cherry)"/>
<rect x="170" y="138" width="18" height="14" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="179" cy="138" r="3" fill="var(--h-cherry)"/>
<polyline points="267.0,151.4 269.5,151.2 272.1,150.9 274.6,150.5 277.1,150.1 279.7,149.4 282.2,148.7 284.7,147.7 287.3,146.5 289.8,145.1 292.3,143.3 294.9,141.3 297.4,138.9 299.9,136.2 302.5,133.0 305.0,129.6 307.5,125.8 310.1,121.6 312.6,117.2 315.1,112.6 317.7,107.8 320.2,103.0 322.7,98.3 325.3,93.7 327.8,89.5 330.3,85.6 332.9,82.3 335.4,79.6 337.9,77.6 340.5,76.4 343.0,76.0 345.5,76.4 348.1,77.6 350.6,79.6 353.1,82.3 355.7,85.6 358.2,89.5 360.7,93.7 363.3,98.3 365.8,103.0 368.3,107.8 370.9,112.6 373.4,117.2 375.9,121.6 378.5,125.8 381.0,129.6 383.5,133.0 386.1,136.2 388.6,138.9 391.1,141.3 393.7,143.3 396.2,145.1 398.7,146.5 401.3,147.7 403.8,148.7 406.3,149.4 408.9,150.1 411.4,150.5 413.9,150.9 416.5,151.2 419.0,151.4" fill="none" stroke="var(--h-cherry)" stroke-width="2.5"/>
<polygon points="312.6,152.0 312.6,117.2 315.1,112.6 317.7,107.8 320.2,103.0 322.7,98.3 325.3,93.7 327.8,89.5 330.3,85.6 332.9,82.3 335.4,79.6 337.9,77.6 340.5,76.4 343.0,76.0 345.5,76.4 348.1,77.6 350.6,79.6 353.1,82.3 353.1,152.0" fill="var(--h-cherry)" opacity="0.3"/>
<text x="337" y="146" text-anchor="middle" font-size="9.5" font-weight="700" fill="var(--h-cherry)">area</text>
<polyline points="495.0,152.0 497.5,151.9 500.1,151.9 502.6,151.8 505.1,151.8 507.7,151.7 510.2,151.5 512.7,151.3 515.3,151.1 517.8,150.8 520.3,150.5 522.9,150.0 525.4,149.5 527.9,148.8 530.5,148.0 533.0,147.1 535.5,146.0 538.1,144.7 540.6,143.3 543.1,141.6 545.7,139.8 548.2,137.8 550.7,135.5 553.3,133.1 555.8,130.5 558.3,127.7 560.9,124.8 563.4,121.8 565.9,118.7 568.5,115.6 571.0,112.4 573.5,109.3 576.1,106.2 578.6,103.2 581.1,100.3 583.7,97.5 586.2,94.9 588.7,92.5 591.3,90.2 593.8,88.2 596.3,86.4 598.9,84.7 601.4,83.3 603.9,82.0 606.5,80.9 609.0,80.0 611.5,79.2 614.1,78.5 616.6,78.0 619.1,77.5 621.7,77.2 624.2,76.9 626.7,76.7 629.3,76.5 631.8,76.3 634.3,76.2 636.9,76.2 639.4,76.1 641.9,76.1 644.5,76.0 647.0,76.0" fill="none" stroke="var(--h-cherry)" stroke-width="2.5"/>
<line x1="491" y1="76" x2="651" y2="76" stroke="var(--h-steel)" stroke-width="1" stroke-dasharray="3 3"/>
<text x="487" y="80" text-anchor="end" font-size="9.5" fill="var(--h-space)">1</text>
<text x="340" y="216" text-anchor="middle" font-size="12" fill="var(--h-graphite)">For a continuous variable the probability of any exact value is zero. Only ranges have probability.</text>
</svg>

The **PMF** gives the probability of each separate value, and only makes sense
for discrete variables. The **PDF** describes a continuous variable, where
probability lives in areas rather than points. The **CDF** gives the probability
of being at or below a value, and it is what percentiles are built from.

!!! warning "The height of a PDF is not a probability"

    It can exceed 1. Only the area under the curve is a probability, which is
    why $P(X = x) = 0$ exactly for a continuous variable. Asking for the
    probability that somebody is exactly 175.000 centimetres tall is asking for
    the area of a line.

## Describing a distribution with numbers

**Expected value** is the long run average, weighting each value by how likely
it is.

$$
E[X] = \sum_x x \, P(X = x)
$$

**Variance** is the average squared distance from that centre, and **standard
deviation** is its square root, which puts it back into the units of the data.

For the scores 70, 75, 80, 85 and 90, the mean is 80, the variance is 50 and the
standard deviation is about 7.07. So a typical score sits about 7 points from
the average.

```python
scores = np.array([70, 75, 80, 85, 90])
print(scores.mean(), scores.var(), scores.std())
```

Squaring is what makes variance awkward to interpret and useful to optimise. It
is differentiable everywhere and it punishes large errors far more than small
ones.

!!! ml "ML connection"

    Mean squared error is the variance of your prediction errors. When you
    minimise MSE you are shrinking the spread of how wrong you are, and the
    squaring is why one catastrophic prediction hurts more than ten mediocre
    ones. That is a choice, not a law, and in trimester two you will meet losses
    that make a different one.

### Percentiles

The p-th percentile is the value below which p percent of the data falls. The
25th, 50th and 75th are the quartiles, and the gap between the outer two is the
interquartile range.

```python
q1, med, q3 = np.percentile(scores, [25, 50, 75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
```

Those last two numbers are the whiskers of the box plot you drew last week, and
anything outside them is conventionally called an outlier. Conventionally, not
truly. It is a rule of thumb, not a verdict.

### When the mean stops being typical

<svg viewBox="0 0 680 246" role="img" aria-labelledby="sk-title sk-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="sk-title">Mean and median on a skewed distribution</title>
<desc id="sk-desc">On a right skewed distribution such as salaries, the long tail pulls the mean to the right of the median, so the mean no longer describes a typical value.</desc>
<line x1="60" y1="180" x2="640" y2="180" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="60" y1="180" x2="60" y2="40" stroke="var(--h-steel)" stroke-width="1.5"/>
<polygon points="60.0,180.0 60.0,180.0 64.8,178.2 69.7,174.7 74.5,170.2 79.3,165.1 84.2,159.7 89.0,153.9 93.8,148.0 98.7,142.1 103.5,136.2 108.3,130.4 113.2,124.6 118.0,119.1 122.8,113.7 127.7,108.5 132.5,103.5 137.3,98.8 142.2,94.3 147.0,90.1 151.8,86.1 156.7,82.4 161.5,79.0 166.3,75.8 171.2,72.8 176.0,70.2 180.8,67.7 185.7,65.5 190.5,63.5 195.3,61.8 200.2,60.3 205.0,59.0 209.8,57.9 214.7,56.9 219.5,56.2 224.3,55.7 229.2,55.3 234.0,55.1 238.8,55.0 243.7,55.1 248.5,55.3 253.3,55.7 258.2,56.1 263.0,56.7 267.8,57.4 272.7,58.2 277.5,59.1 282.3,60.1 287.2,61.2 292.0,62.3 296.8,63.5 301.7,64.8 306.5,66.1 311.3,67.5 316.2,69.0 321.0,70.4 325.8,72.0 330.7,73.5 335.5,75.1 340.3,76.7 345.2,78.3 350.0,80.0 354.8,81.7 359.7,83.4 364.5,85.1 369.3,86.8 374.2,88.5 379.0,90.2 383.8,91.9 388.7,93.6 393.5,95.3 398.3,97.0 403.2,98.7 408.0,100.4 412.8,102.1 417.7,103.8 422.5,105.4 427.3,107.0 432.2,108.6 437.0,110.3 441.8,111.8 446.7,113.4 451.5,114.9 456.3,116.5 461.2,118.0 466.0,119.4 470.8,120.9 475.7,122.3 480.5,123.8 485.3,125.2 490.2,126.5 495.0,127.9 499.8,129.2 504.7,130.5 509.5,131.8 514.3,133.0 519.2,134.2 524.0,135.4 528.8,136.6 533.7,137.8 538.5,138.9 543.3,140.0 548.2,141.1 553.0,142.1 557.8,143.2 562.7,144.2 567.5,145.2 572.3,146.1 577.2,147.1 582.0,148.0 586.8,148.9 591.7,149.8 596.5,150.6 601.3,151.5 606.2,152.3 611.0,153.1 615.8,153.9 620.7,154.6 625.5,155.4 630.3,156.1 635.2,156.8 640.0,157.5 640.0,180.0" fill="var(--h-cherry)" opacity="0.18"/>
<polyline points="60.0,180.0 64.8,178.2 69.7,174.7 74.5,170.2 79.3,165.1 84.2,159.7 89.0,153.9 93.8,148.0 98.7,142.1 103.5,136.2 108.3,130.4 113.2,124.6 118.0,119.1 122.8,113.7 127.7,108.5 132.5,103.5 137.3,98.8 142.2,94.3 147.0,90.1 151.8,86.1 156.7,82.4 161.5,79.0 166.3,75.8 171.2,72.8 176.0,70.2 180.8,67.7 185.7,65.5 190.5,63.5 195.3,61.8 200.2,60.3 205.0,59.0 209.8,57.9 214.7,56.9 219.5,56.2 224.3,55.7 229.2,55.3 234.0,55.1 238.8,55.0 243.7,55.1 248.5,55.3 253.3,55.7 258.2,56.1 263.0,56.7 267.8,57.4 272.7,58.2 277.5,59.1 282.3,60.1 287.2,61.2 292.0,62.3 296.8,63.5 301.7,64.8 306.5,66.1 311.3,67.5 316.2,69.0 321.0,70.4 325.8,72.0 330.7,73.5 335.5,75.1 340.3,76.7 345.2,78.3 350.0,80.0 354.8,81.7 359.7,83.4 364.5,85.1 369.3,86.8 374.2,88.5 379.0,90.2 383.8,91.9 388.7,93.6 393.5,95.3 398.3,97.0 403.2,98.7 408.0,100.4 412.8,102.1 417.7,103.8 422.5,105.4 427.3,107.0 432.2,108.6 437.0,110.3 441.8,111.8 446.7,113.4 451.5,114.9 456.3,116.5 461.2,118.0 466.0,119.4 470.8,120.9 475.7,122.3 480.5,123.8 485.3,125.2 490.2,126.5 495.0,127.9 499.8,129.2 504.7,130.5 509.5,131.8 514.3,133.0 519.2,134.2 524.0,135.4 528.8,136.6 533.7,137.8 538.5,138.9 543.3,140.0 548.2,141.1 553.0,142.1 557.8,143.2 562.7,144.2 567.5,145.2 572.3,146.1 577.2,147.1 582.0,148.0 586.8,148.9 591.7,149.8 596.5,150.6 601.3,151.5 606.2,152.3 611.0,153.1 615.8,153.9 620.7,154.6 625.5,155.4 630.3,156.1 635.2,156.8 640.0,157.5" fill="none" stroke="var(--h-cherry)" stroke-width="2.5"/>
<line x1="138" y1="180" x2="138" y2="52" stroke="var(--md-default-fg-color)" stroke-width="2" stroke-dasharray="5 4"/>
<text x="138" y="44" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">median</text>
<line x1="196" y1="180" x2="196" y2="76" stroke="var(--h-cherry)" stroke-width="2"/>
<text x="238" y="70" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-cherry)">mean</text>
<text x="470" y="150" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">a few very large values live out here</text>
<text x="340" y="212" text-anchor="middle" font-size="12" fill="var(--h-graphite)">The tail drags the mean away from the middle. Half the people are still below the median.</text>
<text x="340" y="232" text-anchor="middle" font-size="12" fill="var(--h-graphite)">This is why you filled missing salaries with the median on the Pandas page.</text>
</svg>

On a symmetric distribution the mean and median sit together. On a skewed one
they separate, and the mean stops describing anybody in particular.

Report the median for skewed data, and report both when you want to show that
the skew exists. A dataset where the mean is far above the median is telling you
something before you have plotted anything.

## The distributions worth knowing

<svg viewBox="0 0 680 320" role="img" aria-labelledby="ga-title ga-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="ga-title">Six distributions and what each one models</title>
<desc id="ga-desc">Uniform for equally likely values. Normal for measurements clustered around an average. Bernoulli for one yes or no trial. Binomial for the number of successes in many trials. Poisson for counts of rare events. Exponential for waiting times.</desc>
<rect x="5" y="14" width="214" height="138" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="112" y="38" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">Uniform</text>
<text x="112" y="142" text-anchor="middle" font-size="10.5" fill="var(--h-space)">every value equally likely</text>
<line x1="33" y1="120" x2="193" y2="120" stroke="var(--h-steel)" stroke-width="1.2"/>
<rect x="233" y="14" width="214" height="138" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="340" y="38" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">Normal</text>
<text x="340" y="142" text-anchor="middle" font-size="10.5" fill="var(--h-space)">clustered around an average</text>
<line x1="261" y1="120" x2="421" y2="120" stroke="var(--h-steel)" stroke-width="1.2"/>
<rect x="461" y="14" width="214" height="138" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="568" y="38" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">Bernoulli</text>
<text x="568" y="142" text-anchor="middle" font-size="10.5" fill="var(--h-space)">one yes or no trial</text>
<line x1="489" y1="120" x2="649" y2="120" stroke="var(--h-steel)" stroke-width="1.2"/>
<rect x="5" y="170" width="214" height="138" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="112" y="194" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">Binomial</text>
<text x="112" y="298" text-anchor="middle" font-size="10.5" fill="var(--h-space)">successes out of n trials</text>
<line x1="33" y1="276" x2="193" y2="276" stroke="var(--h-steel)" stroke-width="1.2"/>
<rect x="233" y="170" width="214" height="138" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="340" y="194" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">Poisson</text>
<text x="340" y="298" text-anchor="middle" font-size="10.5" fill="var(--h-space)">how many rare events</text>
<line x1="261" y1="276" x2="421" y2="276" stroke="var(--h-steel)" stroke-width="1.2"/>
<rect x="461" y="170" width="214" height="138" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="568" y="194" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">Exponential</text>
<text x="568" y="298" text-anchor="middle" font-size="10.5" fill="var(--h-space)">how long until the next one</text>
<line x1="489" y1="276" x2="649" y2="276" stroke="var(--h-steel)" stroke-width="1.2"/>
<rect x="37.0" y="73.8" width="22.0" height="46.2" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="62.0" y="73.8" width="22.0" height="46.2" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="87.0" y="73.8" width="22.0" height="46.2" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="112.0" y="73.8" width="22.0" height="46.2" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="137.0" y="73.8" width="22.0" height="46.2" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="162.0" y="73.8" width="22.0" height="46.2" fill="var(--h-cherry)" opacity="0.85"/>
<polyline points="265.0,119.1 268.0,118.8 271.1,118.3 274.1,117.7 277.2,116.9 280.2,115.9 283.2,114.6 286.3,113.0 289.3,111.1 292.4,108.8 295.4,106.1 298.4,103.0 301.5,99.5 304.5,95.6 307.6,91.4 310.6,87.0 313.6,82.3 316.7,77.6 319.7,73.0 322.8,68.6 325.8,64.5 328.8,60.9 331.9,58.0 334.9,55.8 338.0,54.5 341.0,54.0 344.0,54.5 347.1,55.8 350.1,58.0 353.2,60.9 356.2,64.5 359.2,68.6 362.3,73.0 365.3,77.6 368.4,82.3 371.4,87.0 374.4,91.4 377.5,95.6 380.5,99.5 383.6,103.0 386.6,106.1 389.6,108.8 392.7,111.1 395.7,113.0 398.8,114.6 401.8,115.9 404.8,116.9 407.9,117.7 410.9,118.3 414.0,118.8 417.0,119.1" fill="none" stroke="var(--h-cherry)" stroke-width="2.5"/>
<rect x="493.0" y="96.9" width="72.0" height="23.1" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="568.0" y="70.5" width="72.0" height="49.5" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="37.0" y="270.7" width="18.4" height="5.3" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="58.4" y="258.8" width="18.4" height="17.2" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="79.9" y="239.7" width="18.4" height="36.3" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="101.3" y="223.2" width="18.4" height="52.8" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="122.7" y="235.1" width="18.4" height="40.9" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="144.1" y="256.2" width="18.4" height="19.8" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="165.6" y="269.4" width="18.4" height="6.6" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="265.0" y="259.5" width="18.4" height="16.5" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="286.4" y="235.1" width="18.4" height="40.9" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="307.9" y="223.2" width="18.4" height="52.8" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="329.3" y="236.4" width="18.4" height="39.6" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="350.7" y="252.9" width="18.4" height="23.1" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="372.1" y="265.4" width="18.4" height="10.6" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="393.6" y="272.0" width="18.4" height="4.0" fill="var(--h-cherry)" opacity="0.85"/>
<polyline points="493.0,210.0 496.0,214.1 499.1,217.9 502.1,221.5 505.2,224.9 508.2,228.1 511.2,231.0 514.3,233.8 517.3,236.4 520.4,238.9 523.4,241.2 526.4,243.4 529.5,245.4 532.5,247.3 535.6,249.1 538.6,250.7 541.6,252.3 544.7,253.8 547.7,255.1 550.8,256.4 553.8,257.6 556.8,258.8 559.9,259.9 562.9,260.9 566.0,261.8 569.0,262.7 572.0,263.5 575.1,264.3 578.1,265.0 581.2,265.7 584.2,266.3 587.2,266.9 590.3,267.5 593.3,268.0 596.4,268.5 599.4,269.0 602.4,269.4 605.5,269.8 608.5,270.2 611.6,270.6 614.6,270.9 617.6,271.2 620.7,271.5 623.7,271.8 626.8,272.1 629.8,272.3 632.8,272.5 635.9,272.7 638.9,272.9 642.0,273.1 645.0,273.3" fill="none" stroke="var(--h-cherry)" stroke-width="2.5"/>
<text x="340" y="300" text-anchor="middle" font-size="12" fill="var(--h-graphite)">The three outlined in red are the ones you will use constantly. Recognise the other three and move on.</text>
</svg>

### Normal

The bell curve. Symmetric, described entirely by a mean $\mu$ and a standard
deviation $\sigma$.

The rule worth memorising: **68, 95, 99.7**. About 68 percent of values fall
within one standard deviation of the mean, 95 percent within two, and 99.7
percent within three. That is how you judge in your head whether a value is
unusual.

```python
heights = rng.normal(loc=170, scale=8, size=10_000)
within_one = np.mean(np.abs(heights - 170) < 8)
print(within_one)
```

```
0.682...
```

### Bernoulli

One trial with two outcomes, described by a single number $p$. A coin flip, a
click, a fraud or not fraud.

This is the distribution every binary classifier outputs. When logistic
regression returns 0.85, it is handing you a Bernoulli distribution.

### Binomial

The number of successes in $n$ independent Bernoulli trials. Mean $np$, variance
$np(1-p)$.

```python
successes = rng.binomial(n=10, p=0.5, size=10_000)
print(successes.mean())
```

Useful the moment you ask whether a result could have happened by chance. If a
model gets 60 out of 100 right and guessing gives 50, the binomial tells you how
often luck alone would manage 60.

### Three to recognise

**Uniform** gives every value in a range the same chance, and is what you get
from `rng.random()`. It is used for random initialisation and for shuffling.

**Poisson** counts rare events in a fixed window: emails per hour, arrivals per
day. One parameter, $\lambda$, which is both the mean and the variance.

**Exponential** measures the waiting time between those events. If the counts are
Poisson, the gaps are exponential.

You will not use these three often in this programme. Know what they model so
you recognise them when they appear.

## Sampling and seeing the shape

The fastest way to understand a distribution is to draw from it and plot it.

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
sns.histplot(rng.normal(0, 1, 5000), bins=40, ax=axes[0])
axes[0].set_title("Normal")
sns.histplot(rng.binomial(10, 0.5, 5000), bins=11, ax=axes[1])
axes[1].set_title("Binomial, n=10, p=0.5")
sns.histplot(rng.exponential(1.0, 5000), bins=40, ax=axes[2])
axes[2].set_title("Exponential")
fig.tight_layout()
```

Change the parameters and run it again. Ten minutes of this teaches more than
reading the formulas twice.

## Why the bell curve is everywhere

The **Central Limit Theorem** says that if you take averages of samples, those
averages tend towards a normal distribution as the sample grows, almost
regardless of the shape you started with.

<svg viewBox="0 0 680 256" role="img" aria-labelledby="cl-title cl-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="cl-title">The distribution of an average as the sample grows</title>
<desc id="cl-desc">Averages of one observation are spread widely. Averages of ten are narrower. Averages of a hundred cluster tightly around the true value, and the shape becomes a bell whatever the original data looked like.</desc>
<line x1="50" y1="186" x2="650" y2="186" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="350" y1="186" x2="350" y2="34" stroke="var(--md-default-fg-color)" stroke-width="1.5" stroke-dasharray="5 4"/>
<text x="350" y="26" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--md-default-fg-color)">the true value</text>
<polyline points="50.0,148.6 53.8,147.3 57.5,145.9 61.2,144.6 65.0,143.2 68.8,141.7 72.5,140.3 76.2,138.8 80.0,137.3 83.8,135.8 87.5,134.2 91.2,132.6 95.0,131.0 98.8,129.4 102.5,127.7 106.2,126.0 110.0,124.3 113.8,122.6 117.5,120.9 121.2,119.1 125.0,117.3 128.8,115.5 132.5,113.7 136.2,111.9 140.0,110.0 143.8,108.2 147.5,106.3 151.2,104.5 155.0,102.6 158.8,100.7 162.5,98.8 166.2,96.9 170.0,95.0 173.8,93.1 177.5,91.2 181.2,89.3 185.0,87.5 188.8,85.6 192.5,83.7 196.2,81.8 200.0,80.0 203.8,78.2 207.5,76.4 211.2,74.6 215.0,72.8 218.8,71.0 222.5,69.3 226.2,67.6 230.0,65.9 233.8,64.2 237.5,62.6 241.2,61.0 245.0,59.5 248.8,57.9 252.5,56.5 256.2,55.0 260.0,53.6 263.8,52.3 267.5,51.0 271.2,49.7 275.0,48.5 278.8,47.3 282.5,46.2 286.2,45.1 290.0,44.1 293.8,43.1 297.5,42.2 301.2,41.4 305.0,40.6 308.8,39.9 312.5,39.2 316.2,38.6 320.0,38.1 323.8,37.6 327.5,37.2 331.2,36.8 335.0,36.5 338.8,36.3 342.5,36.1 346.2,36.0 350.0,36.0 353.8,36.0 357.5,36.1 361.2,36.3 365.0,36.5 368.8,36.8 372.5,37.2 376.2,37.6 380.0,38.1 383.8,38.6 387.5,39.2 391.2,39.9 395.0,40.6 398.8,41.4 402.5,42.2 406.2,43.1 410.0,44.1 413.8,45.1 417.5,46.2 421.2,47.3 425.0,48.5 428.8,49.7 432.5,51.0 436.2,52.3 440.0,53.6 443.8,55.0 447.5,56.5 451.2,57.9 455.0,59.5 458.8,61.0 462.5,62.6 466.2,64.2 470.0,65.9 473.8,67.6 477.5,69.3 481.2,71.0 485.0,72.8 488.8,74.6 492.5,76.4 496.2,78.2 500.0,80.0 503.8,81.8 507.5,83.7 511.2,85.6 515.0,87.5 518.8,89.3 522.5,91.2 526.2,93.1 530.0,95.0 533.8,96.9 537.5,98.8 541.2,100.7 545.0,102.6 548.8,104.5 552.5,106.3 556.2,108.2 560.0,110.0 563.8,111.9 567.5,113.7 571.2,115.5 575.0,117.3 578.8,119.1 582.5,120.9 586.2,122.6 590.0,124.3 593.8,126.0 597.5,127.7 601.2,129.4 605.0,131.0 608.8,132.6 612.5,134.2 616.2,135.8 620.0,137.3 623.8,138.8 627.5,140.3 631.2,141.7 635.0,143.2 638.8,144.6 642.5,145.9 646.2,147.3 650.0,148.6" fill="none" stroke="var(--h-cherry)" stroke-width="2.5" opacity="0.22"/>
<polyline points="50.0,185.7 53.8,185.7 57.5,185.7 61.2,185.6 65.0,185.5 68.8,185.4 72.5,185.4 76.2,185.3 80.0,185.1 83.8,185.0 87.5,184.9 91.2,184.7 95.0,184.5 98.8,184.3 102.5,184.0 106.2,183.8 110.0,183.5 113.8,183.1 117.5,182.7 121.2,182.3 125.0,181.8 128.8,181.3 132.5,180.7 136.2,180.1 140.0,179.4 143.8,178.6 147.5,177.8 151.2,176.9 155.0,175.9 158.8,174.8 162.5,173.6 166.2,172.3 170.0,170.9 173.8,169.4 177.5,167.8 181.2,166.1 185.0,164.2 188.8,162.2 192.5,160.1 196.2,157.9 200.0,155.5 203.8,153.1 207.5,150.4 211.2,147.7 215.0,144.8 218.8,141.7 222.5,138.6 226.2,135.3 230.0,131.9 233.8,128.4 237.5,124.8 241.2,121.1 245.0,117.3 248.8,113.5 252.5,109.5 256.2,105.5 260.0,101.5 263.8,97.5 267.5,93.4 271.2,89.3 275.0,85.3 278.8,81.3 282.5,77.4 286.2,73.5 290.0,69.8 293.8,66.1 297.5,62.6 301.2,59.2 305.0,56.1 308.8,53.0 312.5,50.2 316.2,47.6 320.0,45.3 323.8,43.1 327.5,41.3 331.2,39.7 335.0,38.4 338.8,37.3 342.5,36.6 346.2,36.1 350.0,36.0 353.8,36.1 357.5,36.6 361.2,37.3 365.0,38.4 368.8,39.7 372.5,41.3 376.2,43.1 380.0,45.3 383.8,47.6 387.5,50.2 391.2,53.0 395.0,56.1 398.8,59.2 402.5,62.6 406.2,66.1 410.0,69.8 413.8,73.5 417.5,77.4 421.2,81.3 425.0,85.3 428.8,89.3 432.5,93.4 436.2,97.5 440.0,101.5 443.8,105.5 447.5,109.5 451.2,113.5 455.0,117.3 458.8,121.1 462.5,124.8 466.2,128.4 470.0,131.9 473.8,135.3 477.5,138.6 481.2,141.7 485.0,144.8 488.8,147.7 492.5,150.4 496.2,153.1 500.0,155.5 503.8,157.9 507.5,160.1 511.2,162.2 515.0,164.2 518.8,166.1 522.5,167.8 526.2,169.4 530.0,170.9 533.8,172.3 537.5,173.6 541.2,174.8 545.0,175.9 548.8,176.9 552.5,177.8 556.2,178.6 560.0,179.4 563.8,180.1 567.5,180.7 571.2,181.3 575.0,181.8 578.8,182.3 582.5,182.7 586.2,183.1 590.0,183.5 593.8,183.8 597.5,184.0 601.2,184.3 605.0,184.5 608.8,184.7 612.5,184.9 616.2,185.0 620.0,185.1 623.8,185.3 627.5,185.4 631.2,185.4 635.0,185.5 638.8,185.6 642.5,185.7 646.2,185.7 650.0,185.7" fill="none" stroke="var(--h-cherry)" stroke-width="2.5" opacity="0.45"/>
<polyline points="50.0,186.0 53.8,186.0 57.5,186.0 61.2,186.0 65.0,186.0 68.8,186.0 72.5,186.0 76.2,186.0 80.0,186.0 83.8,186.0 87.5,186.0 91.2,186.0 95.0,186.0 98.8,186.0 102.5,186.0 106.2,186.0 110.0,186.0 113.8,186.0 117.5,186.0 121.2,186.0 125.0,186.0 128.8,186.0 132.5,186.0 136.2,186.0 140.0,186.0 143.8,186.0 147.5,186.0 151.2,186.0 155.0,186.0 158.8,186.0 162.5,186.0 166.2,186.0 170.0,186.0 173.8,186.0 177.5,186.0 181.2,186.0 185.0,186.0 188.8,186.0 192.5,186.0 196.2,186.0 200.0,186.0 203.8,186.0 207.5,186.0 211.2,186.0 215.0,186.0 218.8,185.9 222.5,185.9 226.2,185.9 230.0,185.8 233.8,185.7 237.5,185.6 241.2,185.3 245.0,185.1 248.8,184.6 252.5,184.1 256.2,183.3 260.0,182.4 263.8,181.1 267.5,179.4 271.2,177.3 275.0,174.7 278.8,171.4 282.5,167.5 286.2,162.8 290.0,157.3 293.8,150.9 297.5,143.7 301.2,135.6 305.0,126.8 308.8,117.3 312.5,107.4 316.2,97.1 320.0,86.8 323.8,76.7 327.5,67.1 331.2,58.4 335.0,50.7 338.8,44.5 342.5,39.8 346.2,37.0 350.0,36.0 353.8,37.0 357.5,39.8 361.2,44.5 365.0,50.7 368.8,58.4 372.5,67.1 376.2,76.7 380.0,86.8 383.8,97.1 387.5,107.4 391.2,117.3 395.0,126.8 398.8,135.6 402.5,143.7 406.2,150.9 410.0,157.3 413.8,162.8 417.5,167.5 421.2,171.4 425.0,174.7 428.8,177.3 432.5,179.4 436.2,181.1 440.0,182.4 443.8,183.3 447.5,184.1 451.2,184.6 455.0,185.1 458.8,185.3 462.5,185.6 466.2,185.7 470.0,185.8 473.8,185.9 477.5,185.9 481.2,185.9 485.0,186.0 488.8,186.0 492.5,186.0 496.2,186.0 500.0,186.0 503.8,186.0 507.5,186.0 511.2,186.0 515.0,186.0 518.8,186.0 522.5,186.0 526.2,186.0 530.0,186.0 533.8,186.0 537.5,186.0 541.2,186.0 545.0,186.0 548.8,186.0 552.5,186.0 556.2,186.0 560.0,186.0 563.8,186.0 567.5,186.0 571.2,186.0 575.0,186.0 578.8,186.0 582.5,186.0 586.2,186.0 590.0,186.0 593.8,186.0 597.5,186.0 601.2,186.0 605.0,186.0 608.8,186.0 612.5,186.0 616.2,186.0 620.0,186.0 623.8,186.0 627.5,186.0 631.2,186.0 635.0,186.0 638.8,186.0 642.5,186.0 646.2,186.0 650.0,186.0" fill="none" stroke="var(--h-cherry)" stroke-width="2.5" opacity="1.0"/>
<text x="120" y="150" font-size="11.5" font-weight="700" fill="var(--h-cherry)" opacity="0.5">n = 1</text>
<text x="236" y="96" font-size="11.5" font-weight="700" fill="var(--h-cherry)" opacity="0.7">n = 10</text>
<text x="368" y="52" font-size="11.5" font-weight="700" fill="var(--h-cherry)">n = 100</text>
<text x="340" y="216" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Averages of larger samples land closer to the truth, and the spread shrinks with the square root of n.</text>
<text x="340" y="236" text-anchor="middle" font-size="12" fill="var(--h-graphite)">To halve the uncertainty you need four times the data.</text>
</svg>

```python
rng = np.random.default_rng(1)
population = rng.exponential(1.0, 1_000_000)
means_10 = [rng.choice(population, 10).mean() for _ in range(2000)]
means_100 = [rng.choice(population, 100).mean() for _ in range(2000)]
print(np.std(means_10), np.std(means_100))
```

The population is strongly skewed and the averages are not. The standard
deviation of the averages falls by roughly a factor of three when the sample
grows from 10 to 100, because it shrinks with the square root of $n$.

Two consequences you will use constantly. Averages are far more trustworthy than
single observations. And to halve your uncertainty you need four times the data,
which is why collecting more of it has diminishing returns.

## Your test score is an estimate

This is the practical payoff of everything above, and it is the idea that will
save you from the most embarrassing mistakes in trimester two.

When you measure accuracy on a test set, you are measuring a sample. Run the
same model on a different sample and you get a different number. The question is
whether a difference you observe is real or is noise.

```python
rng = np.random.default_rng(7)
true_accuracy = 0.90
for n in (100, 1000, 10_000):
    trials = rng.binomial(n, true_accuracy, 2000) / n
    print(n, round(trials.min(), 3), round(trials.max(), 3))
```

```
100 0.81 0.98
1000 0.868 0.932
10000 0.891 0.911
```

The model never changed. On 100 test rows the measured accuracy wanders between
81 and 98 percent purely by chance. So an improvement from 91 to 92 percent on a
small test set is not an improvement, it is weather.

!!! warning "The rough rule"

    The uncertainty on an accuracy measured over $n$ items is roughly
    $\sqrt{p(1-p)/n}$. At 90 percent accuracy on 100 items that is about 3
    percentage points, so differences smaller than about 6 points mean nothing.
    Compute it before you celebrate.

## Is a difference real?

The last section asked whether a change in accuracy is real or just noise. The
same question turns up all over data work. Customers who signed up on the
website come back more often than customers who signed up by phone. Students
who revised a new way scored higher. Is that a real pattern, or the kind of gap
that luck produces all the time?

A **hypothesis test** answers this with one idea: assume nothing is going on,
then ask how surprising your data would be.

The "nothing is going on" story has a name, the **null hypothesis**. For two
groups it says the group makes no difference. Both groups come from the same
population, and any gap you see is luck in who happened to land where.

### Answer it by shuffling first

Here is a concrete case. One class of 12 students tried a new way of revising.
Another class of 12 revised the usual way. These are their exam scores.

```python
import numpy as np

new_way = np.array([78, 85, 69, 91, 74, 88, 82, 93, 71, 84, 79, 90])
usual = np.array([74, 82, 66, 78, 71, 86, 70, 77, 80, 64, 73, 79])
print(new_way.mean(), usual.mean())
```

```
82.0 75.0
```

A gap of 7 points. Now suppose the revision method made no difference at all.
Then the labels "new way" and "usual" are just stickers. You could peel them
off, shuffle all 24 scores, deal them back into two groups of 12, and nothing
meaningful would change. So let's do exactly that, ten thousand times, and see
how big the gap usually is when it is pure luck.

```python
rng = np.random.default_rng(0)
observed = new_way.mean() - usual.mean()
pooled = np.concatenate([new_way, usual])

diffs = []
for _ in range(10_000):
    rng.shuffle(pooled)
    diffs.append(pooled[:12].mean() - pooled[12:].mean())
diffs = np.array(diffs)

print((np.abs(diffs) >= abs(observed)).mean())
```

```
0.0321
```

This is a **permutation test**, and the picture shows what it did.

<svg viewBox="0 0 680 300" role="img" aria-labelledby="pm-title pm-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="pm-title">Shuffled differences and the p-value</title>
<desc id="pm-desc">A histogram of 10,000 differences in mean score produced by shuffling the group labels. Most are within 5 points of zero. The real difference of 7 points sits far out on the right, and only about 3 percent of shuffles reach 7 points or more in either direction. That share is the p-value.</desc>
<rect x="95.6" y="213.0" width="20.3" height="1.0" rx="2" fill="var(--h-cherry)" opacity="1"/>
<rect x="117.9" y="212.2" width="20.3" height="1.8" rx="2" fill="var(--h-cherry)" opacity="1"/>
<rect x="140.2" y="207.2" width="20.3" height="6.8" rx="2" fill="var(--h-cherry)" opacity="1"/>
<rect x="162.5" y="201.1" width="20.3" height="12.9" rx="2" fill="var(--h-cherry)" opacity="1"/>
<rect x="184.8" y="188.8" width="20.3" height="25.2" rx="2" fill="var(--h-steel)" opacity="0.45"/>
<rect x="207.2" y="165.6" width="20.3" height="48.4" rx="2" fill="var(--h-steel)" opacity="0.45"/>
<rect x="229.5" y="137.5" width="20.3" height="76.5" rx="2" fill="var(--h-steel)" opacity="0.45"/>
<rect x="251.8" y="112.8" width="20.3" height="101.2" rx="2" fill="var(--h-steel)" opacity="0.45"/>
<rect x="274.1" y="82.8" width="20.3" height="131.2" rx="2" fill="var(--h-steel)" opacity="0.45"/>
<rect x="296.4" y="62.0" width="20.3" height="152.0" rx="2" fill="var(--h-steel)" opacity="0.45"/>
<rect x="318.7" y="78.9" width="20.3" height="135.1" rx="2" fill="var(--h-steel)" opacity="0.45"/>
<rect x="341.0" y="52.0" width="20.3" height="162.0" rx="2" fill="var(--h-steel)" opacity="0.45"/>
<rect x="363.3" y="52.4" width="20.3" height="161.6" rx="2" fill="var(--h-steel)" opacity="0.45"/>
<rect x="385.6" y="81.9" width="20.3" height="132.1" rx="2" fill="var(--h-steel)" opacity="0.45"/>
<rect x="407.9" y="116.5" width="20.3" height="97.5" rx="2" fill="var(--h-steel)" opacity="0.45"/>
<rect x="430.2" y="145.4" width="20.3" height="68.6" rx="2" fill="var(--h-steel)" opacity="0.45"/>
<rect x="452.5" y="163.8" width="20.3" height="50.2" rx="2" fill="var(--h-steel)" opacity="0.45"/>
<rect x="474.8" y="186.1" width="20.3" height="27.9" rx="2" fill="var(--h-steel)" opacity="0.45"/>
<rect x="497.2" y="200.6" width="20.3" height="13.4" rx="2" fill="var(--h-cherry)" opacity="1"/>
<rect x="519.5" y="207.9" width="20.3" height="6.1" rx="2" fill="var(--h-cherry)" opacity="1"/>
<rect x="541.8" y="211.2" width="20.3" height="2.8" rx="2" fill="var(--h-cherry)" opacity="1"/>
<rect x="564.1" y="213.6" width="20.3" height="0.4" rx="2" fill="var(--h-cherry)" opacity="1"/>
<rect x="608.7" y="213.9" width="20.3" height="0.1" rx="2" fill="var(--h-cherry)" opacity="1"/>
<line x1="50" y1="214" x2="630" y2="214" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="116.9" y1="214" x2="116.9" y2="219" stroke="var(--h-steel)" stroke-width="1.5"/>
<text x="116.9" y="233" text-anchor="middle" font-size="11" fill="var(--h-space)">-10</text>
<line x1="228.5" y1="214" x2="228.5" y2="219" stroke="var(--h-steel)" stroke-width="1.5"/>
<text x="228.5" y="233" text-anchor="middle" font-size="11" fill="var(--h-space)">-5</text>
<line x1="340.0" y1="214" x2="340.0" y2="219" stroke="var(--h-steel)" stroke-width="1.5"/>
<text x="340.0" y="233" text-anchor="middle" font-size="11" fill="var(--h-space)">0</text>
<line x1="451.5" y1="214" x2="451.5" y2="219" stroke="var(--h-steel)" stroke-width="1.5"/>
<text x="451.5" y="233" text-anchor="middle" font-size="11" fill="var(--h-space)">5</text>
<line x1="563.1" y1="214" x2="563.1" y2="219" stroke="var(--h-steel)" stroke-width="1.5"/>
<text x="563.1" y="233" text-anchor="middle" font-size="11" fill="var(--h-space)">10</text>
<text x="340" y="252" text-anchor="middle" font-size="11.5" fill="var(--h-space)">difference in mean score after shuffling the labels (points)</text>
<line x1="496.2" y1="214" x2="496.2" y2="40" stroke="var(--md-default-fg-color)" stroke-width="2"/>
<text x="504.2" y="50" text-anchor="start" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">what we saw:</text>
<text x="504.2" y="66" text-anchor="start" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">7 points</text>
<line x1="183.8" y1="214" x2="183.8" y2="70" stroke="var(--md-default-fg-color)" stroke-width="1.5" stroke-dasharray="5 4"/>
<text x="175.8" y="64" text-anchor="end" font-size="11" fill="var(--h-space)">just as extreme the other way</text>
<text x="340.0" y="30" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">luck alone usually lands here</text>
<rect x="110" y="262" width="460" height="30" rx="6" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="340" y="277" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-cherry)" dy="0.36em">Red bars: about 3 in 100 shuffles. That share is the p-value, p ≈ 0.03.</text>
</svg>

Most shuffles give a gap within about 5 points of zero. Only about 3 in 100
reach 7 points or more, in either direction. That fraction is the **p-value**.

> The p-value is the probability of seeing a difference at least as big as
> yours, **if nothing were going on**.

A small p-value means luck rarely produces something like this, so the "nothing
is going on" story becomes hard to believe. A large p-value means luck produces
this all the time, so your data gives you no reason to doubt that story.

We counted gaps in both directions because we did not know in advance which
class would do better, so a gap of minus 7 would have been just as surprising.
This is called a two-sided test. It is SciPy's default, and the one to use
unless you have a strong reason not to.

### The 0.05 line

By convention, a result is called **statistically significant** when $p < 0.05$,
meaning luck alone would produce something this extreme less than 1 time in 20.
There is nothing magic about 0.05. It is a habit the field settled on, and
$p = 0.049$ and $p = 0.051$ are the same evidence. Report the number, not just
which side of the line it fell on.

### Two ways to be wrong

Whatever you conclude, you can be wrong in one of two ways.

| | Nothing is really going on | Something really is |
|---|---|---|
| You say "real difference" | **False alarm** | Correct |
| You say "could be luck" | Correct | **Missed effect** |

The 0.05 line is the false alarm rate you agree to live with: when nothing is
going on, 1 test in 20 will still come out significant. Missed effects happen
mostly when data is small. A real 7 point gain measured on 4 students per class
can easily look like luck. These are the false positives and false negatives of
the medical test again. Statisticians call them Type I and Type II errors.

Every test on this page follows the same three steps.

1. Pick a number that measures the difference, such as a gap in means or a gap
   between counts.
2. Work out how that number behaves when nothing is going on.
3. See how far out your real number falls. That is the p-value.

Shuffling does step 2 by brute force. The tests below do it with a formula,
which is faster and is what you will see in libraries and reports. When you are
unsure about a formula, you can always check it against shuffling.

## Two categories: the chi-square test

Often both things you care about are categories. Did the customer come back,
yes or no? How did they sign up: by phone, in a store, or on the website? The
question is whether the two are related.

Start by counting. `pd.crosstab` builds a **contingency table**, with one row
for each value of one column, one column for each value of the other, and a
count in every cell.

```python
import pandas as pd

print(pd.crosstab(df["signup"], df["came_back"], margins=True))
```

```
came_back   No  Yes  All
signup
phone       52   48  100
store       50   30   80
web         48   72  120
All        150  150  300
```

Each row is a conditional probability waiting to be read.
$P(\text{came back} \mid \text{web}) = 72/120 = 60$ percent, against 48 percent
for phone and 37.5 percent for store. Passing `normalize="index"` to
`pd.crosstab` gives you these rates directly. They are clearly different. The
question is the same as before: is that more than luck would produce among 300
people?

### What "no relationship" would look like

If the sign-up route and coming back were independent, the multiplication rule
from the top of this page would hold:
$P(\text{web and came back}) = P(\text{web}) \cdot P(\text{came back})$.
Multiply by $n$ to turn that probability into a count, and you get the
**expected count** for each cell.

$$
E = n \cdot \frac{\text{row total}}{n} \cdot \frac{\text{column total}}{n}
  = \frac{\text{row total} \times \text{column total}}{n}
$$

For web and came back that is $120 \times 150 / 300 = 60$. In plain words: half
of everyone came back, so if the route made no difference, half of the 120
website customers would have come back too. We saw 72.

<svg viewBox="0 0 680 342" role="img" aria-labelledby="cs-title cs-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="cs-title">Observed counts compared with expected counts</title>
<desc id="cs-desc">Two tables side by side. On the left, the counts we observed: phone 52 no and 48 yes, store 50 no and 30 yes, web 48 no and 72 yes. On the right, the counts we would expect if sign-up route made no difference: 50 and 50, 40 and 40, 60 and 60. Phone is close to expected. Store and web are far off, and they supply almost all of the chi-square statistic of 9.96.</desc>
<text x="155" y="22" text-anchor="middle" font-size="13.5" font-weight="700" fill="var(--md-default-fg-color)">What we counted</text>
<text x="155" y="40" text-anchor="middle" font-size="11" fill="var(--h-space)">observed, O</text>
<text x="150.0" y="68" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-space)" dy="0.36em">came back: no</text>
<text x="240.0" y="68" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-space)" dy="0.36em">yes</text>
<text x="98" y="104.0" text-anchor="end" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)" dy="0.36em">phone</text>
<rect x="110" y="84" width="80" height="40" rx="7" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="150.0" y="104.0" text-anchor="middle" font-size="15" font-weight="700" fill="var(--md-default-fg-color)" dy="0.36em">52</text>
<rect x="200" y="84" width="80" height="40" rx="7" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="240.0" y="104.0" text-anchor="middle" font-size="15" font-weight="700" fill="var(--md-default-fg-color)" dy="0.36em">48</text>
<text x="98" y="152.0" text-anchor="end" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)" dy="0.36em">store</text>
<rect x="110" y="132" width="80" height="40" rx="7" fill="var(--h-cherry-wash)" stroke="var(--h-cherry)" stroke-width="1.5"/>
<text x="150.0" y="152.0" text-anchor="middle" font-size="15" font-weight="700" fill="var(--h-cherry)" dy="0.36em">50</text>
<rect x="200" y="132" width="80" height="40" rx="7" fill="var(--h-cherry-wash)" stroke="var(--h-cherry)" stroke-width="1.5"/>
<text x="240.0" y="152.0" text-anchor="middle" font-size="15" font-weight="700" fill="var(--h-cherry)" dy="0.36em">30</text>
<text x="98" y="200.0" text-anchor="end" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)" dy="0.36em">web</text>
<rect x="110" y="180" width="80" height="40" rx="7" fill="var(--h-cherry-wash)" stroke="var(--h-cherry)" stroke-width="1.5"/>
<text x="150.0" y="200.0" text-anchor="middle" font-size="15" font-weight="700" fill="var(--h-cherry)" dy="0.36em">48</text>
<rect x="200" y="180" width="80" height="40" rx="7" fill="var(--h-cherry-wash)" stroke="var(--h-cherry)" stroke-width="1.5"/>
<text x="240.0" y="200.0" text-anchor="middle" font-size="15" font-weight="700" fill="var(--h-cherry)" dy="0.36em">72</text>
<text x="515" y="22" text-anchor="middle" font-size="13.5" font-weight="700" fill="var(--md-default-fg-color)">What no relationship looks like</text>
<text x="515" y="40" text-anchor="middle" font-size="11" fill="var(--h-space)">expected, E = row total × column total / n</text>
<text x="510.0" y="68" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-space)" dy="0.36em">came back: no</text>
<text x="600.0" y="68" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-space)" dy="0.36em">yes</text>
<text x="458" y="104.0" text-anchor="end" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)" dy="0.36em">phone</text>
<rect x="470" y="84" width="80" height="40" rx="7" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="510.0" y="104.0" text-anchor="middle" font-size="15" font-weight="700" fill="var(--md-default-fg-color)" dy="0.36em">50</text>
<rect x="560" y="84" width="80" height="40" rx="7" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="600.0" y="104.0" text-anchor="middle" font-size="15" font-weight="700" fill="var(--md-default-fg-color)" dy="0.36em">50</text>
<text x="458" y="152.0" text-anchor="end" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)" dy="0.36em">store</text>
<rect x="470" y="132" width="80" height="40" rx="7" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="510.0" y="152.0" text-anchor="middle" font-size="15" font-weight="700" fill="var(--md-default-fg-color)" dy="0.36em">40</text>
<rect x="560" y="132" width="80" height="40" rx="7" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="600.0" y="152.0" text-anchor="middle" font-size="15" font-weight="700" fill="var(--md-default-fg-color)" dy="0.36em">40</text>
<text x="458" y="200.0" text-anchor="end" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)" dy="0.36em">web</text>
<rect x="470" y="180" width="80" height="40" rx="7" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="510.0" y="200.0" text-anchor="middle" font-size="15" font-weight="700" fill="var(--md-default-fg-color)" dy="0.36em">60</text>
<rect x="560" y="180" width="80" height="40" rx="7" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="600.0" y="200.0" text-anchor="middle" font-size="15" font-weight="700" fill="var(--md-default-fg-color)" dy="0.36em">60</text>
<line x1="300" y1="160" x2="352" y2="160" stroke="var(--h-steel)" stroke-width="1.5"/>
<polyline points="345,154 353,160 345,166" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<text x="326" y="148" text-anchor="middle" font-size="11" fill="var(--h-space)">compare</text>
<text x="340" y="246" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">Each cell adds (O − E)² / E to the total</text>
<rect x="70" y="258" width="160" height="34" rx="8" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="150" y="275" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)" dy="0.36em">phone: 0.08 + 0.08</text>
<rect x="260" y="258" width="160" height="34" rx="8" fill="var(--h-cherry)"/>
<text x="340" y="275" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff" dy="0.36em">store: 2.5 + 2.5</text>
<rect x="450" y="258" width="160" height="34" rx="8" fill="var(--h-cherry)"/>
<text x="530" y="275" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff" dy="0.36em">web: 2.4 + 2.4</text>
<text x="340" y="322" text-anchor="middle" font-size="12.5" fill="var(--h-graphite)">Total χ² = 9.96. Luck alone would typically give about 2 here.</text>
</svg>

### Measuring the gap

The **chi-square statistic** adds up how far each cell is from its expected
count.

$$
\chi^2 = \sum_{\text{all cells}} \frac{(O - E)^2}{E}
$$

The difference is squared so that too many and too few both count. It is
divided by $E$ so that a gap of 12 matters more in a cell expecting 60 than in a
cell expecting 6000. Here the total is 9.96, and almost all of it comes from the
store and web rows.

Is 9.96 big? Here is a handy rule: when nothing is going on, $\chi^2$ averages
about its **degrees of freedom**, which for a table is

$$
\text{df} = (\text{rows} - 1)(\text{columns} - 1) = (3 - 1)(2 - 1) = 2
$$

The name makes sense once you see it: with the row and column totals fixed, you
can choose only 2 cells freely, and the rest are forced. So luck typically gives
a value around 2, and 9.96 is far out in the tail. SciPy does the whole
calculation in one call.

```python
from scipy import stats

table = pd.crosstab(df["signup"], df["came_back"])
chi2, p, dof, expected = stats.chi2_contingency(table)
print(round(chi2, 2), round(p, 4), dof)
print(expected)
```

```
9.96 0.0069 2
[[50. 50.]
 [40. 40.]
 [60. 60.]]
```

Notice that the table passed to the test has no `margins=True`. The totals are
not data, and counting them as cells would give a wrong answer.

A p-value of 0.007 means luck alone would produce a gap this large about 7 times
in 1000. So the sign-up route and coming back look genuinely related.

### What to keep in mind

- **Feed it counts, never percentages.** 60 percent of 10 people and 60 percent
  of 10,000 people are very different amounts of evidence, and only the counts
  carry that difference.
- **Each individual is counted once**, in exactly one cell, and individuals do
  not influence each other.
- **Expected counts should be at least about 5** in every cell. With smaller
  ones the formula stops being reliable. Merge rare categories, or for a 2 by 2
  table use `stats.fisher_exact`.
- **For a 2 by 2 table, SciPy applies the Yates correction by default.** It
  shrinks each $|O - E|$ by 0.5 before squaring, which makes the test a little
  more cautious with small counts. Your hand calculation will not match SciPy
  unless you pass `correction=False`. Both are acceptable; just know which one
  you used.
- **It tells you whether, not how much.** The finding people care about is "60
  percent against 37.5 percent". The p-value only tells you that gap is unlikely
  to be luck.

## Two groups of numbers: Welch's t-test

When the thing you measure is a number, such as a score, a bill or a number of
months, you compare means. Back to the two revision classes and their 7 point
gap.

Whether 7 points is a lot depends on how noisy the scores are. A 7 point gap
between groups whose scores barely vary is striking. A 7 point gap between
groups whose scores swing by 30 points is nothing.

<svg viewBox="0 0 680 300" role="img" aria-labelledby="wt-title wt-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="wt-title">The same gap with little noise and with lots of noise</title>
<desc id="wt-desc">Two panels. In both, two bell curves have their centres the same distance apart. On the left the curves are narrow and barely overlap, so the gap is clear and t is large. On the right the curves are wide and overlap heavily, so the same gap could easily be luck and t is small.</desc>
<text x="180.0" y="24" text-anchor="middle" font-size="13.5" font-weight="700" fill="var(--md-default-fg-color)">Little noise</text>
<polygon points="40.0,200 40.0,200.0 42.0,200.0 44.0,200.0 46.0,200.0 48.0,200.0 50.0,200.0 52.0,200.0 54.0,200.0 56.0,200.0 58.0,200.0 60.0,200.0 62.0,200.0 64.0,200.0 66.0,200.0 68.0,200.0 70.0,200.0 72.0,200.0 74.0,200.0 76.0,200.0 78.0,200.0 80.0,200.0 82.0,200.0 84.0,200.0 86.0,200.0 88.0,200.0 90.0,200.0 92.0,200.0 94.0,200.0 96.0,200.0 98.0,200.0 100.0,200.0 102.0,200.0 104.0,199.9 106.0,199.9 108.0,199.8 110.0,199.6 112.0,199.4 114.0,199.0 116.0,198.3 118.0,197.4 120.0,196.1 122.0,194.1 124.0,191.4 126.0,187.7 128.0,182.8 130.0,176.6 132.0,168.7 134.0,159.2 136.0,147.9 138.0,135.0 140.0,120.7 142.0,105.4 144.0,89.7 146.0,74.2 148.0,59.7 150.0,47.1 152.0,37.0 154.0,30.2 156.0,27.0 158.0,27.6 160.0,32.1 162.0,40.1 164.0,51.1 166.0,64.4 168.0,79.3 170.0,94.9 172.0,110.6 174.0,125.6 176.0,139.4 178.0,151.8 180.0,162.5 182.0,171.5 184.0,178.8 186.0,184.6 188.0,189.0 190.0,192.4 192.0,194.8 194.0,196.6 196.0,197.8 198.0,198.6 200.0,199.1 202.0,199.5 204.0,199.7 206.0,199.8 208.0,199.9 210.0,199.9 212.0,200.0 214.0,200.0 216.0,200.0 218.0,200.0 220.0,200.0 222.0,200.0 224.0,200.0 226.0,200.0 228.0,200.0 230.0,200.0 232.0,200.0 234.0,200.0 236.0,200.0 238.0,200.0 240.0,200.0 242.0,200.0 244.0,200.0 246.0,200.0 248.0,200.0 250.0,200.0 252.0,200.0 254.0,200.0 256.0,200.0 258.0,200.0 260.0,200.0 262.0,200.0 264.0,200.0 266.0,200.0 268.0,200.0 270.0,200.0 272.0,200.0 274.0,200.0 276.0,200.0 278.0,200.0 280.0,200.0 282.0,200.0 284.0,200.0 286.0,200.0 288.0,200.0 290.0,200.0 292.0,200.0 294.0,200.0 296.0,200.0 298.0,200.0 300.0,200.0 302.0,200.0 304.0,200.0 306.0,200.0 308.0,200.0 310.0,200.0 312.0,200.0 314.0,200.0 316.0,200.0 318.0,200.0 320.0,200.0 320.0,200" fill="var(--h-steel)" opacity="0.18"/>
<polyline points="40.0,200.0 42.0,200.0 44.0,200.0 46.0,200.0 48.0,200.0 50.0,200.0 52.0,200.0 54.0,200.0 56.0,200.0 58.0,200.0 60.0,200.0 62.0,200.0 64.0,200.0 66.0,200.0 68.0,200.0 70.0,200.0 72.0,200.0 74.0,200.0 76.0,200.0 78.0,200.0 80.0,200.0 82.0,200.0 84.0,200.0 86.0,200.0 88.0,200.0 90.0,200.0 92.0,200.0 94.0,200.0 96.0,200.0 98.0,200.0 100.0,200.0 102.0,200.0 104.0,199.9 106.0,199.9 108.0,199.8 110.0,199.6 112.0,199.4 114.0,199.0 116.0,198.3 118.0,197.4 120.0,196.1 122.0,194.1 124.0,191.4 126.0,187.7 128.0,182.8 130.0,176.6 132.0,168.7 134.0,159.2 136.0,147.9 138.0,135.0 140.0,120.7 142.0,105.4 144.0,89.7 146.0,74.2 148.0,59.7 150.0,47.1 152.0,37.0 154.0,30.2 156.0,27.0 158.0,27.6 160.0,32.1 162.0,40.1 164.0,51.1 166.0,64.4 168.0,79.3 170.0,94.9 172.0,110.6 174.0,125.6 176.0,139.4 178.0,151.8 180.0,162.5 182.0,171.5 184.0,178.8 186.0,184.6 188.0,189.0 190.0,192.4 192.0,194.8 194.0,196.6 196.0,197.8 198.0,198.6 200.0,199.1 202.0,199.5 204.0,199.7 206.0,199.8 208.0,199.9 210.0,199.9 212.0,200.0 214.0,200.0 216.0,200.0 218.0,200.0 220.0,200.0 222.0,200.0 224.0,200.0 226.0,200.0 228.0,200.0 230.0,200.0 232.0,200.0 234.0,200.0 236.0,200.0 238.0,200.0 240.0,200.0 242.0,200.0 244.0,200.0 246.0,200.0 248.0,200.0 250.0,200.0 252.0,200.0 254.0,200.0 256.0,200.0 258.0,200.0 260.0,200.0 262.0,200.0 264.0,200.0 266.0,200.0 268.0,200.0 270.0,200.0 272.0,200.0 274.0,200.0 276.0,200.0 278.0,200.0 280.0,200.0 282.0,200.0 284.0,200.0 286.0,200.0 288.0,200.0 290.0,200.0 292.0,200.0 294.0,200.0 296.0,200.0 298.0,200.0 300.0,200.0 302.0,200.0 304.0,200.0 306.0,200.0 308.0,200.0 310.0,200.0 312.0,200.0 314.0,200.0 316.0,200.0 318.0,200.0 320.0,200.0" fill="none" stroke="var(--h-steel)" stroke-width="2.5"/>
<line x1="156.7" y1="200" x2="156.7" y2="58" stroke="var(--h-steel)" stroke-width="1.5" stroke-dasharray="4 3"/>
<polygon points="40.0,200 40.0,200.0 42.0,200.0 44.0,200.0 46.0,200.0 48.0,200.0 50.0,200.0 52.0,200.0 54.0,200.0 56.0,200.0 58.0,200.0 60.0,200.0 62.0,200.0 64.0,200.0 66.0,200.0 68.0,200.0 70.0,200.0 72.0,200.0 74.0,200.0 76.0,200.0 78.0,200.0 80.0,200.0 82.0,200.0 84.0,200.0 86.0,200.0 88.0,200.0 90.0,200.0 92.0,200.0 94.0,200.0 96.0,200.0 98.0,200.0 100.0,200.0 102.0,200.0 104.0,200.0 106.0,200.0 108.0,200.0 110.0,200.0 112.0,200.0 114.0,200.0 116.0,200.0 118.0,200.0 120.0,200.0 122.0,200.0 124.0,200.0 126.0,200.0 128.0,200.0 130.0,200.0 132.0,200.0 134.0,200.0 136.0,200.0 138.0,200.0 140.0,200.0 142.0,200.0 144.0,200.0 146.0,200.0 148.0,200.0 150.0,199.9 152.0,199.9 154.0,199.8 156.0,199.7 158.0,199.5 160.0,199.1 162.0,198.6 164.0,197.8 166.0,196.6 168.0,194.8 170.0,192.4 172.0,189.0 174.0,184.6 176.0,178.8 178.0,171.5 180.0,162.5 182.0,151.8 184.0,139.4 186.0,125.6 188.0,110.6 190.0,94.9 192.0,79.3 194.0,64.4 196.0,51.1 198.0,40.1 200.0,32.1 202.0,27.6 204.0,27.0 206.0,30.2 208.0,37.0 210.0,47.1 212.0,59.7 214.0,74.2 216.0,89.7 218.0,105.4 220.0,120.7 222.0,135.0 224.0,147.9 226.0,159.2 228.0,168.7 230.0,176.6 232.0,182.8 234.0,187.7 236.0,191.4 238.0,194.1 240.0,196.1 242.0,197.4 244.0,198.3 246.0,199.0 248.0,199.4 250.0,199.6 252.0,199.8 254.0,199.9 256.0,199.9 258.0,200.0 260.0,200.0 262.0,200.0 264.0,200.0 266.0,200.0 268.0,200.0 270.0,200.0 272.0,200.0 274.0,200.0 276.0,200.0 278.0,200.0 280.0,200.0 282.0,200.0 284.0,200.0 286.0,200.0 288.0,200.0 290.0,200.0 292.0,200.0 294.0,200.0 296.0,200.0 298.0,200.0 300.0,200.0 302.0,200.0 304.0,200.0 306.0,200.0 308.0,200.0 310.0,200.0 312.0,200.0 314.0,200.0 316.0,200.0 318.0,200.0 320.0,200.0 320.0,200" fill="var(--h-cherry)" opacity="0.18"/>
<polyline points="40.0,200.0 42.0,200.0 44.0,200.0 46.0,200.0 48.0,200.0 50.0,200.0 52.0,200.0 54.0,200.0 56.0,200.0 58.0,200.0 60.0,200.0 62.0,200.0 64.0,200.0 66.0,200.0 68.0,200.0 70.0,200.0 72.0,200.0 74.0,200.0 76.0,200.0 78.0,200.0 80.0,200.0 82.0,200.0 84.0,200.0 86.0,200.0 88.0,200.0 90.0,200.0 92.0,200.0 94.0,200.0 96.0,200.0 98.0,200.0 100.0,200.0 102.0,200.0 104.0,200.0 106.0,200.0 108.0,200.0 110.0,200.0 112.0,200.0 114.0,200.0 116.0,200.0 118.0,200.0 120.0,200.0 122.0,200.0 124.0,200.0 126.0,200.0 128.0,200.0 130.0,200.0 132.0,200.0 134.0,200.0 136.0,200.0 138.0,200.0 140.0,200.0 142.0,200.0 144.0,200.0 146.0,200.0 148.0,200.0 150.0,199.9 152.0,199.9 154.0,199.8 156.0,199.7 158.0,199.5 160.0,199.1 162.0,198.6 164.0,197.8 166.0,196.6 168.0,194.8 170.0,192.4 172.0,189.0 174.0,184.6 176.0,178.8 178.0,171.5 180.0,162.5 182.0,151.8 184.0,139.4 186.0,125.6 188.0,110.6 190.0,94.9 192.0,79.3 194.0,64.4 196.0,51.1 198.0,40.1 200.0,32.1 202.0,27.6 204.0,27.0 206.0,30.2 208.0,37.0 210.0,47.1 212.0,59.7 214.0,74.2 216.0,89.7 218.0,105.4 220.0,120.7 222.0,135.0 224.0,147.9 226.0,159.2 228.0,168.7 230.0,176.6 232.0,182.8 234.0,187.7 236.0,191.4 238.0,194.1 240.0,196.1 242.0,197.4 244.0,198.3 246.0,199.0 248.0,199.4 250.0,199.6 252.0,199.8 254.0,199.9 256.0,199.9 258.0,200.0 260.0,200.0 262.0,200.0 264.0,200.0 266.0,200.0 268.0,200.0 270.0,200.0 272.0,200.0 274.0,200.0 276.0,200.0 278.0,200.0 280.0,200.0 282.0,200.0 284.0,200.0 286.0,200.0 288.0,200.0 290.0,200.0 292.0,200.0 294.0,200.0 296.0,200.0 298.0,200.0 300.0,200.0 302.0,200.0 304.0,200.0 306.0,200.0 308.0,200.0 310.0,200.0 312.0,200.0 314.0,200.0 316.0,200.0 318.0,200.0 320.0,200.0" fill="none" stroke="var(--h-cherry)" stroke-width="2.5"/>
<line x1="203.3" y1="200" x2="203.3" y2="58" stroke="var(--h-cherry)" stroke-width="1.5" stroke-dasharray="4 3"/>
<line x1="156.7" y1="52" x2="203.3" y2="52" stroke="var(--md-default-fg-color)" stroke-width="1.5"/>
<text x="180.0" y="42" text-anchor="middle" font-size="11" font-weight="700" fill="var(--md-default-fg-color)" dy="0.36em">gap</text>
<line x1="40" y1="200" x2="320" y2="200" stroke="var(--h-steel)" stroke-width="1.5"/>
<text x="180.0" y="226" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">Same gap, groups barely overlap</text>
<text x="180.0" y="244" text-anchor="middle" font-size="12" fill="var(--h-graphite)">t is large, small p-value</text>
<text x="500.0" y="24" text-anchor="middle" font-size="13.5" font-weight="700" fill="var(--md-default-fg-color)">Lots of noise</text>
<polygon points="360.0,200 360.0,198.6 362.0,198.4 364.0,198.2 366.0,198.0 368.0,197.7 370.0,197.4 372.0,197.1 374.0,196.8 376.0,196.4 378.0,196.0 380.0,195.6 382.0,195.1 384.0,194.6 386.0,194.0 388.0,193.4 390.0,192.8 392.0,192.1 394.0,191.4 396.0,190.6 398.0,189.7 400.0,188.9 402.0,187.9 404.0,186.9 406.0,185.9 408.0,184.8 410.0,183.7 412.0,182.5 414.0,181.3 416.0,180.0 418.0,178.7 420.0,177.3 422.0,175.9 424.0,174.5 426.0,173.1 428.0,171.6 430.0,170.1 432.0,168.7 434.0,167.2 436.0,165.7 438.0,164.2 440.0,162.7 442.0,161.3 444.0,159.9 446.0,158.5 448.0,157.2 450.0,155.9 452.0,154.7 454.0,153.5 456.0,152.4 458.0,151.4 460.0,150.5 462.0,149.7 464.0,148.9 466.0,148.3 468.0,147.7 470.0,147.3 472.0,147.0 474.0,146.8 476.0,146.7 478.0,146.7 480.0,146.8 482.0,147.1 484.0,147.5 486.0,147.9 488.0,148.5 490.0,149.2 492.0,149.9 494.0,150.8 496.0,151.7 498.0,152.8 500.0,153.9 502.0,155.1 504.0,156.3 506.0,157.6 508.0,159.0 510.0,160.3 512.0,161.8 514.0,163.2 516.0,164.7 518.0,166.2 520.0,167.7 522.0,169.2 524.0,170.6 526.0,172.1 528.0,173.6 530.0,175.0 532.0,176.4 534.0,177.8 536.0,179.1 538.0,180.4 540.0,181.7 542.0,182.9 544.0,184.1 546.0,185.2 548.0,186.2 550.0,187.3 552.0,188.2 554.0,189.2 556.0,190.0 558.0,190.8 560.0,191.6 562.0,192.3 564.0,193.0 566.0,193.6 568.0,194.2 570.0,194.8 572.0,195.3 574.0,195.7 576.0,196.1 578.0,196.5 580.0,196.9 582.0,197.2 584.0,197.5 586.0,197.8 588.0,198.0 590.0,198.3 592.0,198.5 594.0,198.6 596.0,198.8 598.0,198.9 600.0,199.1 602.0,199.2 604.0,199.3 606.0,199.4 608.0,199.5 610.0,199.5 612.0,199.6 614.0,199.6 616.0,199.7 618.0,199.7 620.0,199.8 622.0,199.8 624.0,199.8 626.0,199.9 628.0,199.9 630.0,199.9 632.0,199.9 634.0,199.9 636.0,199.9 638.0,199.9 640.0,200.0 640.0,200" fill="var(--h-steel)" opacity="0.18"/>
<polyline points="360.0,198.6 362.0,198.4 364.0,198.2 366.0,198.0 368.0,197.7 370.0,197.4 372.0,197.1 374.0,196.8 376.0,196.4 378.0,196.0 380.0,195.6 382.0,195.1 384.0,194.6 386.0,194.0 388.0,193.4 390.0,192.8 392.0,192.1 394.0,191.4 396.0,190.6 398.0,189.7 400.0,188.9 402.0,187.9 404.0,186.9 406.0,185.9 408.0,184.8 410.0,183.7 412.0,182.5 414.0,181.3 416.0,180.0 418.0,178.7 420.0,177.3 422.0,175.9 424.0,174.5 426.0,173.1 428.0,171.6 430.0,170.1 432.0,168.7 434.0,167.2 436.0,165.7 438.0,164.2 440.0,162.7 442.0,161.3 444.0,159.9 446.0,158.5 448.0,157.2 450.0,155.9 452.0,154.7 454.0,153.5 456.0,152.4 458.0,151.4 460.0,150.5 462.0,149.7 464.0,148.9 466.0,148.3 468.0,147.7 470.0,147.3 472.0,147.0 474.0,146.8 476.0,146.7 478.0,146.7 480.0,146.8 482.0,147.1 484.0,147.5 486.0,147.9 488.0,148.5 490.0,149.2 492.0,149.9 494.0,150.8 496.0,151.7 498.0,152.8 500.0,153.9 502.0,155.1 504.0,156.3 506.0,157.6 508.0,159.0 510.0,160.3 512.0,161.8 514.0,163.2 516.0,164.7 518.0,166.2 520.0,167.7 522.0,169.2 524.0,170.6 526.0,172.1 528.0,173.6 530.0,175.0 532.0,176.4 534.0,177.8 536.0,179.1 538.0,180.4 540.0,181.7 542.0,182.9 544.0,184.1 546.0,185.2 548.0,186.2 550.0,187.3 552.0,188.2 554.0,189.2 556.0,190.0 558.0,190.8 560.0,191.6 562.0,192.3 564.0,193.0 566.0,193.6 568.0,194.2 570.0,194.8 572.0,195.3 574.0,195.7 576.0,196.1 578.0,196.5 580.0,196.9 582.0,197.2 584.0,197.5 586.0,197.8 588.0,198.0 590.0,198.3 592.0,198.5 594.0,198.6 596.0,198.8 598.0,198.9 600.0,199.1 602.0,199.2 604.0,199.3 606.0,199.4 608.0,199.5 610.0,199.5 612.0,199.6 614.0,199.6 616.0,199.7 618.0,199.7 620.0,199.8 622.0,199.8 624.0,199.8 626.0,199.9 628.0,199.9 630.0,199.9 632.0,199.9 634.0,199.9 636.0,199.9 638.0,199.9 640.0,200.0" fill="none" stroke="var(--h-steel)" stroke-width="2.5"/>
<line x1="476.7" y1="200" x2="476.7" y2="58" stroke="var(--h-steel)" stroke-width="1.5" stroke-dasharray="4 3"/>
<polygon points="360.0,200 360.0,200.0 362.0,199.9 364.0,199.9 366.0,199.9 368.0,199.9 370.0,199.9 372.0,199.9 374.0,199.9 376.0,199.8 378.0,199.8 380.0,199.8 382.0,199.7 384.0,199.7 386.0,199.6 388.0,199.6 390.0,199.5 392.0,199.5 394.0,199.4 396.0,199.3 398.0,199.2 400.0,199.1 402.0,198.9 404.0,198.8 406.0,198.6 408.0,198.5 410.0,198.3 412.0,198.0 414.0,197.8 416.0,197.5 418.0,197.2 420.0,196.9 422.0,196.5 424.0,196.1 426.0,195.7 428.0,195.3 430.0,194.8 432.0,194.2 434.0,193.6 436.0,193.0 438.0,192.3 440.0,191.6 442.0,190.8 444.0,190.0 446.0,189.2 448.0,188.2 450.0,187.3 452.0,186.2 454.0,185.2 456.0,184.1 458.0,182.9 460.0,181.7 462.0,180.4 464.0,179.1 466.0,177.8 468.0,176.4 470.0,175.0 472.0,173.6 474.0,172.1 476.0,170.6 478.0,169.2 480.0,167.7 482.0,166.2 484.0,164.7 486.0,163.2 488.0,161.8 490.0,160.3 492.0,159.0 494.0,157.6 496.0,156.3 498.0,155.1 500.0,153.9 502.0,152.8 504.0,151.7 506.0,150.8 508.0,149.9 510.0,149.2 512.0,148.5 514.0,147.9 516.0,147.5 518.0,147.1 520.0,146.8 522.0,146.7 524.0,146.7 526.0,146.8 528.0,147.0 530.0,147.3 532.0,147.7 534.0,148.3 536.0,148.9 538.0,149.7 540.0,150.5 542.0,151.4 544.0,152.4 546.0,153.5 548.0,154.7 550.0,155.9 552.0,157.2 554.0,158.5 556.0,159.9 558.0,161.3 560.0,162.7 562.0,164.2 564.0,165.7 566.0,167.2 568.0,168.7 570.0,170.1 572.0,171.6 574.0,173.1 576.0,174.5 578.0,175.9 580.0,177.3 582.0,178.7 584.0,180.0 586.0,181.3 588.0,182.5 590.0,183.7 592.0,184.8 594.0,185.9 596.0,186.9 598.0,187.9 600.0,188.9 602.0,189.7 604.0,190.6 606.0,191.4 608.0,192.1 610.0,192.8 612.0,193.4 614.0,194.0 616.0,194.6 618.0,195.1 620.0,195.6 622.0,196.0 624.0,196.4 626.0,196.8 628.0,197.1 630.0,197.4 632.0,197.7 634.0,198.0 636.0,198.2 638.0,198.4 640.0,198.6 640.0,200" fill="var(--h-cherry)" opacity="0.18"/>
<polyline points="360.0,200.0 362.0,199.9 364.0,199.9 366.0,199.9 368.0,199.9 370.0,199.9 372.0,199.9 374.0,199.9 376.0,199.8 378.0,199.8 380.0,199.8 382.0,199.7 384.0,199.7 386.0,199.6 388.0,199.6 390.0,199.5 392.0,199.5 394.0,199.4 396.0,199.3 398.0,199.2 400.0,199.1 402.0,198.9 404.0,198.8 406.0,198.6 408.0,198.5 410.0,198.3 412.0,198.0 414.0,197.8 416.0,197.5 418.0,197.2 420.0,196.9 422.0,196.5 424.0,196.1 426.0,195.7 428.0,195.3 430.0,194.8 432.0,194.2 434.0,193.6 436.0,193.0 438.0,192.3 440.0,191.6 442.0,190.8 444.0,190.0 446.0,189.2 448.0,188.2 450.0,187.3 452.0,186.2 454.0,185.2 456.0,184.1 458.0,182.9 460.0,181.7 462.0,180.4 464.0,179.1 466.0,177.8 468.0,176.4 470.0,175.0 472.0,173.6 474.0,172.1 476.0,170.6 478.0,169.2 480.0,167.7 482.0,166.2 484.0,164.7 486.0,163.2 488.0,161.8 490.0,160.3 492.0,159.0 494.0,157.6 496.0,156.3 498.0,155.1 500.0,153.9 502.0,152.8 504.0,151.7 506.0,150.8 508.0,149.9 510.0,149.2 512.0,148.5 514.0,147.9 516.0,147.5 518.0,147.1 520.0,146.8 522.0,146.7 524.0,146.7 526.0,146.8 528.0,147.0 530.0,147.3 532.0,147.7 534.0,148.3 536.0,148.9 538.0,149.7 540.0,150.5 542.0,151.4 544.0,152.4 546.0,153.5 548.0,154.7 550.0,155.9 552.0,157.2 554.0,158.5 556.0,159.9 558.0,161.3 560.0,162.7 562.0,164.2 564.0,165.7 566.0,167.2 568.0,168.7 570.0,170.1 572.0,171.6 574.0,173.1 576.0,174.5 578.0,175.9 580.0,177.3 582.0,178.7 584.0,180.0 586.0,181.3 588.0,182.5 590.0,183.7 592.0,184.8 594.0,185.9 596.0,186.9 598.0,187.9 600.0,188.9 602.0,189.7 604.0,190.6 606.0,191.4 608.0,192.1 610.0,192.8 612.0,193.4 614.0,194.0 616.0,194.6 618.0,195.1 620.0,195.6 622.0,196.0 624.0,196.4 626.0,196.8 628.0,197.1 630.0,197.4 632.0,197.7 634.0,198.0 636.0,198.2 638.0,198.4 640.0,198.6" fill="none" stroke="var(--h-cherry)" stroke-width="2.5"/>
<line x1="523.3" y1="200" x2="523.3" y2="58" stroke="var(--h-cherry)" stroke-width="1.5" stroke-dasharray="4 3"/>
<line x1="476.7" y1="52" x2="523.3" y2="52" stroke="var(--md-default-fg-color)" stroke-width="1.5"/>
<text x="500.0" y="42" text-anchor="middle" font-size="11" font-weight="700" fill="var(--md-default-fg-color)" dy="0.36em">gap</text>
<line x1="360" y1="200" x2="640" y2="200" stroke="var(--h-steel)" stroke-width="1.5"/>
<text x="500.0" y="226" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">Same gap, groups overlap a lot</text>
<text x="500.0" y="244" text-anchor="middle" font-size="12" fill="var(--h-graphite)">t is small, large p-value</text>
<rect x="90" y="262" width="500" height="30" rx="6" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="340" y="277" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)" dy="0.36em">t = gap ÷ noise. More data shrinks the noise.</text>
</svg>

The **t statistic** puts signal and noise into one number.

$$
t = \frac{\bar{x}_1 - \bar{x}_2}{\text{SE}},
\qquad
\text{SE} = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
$$

The top is the signal, the gap between the two means. The bottom is the noise,
called the **standard error** of the gap: how much that gap would wobble if you
ran the whole study again with new students. Each group contributes its own
variance $s^2$ divided by its own size $n$. This is the same square root of $n$
you met in the rough rule for accuracy. More data means less wobble, so the same
gap produces a bigger $t$.

```python
gap = new_way.mean() - usual.mean()
se = np.sqrt(new_way.var(ddof=1) / 12 + usual.var(ddof=1) / 12)
print(round(se, 2), round(gap / se, 2))
```

```
2.97 2.36
```

The gap is about 2.4 times the size of its own wobble. When nothing is going on,
$t$ mostly lands between about $-2$ and $2$, for the same reason averages form a
bell curve. So 2.36 sits just outside the usual range. SciPy turns that into a
p-value.

```python
result = stats.ttest_ind(new_way, usual, equal_var=False)
print(round(result.statistic, 2), round(result.pvalue, 3))
```

```
2.36 0.028
```

Shuffling gave 0.032. The formula gives 0.028. Two completely different routes,
nearly the same answer. That agreement is the whole point: the formula is a fast
shortcut for the shuffling idea.

### Why Welch

The original version, Student's t-test, assumes both groups have the same
spread. Welch's version does not. Each group keeps its own variance in the
standard error, exactly as in the formula above. Real groups rarely have
identical spreads, and when they do, Welch gives practically the same answer
anyway. So pass `equal_var=False` every time. SciPy's default is still the old
version, so remembering it is your job.

Welch also adjusts the degrees of freedom with a slightly messy formula that
SciPy handles for you (`result.df` is about 21.2 here). You do not need to
compute it by hand.

### What to keep in mind

- **It compares means, so outliers push it around.** One customer with a bill
  of 10,000 can create a difference or hide one. Look at a box plot of each group
  before you trust the number.
- **It leans on the central limit theorem.** With about 30 or more in each
  group, the means behave like a bell curve even when the data is skewed, and the
  p-value is trustworthy. With small, skewed groups, also compare the medians, or
  run the permutation test, which makes no assumption about shape.
- **Observations must be independent.** If the same people are measured twice,
  before and after, the groups are not separate. Use the paired version,
  `stats.ttest_rel`.
- **It tells you whether, not how much.** Always report the gap in means next to
  the p-value.

## Reading a p-value without fooling yourself

Running a test takes one line of code. Reading the result well is the real
skill, and five mistakes catch almost everyone.

**1. The p-value is not the probability that nothing is going on.** A p-value of
0.03 says: if nothing were going on, data like this would turn up 3 percent of
the time. That is $P(\text{data} \mid \text{null})$. People hear
$P(\text{null} \mid \text{data})$. You have seen this swap before, in the medical
test, where $P(\text{positive} \mid \text{healthy})$ was 5 percent and
$P(\text{healthy} \mid \text{positive})$ was 85 percent. It is the same trap in a
new place.

**2. Significant does not mean important.** With enough data, even tiny
differences become significant. So always report the **effect size**, the
difference itself in units people understand: percentage points, euros, exam
points. Then ask whether a difference of that size matters to anyone.

**3. Not significant does not mean no effect.** A large p-value means you could
not tell, which is not the same as there being nothing to tell. Small samples
miss real effects all the time. "No evidence of a difference" is not "evidence
of no difference".

The size of the difference and the p-value answer two separate questions, and
you need both.

<svg viewBox="0 0 680 340" role="img" aria-labelledby="pg-title pg-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="pg-title">Effect size and p-value answer different questions</title>
<desc id="pg-desc">A two by two grid. Rows are the size of the difference, large or small. Columns are the p-value, small or large. Large difference with small p-value: real and worth acting on. Small difference with small p-value: real but possibly too small to matter, which happens with huge samples. Large difference with large p-value: could matter but there is too little data to tell. Small difference with large p-value: probably nothing there.</desc>
<text x="275.0" y="30" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">small p-value</text>
<text x="537.0" y="30" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">large p-value</text>
<text x="275.0" y="44" text-anchor="middle" font-size="11" fill="var(--h-space)">luck rarely does this</text>
<text x="537.0" y="44" text-anchor="middle" font-size="11" fill="var(--h-space)">luck often does this</text>
<text x="70" y="118.0" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">large</text>
<text x="70" y="135.0" text-anchor="middle" font-size="11" fill="var(--h-space)">difference</text>
<text x="70" y="262.0" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">small</text>
<text x="70" y="279.0" text-anchor="middle" font-size="11" fill="var(--h-space)">difference</text>
<rect x="150" y="56" width="250" height="132" rx="10" fill="var(--h-cherry)"/>
<text x="275.0" y="110" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff" dy="0.36em">Real and worth acting on</text>
<text x="275.0" y="140" text-anchor="middle" font-size="11.5" fill="#ffffff" dy="0.36em">Return rate 60% vs 38%,</text>
<text x="275.0" y="158" text-anchor="middle" font-size="11.5" fill="#ffffff" dy="0.36em">300 people, p = 0.007</text>
<rect x="412" y="56" width="250" height="132" rx="10" fill="var(--h-cherry-wash)" stroke="var(--h-cherry)" stroke-width="2"/>
<text x="537.0" y="84" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-cherry)" dy="0.36em">WATCH OUT</text>
<text x="537.0" y="110" text-anchor="middle" font-size="13" font-weight="700" fill="var(--h-cherry)" dy="0.36em">Could matter, too little data</text>
<text x="537.0" y="140" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)" dy="0.36em">Scores 82 vs 70, only 4</text>
<text x="537.0" y="158" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)" dy="0.36em">students per group, p ≈ 0.14</text>
<rect x="150" y="200" width="250" height="132" rx="10" fill="var(--h-cherry-wash)" stroke="var(--h-cherry)" stroke-width="2"/>
<text x="275.0" y="228" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-cherry)" dy="0.36em">WATCH OUT</text>
<text x="275.0" y="254" text-anchor="middle" font-size="13" font-weight="700" fill="var(--h-cherry)" dy="0.36em">Real, but maybe too small to care</text>
<text x="275.0" y="284" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)" dy="0.36em">Return rate 50.4% vs 50.0%,</text>
<text x="275.0" y="302" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)" dy="0.36em">a million per group, p &lt; 0.001</text>
<rect x="412" y="200" width="250" height="132" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="537.0" y="254" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)" dy="0.36em">Probably nothing there</text>
<text x="537.0" y="284" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)" dy="0.36em">Scores 75.3 vs 75.0,</text>
<text x="537.0" y="302" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)" dy="0.36em">500 per group, p ≈ 0.6</text>
</svg>

**4. Run enough tests and something will come out significant.** At 0.05, 1 test
in 20 is significant by luck alone. Here are 40 comparisons where both groups
come from exactly the same distribution, so nothing is going on in any of them.

```python
rng = np.random.default_rng(4)
false_alarms = 0
for _ in range(40):
    a, b = rng.normal(0, 1, 50), rng.normal(0, 1, 50)
    false_alarms += stats.ttest_ind(a, b, equal_var=False).pvalue < 0.05
print(false_alarms)
```

```
2
```

Two "discoveries", both pure luck. The simplest protection is the **Bonferroni
correction**: divide 0.05 by the number of tests you ran. With 40 tests, only
trust results with $p < 0.05 / 40 = 0.00125$. It is strict, but it is honest.

**5. One column at a time misses combinations.** A test looks at one column
against the target, alone. A column that seems useless on its own can matter in
combination with another. Two columns that both look strong may be telling you
the same thing twice. A model sees all the columns together; a test sees one.

!!! ml "ML connection"

    In a project you will often run a chi-square test on every categorical
    column and a t-test on every numeric column against the target, and then
    think about dropping the columns with large p-values. Treat the results as a
    screen, not a verdict. A small p-value says the column is related to the
    target, so a model may find it useful. A large p-value says you could not see
    a relationship in that column on its own, not that it is useless. Before you
    drop anything, look at the effect size, remember how many tests you ran, and
    remember that the model will see the columns together. When in doubt, keep
    the column and let the model's test score decide.

## Likelihood, and where loss functions come from

Here is the idea that connects this page to every model you will train.

A model that outputs probabilities can be scored by asking one question: what
probability did it assign to the answer that actually happened? High is good.
That number is the **likelihood**.

Multiply those probabilities across thousands of examples and you get a number
so small that a computer cannot represent it. So we take logarithms, which turn
products into sums, and then flip the sign so that smaller is better.

$$
\text{loss} = -\frac{1}{n} \sum_i \log p_i
$$

where $p_i$ is the probability the model gave to the correct answer for example
$i$. That is cross-entropy, and it is the loss function behind almost every
classifier you will build.

<svg viewBox="0 0 680 262" role="img" aria-labelledby="nl-title nl-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="nl-title">The cost of being confidently wrong</title>
<desc id="nl-desc">Negative log of the probability the model gave to the correct answer. Near probability one the cost is almost zero. As the probability approaches zero the cost rises without limit.</desc>
<line x1="70" y1="196" x2="640" y2="196" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="70" y1="196" x2="70" y2="36" stroke="var(--h-steel)" stroke-width="1.5"/>
<text x="355" y="224" text-anchor="middle" font-size="12" fill="var(--h-graphite)">probability the model gave to the right answer</text>
<text x="78" y="46" font-size="12" fill="var(--h-graphite)">cost</text>
<text x="70" y="212" text-anchor="middle" font-size="10" fill="var(--h-space)">0</text>
<text x="640" y="212" text-anchor="middle" font-size="10" fill="var(--h-space)">1</text>
<polyline points="78.5,53.2 81.4,63.0 84.2,70.6 87.1,76.8 90.0,82.0 92.8,86.6 95.7,90.6 98.5,94.1 101.3,97.4 104.2,100.3 107.1,103.1 109.9,105.6 112.8,107.9 115.6,110.1 118.5,112.2 121.3,114.1 124.2,116.0 127.0,117.7 129.8,119.4 132.7,121.0 135.6,122.5 138.4,123.9 141.2,125.3 144.1,126.6 146.9,127.9 149.8,129.2 152.6,130.3 155.5,131.5 158.3,132.6 161.2,133.7 164.1,134.7 166.9,135.8 169.8,136.7 172.6,137.7 175.4,138.6 178.3,139.5 181.2,140.4 184.0,141.3 186.8,142.1 189.7,142.9 192.6,143.7 195.4,144.5 198.2,145.3 201.1,146.0 203.9,146.8 206.8,147.5 209.7,148.2 212.5,148.9 215.3,149.5 218.2,150.2 221.1,150.8 223.9,151.5 226.8,152.1 229.6,152.7 232.4,153.3 235.3,153.9 238.1,154.5 241.0,155.1 243.8,155.6 246.7,156.2 249.6,156.7 252.4,157.3 255.2,157.8 258.1,158.3 261.0,158.8 263.8,159.3 266.6,159.8 269.5,160.3 272.4,160.8 275.2,161.3 278.0,161.7 280.9,162.2 283.8,162.7 286.6,163.1 289.5,163.5 292.3,164.0 295.1,164.4 298.0,164.8 300.9,165.3 303.7,165.7 306.5,166.1 309.4,166.5 312.2,166.9 315.1,167.3 317.9,167.7 320.8,168.1 323.6,168.5 326.5,168.9 329.4,169.2 332.2,169.6 335.1,170.0 337.9,170.3 340.8,170.7 343.6,171.0 346.4,171.4 349.3,171.7 352.1,172.1 355.0,172.4 357.9,172.8 360.7,173.1 363.6,173.4 366.4,173.8 369.2,174.1 372.1,174.4 375.0,174.7 377.8,175.0 380.7,175.4 383.5,175.7 386.4,176.0 389.2,176.3 392.0,176.6 394.9,176.9 397.8,177.2 400.6,177.5 403.4,177.8 406.3,178.1 409.1,178.3 412.0,178.6 414.8,178.9 417.7,179.2 420.6,179.5 423.4,179.7 426.2,180.0 429.1,180.3 431.9,180.6 434.8,180.8 437.7,181.1 440.5,181.4 443.4,181.6 446.2,181.9 449.1,182.1 451.9,182.4 454.8,182.6 457.6,182.9 460.5,183.1 463.3,183.4 466.1,183.6 469.0,183.9 471.8,184.1 474.7,184.4 477.5,184.6 480.4,184.8 483.2,185.1 486.1,185.3 488.9,185.5 491.8,185.8 494.6,186.0 497.5,186.2 500.4,186.4 503.2,186.7 506.1,186.9 508.9,187.1 511.8,187.3 514.6,187.6 517.5,187.8 520.3,188.0 523.2,188.2 526.0,188.4 528.9,188.6 531.7,188.8 534.5,189.0 537.4,189.3 540.2,189.5 543.1,189.7 546.0,189.9 548.8,190.1 551.6,190.3 554.5,190.5 557.3,190.7 560.2,190.9 563.0,191.1 565.9,191.3 568.8,191.5 571.6,191.7 574.5,191.8 577.3,192.0 580.2,192.2 583.0,192.4 585.9,192.6 588.7,192.8 591.6,193.0 594.4,193.2 597.2,193.3 600.1,193.5 603.0,193.7 605.8,193.9 608.6,194.1 611.5,194.3 614.4,194.4 617.2,194.6 620.0,194.8 622.9,195.0 625.8,195.1 628.6,195.3 631.5,195.5 634.3,195.7 637.1,195.8 640.0,196.0" fill="none" stroke="var(--h-cherry)" stroke-width="3"/>
<circle cx="583.0" cy="192.4" r="5" fill="var(--h-cherry)"/>
<circle cx="355.0" cy="172.4" r="5" fill="var(--h-cherry)"/>
<circle cx="98.5" cy="94.1" r="5" fill="var(--h-cherry)"/>
<text x="573" y="178" text-anchor="end" font-size="11.5" font-weight="700" fill="var(--h-graphite)">confident and right</text>
<text x="355" y="156" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-graphite)">a coin flip</text>
<text x="110" y="98" text-anchor="start" font-size="11.5" font-weight="700" fill="var(--h-graphite)">confident and wrong</text>
<text x="340" y="250" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Being 90 percent sure and right costs almost nothing. Being 95 percent sure and wrong costs 30 times as much.</text>
</svg>

The shape of the negative logarithm is the whole story. Give the right answer 90
percent and the cost is 0.105. Give it 5 percent and the cost is 3.0, thirty
times more. As the probability approaches zero the cost grows without limit.

So the function does not merely punish being wrong. It punishes being confident
and wrong, and it rewards honest uncertainty. A model that says 60 percent and is
occasionally wrong scores better than one that says 99 percent and is
occasionally wrong.

```python
p_correct = np.array([0.9, 0.8, 0.6, 0.05])
print(-np.log(p_correct))
print(-np.log(p_correct).mean())
```

```
[0.105 0.223 0.511 2.996]
0.959
```

One bad prediction dominates the average. That single line explains most of what
you will see when you start watching training curves.

## Correlation, and its limits

Covariance measures whether two variables move together. Correlation is
covariance rescaled to sit between -1 and 1, which makes it comparable across
different units.

```python
x = rng.normal(0, 1, 1000)
y = 2 * x + rng.normal(0, 1, 1000)
print(np.corrcoef(x, y)[0, 1])
```

```
0.89...
```

Two warnings, both of which will cost somebody marks this trimester.

**Correlation only measures straight line relationships.** A perfect U shape can
have a correlation near zero. This is why you plot before you trust the number,
which is the same lesson as Anscombe's quartet from the Plotting page.

**Correlation is not causation.** Ice cream sales correlate with drownings. Both
are caused by summer. A model does not care, because a correlated feature
predicts perfectly well until the underlying cause shifts and the relationship
quietly stops holding.

## Common confusions

??? note "Disjoint and independent"

    Disjoint means they cannot both happen. Independent means one tells you
    nothing about the other. Disjoint events with non-zero probability are
    always dependent.

??? note "$P(A \mid B)$ and $P(B \mid A)$"

    These are not the same number and are often wildly different. The
    probability of testing positive given illness was 95 percent. The
    probability of illness given a positive test was 15. Swapping these two is
    the single most common probability error in the wild.

??? note "The PDF height looks like a probability"

    It is not, and it can exceed 1. Probability is the area under the curve
    between two values.

??? note "A big correlation means one causes the other"

    It means they move together. It says nothing at all about why.

??? note "My model improved from 91 to 92 percent"

    On how many test items? Work out the uncertainty first. On a few hundred
    rows that difference is indistinguishable from noise.

??? note "Dividing by n or by n minus 1"

    Use $n$ when your data is the whole population, and $n-1$ when it is a
    sample you are using to estimate a wider population. NumPy defaults to $n$,
    and pandas defaults to $n-1$, which is why the two sometimes disagree by a
    little. For large datasets the difference is negligible.

??? note "The p-value is the chance my result is a fluke"

    Close, but backwards. It is the chance of data at least this extreme
    assuming it is a fluke. Whether it actually is a fluke also depends on how
    plausible a real effect was to begin with, just as in the medical test.

??? note "The p-value is tiny, so the difference must be big"

    The p-value mixes the size of the difference with the amount of data. With a
    million rows, a difference nobody would notice can still give $p < 0.001$.
    Look at the difference itself to judge whether it matters.

??? note "The p-value is 0.4, so the groups are the same"

    It means the data could not tell them apart. With small groups, real
    differences often give large p-values. Absence of evidence is not evidence
    of absence.

## Check yourself

Use NumPy rather than algebra wherever you can. Simulating is a legitimate way
to answer these.

1. Simulate 100,000 rolls of two dice. Estimate the probability that the sum is
   7, and compare it with the exact answer.
2. Change the disease prevalence in the medical test simulation from 1 percent
   to 30 percent. Recompute the chance of being ill given a positive test, and
   explain in one sentence why it moved so much.
3. Draw 10,000 values from a normal distribution of your choice and verify the
   68, 95, 99.7 rule yourself.
4. Take an exponential population, then plot the distribution of the mean of 5
   samples and of 50 samples. Describe what changes.
5. A model is 85 percent accurate. How large must your test set be before a 2
   percentage point improvement is distinguishable from noise? Answer by
   simulation.
6. Compute the cross-entropy loss for a model that gives the right answer 0.99,
   0.99, 0.99 and 0.01, and for one that gives 0.7 four times. Which is the
   better model, and does the number agree with your instinct?
7. Run the permutation test on `new_way` and `usual` with 1,000, 10,000 and
   100,000 shuffles and compare each p-value with
   `stats.ttest_ind(new_way, usual, equal_var=False)`. Then change one score in
   `usual` from 80 to 40 and run both again. Which method moves more, and why?
8. Put the sign-up table `[[52, 48], [50, 30], [48, 72]]` in a NumPy array.
   Compute the expected counts yourself from the row totals, column totals and
   $n$ (hint: `np.outer`), then the chi-square statistic. Check both against
   `stats.chi2_contingency`. Repeat with a 2 by 2 table of your own and explain
   why your answer and SciPy's no longer agree.

## Quick reference

| Idea | Formula or code |
|---|---|
| Complement | $P(A^c) = 1 - P(A)$ |
| Addition | $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ |
| Multiplication | $P(A \cap B) = P(A) P(B \mid A)$ |
| Independence | $P(A \cap B) = P(A) P(B)$ |
| Conditional | $P(A \mid B) = P(A \cap B) / P(B)$ |
| Bayes | $P(A \mid B) = P(B \mid A) P(A) / P(B)$ |
| Mean, variance, sd | `a.mean()`, `a.var()`, `a.std()` |
| Percentiles | `np.percentile(a, [25, 50, 75])` |
| Correlation | `np.corrcoef(x, y)[0, 1]` |
| Seeded generator | `rng = np.random.default_rng(42)` |
| Normal sample | `rng.normal(loc, scale, size)` |
| Binomial sample | `rng.binomial(n, p, size)` |
| Uniform sample | `rng.random(size)` |
| Cross-entropy | `-np.log(p_correct).mean()` |
| Accuracy uncertainty | $\sqrt{p(1-p)/n}$ |
| Contingency table | `pd.crosstab(df["a"], df["b"])` |
| Rates per row | `pd.crosstab(df["a"], df["b"], normalize="index")` |
| Expected count | row total $\times$ column total $/ n$ |
| Chi-square statistic | $\sum (O - E)^2 / E$, df $= (r-1)(c-1)$ |
| Chi-square test | `chi2, p, dof, expected = stats.chi2_contingency(table)` |
| Standard error of a gap | $\sqrt{s_1^2/n_1 + s_2^2/n_2}$ |
| Welch's t-test | `stats.ttest_ind(a, b, equal_var=False)` |
| Permutation p-value | `(np.abs(diffs) >= abs(observed)).mean()` |
| Bonferroni | trust only $p < 0.05 / \text{number of tests}$ |

## Questions to sit with

1. The test in the Bayes example was 95 percent accurate and a positive result
   still meant an 85 percent chance of being healthy. If you were designing a
   national screening programme, what would you do with that fact?
2. Cross-entropy punishes confident wrong answers far more than uncertain ones.
   What behaviour does that encourage in a model, and can you think of a
   situation where you would not want it?
3. Minimising mean squared error means treating one error of size 10 as worse
   than ten errors of size 1. Name a problem where that is right, and one where
   it is badly wrong.
4. To halve your uncertainty you need four times the data. What does that imply
   about projects that promise big gains from collecting a bit more?
5. A feature correlates strongly with your target and you have no idea why.
   Do you use it? Does your answer change if the model decides who gets a loan?
6. A chi-square test of one column against your target gives $p = 0.3$. Do you
   drop the column? What would you want to know before you decide?

## Next

That completes Maths and tools. You can represent data, shape it, see it, and
now describe how certain you are about it.

The next group moves from clean examples to real data, where nothing arrives in
the shape you want.

[Back to the group overview](index.md){ .h-button }

Worth your time:
[Seeing Theory](https://seeing-theory.brown.edu/) is an interactive visual
introduction to everything on this page, and
[3Blue1Brown on Bayes' theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM)
explains the medical test example better than any text can.

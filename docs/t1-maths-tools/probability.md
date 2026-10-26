# Probability

<span class="badge badge--time">80 min</span>
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

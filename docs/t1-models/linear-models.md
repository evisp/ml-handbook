# Linear Models

<span class="badge badge--time">85 min</span>
<span class="badge badge--level">Classic models</span>
<span class="badge">Needs: Classification</span>

A linear model adds things up. That is the whole idea, and it is why these
models are still the default wherever somebody has to explain the answer to a
regulator, a doctor or a customer.

## Why this matters

Linear models are where you learn to read a model rather than only run one. The
coefficients are the model: there is nothing else inside. That makes them the
right place to meet regularisation, feature selection and explainability,
because you can check what every technique did by looking.

They are also the floor. On tabular data a linear model is often within a few
points of a much heavier one, and when it is not, the gap tells you something
about your problem.

## What a linear model is

$$
\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$

Each feature gets a weight. Multiply, add, done. The weights $w$ are called
coefficients and $b$ is the intercept.

<svg viewBox="0 0 680 300" role="img" aria-labelledby="l1-title l1-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="l1-title">How a linear model produces one prediction</title>
<desc id="l1-desc">Each feature value is multiplied by its coefficient to give a contribution. The contributions are added to the intercept to give the prediction.</desc>
<defs><marker id="l1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<text x="30" y="30" font-size="11.5" font-weight="700" fill="var(--h-graphite)">feature</text>
<text x="180" y="30" font-size="11.5" font-weight="700" fill="var(--h-graphite)">value</text>
<text x="252" y="30" font-size="11.5" font-weight="700" fill="var(--h-graphite)">coefficient</text>
<text x="430" y="30" font-size="11.5" font-weight="700" fill="var(--h-cherry)">contribution</text>
<text x="30" y="60" font-size="12" fill="var(--md-default-fg-color)">tenure</text>
<text x="196" y="60" text-anchor="end" font-size="12" fill="var(--h-graphite)">24</text>
<text x="316" y="60" text-anchor="end" font-size="12" fill="var(--h-graphite)">-1.6</text>
<rect x="369" y="47" width="61" height="18" rx="4" fill="var(--h-cherry)" opacity="0.45"/>
<text x="361" y="60" text-anchor="end" font-size="11" font-weight="700" fill="var(--h-graphite)">-38</text>
<text x="30" y="98" font-size="12" fill="var(--md-default-fg-color)">MonthlyCharges</text>
<text x="196" y="98" text-anchor="end" font-size="12" fill="var(--h-graphite)">79</text>
<text x="316" y="98" text-anchor="end" font-size="12" fill="var(--h-graphite)">0.53</text>
<rect x="430" y="85" width="67" height="18" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<text x="505" y="98" text-anchor="start" font-size="11" font-weight="700" fill="var(--h-cherry)">+42</text>
<text x="30" y="136" font-size="12" fill="var(--md-default-fg-color)">is_fiber</text>
<text x="196" y="136" text-anchor="end" font-size="12" fill="var(--h-graphite)">1</text>
<text x="316" y="136" text-anchor="end" font-size="12" fill="var(--h-graphite)">31.0</text>
<rect x="430" y="123" width="50" height="18" rx="4" fill="var(--h-cherry)" opacity="0.85"/>
<text x="488" y="136" text-anchor="start" font-size="11" font-weight="700" fill="var(--h-cherry)">+31</text>
<text x="30" y="174" font-size="12" fill="var(--md-default-fg-color)">n_services</text>
<text x="196" y="174" text-anchor="end" font-size="12" fill="var(--h-graphite)">4</text>
<text x="316" y="174" text-anchor="end" font-size="12" fill="var(--h-graphite)">-4.8</text>
<rect x="400" y="161" width="30" height="18" rx="4" fill="var(--h-cherry)" opacity="0.45"/>
<text x="392" y="174" text-anchor="end" font-size="11" font-weight="700" fill="var(--h-graphite)">-19</text>
<line x1="430" y1="40" x2="430" y2="200" stroke="var(--h-surface-line)" stroke-width="1.5" stroke-dasharray="4 3"/>
<text x="430" y="216" text-anchor="middle" font-size="10" fill="var(--h-space)">0</text>
<line x1="30" y1="204" x2="640" y2="204" stroke="var(--h-surface-line)" stroke-width="1.5"/>
<text x="30" y="230" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">intercept</text>
<text x="316" y="230" text-anchor="end" font-size="12" fill="var(--h-graphite)">+ 58</text>
<rect x="430" y="246" width="150" height="34" rx="8" fill="var(--h-cherry)"/>
<text x="505" y="263" dy="0.36em" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">prediction: 74</text>
<text x="30" y="266" font-size="12" fill="var(--h-graphite)">add everything up</text>
<text x="30" y="286" font-size="11.5" fill="var(--h-space)">the coefficients are the model. There is nothing else inside it.</text>
</svg>

Training means choosing the coefficients that make the predictions as close as
possible to the truth, where "close" means the mean squared error you met in
Probability.

!!! ml "ML connection"

    You have seen this picture before, as a single unit on the What Training
    Actually Does page. A linear model is one of those units, and a neural
    network is many of them stacked with a squashing function in between.

    Understanding this one properly is not preparation for the real thing. It
    is the real thing, at its smallest.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
print(model.intercept_)
for name, coef in zip(X_train.columns, model.coef_):
    print(f"{name:25s} {coef:+.3f}")
```

Read a coefficient like this: **holding everything else constant, one extra unit
of this feature changes the prediction by this much.** A coefficient of $-1.6$
on tenure means each extra month of tenure lowers the prediction by 1.6.

!!! warning "Coefficients are only comparable if the features are scaled"

    A coefficient of 0.53 on monthly charges and 31 on a yes or no flag does not
    mean the flag matters sixty times more. They are measured in different
    units.

    Scale the features first, as you did on the Data Preparation page, and the
    coefficients become comparable. For regularised models it is not optional at
    all, for reasons in the next section.

## Judging a regression

Classification gave you accuracy, precision and recall. Regression predicts a
number, so the question changes to how far off you were.

<svg viewBox="0 0 680 276" role="img" aria-labelledby="l2-title l2-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="l2-title">Residuals, and what one large error does to each metric</title>
<desc id="l2-desc">The vertical distance from each point to the line is its residual. Adding one badly wrong point barely moves the mean absolute error and moves the root mean squared error a great deal.</desc>
<rect x="10" y="26" width="320" height="170" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="170" y="18" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">residuals</text>
<line x1="28" y1="176" x2="318" y2="54" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<line x1="34" y1="171" x2="34" y2="170" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.7"/>
<circle cx="34" cy="171" r="3.6" fill="var(--h-cherry)"/>
<line x1="58" y1="174" x2="58" y2="160" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.7"/>
<circle cx="58" cy="174" r="3.6" fill="var(--h-cherry)"/>
<line x1="82" y1="140" x2="82" y2="150" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.7"/>
<circle cx="82" cy="140" r="3.6" fill="var(--h-cherry)"/>
<line x1="106" y1="151" x2="106" y2="140" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.7"/>
<circle cx="106" cy="151" r="3.6" fill="var(--h-cherry)"/>
<line x1="130" y1="127" x2="130" y2="130" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.7"/>
<circle cx="130" cy="127" r="3.6" fill="var(--h-cherry)"/>
<line x1="154" y1="117" x2="154" y2="120" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.7"/>
<circle cx="154" cy="117" r="3.6" fill="var(--h-cherry)"/>
<line x1="178" y1="130" x2="178" y2="110" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.7"/>
<circle cx="178" cy="130" r="3.6" fill="var(--h-cherry)"/>
<line x1="202" y1="101" x2="202" y2="99" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.7"/>
<circle cx="202" cy="101" r="3.6" fill="var(--h-cherry)"/>
<line x1="226" y1="89" x2="226" y2="89" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.7"/>
<circle cx="226" cy="89" r="3.6" fill="var(--h-cherry)"/>
<line x1="250" y1="87" x2="250" y2="79" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.7"/>
<circle cx="250" cy="87" r="3.6" fill="var(--h-cherry)"/>
<line x1="274" y1="82" x2="274" y2="69" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.7"/>
<circle cx="274" cy="82" r="3.6" fill="var(--h-cherry)"/>
<line x1="298" y1="59" x2="298" y2="59" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.7"/>
<circle cx="298" cy="59" r="3.6" fill="var(--h-cherry)"/>
<text x="170" y="214" text-anchor="middle" font-size="11" fill="var(--h-graphite)">each red line is one error the model made</text>
<rect x="350" y="26" width="320" height="170" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="510" y="18" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">one point goes badly wrong</text>
<line x1="368" y1="176" x2="658" y2="54" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<line x1="374" y1="171" x2="374" y2="170" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.5"/>
<circle cx="374" cy="171" r="3.6" fill="var(--h-cherry)" opacity="0.6"/>
<line x1="398" y1="174" x2="398" y2="160" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.5"/>
<circle cx="398" cy="174" r="3.6" fill="var(--h-cherry)" opacity="0.6"/>
<line x1="422" y1="140" x2="422" y2="150" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.5"/>
<circle cx="422" cy="140" r="3.6" fill="var(--h-cherry)" opacity="0.6"/>
<line x1="446" y1="151" x2="446" y2="140" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.5"/>
<circle cx="446" cy="151" r="3.6" fill="var(--h-cherry)" opacity="0.6"/>
<line x1="470" y1="127" x2="470" y2="130" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.5"/>
<circle cx="470" cy="127" r="3.6" fill="var(--h-cherry)" opacity="0.6"/>
<line x1="494" y1="117" x2="494" y2="120" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.5"/>
<circle cx="494" cy="117" r="3.6" fill="var(--h-cherry)" opacity="0.6"/>
<line x1="518" y1="130" x2="518" y2="110" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.5"/>
<circle cx="518" cy="130" r="3.6" fill="var(--h-cherry)" opacity="0.6"/>
<line x1="542" y1="58" x2="542" y2="99" stroke="var(--h-cherry)" stroke-width="2.5"/>
<circle cx="542" cy="58" r="5" fill="var(--h-cherry)"/>
<line x1="566" y1="89" x2="566" y2="89" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.5"/>
<circle cx="566" cy="89" r="3.6" fill="var(--h-cherry)" opacity="0.6"/>
<line x1="590" y1="87" x2="590" y2="79" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.5"/>
<circle cx="590" cy="87" r="3.6" fill="var(--h-cherry)" opacity="0.6"/>
<line x1="614" y1="82" x2="614" y2="69" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.5"/>
<circle cx="614" cy="82" r="3.6" fill="var(--h-cherry)" opacity="0.6"/>
<line x1="638" y1="59" x2="638" y2="59" stroke="var(--h-cherry)" stroke-width="1.5" opacity="0.5"/>
<circle cx="638" cy="59" r="3.6" fill="var(--h-cherry)" opacity="0.6"/>
<text x="510" y="214" text-anchor="middle" font-size="11" fill="var(--h-graphite)">MAE barely moves. RMSE jumps, because of the squaring.</text>
<text x="340" y="248" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Use MAE when every euro of error costs the same. Use RMSE when one large mistake is worse than several small ones.</text>
<text x="340" y="268" text-anchor="middle" font-size="12" fill="var(--h-graphite)">This is the squaring you met in Probability, where MSE was the variance of your errors.</text>
</svg>

**MAE**, mean absolute error, is the average size of the error, in the original
units. Easy to explain: "on average we are 47 euro out".

**MSE**, mean squared error, squares each error before averaging. Harder to read,
because the units are squared, and it is what the model actually minimises.

**RMSE** is the square root of MSE, which puts it back into the original units
while keeping the squaring's effect: large errors count for much more than small
ones.

**R squared** answers a different question: what fraction of the variation in the
target does the model explain? 1.0 is perfect, 0.0 is no better than always
predicting the mean, and negative is worse than that.

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pred = model.predict(X_test)
print("MAE ", mean_absolute_error(y_test, pred))
print("RMSE", np.sqrt(mean_squared_error(y_test, pred)))
print("R2  ", r2_score(y_test, pred))
```

Which to report: **MAE** when every unit of error costs the same, **RMSE** when
one large mistake is worse than several small ones, and **R squared** alongside
either, because it is the only one that is comparable across different problems.

An RMSE of 47 means nothing on its own. An RMSE of 47 where the target ranges
from 20 to 8000 is excellent, and where it ranges from 0 to 100 is terrible.

## Regularisation

The Classification page showed you overfitting. This is the fix.

An unconstrained linear model will happily give a feature an enormous
coefficient if that squeezes a little more accuracy out of the training data.
Large coefficients mean the prediction swings wildly on small changes in input,
which is overfitting in its most readable form.

Regularisation adds a penalty for large coefficients, so the model has to trade
fit against simplicity.

$$
\text{minimise} \quad \text{error} + \alpha \times \text{penalty on the coefficients}
$$

Alpha controls the trade. Zero gives you plain linear regression. Very large
drives every coefficient towards zero and the model predicts the mean.

The two common penalties behave very differently, and the reason is geometric.

<svg viewBox="0 0 680 286" role="img" aria-labelledby="l3-title l3-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="l3-title">Why Lasso sets coefficients to exactly zero and Ridge does not</title>
<desc id="l3-desc">The penalty region for Ridge is a circle and for Lasso a diamond. The solution is where the error contours first touch that region. A diamond has corners on the axes, so the touch point often has one coefficient at exactly zero.</desc>
<rect x="10" y="30" width="320" height="180" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="170" y="22" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">Ridge, L2</text>
<line x1="40" y1="140" x2="300" y2="140" stroke="var(--h-steel)" stroke-width="1.2"/>
<line x1="120" y1="195" x2="120" y2="40" stroke="var(--h-steel)" stroke-width="1.2"/>
<circle cx="120" cy="140" r="42" fill="var(--h-cherry)" opacity="0.18" stroke="var(--h-cherry)" stroke-width="2"/>
<polyline points="260.8,63.5 261.5,65.0 261.9,66.7 262.0,68.4 261.9,70.2 261.5,72.1 260.9,74.0 260.0,75.9 258.8,77.8 257.5,79.7 255.9,81.6 254.1,83.4 252.2,85.2 250.1,86.8 247.8,88.3 245.5,89.7 243.1,90.9 240.6,92.0 238.1,92.9 235.6,93.6 233.1,94.1 230.6,94.4 228.3,94.5 226.0,94.4 223.9,94.1 222.0,93.6 220.2,93.0 218.6,92.1 217.2,91.0 216.1,89.8 215.2,88.5 214.5,87.0 214.1,85.3 214.0,83.6 214.1,81.8 214.5,79.9 215.1,78.0 216.0,76.1 217.2,74.2 218.5,72.3 220.1,70.4 221.9,68.6 223.8,66.8 225.9,65.2 228.2,63.7 230.5,62.3 232.9,61.1 235.4,60.0 237.9,59.1 240.4,58.4 242.9,57.9 245.4,57.6 247.7,57.5 250.0,57.6 252.1,57.9 254.0,58.4 255.8,59.0 257.4,59.9 258.8,61.0 259.9,62.2 260.8,63.5" fill="none" stroke="var(--h-steel)" stroke-width="1.4" opacity="0.7"/>
<polyline points="276.6,54.9 277.7,57.4 278.4,60.2 278.6,63.1 278.4,66.2 277.8,69.3 276.7,72.6 275.2,75.8 273.2,79.1 270.9,82.3 268.3,85.5 265.3,88.6 262.0,91.5 258.4,94.3 254.6,96.8 250.7,99.2 246.6,101.2 242.4,103.0 238.1,104.6 233.9,105.7 229.7,106.6 225.5,107.1 221.6,107.3 217.8,107.2 214.2,106.7 210.9,105.9 207.9,104.7 205.2,103.2 202.9,101.5 200.9,99.4 199.4,97.1 198.3,94.6 197.6,91.8 197.4,88.9 197.6,85.8 198.2,82.7 199.3,79.4 200.8,76.2 202.8,72.9 205.1,69.7 207.7,66.5 210.7,63.4 214.0,60.5 217.6,57.7 221.4,55.2 225.3,52.8 229.4,50.8 233.6,49.0 237.9,47.4 242.1,46.3 246.3,45.4 250.5,44.9 254.4,44.7 258.2,44.8 261.8,45.3 265.1,46.1 268.1,47.3 270.8,48.8 273.1,50.5 275.1,52.6 276.6,54.9" fill="none" stroke="var(--h-steel)" stroke-width="1.4" opacity="0.7"/>
<polyline points="292.4,46.3 294.0,49.9 294.9,53.7 295.3,57.8 295.0,62.1 294.0,66.6 292.5,71.1 290.4,75.8 287.7,80.4 284.4,84.9 280.7,89.4 276.4,93.7 271.8,97.9 266.8,101.8 261.4,105.4 255.8,108.6 250.0,111.6 244.1,114.1 238.1,116.2 232.2,117.9 226.2,119.1 220.4,119.9 214.8,120.2 209.5,119.9 204.5,119.2 199.8,118.1 195.5,116.4 191.8,114.4 188.5,111.9 185.8,109.0 183.6,105.7 182.0,102.1 181.1,98.3 180.7,94.2 181.0,89.9 182.0,85.4 183.5,80.9 185.6,76.2 188.3,71.6 191.6,67.1 195.3,62.6 199.6,58.3 204.2,54.1 209.2,50.2 214.6,46.6 220.2,43.4 226.0,40.4 231.9,37.9 237.9,35.8 243.8,34.1 249.8,32.9 255.6,32.1 261.2,31.8 266.5,32.1 271.5,32.8 276.2,33.9 280.5,35.6 284.2,37.6 287.5,40.1 290.2,43.0 292.4,46.3" fill="none" stroke="var(--h-steel)" stroke-width="1.4" opacity="0.7"/>
<circle cx="238" cy="76" r="3.5" fill="var(--h-steel)"/>
<text x="238" y="56" text-anchor="middle" font-size="10" fill="var(--h-space)">best fit, no penalty</text>
<circle cx="153" cy="114" r="5" fill="var(--h-cherry)"/>
<text x="170" y="232" text-anchor="middle" font-size="11" fill="var(--h-graphite)">a circle has no corners</text>
<rect x="350" y="30" width="320" height="180" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="510" y="22" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">Lasso, L1</text>
<line x1="380" y1="140" x2="640" y2="140" stroke="var(--h-steel)" stroke-width="1.2"/>
<line x1="460" y1="195" x2="460" y2="40" stroke="var(--h-steel)" stroke-width="1.2"/>
<polygon points="460,94 506,140 460,186 414,140" fill="var(--h-cherry)" opacity="0.18" stroke="var(--h-cherry)" stroke-width="2"/>
<polyline points="600.8,63.5 601.5,65.0 601.9,66.7 602.0,68.4 601.9,70.2 601.5,72.1 600.9,74.0 600.0,75.9 598.8,77.8 597.5,79.7 595.9,81.6 594.1,83.4 592.2,85.2 590.1,86.8 587.8,88.3 585.5,89.7 583.1,90.9 580.6,92.0 578.1,92.9 575.6,93.6 573.1,94.1 570.6,94.4 568.3,94.5 566.0,94.4 563.9,94.1 562.0,93.6 560.2,93.0 558.6,92.1 557.2,91.0 556.1,89.8 555.2,88.5 554.5,87.0 554.1,85.3 554.0,83.6 554.1,81.8 554.5,79.9 555.1,78.0 556.0,76.1 557.2,74.2 558.5,72.3 560.1,70.4 561.9,68.6 563.8,66.8 565.9,65.2 568.2,63.7 570.5,62.3 572.9,61.1 575.4,60.0 577.9,59.1 580.4,58.4 582.9,57.9 585.4,57.6 587.7,57.5 590.0,57.6 592.1,57.9 594.0,58.4 595.8,59.0 597.4,59.9 598.8,61.0 599.9,62.2 600.8,63.5" fill="none" stroke="var(--h-steel)" stroke-width="1.4" opacity="0.7"/>
<polyline points="616.6,54.9 617.7,57.4 618.4,60.2 618.6,63.1 618.4,66.2 617.8,69.3 616.7,72.6 615.2,75.8 613.2,79.1 610.9,82.3 608.3,85.5 605.3,88.6 602.0,91.5 598.4,94.3 594.6,96.8 590.7,99.2 586.6,101.2 582.4,103.0 578.1,104.6 573.9,105.7 569.7,106.6 565.5,107.1 561.6,107.3 557.8,107.2 554.2,106.7 550.9,105.9 547.9,104.7 545.2,103.2 542.9,101.5 540.9,99.4 539.4,97.1 538.3,94.6 537.6,91.8 537.4,88.9 537.6,85.8 538.2,82.7 539.3,79.4 540.8,76.2 542.8,72.9 545.1,69.7 547.7,66.5 550.7,63.4 554.0,60.5 557.6,57.7 561.4,55.2 565.3,52.8 569.4,50.8 573.6,49.0 577.9,47.4 582.1,46.3 586.3,45.4 590.5,44.9 594.4,44.7 598.2,44.8 601.8,45.3 605.1,46.1 608.1,47.3 610.8,48.8 613.1,50.5 615.1,52.6 616.6,54.9" fill="none" stroke="var(--h-steel)" stroke-width="1.4" opacity="0.7"/>
<polyline points="632.4,46.3 634.0,49.9 634.9,53.7 635.3,57.8 635.0,62.1 634.0,66.6 632.5,71.1 630.4,75.8 627.7,80.4 624.4,84.9 620.7,89.4 616.4,93.7 611.8,97.9 606.8,101.8 601.4,105.4 595.8,108.6 590.0,111.6 584.1,114.1 578.1,116.2 572.2,117.9 566.2,119.1 560.4,119.9 554.8,120.2 549.5,119.9 544.5,119.2 539.8,118.1 535.5,116.4 531.8,114.4 528.5,111.9 525.8,109.0 523.6,105.7 522.0,102.1 521.1,98.3 520.7,94.2 521.0,89.9 522.0,85.4 523.5,80.9 525.6,76.2 528.3,71.6 531.6,67.1 535.3,62.6 539.6,58.3 544.2,54.1 549.2,50.2 554.6,46.6 560.2,43.4 566.0,40.4 571.9,37.9 577.9,35.8 583.8,34.1 589.8,32.9 595.6,32.1 601.2,31.8 606.5,32.1 611.5,32.8 616.2,33.9 620.5,35.6 624.2,37.6 627.5,40.1 630.2,43.0 632.4,46.3" fill="none" stroke="var(--h-steel)" stroke-width="1.4" opacity="0.7"/>
<circle cx="578" cy="76" r="3.5" fill="var(--h-steel)"/>
<text x="578" y="56" text-anchor="middle" font-size="10" fill="var(--h-space)">best fit, no penalty</text>
<circle cx="460" cy="94" r="5" fill="var(--h-cherry)"/>
<text x="510" y="232" text-anchor="middle" font-size="11" fill="var(--h-graphite)">a diamond has corners on the axes</text>
<text x="170" y="252" text-anchor="middle" font-size="11" fill="var(--h-graphite)">both coefficients stay non-zero, just smaller</text>
<text x="510" y="252" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-cherry)">the touch point sits on the axis: one coefficient is exactly 0</text>
<text x="340" y="278" text-anchor="middle" font-size="12" fill="var(--h-graphite)">That is the whole difference. Ridge shrinks every coefficient. Lasso removes some of them entirely.</text>
</svg>

**Ridge** penalises the squared coefficients. Its constraint region is round, so
the solution almost never lands exactly on an axis. Everything shrinks, nothing
disappears.

**Lasso** penalises the absolute values. Its region is a diamond with corners on
the axes, and corners are exactly where the error contours tend to touch first.
When that happens a coefficient becomes exactly zero.

<svg viewBox="0 0 680 276" role="img" aria-labelledby="l4-title l4-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="l4-title">What happens to the coefficients as the penalty grows</title>
<desc id="l4-desc">With Ridge, every coefficient shrinks towards zero but never reaches it. With Lasso, coefficients hit exactly zero one after another, which is feature selection.</desc>
<rect x="10" y="30" width="320" height="170" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="170" y="22" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">Ridge</text>
<line x1="50" y1="115" x2="300" y2="115" stroke="var(--h-steel)" stroke-width="1.2" stroke-dasharray="3 3"/>
<line x1="50" y1="40" x2="50" y2="177" stroke="var(--h-steel)" stroke-width="1.4"/>
<text x="42" y="119" text-anchor="end" font-size="10" fill="var(--h-space)">0</text>
<text x="170" y="190" text-anchor="middle" font-size="10.5" fill="var(--h-space)">penalty strength, alpha</text>
<polyline points="50.0,47.0 55.0,50.4 60.0,53.7 65.0,56.8 70.0,59.8 75.0,62.6 80.0,65.2 85.0,67.7 90.0,70.1 95.0,72.4 100.0,74.6 105.0,76.6 110.0,78.6 115.0,80.4 120.0,82.2 125.0,83.8 130.0,85.4 135.0,86.9 140.0,88.3 145.0,89.7 150.0,91.0 155.0,92.2 160.0,93.3 165.0,94.4 170.0,95.5 175.0,96.5 180.0,97.4 185.0,98.3 190.0,99.1 195.0,99.9 200.0,100.7 205.0,101.4 210.0,102.1 215.0,102.8 220.0,103.4 225.0,104.0 230.0,104.5 235.0,105.1 240.0,105.6 245.0,106.1 250.0,106.5 255.0,106.9 260.0,107.3 265.0,107.7 270.0,108.1 275.0,108.4 280.0,108.8 285.0,109.1 290.0,109.4 295.0,109.7 300.0,109.9" fill="none" stroke="var(--h-cherry)" stroke-width="2" opacity="0.90"/>
<polyline points="50.0,69.0 55.0,71.3 60.0,73.5 65.0,75.6 70.0,77.6 75.0,79.5 80.0,81.3 85.0,83.0 90.0,84.7 95.0,86.2 100.0,87.7 105.0,89.0 110.0,90.4 115.0,91.6 120.0,92.8 125.0,93.9 130.0,95.0 135.0,96.0 140.0,97.0 145.0,97.9 150.0,98.7 155.0,99.6 160.0,100.3 165.0,101.1 170.0,101.8 175.0,102.5 180.0,103.1 185.0,103.7 190.0,104.3 195.0,104.8 200.0,105.3 205.0,105.8 210.0,106.3 215.0,106.7 220.0,107.1 225.0,107.5 230.0,107.9 235.0,108.3 240.0,108.6 245.0,108.9 250.0,109.3 255.0,109.5 260.0,109.8 265.0,110.1 270.0,110.3 275.0,110.6 280.0,110.8 285.0,111.0 290.0,111.2 295.0,111.4 300.0,111.6" fill="none" stroke="var(--h-cherry)" stroke-width="2" opacity="0.78"/>
<polyline points="50.0,85.0 55.0,86.5 60.0,88.0 65.0,89.3 70.0,90.6 75.0,91.9 80.0,93.0 85.0,94.2 90.0,95.2 95.0,96.2 100.0,97.2 105.0,98.1 110.0,98.9 115.0,99.7 120.0,100.5 125.0,101.2 130.0,101.9 135.0,102.6 140.0,103.2 145.0,103.8 150.0,104.4 155.0,104.9 160.0,105.4 165.0,105.9 170.0,106.4 175.0,106.8 180.0,107.2 185.0,107.6 190.0,108.0 195.0,108.4 200.0,108.7 205.0,109.0 210.0,109.3 215.0,109.6 220.0,109.9 225.0,110.1 230.0,110.4 235.0,110.6 240.0,110.8 245.0,111.1 250.0,111.3 255.0,111.4 260.0,111.6 265.0,111.8 270.0,112.0 275.0,112.1 280.0,112.3 285.0,112.4 290.0,112.5 295.0,112.7 300.0,112.8" fill="none" stroke="var(--h-cherry)" stroke-width="2" opacity="0.66"/>
<polyline points="50.0,155.0 55.0,153.0 60.0,151.0 65.0,149.2 70.0,147.5 75.0,145.8 80.0,144.3 85.0,142.8 90.0,141.4 95.0,140.1 100.0,138.8 105.0,137.6 110.0,136.4 115.0,135.3 120.0,134.3 125.0,133.3 130.0,132.4 135.0,131.5 140.0,130.7 145.0,129.9 150.0,129.1 155.0,128.4 160.0,127.7 165.0,127.1 170.0,126.5 175.0,125.9 180.0,125.3 185.0,124.8 190.0,124.3 195.0,123.9 200.0,123.4 205.0,123.0 210.0,122.6 215.0,122.2 220.0,121.8 225.0,121.5 230.0,121.2 235.0,120.8 240.0,120.5 245.0,120.3 250.0,120.0 255.0,119.7 260.0,119.5 265.0,119.3 270.0,119.1 275.0,118.9 280.0,118.7 285.0,118.5 290.0,118.3 295.0,118.1 300.0,118.0" fill="none" stroke="var(--h-cherry)" stroke-width="2" opacity="0.54"/>
<polyline points="50.0,137.0 55.0,135.9 60.0,134.8 65.0,133.8 70.0,132.9 75.0,132.0 80.0,131.1 85.0,130.3 90.0,129.5 95.0,128.8 100.0,128.1 105.0,127.4 110.0,126.8 115.0,126.2 120.0,125.6 125.0,125.1 130.0,124.6 135.0,124.1 140.0,123.6 145.0,123.2 150.0,122.8 155.0,122.4 160.0,122.0 165.0,121.7 170.0,121.3 175.0,121.0 180.0,120.7 185.0,120.4 190.0,120.1 195.0,119.9 200.0,119.6 205.0,119.4 210.0,119.2 215.0,119.0 220.0,118.8 225.0,118.6 230.0,118.4 235.0,118.2 240.0,118.0 245.0,117.9 250.0,117.7 255.0,117.6 260.0,117.5 265.0,117.4 270.0,117.2 275.0,117.1 280.0,117.0 285.0,116.9 290.0,116.8 295.0,116.7 300.0,116.6" fill="none" stroke="var(--h-cherry)" stroke-width="2" opacity="0.42"/>
<rect x="350" y="30" width="320" height="170" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="510" y="22" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">Lasso</text>
<line x1="390" y1="115" x2="640" y2="115" stroke="var(--h-steel)" stroke-width="1.2" stroke-dasharray="3 3"/>
<line x1="390" y1="40" x2="390" y2="177" stroke="var(--h-steel)" stroke-width="1.4"/>
<text x="382" y="119" text-anchor="end" font-size="10" fill="var(--h-space)">0</text>
<text x="510" y="190" text-anchor="middle" font-size="10.5" fill="var(--h-space)">penalty strength, alpha</text>
<polyline points="390.0,47.0 395.0,52.4 400.0,57.9 405.0,63.3 410.0,68.8 415.0,74.2 420.0,79.6 425.0,85.1 430.0,90.5 435.0,96.0 440.0,101.4 445.0,106.8 450.0,112.3 455.0,115.0 460.0,115.0 465.0,115.0 470.0,115.0 475.0,115.0 480.0,115.0 485.0,115.0 490.0,115.0 495.0,115.0 500.0,115.0 505.0,115.0 510.0,115.0 515.0,115.0 520.0,115.0 525.0,115.0 530.0,115.0 535.0,115.0 540.0,115.0 545.0,115.0 550.0,115.0 555.0,115.0 560.0,115.0 565.0,115.0 570.0,115.0 575.0,115.0 580.0,115.0 585.0,115.0 590.0,115.0 595.0,115.0 600.0,115.0 605.0,115.0 610.0,115.0 615.0,115.0 620.0,115.0 625.0,115.0 630.0,115.0 635.0,115.0 640.0,115.0" fill="none" stroke="var(--h-cherry)" stroke-width="2" opacity="0.90"/>
<polyline points="390.0,69.0 395.0,71.2 400.0,73.5 405.0,75.7 410.0,78.0 415.0,80.2 420.0,82.5 425.0,84.7 430.0,87.0 435.0,89.2 440.0,91.4 445.0,93.7 450.0,95.9 455.0,98.2 460.0,100.4 465.0,102.7 470.0,104.9 475.0,107.1 480.0,109.4 485.0,111.6 490.0,113.9 495.0,115.0 500.0,115.0 505.0,115.0 510.0,115.0 515.0,115.0 520.0,115.0 525.0,115.0 530.0,115.0 535.0,115.0 540.0,115.0 545.0,115.0 550.0,115.0 555.0,115.0 560.0,115.0 565.0,115.0 570.0,115.0 575.0,115.0 580.0,115.0 585.0,115.0 590.0,115.0 595.0,115.0 600.0,115.0 605.0,115.0 610.0,115.0 615.0,115.0 620.0,115.0 625.0,115.0 630.0,115.0 635.0,115.0 640.0,115.0" fill="none" stroke="var(--h-cherry)" stroke-width="2" opacity="0.78"/>
<polyline points="390.0,85.0 395.0,86.1 400.0,87.1 405.0,88.2 410.0,89.2 415.0,90.3 420.0,91.3 425.0,92.4 430.0,93.4 435.0,94.5 440.0,95.5 445.0,96.6 450.0,97.6 455.0,98.7 460.0,99.7 465.0,100.8 470.0,101.8 475.0,102.9 480.0,103.9 485.0,105.0 490.0,106.1 495.0,107.1 500.0,108.2 505.0,109.2 510.0,110.3 515.0,111.3 520.0,112.4 525.0,113.4 530.0,114.5 535.0,115.0 540.0,115.0 545.0,115.0 550.0,115.0 555.0,115.0 560.0,115.0 565.0,115.0 570.0,115.0 575.0,115.0 580.0,115.0 585.0,115.0 590.0,115.0 595.0,115.0 600.0,115.0 605.0,115.0 610.0,115.0 615.0,115.0 620.0,115.0 625.0,115.0 630.0,115.0 635.0,115.0 640.0,115.0" fill="none" stroke="var(--h-cherry)" stroke-width="2" opacity="0.66"/>
<polyline points="390.0,155.0 395.0,153.9 400.0,152.8 405.0,151.7 410.0,150.6 415.0,149.5 420.0,148.4 425.0,147.3 430.0,146.2 435.0,145.1 440.0,144.0 445.0,142.9 450.0,141.8 455.0,140.8 460.0,139.7 465.0,138.6 470.0,137.5 475.0,136.4 480.0,135.3 485.0,134.2 490.0,133.1 495.0,132.0 500.0,130.9 505.0,129.8 510.0,128.7 515.0,127.6 520.0,126.5 525.0,125.4 530.0,124.3 535.0,123.2 540.0,122.1 545.0,121.0 550.0,119.9 555.0,118.8 560.0,117.7 565.0,116.6 570.0,115.5 575.0,115.0 580.0,115.0 585.0,115.0 590.0,115.0 595.0,115.0 600.0,115.0 605.0,115.0 610.0,115.0 615.0,115.0 620.0,115.0 625.0,115.0 630.0,115.0 635.0,115.0 640.0,115.0" fill="none" stroke="var(--h-cherry)" stroke-width="2" opacity="0.54"/>
<polyline points="390.0,137.0 395.0,136.5 400.0,136.0 405.0,135.5 410.0,135.0 415.0,134.5 420.0,134.0 425.0,133.5 430.0,133.0 435.0,132.6 440.0,132.1 445.0,131.6 450.0,131.1 455.0,130.6 460.0,130.1 465.0,129.6 470.0,129.1 475.0,128.6 480.0,128.1 485.0,127.6 490.0,127.1 495.0,126.6 500.0,126.1 505.0,125.6 510.0,125.1 515.0,124.6 520.0,124.1 525.0,123.7 530.0,123.2 535.0,122.7 540.0,122.2 545.0,121.7 550.0,121.2 555.0,120.7 560.0,120.2 565.0,119.7 570.0,119.2 575.0,118.7 580.0,118.2 585.0,117.7 590.0,117.2 595.0,116.7 600.0,116.2 605.0,115.7 610.0,115.2 615.0,115.0 620.0,115.0 625.0,115.0 630.0,115.0 635.0,115.0 640.0,115.0" fill="none" stroke="var(--h-cherry)" stroke-width="2" opacity="0.42"/>
<text x="170" y="222" text-anchor="middle" font-size="11" fill="var(--h-graphite)">everything shrinks, nothing disappears</text>
<text x="510" y="222" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-cherry)">features drop out one by one</text>
<text x="340" y="252" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Lasso is therefore a feature selector as well as a regulariser: read off which coefficients survived.</text>
<text x="340" y="270" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Alpha at zero gives you plain linear regression. Alpha very large gives you a model that predicts the mean.</text>
</svg>

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ridge = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
lasso = Pipeline([("scale", StandardScaler()), ("model", Lasso(alpha=0.1))])
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
kept = (lasso.named_steps["model"].coef_ != 0).sum()
print(f"lasso kept {kept} of {X_train.shape[1]} features")
```

**Use Ridge** when you believe most features contribute something, and when
several are correlated. Ridge spreads the weight across a correlated group
rather than picking one arbitrarily.

**Use Lasso** when you suspect most features are useless and you want the model
to tell you which. The surviving coefficients are a feature selection.

**Use ElasticNet** when you want both, which is a mix of the two penalties.

!!! warning "Scaling is mandatory here, not advisory"

    The penalty is applied to the coefficients. A feature measured in thousands
    gets a tiny coefficient and is barely penalised. A feature measured in
    single digits gets a large one and is punished hard.

    Without scaling, alpha means something different for every column and the
    model's choices are driven by your units rather than by your data. Put the
    scaler in the pipeline and this cannot happen.

Choose alpha with cross-validation rather than by eye.

```python
from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 13))
```

## Explaining a prediction

Coefficients tell you what the model does in general. They do not tell you why
it said what it said about **this** customer.

SHAP answers that. For a single prediction it splits the difference between the
model's output and its average output into one contribution per feature.

That is precisely the diagram at the top of this page: value times coefficient,
one bar per feature. For a linear model SHAP reproduces it exactly. The reason
SHAP is worth learning is that it produces the same picture for a random forest,
a gradient boosted tree or a neural network, where no coefficients exist.

```python
import shap

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[0])
shap.plots.beeswarm(shap_values)
```

Two plots, two questions.

**Waterfall** explains one prediction. It starts at the average and adds each
feature's push up or down until it reaches this customer's answer. This is what
you show to the person who has to act on the prediction.

**Beeswarm** explains the model. Every point is one customer, features are
ordered by importance, and the colour shows whether the feature value was high
or low. It tells you not just which features matter but in which direction.

!!! warning "An explanation is not a cause"

    SHAP tells you what the model used. It does not tell you what is true about
    the world.

    If the model leans on a feature that happens to correlate with the outcome,
    SHAP will faithfully report that, and somebody will read it as a finding
    about customers rather than a finding about the model. This is the
    correlation and causation problem from Probability, arriving in a more
    persuasive costume.

## From regression to classification

Logistic regression is the same linear model with one extra step.

The linear part produces a score that can be any number. The sigmoid squashes it
into the range 0 to 1, which makes it a probability.

<svg viewBox="0 0 680 264" role="img" aria-labelledby="l5-title l5-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="l5-title">The sigmoid turns a linear score into a probability</title>
<desc id="l5-desc">The linear part can produce any number. The sigmoid squashes it into the range zero to one. A threshold, usually a half, then turns that probability into a label.</desc>
<line x1="60" y1="120" x2="640" y2="120" stroke="var(--h-steel)" stroke-width="1.2" stroke-dasharray="3 3"/>
<line x1="350" y1="40" x2="350" y2="200" stroke="var(--h-steel)" stroke-width="1.2"/>
<line x1="60" y1="200" x2="640" y2="200" stroke="var(--h-steel)" stroke-width="1.4"/>
<text x="52" y="204" text-anchor="end" font-size="10" fill="var(--h-space)">0</text>
<text x="52" y="44" text-anchor="end" font-size="10" fill="var(--h-space)">1</text>
<text x="52" y="124" text-anchor="end" font-size="10" fill="var(--h-space)">0.5</text>
<polyline points="60.0,199.6 64.8,199.6 69.7,199.5 74.5,199.5 79.3,199.4 84.2,199.3 89.0,199.3 93.8,199.2 98.7,199.1 103.5,199.0 108.3,198.9 113.2,198.8 118.0,198.7 122.8,198.6 127.7,198.4 132.5,198.2 137.3,198.1 142.2,197.9 147.0,197.6 151.8,197.4 156.7,197.1 161.5,196.8 166.3,196.5 171.2,196.1 176.0,195.7 180.8,195.3 185.7,194.8 190.5,194.3 195.3,193.7 200.2,193.1 205.0,192.4 209.8,191.7 214.7,190.8 219.5,189.9 224.3,188.9 229.2,187.9 234.0,186.7 238.8,185.4 243.7,184.0 248.5,182.5 253.3,180.9 258.2,179.2 263.0,177.3 267.8,175.3 272.7,173.1 277.5,170.8 282.3,168.3 287.2,165.7 292.0,163.0 296.8,160.0 301.7,157.0 306.5,153.8 311.3,150.4 316.2,146.9 321.0,143.3 325.8,139.6 330.7,135.8 335.5,131.9 340.3,128.0 345.2,124.0 350.0,120.0 354.8,116.0 359.7,112.0 364.5,108.1 369.3,104.2 374.2,100.4 379.0,96.7 383.8,93.1 388.7,89.6 393.5,86.2 398.3,83.0 403.2,80.0 408.0,77.0 412.8,74.3 417.7,71.7 422.5,69.2 427.3,66.9 432.2,64.7 437.0,62.7 441.8,60.8 446.7,59.1 451.5,57.5 456.3,56.0 461.2,54.6 466.0,53.3 470.8,52.1 475.7,51.1 480.5,50.1 485.3,49.2 490.2,48.3 495.0,47.6 499.8,46.9 504.7,46.3 509.5,45.7 514.3,45.2 519.2,44.7 524.0,44.3 528.8,43.9 533.7,43.5 538.5,43.2 543.3,42.9 548.2,42.6 553.0,42.4 557.8,42.1 562.7,41.9 567.5,41.8 572.3,41.6 577.2,41.4 582.0,41.3 586.8,41.2 591.7,41.1 596.5,41.0 601.3,40.9 606.2,40.8 611.0,40.7 615.8,40.7 620.7,40.6 625.5,40.5 630.3,40.5 635.2,40.4 640.0,40.4" fill="none" stroke="var(--h-cherry)" stroke-width="3"/>
<text x="350" y="216" text-anchor="middle" font-size="11" fill="var(--h-space)">0</text>
<text x="350" y="232" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-graphite)">the linear score, w times x plus b, can be any number</text>
<text x="120" y="188" font-size="11" font-weight="700" fill="var(--h-graphite)">very negative: almost certainly class 0</text>
<text x="620" y="56" text-anchor="end" font-size="11" font-weight="700" fill="var(--h-graphite)">very positive: almost certainly class 1</text>
<rect x="300" y="108" width="100" height="24" rx="5" fill="var(--h-cherry)" opacity="0.18"/>
<text x="350" y="100" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-cherry)">the threshold lives here</text>
<text x="340" y="256" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Linear regression stops at the score. Logistic regression pushes it through this curve to get a probability.</text>
</svg>

$$
p = \frac{1}{1 + e^{-(w \cdot x + b)}}
$$

```python
from sklearn.linear_model import LogisticRegression

clf = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])
clf.fit(X_train, y_train)
print(clf.predict_proba(X_test)[:3])
```

Despite the name it is a classifier, not a regressor. The confusing name is
historical and you will have to live with it.

The coefficients still mean something, just not the thing they mean in linear
regression. A logistic coefficient is a change in the **log odds**. Exponentiate
it and you get an odds ratio, which is how it is reported in medicine.

```python
import numpy as np

odds_ratio = np.exp(clf.named_steps["model"].coef_[0])
```

An odds ratio of 1.8 on a feature means the odds of the positive class are 1.8
times higher when that feature increases by one unit, holding the rest constant.

### The decision boundary

Where the probability equals 0.5, the linear score is zero. That set of points is
a straight line in two dimensions, a plane in three, and a flat surface in more.

This is what the Classification page drew as the linear family's boundary. It is
a hard constraint: if your classes cannot be separated by a straight cut, no
amount of tuning a logistic regression will fix it, and you need either better
features or a different family.

Regularisation applies here too, through the `C` parameter, which is the inverse
of alpha. Small `C` means strong regularisation.

## Support vector machines

An SVM asks a sharper question. Many lines separate two groups. Which one should
you choose?

<svg viewBox="0 0 680 266" role="img" aria-labelledby="l6-title l6-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="l6-title">A support vector machine maximises the margin</title>
<desc id="l6-desc">Many lines separate the two groups. The support vector machine chooses the one with the widest gap, and only the points sitting on the edge of that gap affect where it goes.</desc>
<rect x="10" y="26" width="320" height="176" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="170" y="18" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">many lines separate them</text>
<rect x="350" y="26" width="320" height="176" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="510" y="18" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">one has the widest gap</text>
<circle cx="60" cy="70" r="5" fill="var(--h-cherry)"/>
<circle cx="400" cy="70" r="5" fill="var(--h-cherry)"/>
<circle cx="92" cy="52" r="5" fill="var(--h-cherry)"/>
<circle cx="432" cy="52" r="5" fill="var(--h-cherry)"/>
<circle cx="120" cy="88" r="5" fill="var(--h-cherry)"/>
<circle cx="460" cy="88" r="5" fill="var(--h-cherry)"/>
<circle cx="78" cy="104" r="5" fill="var(--h-cherry)"/>
<circle cx="418" cy="104" r="5" fill="var(--h-cherry)"/>
<circle cx="140" cy="58" r="5" fill="var(--h-cherry)"/>
<circle cx="480" cy="58" r="5" fill="var(--h-cherry)"/>
<circle cx="104" cy="124" r="5" fill="var(--h-cherry)"/>
<circle cx="444" cy="124" r="5" fill="var(--h-cherry)"/>
<circle cx="214" cy="168" r="5" fill="var(--h-steel)" opacity="0.6"/>
<circle cx="554" cy="168" r="5" fill="var(--h-steel)" opacity="0.6"/>
<circle cx="250" cy="140" r="5" fill="var(--h-steel)" opacity="0.6"/>
<circle cx="590" cy="140" r="5" fill="var(--h-steel)" opacity="0.6"/>
<circle cx="196" cy="132" r="5" fill="var(--h-steel)" opacity="0.6"/>
<circle cx="536" cy="132" r="5" fill="var(--h-steel)" opacity="0.6"/>
<circle cx="268" cy="176" r="5" fill="var(--h-steel)" opacity="0.6"/>
<circle cx="608" cy="176" r="5" fill="var(--h-steel)" opacity="0.6"/>
<circle cx="232" cy="112" r="5" fill="var(--h-steel)" opacity="0.6"/>
<circle cx="572" cy="112" r="5" fill="var(--h-steel)" opacity="0.6"/>
<circle cx="282" cy="120" r="5" fill="var(--h-steel)" opacity="0.6"/>
<circle cx="622" cy="120" r="5" fill="var(--h-steel)" opacity="0.6"/>
<line x1="40" y1="196" x2="290" y2="36" stroke="var(--h-steel)" stroke-width="1.6" opacity="0.6"/>
<line x1="24" y1="196" x2="274" y2="36" stroke="var(--h-steel)" stroke-width="1.6" opacity="0.6"/>
<line x1="56" y1="196" x2="306" y2="36" stroke="var(--h-steel)" stroke-width="1.6" opacity="0.6"/>
<line x1="386" y1="196" x2="646" y2="52" stroke="var(--md-default-fg-color)" stroke-width="2.6"/>
<line x1="366" y1="186" x2="626" y2="42" stroke="var(--h-cherry)" stroke-width="1.6" stroke-dasharray="5 4"/>
<line x1="406" y1="206" x2="666" y2="62" stroke="var(--h-cherry)" stroke-width="1.6" stroke-dasharray="5 4"/>
<circle cx="444" cy="124" r="9" fill="none" stroke="var(--h-cherry)" stroke-width="2.5"/>
<circle cx="478" cy="104" r="9" fill="none" stroke="var(--h-cherry)" stroke-width="2.5"/>
<circle cx="532" cy="152" r="9" fill="none" stroke="var(--h-cherry)" stroke-width="2.5"/>
<text x="510" y="224" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-cherry)">the circled points are the support vectors</text>
<text x="340" y="248" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Move any other point and the boundary does not budge. Move a support vector and it does.</text>
</svg>

The answer is the one with the widest gap, called the margin. The points sitting
on the edge of that gap are the **support vectors**, and they are the only ones
that matter. Move any other point and the boundary does not change.

`C` controls how much you are willing to bend. A large `C` insists on
classifying every training point correctly, giving a narrow margin and probably
overfitting. A small `C` accepts some mistakes in exchange for a wider, more
robust margin.

### Kernels

A straight boundary is often not enough. Rather than adding polynomial features
by hand, an SVM can use a kernel, which computes what the answer would be in a
higher dimensional space without ever building it.

<svg viewBox="0 0 680 256" role="img" aria-labelledby="l7-title l7-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="l7-title">What each kernel can separate</title>
<desc id="l7-desc">A linear kernel draws a straight boundary. A polynomial kernel draws a curve. An RBF kernel can draw closed regions around clusters of points.</desc>
<rect x="5" y="26" width="214" height="150" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="112" y="18" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">linear</text>
<text x="112" y="196" text-anchor="middle" font-size="10.5" fill="var(--h-space)">a straight boundary</text>
<line x1="23" y1="160" x2="201" y2="44" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<circle cx="23" cy="49" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="92" cy="126" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="46" cy="49" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="63" cy="136" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="48" cy="133" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="142" cy="52" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="118" cy="94" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="96" cy="168" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="38" cy="37" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="192" cy="88" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="57" cy="78" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="87" cy="162" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="59" cy="63" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="127" cy="107" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="195" cy="146" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="86" cy="61" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="173" cy="167" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="27" cy="132" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="179" cy="121" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="61" cy="141" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<rect x="233" y="26" width="214" height="150" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="340" y="18" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">polynomial</text>
<text x="340" y="196" text-anchor="middle" font-size="10.5" fill="var(--h-space)">a smooth curve</text>
<polyline points="249,60 252,67 255,73 258,79 261,85 264,91 267,96 270,101 273,106 276,111 279,115 282,119 285,123 288,126 291,130 294,133 297,136 300,138 303,141 306,143 309,144 312,146 315,147 318,148 321,149 324,150 327,150 330,150 333,150 336,149 339,149 342,148 345,146 348,145 351,143 354,141 357,139 360,137 363,134 366,131 369,128 372,124 375,120 378,116 381,112 384,108 387,103 390,98 393,93 396,87 399,81 402,75 405,69 408,63 411,56 414,49 417,42 420,34 423,26 426,18 429,10 432,1" fill="none" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<circle cx="382" cy="60" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="366" cy="63" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="257" cy="79" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="310" cy="87" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="367" cy="146" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="424" cy="165" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="430" cy="140" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="305" cy="58" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="363" cy="107" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="372" cy="47" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="284" cy="157" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="358" cy="127" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="266" cy="87" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="415" cy="121" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="429" cy="60" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="332" cy="141" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="330" cy="140" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="351" cy="50" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="267" cy="166" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="271" cy="60" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="461" y="26" width="214" height="150" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="568" y="18" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">RBF</text>
<text x="568" y="196" text-anchor="middle" font-size="10.5" fill="var(--h-space)">closed regions</text>
<ellipse cx="531" cy="100" rx="38" ry="34" fill="none" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<ellipse cx="613" cy="126" rx="30" ry="26" fill="none" stroke="var(--md-default-fg-color)" stroke-width="2.5"/>
<circle cx="538" cy="130" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="632" cy="98" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="568" cy="140" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="528" cy="84" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="589" cy="119" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="484" cy="49" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="502" cy="131" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="603" cy="164" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="617" cy="68" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="616" cy="166" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="540" cy="35" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="545" cy="72" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="509" cy="40" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="500" cy="150" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="486" cy="116" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="515" cy="60" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="626" cy="74" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="510" cy="95" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="603" cy="104" r="3.6" fill="var(--h-cherry)" opacity="0.85"/>
<circle cx="633" cy="88" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="572" cy="130" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<circle cx="561" cy="130" r="3.6" fill="var(--h-steel)" opacity="0.5"/>
<text x="340" y="226" text-anchor="middle" font-size="12" fill="var(--h-graphite)">More flexible is not better. A kernel that can draw any shape can also draw the noise.</text>
<text x="340" y="246" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Start linear. Move to RBF only when the linear boundary is clearly not enough.</text>
</svg>

```python
from sklearn.svm import SVC

for kernel in ("linear", "poly", "rbf"):
    svm = Pipeline([("scale", StandardScaler()), ("model", SVC(kernel=kernel))])
    svm.fit(X_train, y_train)
    print(kernel, svm.score(X_test, y_test))
```

| Kernel | Boundary | Try it when |
|---|---|---|
| `linear` | A straight cut | Many features, or you want speed and interpretability |
| `poly` | A smooth curve | You expect interactions between features |
| `rbf` | Closed regions around clusters | The classes are tangled and you have the data to support it |

`rbf` is the usual default and has a second parameter, `gamma`, which controls
how far the influence of one point reaches. High gamma means each point only
affects its immediate neighbourhood, which produces a wiggly boundary that
overfits. Low gamma approaches a straight line.

So `C` and `gamma` both control complexity, and they interact. Tune them
together with a grid search rather than one at a time.

!!! warning "SVMs need scaling and do not scale up"

    The margin is measured as a distance, so an unscaled feature with a large
    range dominates it entirely. Always scale.

    Training time grows roughly with the square of the number of rows, so an SVM
    on a hundred thousand rows is painful and on a million is impractical. On
    the churn data it is fine. On something larger, reach for a linear model or
    a tree ensemble.

## Evaluating the classifiers

Everything from the Classification page applies unchanged: confusion matrix,
precision, recall, F1, ROC AUC, and choosing the metric before you see the
scores.

```python
from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_test, clf.predict(X_test)))
print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
```

One addition specific to this page. `SVC` does not give you probabilities by
default, because it produces distances from the boundary rather than
probabilities. Pass `probability=True` if you need them, and expect training to
take noticeably longer.

Compare every model against the dummy baseline, and compare them with
cross-validation rather than a single split, so that you are not reading noise.

```python
from sklearn.model_selection import cross_val_score

for name, m in [("logistic", clf), ("svm", svm), ("ridge", ridge)]:
    scores = cross_val_score(m, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"{name:10s} {scores.mean():.3f} plus or minus {scores.std():.3f}")
```

Print the spread as well as the mean. A model that scores 0.81 give or take 0.01
is better than one that scores 0.82 give or take 0.05, and the second number is
the one everybody forgets.

## When it goes wrong

??? note "`ConvergenceWarning: lbfgs failed to converge`"

    Usually unscaled features. Put a `StandardScaler` in the pipeline. If it
    persists, raise `max_iter`.

??? note "Lasso set every coefficient to zero"

    Alpha is too large, so the model found that predicting the mean was cheaper
    than using any feature. Try a much smaller alpha, on a log scale.

??? note "Ridge and plain linear regression give identical results"

    Alpha is effectively zero relative to the size of your errors, or your
    features are unscaled so the penalty is not biting. Check both.

??? note "The coefficients change completely when I refit"

    Two or more features are highly correlated, so the model can distribute the
    weight between them in many equivalent ways. Ridge stabilises this. It is
    also a sign to check the correlation heatmap you drew during preparation.

??? note "SVM training never finishes"

    Too many rows, or `C` is very large, or `gamma` is very high. Subsample
    first to find workable parameters, then scale up.

??? note "SHAP is extremely slow"

    You are using the general explainer on a large background set. Pass a
    sample of a hundred rows as the background, and use the linear or tree
    explainer when your model is one of those.

## Check yourself

Use the churn data, or any dataset with a numeric target for the regression
parts.

1. Fit a linear regression, print the coefficients, and write one sentence
   interpreting the largest one in plain language.
2. Report MAE, RMSE and R squared on the same predictions. Explain which you
   would put in a report and why.
3. Fit Ridge across a range of alphas and plot how one coefficient changes.
   Describe the shape.
4. Fit Lasso and report how many features survived. Name two that were dropped
   and say whether you agree.
5. Deliberately remove the scaler from a Ridge pipeline and compare the
   coefficients. Explain the difference.
6. Produce a SHAP waterfall for one customer and write the explanation you would
   give to that customer.
7. Fit a logistic regression, convert one coefficient to an odds ratio, and
   state what it means.
8. Compare `linear`, `poly` and `rbf` kernels with cross-validation, reporting
   the mean and the spread. Say which you would ship and why.

## Quick reference

| Task | Code |
|---|---|
| Fit a linear model | `LinearRegression().fit(X, y)` |
| Read the model | `model.coef_`, `model.intercept_` |
| Regression metrics | `mean_absolute_error`, `mean_squared_error`, `r2_score` |
| RMSE | `np.sqrt(mean_squared_error(y, pred))` |
| Ridge, Lasso | `Ridge(alpha=1.0)`, `Lasso(alpha=0.1)` |
| Choose alpha | `RidgeCV(alphas=np.logspace(-3, 3, 13))` |
| Surviving features | `(lasso.coef_ != 0).sum()` |
| Explain one prediction | `shap.plots.waterfall(shap_values[0])` |
| Explain the model | `shap.plots.beeswarm(shap_values)` |
| Logistic regression | `LogisticRegression(max_iter=1000)` |
| Odds ratio | `np.exp(model.coef_[0])` |
| SVM with a kernel | `SVC(kernel="rbf", C=1.0, gamma="scale")` |
| Support vectors | `svm.support_vectors_` |
| Fair comparison | `cross_val_score(pipeline, X, y, cv=5)` |

## Questions to sit with

1. A coefficient says the prediction changes by this much when the feature
   changes by one, holding everything else constant. When is "holding everything
   else constant" impossible in the real world, and what does that do to your
   interpretation?
2. Lasso drops features for you. What could it drop that you would want to keep,
   and how would you notice?
3. SHAP explains what the model used. A customer asks why they were refused.
   Is a SHAP waterfall an adequate answer? Would it satisfy a regulator?
4. A linear model gets 0.80 and an RBF kernel gets 0.82. Which do you ship, and
   what else would you want to know before deciding?
5. Regularisation deliberately makes the fit on training data worse. Explain to
   somebody non-technical why that is the point.

## Next

Linear models draw one straight cut. The next family builds its boundary from
rectangles instead, and on tabular data it usually wins.

[Tree-Based Models](tree-models.md){ .h-button }

Worth your time: the
[scikit-learn linear models guide](https://scikit-learn.org/stable/modules/linear_model.html)
covers every variant, and
[Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/shap.html)
by Christoph Molnar is the clearest written explanation of SHAP.

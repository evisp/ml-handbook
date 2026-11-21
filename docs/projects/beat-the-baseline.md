# Project 4: Beat the baseline

<span class="badge badge--time">Due 2 December</span>
<span class="badge badge--level">Classic models</span>
<span class="badge">Teams of 2 or 3</span>
<span class="badge">Five datasets</span>

Project 3 banned models. This one is built around them. But a model is only
worth something next to the simplest guess you could have made without one, so
that is where you start, and that is the number you have to beat.

## What you are doing

Five teams, five datasets, all from scikit-learn. Two teams predict a number.
Three teams predict a class. Everybody follows the same pipeline, and everybody
answers the same question at the end: how much better than the obvious guess is
your model, give or take?

<svg viewBox="0 0 680 240" role="img" aria-labelledby="p4-f1-title p4-f1-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="p4-f1-title">The pipeline every team follows</title>
<desc id="p4-f1-desc">Seven phases: data, split, baseline, model, tune, evaluate, explain. Baseline and evaluate are highlighted. Cached data, a data contract and ranges carry over from Project 3.</desc>
<defs><marker id="p4a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<rect x="5" y="40" width="86" height="86" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="48" y="68" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">data</text>
<text x="48" y="92" text-anchor="middle" font-size="10" fill="var(--h-graphite)">load, contract,</text>
<text x="48" y="106" text-anchor="middle" font-size="10" fill="var(--h-graphite)">look</text>
<line x1="92" y1="83" x2="99" y2="83" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#p4a)"/>
<rect x="100" y="40" width="86" height="86" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="143" y="68" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">split</text>
<text x="143" y="92" text-anchor="middle" font-size="10" fill="var(--h-graphite)">lock the</text>
<text x="143" y="106" text-anchor="middle" font-size="10" fill="var(--h-graphite)">test set away</text>
<line x1="187" y1="83" x2="194" y2="83" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#p4a)"/>
<rect x="195" y="40" width="86" height="86" rx="10" fill="var(--h-cherry)" stroke="var(--h-cherry)"/>
<text x="238" y="68" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">baseline</text>
<text x="238" y="92" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.9">the dumbest</text>
<text x="238" y="106" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.9">honest guess</text>
<line x1="282" y1="83" x2="289" y2="83" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#p4a)"/>
<rect x="290" y="40" width="86" height="86" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="333" y="68" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">model</text>
<text x="333" y="92" text-anchor="middle" font-size="10" fill="var(--h-graphite)">linear first,</text>
<text x="333" y="106" text-anchor="middle" font-size="10" fill="var(--h-graphite)">then one more</text>
<line x1="377" y1="83" x2="384" y2="83" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#p4a)"/>
<rect x="385" y="40" width="86" height="86" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="428" y="68" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">tune</text>
<text x="428" y="92" text-anchor="middle" font-size="10" fill="var(--h-graphite)">cross-validate,</text>
<text x="428" y="106" text-anchor="middle" font-size="10" fill="var(--h-graphite)">never on test</text>
<line x1="472" y1="83" x2="479" y2="83" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#p4a)"/>
<rect x="480" y="40" width="86" height="86" rx="10" fill="var(--h-cherry)" stroke="var(--h-cherry)"/>
<text x="523" y="68" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">evaluate</text>
<text x="523" y="92" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.9">test once,</text>
<text x="523" y="106" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.9">with a range</text>
<line x1="567" y1="83" x2="574" y2="83" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#p4a)"/>
<rect x="575" y="40" width="100" height="86" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="625" y="68" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">explain</text>
<text x="625" y="92" text-anchor="middle" font-size="10" fill="var(--h-graphite)">where it fails,</text>
<text x="625" y="106" text-anchor="middle" font-size="10" fill="var(--h-graphite)">and why</text>
<rect x="5" y="142" width="670" height="38" rx="8" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="340" y="161" dy="0.36em" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">carried over from Project 3: cached data, a data contract, every number with a range</text>
<text x="340" y="206" text-anchor="middle" font-size="12" fill="var(--h-graphite)">The two red boxes are the habits this project exists to train.</text>
<text x="340" y="226" text-anchor="middle" font-size="12" fill="var(--h-graphite)">A model that cannot beat the baseline has learned nothing you could not have guessed.</text>
</svg>

## What everyone must show

Same objectives for every team. What they mean for your data is in your tab.

| | Objective |
|---|---|
| 0 | Load the data, cache it, write the data contract, look at what is there |
| 1 | Split off a test set before modelling anything, with a fixed seed (stratified if you predict a class) |
| 2 | Score a baseline, exactly the way you will score your models |
| 3 | Train at least two models: a plain linear one, and one with more to tune (regularised, or a kernel) |
| 4 | Tune with cross-validation on the training set |
| 5 | Score the final model on the test set once, with a bootstrap range, next to the baseline |
| 6 | Find where the model fails, show examples, and explain why |
| 7 | Write it up: the notebook in full, the poster in one page |

!!! warning "The test set is opened once"

    Every decision, which features, which model, which hyperparameters, is made
    without looking at the test set. You score on it once, at the end, and
    whatever number comes out is the number you report.

    If you score on the test set, change something, and score again, it has
    become a second validation set, and your result is now an overestimate that
    nobody can correct.

These datasets are small, between 442 and 20,640 rows. Holding out a separate
validation set as well would leave very little to train on, so you tune with
cross-validation instead.

<svg viewBox="0 0 680 190" role="img" aria-labelledby="p4-f2-title p4-f2-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="p4-f2-title">Splitting and cross-validation</title>
<desc id="p4-f2-desc">The data is split once into a training set of about 80 percent and a test set of about 20 percent, which is locked away and opened once. Inside the training set, five-fold cross-validation trains on four folds and scores on the fifth, rotating, to choose and tune models.</desc>
<rect x="5" y="22" width="530" height="30" rx="8" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="270" y="37" dy="0.36em" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">training set, about 80%</text>
<rect x="541" y="22" width="134" height="30" rx="8" fill="var(--h-cherry)" stroke="var(--h-cherry)"/>
<text x="608" y="37" dy="0.36em" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">test, 20%</text>
<g font-size="10" text-anchor="middle">
<rect x="5" y="66" width="102" height="22" rx="5" fill="var(--h-steel)"/>
<rect x="112" y="66" width="102" height="22" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="219" y="66" width="102" height="22" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="326" y="66" width="102" height="22" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="433" y="66" width="102" height="22" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="56" y="77" dy="0.36em" font-weight="700" fill="var(--md-default-fg-color)">score</text>
<rect x="5" y="94" width="102" height="22" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="112" y="94" width="102" height="22" rx="5" fill="var(--h-steel)"/>
<rect x="219" y="94" width="102" height="22" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="326" y="94" width="102" height="22" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="433" y="94" width="102" height="22" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="163" y="105" dy="0.36em" font-weight="700" fill="var(--md-default-fg-color)">score</text>
<text x="270" y="134" fill="var(--h-graphite)">… and so on, five rounds, each fold scored once</text>
</g>
<rect x="541" y="66" width="134" height="50" rx="8" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)" stroke-dasharray="4 3"/>
<text x="608" y="84" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-cherry)">locked</text>
<text x="608" y="100" text-anchor="middle" font-size="10" fill="var(--h-cherry)">opened once, at the end</text>
<text x="340" y="170" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Split once, with a fixed seed. Every choice is made from the cross-validation scores on the left.</text>
</svg>

## What you hand in

Two things, from the same work. The notebook is the full record. The poster is
the finding, for someone who has thirty seconds.

<div class="grid cards" markdown>

-   :material-notebook-outline: **The notebook**

    ---

    - The data contract at the top
    - Every phase of the pipeline, in order, with the decisions written down in
      markdown, not just made in code
    - Baseline and models scored the same way, with ranges
    - At least three failure cases, shown and explained
    - Runs top to bottom from cached data, wifi off, fixed seed

-   :material-file-image-outline: **The poster**

    ---

    - One page, landscape, exported to PDF
    - The finding as the title, a full sentence
    - One main result with its range and its baseline
    - One figure, labelled so it reads without you there
    - Where it fails, what did not work, one takeaway

</div>

The poster follows this layout. Fill the frame; do not redesign it.

<svg viewBox="0 0 680 384" role="img" aria-labelledby="p4-f3-title p4-f3-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="p4-f3-title">Poster layout</title>
<desc id="p4-f3-desc">A landscape page. A title band with the finding as a sentence. Below it, on the left, the question, the data and the approach; in the centre, the main result with its range and baseline; on the right, one key figure. Along the bottom: where it fails, what did not work, and the takeaway.</desc>
<rect x="5" y="8" width="670" height="318" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="20" y="22" width="640" height="42" rx="8" fill="var(--md-default-fg-color)"/>
<text x="340" y="39" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-bg-color)">title: your finding, as one sentence</text>
<text x="340" y="55" text-anchor="middle" font-size="10" fill="var(--md-default-bg-color)" opacity="0.75">team members · dataset · date</text>
<rect x="20" y="76" width="190" height="50" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="115" y="96" text-anchor="middle" font-size="11" font-weight="700" fill="var(--md-default-fg-color)">the question</text>
<text x="115" y="112" text-anchor="middle" font-size="10" fill="var(--h-graphite)">one sentence</text>
<rect x="20" y="134" width="190" height="50" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="115" y="154" text-anchor="middle" font-size="11" font-weight="700" fill="var(--md-default-fg-color)">the data</text>
<text x="115" y="170" text-anchor="middle" font-size="10" fill="var(--h-graphite)">source, rows, target</text>
<rect x="20" y="192" width="190" height="50" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="115" y="212" text-anchor="middle" font-size="11" font-weight="700" fill="var(--md-default-fg-color)">the approach</text>
<text x="115" y="228" text-anchor="middle" font-size="10" fill="var(--h-graphite)">the pipeline in one strip</text>
<rect x="222" y="76" width="220" height="166" rx="8" fill="var(--h-cherry)"/>
<text x="332" y="104" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">main result</text>
<text x="332" y="146" text-anchor="middle" font-size="24" font-weight="700" fill="#ffffff">recall 0.94</text>
<text x="332" y="170" text-anchor="middle" font-size="11" fill="#ffffff" opacity="0.9">give or take 0.04</text>
<text x="332" y="208" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.9">baseline: 0.81, give or take 0.07</text>
<rect x="454" y="76" width="206" height="166" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="557" y="100" text-anchor="middle" font-size="11" font-weight="700" fill="var(--md-default-fg-color)">one key figure</text>
<line x1="478" y1="206" x2="638" y2="206" stroke="var(--h-steel)" stroke-width="1.2"/>
<line x1="478" y1="206" x2="478" y2="118" stroke="var(--h-steel)" stroke-width="1.2"/>
<polyline points="484,190 504,176 524,182 544,158 564,164 584,140 604,146 628,124" fill="none" stroke="var(--h-cherry)" stroke-width="2"/>
<text x="557" y="228" text-anchor="middle" font-size="10" fill="var(--h-graphite)">labelled, readable without you</text>
<rect x="20" y="254" width="206" height="58" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="123" y="277" text-anchor="middle" font-size="11" font-weight="700" fill="var(--md-default-fg-color)">where it fails</text>
<text x="123" y="294" text-anchor="middle" font-size="10" fill="var(--h-graphite)">one honest example</text>
<rect x="237" y="254" width="206" height="58" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="340" y="277" text-anchor="middle" font-size="11" font-weight="700" fill="var(--md-default-fg-color)">what did not work</text>
<text x="340" y="294" text-anchor="middle" font-size="10" fill="var(--h-graphite)">and what it taught you</text>
<rect x="454" y="254" width="206" height="58" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="557" y="277" text-anchor="middle" font-size="11" font-weight="700" fill="var(--md-default-fg-color)">takeaway</text>
<text x="557" y="294" text-anchor="middle" font-size="10" fill="var(--h-graphite)">one sentence</text>
<text x="340" y="352" text-anchor="middle" font-size="12" fill="var(--h-graphite)">The big number in the middle never travels alone: it comes with its range and its baseline.</text>
<text x="340" y="372" text-anchor="middle" font-size="12" fill="var(--h-graphite)">If the poster needs zooming to read, it has too much on it.</text>
</svg>

Hand both in from your team repository:

```text
project-4/
├── data/raw/        # the dataset saved as CSV, committed
├── notebook.ipynb
└── poster.pdf
```

Three of the datasets ship inside scikit-learn. California Housing and German
Credit download the first time you load them. Either way, save the table to
`data/raw/` once and load it from there, so the notebook runs without the
internet.

## Same pipeline, different scoreboard

The pipeline does not change between the two kinds of problem. What changes is
how you keep score.

| | Regression | Classification |
|---|---|---|
| You predict | a number | a class, and the probability behind it |
| Baseline | the training mean, then a simple rule given in your tab | the majority class, then a simple rule given in your tab |
| Main metric | MAE, in the target's own units | precision, recall and F1 on the class you care about |
| Always show | residuals against predictions | the confusion matrix |
| The range | bootstrap the test set | bootstrap the test set |

## Regression: predict a number

=== "Team A: what is a district's house worth?"

    **The question.** Income, house age, rooms and location describe every
    district in California. How well can a linear model turn them into a
    house price, and where does a straight line stop being enough?

    **The data.** California Housing, 20,640 districts from the 1990 census,
    8 numeric columns. Target: median house value, in units of $100,000.

    ```python
    from sklearn import datasets
    housing = datasets.fetch_california_housing(as_frame=True)
    df = housing.frame
    ```

    **Baseline.** First the training mean for every district. Then a one-column
    rule: split median income into ten bands and predict the average value of
    the band, from the training data.

    **Must do.** Beat the income rule on MAE, with a range, and report it in
    dollars. Then report your error separately for cheap, middle and
    expensive districts, and say where the model struggles and why.

    **Watch out for.** The target is capped at 5.0: every district worth more
    than $500,000 is recorded as exactly $500,000. You will see it as a hard
    edge in your residual plot, so decide what to do about it and say so. House
    age is capped too, at 52. A few districts have absurd averages for rooms
    and occupancy, because only a handful of households live there. Latitude and
    longitude matter, but not in a straight line.

    **If you finish early.** Split by geography instead of at random: train on
    the north of the state and test on the south. How much worse does the
    model get, and what does that say about using it somewhere new?

=== "Team B: how fast will the disease progress?"

    **The question.** Ten measurements taken at a first visit, one number a year
    later. With only 442 patients, can a model do meaningfully better than the
    average, and can you prove it is not luck?

    **The data.** The diabetes dataset: age, sex, body mass index, blood
    pressure and six blood serum measurements (`s1` to `s6`) for 442 patients.
    Target: a measure of disease progression one year later.

    ```python
    from sklearn import datasets
    diabetes = datasets.load_diabetes(as_frame=True, scaled=False)
    df = diabetes.frame
    ```

    **Baseline.** First the training mean. Then a one-column rule: split body
    mass index into four bands and predict the average target of the band.

    **Must do.** Compare `LinearRegression`, `Ridge` and `Lasso`, choosing
    `alpha` with cross-validation. Beat the BMI rule on MAE, with a range, and
    say honestly whether the ranges of your best model and the baseline
    overlap. Show which measurements Lasso keeps as `alpha` grows.

    **Watch out for.** Load it with `scaled=False`. The default version was
    scaled using all 442 patients, test set included, which is a small leak.
    Scale it yourself, on the training set only. With so few rows a single
    split can swing your score a lot, so your ranges will be wide, and that is a
    finding, not a failure. `s1` and `s2` are strongly related, which is exactly
    where Ridge and Lasso disagree.

    **If you finish early.** Repeat the whole split 50 times with different
    seeds and plot the 50 test scores. How much could one unlucky split have
    fooled you?

## Classification: predict a class

=== "Team C: is the tumour malignant?"

    **The question.** Thirty measurements of cell nuclei from a scan. A model
    that misses a malignant tumour is far worse than one that raises a false
    alarm. How many malignant cases can you catch, and at what price?

    **The data.** The breast cancer dataset: 569 tumours, 30 numeric
    measurements. Target: malignant or benign.

    ```python
    from sklearn import datasets
    cancer = datasets.load_breast_cancer(as_frame=True)
    df = cancer.frame
    ```

    **Baseline.** Two of them. Always predict the majority class (benign).
    Then a one-column rule: pick a single measurement and a threshold from the
    training data, and call everything above it malignant.

    **Must do.** Beat the one-column rule on recall for malignant tumours,
    with a range, without letting precision collapse. Choose your probability
    threshold, defend it in one paragraph, and show the confusion matrix at
    that threshold.

    **Watch out for.** In this dataset `0` means malignant and `1` means
    benign, the opposite of what you would guess. If you do not flip it, your
    "recall" is measuring the wrong class. The 30 columns are 10 measurements
    each recorded three ways (mean, standard error, worst), so radius,
    perimeter and area say almost the same thing. With accuracy around 95
    percent for almost any model, accuracy tells you nothing here.

    **If you finish early.** Use an L1 penalty on logistic regression and find
    the smallest set of measurements that keeps recall within your range.

=== "Team D: which digit is it?"

    **The question.** Ten classes instead of two, and pixels instead of
    columns with names. When does a straight boundary stop being enough, and
    which digits does every model confuse?

    **The data.** The digits dataset: 1,797 handwritten digits, each an 8 × 8
    image, so 64 pixel columns with values from 0 to 16. Target: the digit,
    0 to 9.

    ```python
    from sklearn import datasets
    digits = datasets.load_digits(as_frame=True)
    df = digits.frame
    ```

    **Baseline.** Always predicting one digit scores about 10 percent, so that
    is not much of a bar. The real baseline: average all training images of
    each digit, and label a new image with the closest average
    (`neighbors.NearestCentroid`). Expect it to be right close to 9 times in 10.

    **Must do.** Beat the average-image rule on macro F1, with a range. Compare
    a linear model with an SVM using the RBF kernel. Show the 10 × 10
    confusion matrix and at least nine misclassified images, and say what they
    have in common.

    **Watch out for.** With ten classes, precision, recall and F1 need an
    `average=` argument: use `"macro"` so every digit counts equally, and say
    so. The images come from a few dozen writers, so a random split puts the
    same handwriting on both sides, and your score is probably a little
    optimistic for a new writer. Scale the pixels before the SVM.

    **If you finish early.** Reshape the 64 weights a linear model learned for
    each digit back into an 8 × 8 image and plot all ten. What is each one
    looking for?

=== "Team E: will this customer repay?"

    **The question.** A bank has to decide who gets a loan. Lending to someone
    who does not repay costs much more than turning away someone who would
    have. Can a model make those decisions more cheaply than a simple rule?

    **The data.** The German Credit dataset: 1,000 loan applicants described by
    20 columns, most of them categories. Target: good or bad credit risk.

    ```python
    from sklearn import datasets
    credit = datasets.fetch_openml("credit-g", version=1, as_frame=True)
    df = credit.frame
    ```

    **Baseline.** Always predict "good". Then a one-column rule: flag an
    applicant as "bad" based on their checking account status alone, choosing
    the rule from the training data.

    **Must do.** Beat the one-column rule on F1 for "bad", with a range. Then
    use the cost the dataset comes with: approving a bad applicant costs 5,
    rejecting a good one costs 1. Choose the probability threshold that gives
    the lowest total cost on cross-validation, and report the test cost next to
    the baseline's.

    **Watch out for.** Thirteen columns are text, so they need encoding, and
    the encoders and scalers must learn from the training set only. A pipeline
    makes that automatic. Only 3 in 10 applicants are "bad", so the always-good
    baseline scores 70 percent accuracy and catches nobody. Two columns,
    `personal_status` (sex combined with marital status) and `foreign_worker`,
    describe who someone is rather than what they did. Should a credit model use
    them?

    **If you finish early.** Drop those two columns, retrain, and measure what
    it costs in money and in F1. Is the difference larger than your range?

## What you will be asked

Every team gets these questions on presentation day. Prepare the answers in
the notebook, not on the morning.

1. What does your baseline score, and how much did you beat it by, give or take?
2. How did you split, and how do you know nothing from the test set leaked in?
3. Show us one case where your model is badly wrong. Why is it wrong there?
4. What did you try that did not work, and what did it tell you?
5. If someone used this model tomorrow, what should worry them?

!!! tip "Before you train anything"

    Each team writes down two numbers on paper: what you expect the baseline
    to score, and what you expect your best model to score. Fold it up and give
    it to me.

    Most teams overestimate the model and underestimate the baseline. Finding
    out which way you were wrong is half of what this project is for.

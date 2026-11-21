# Project 4: Beat the baseline

<span class="badge badge--time">Two weeks</span>
<span class="badge badge--level">Classic models</span>
<span class="badge">Groups of 3 or 4</span>
<span class="badge">Presented 14 December</span>

Project 3 banned models. This one is built around them. But a model is only
worth something next to the simplest guess you could have made without one, so
that is where you start, and that is the number you have to beat.

## What you are doing

Two groups predict a number. Two groups predict a class. Everybody follows the
same pipeline, and everybody answers the same question at the end: how much
better than the obvious guess is your model, give or take?

<svg viewBox="0 0 680 240" role="img" aria-labelledby="p4-f1-title p4-f1-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="p4-f1-title">The pipeline every group follows</title>
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
<text x="333" y="106" text-anchor="middle" font-size="10" fill="var(--h-graphite)">then a tree</text>
<line x1="377" y1="83" x2="384" y2="83" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#p4a)"/>
<rect x="385" y="40" width="86" height="86" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="428" y="68" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">tune</text>
<text x="428" y="92" text-anchor="middle" font-size="10" fill="var(--h-graphite)">on validation,</text>
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

Same objectives for every group. What they mean for your data is in your tab.

| | Objective |
|---|---|
| 0 | Load the data, write the data contract, look at what is there |
| 1 | Split by time into train, validation and test, before modelling anything |
| 2 | Score a baseline, exactly the way you will score your models |
| 3 | Train at least two models: one linear, one tree-based |
| 4 | Tune on the validation set, or with cross-validation on the training set |
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

All four datasets are ordered in time, which makes the split the first place
where projects go wrong.

<svg viewBox="0 0 680 160" role="img" aria-labelledby="p4-f2-title p4-f2-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="p4-f2-title">Splitting by time</title>
<desc id="p4-f2-desc">The earliest data is for training, the next block is for validation, and the latest block is the test set, opened once.</desc>
<defs><marker id="p4b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<rect x="5" y="30" width="430" height="52" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="220" y="52" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">train</text>
<text x="220" y="70" text-anchor="middle" font-size="10" fill="var(--h-graphite)">fit every model here</text>
<rect x="441" y="30" width="110" height="52" rx="10" fill="var(--h-surface)" stroke="var(--h-steel)"/>
<text x="496" y="52" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">validation</text>
<text x="496" y="70" text-anchor="middle" font-size="10" fill="var(--h-graphite)">choose and tune</text>
<rect x="557" y="30" width="118" height="52" rx="10" fill="var(--h-cherry)" stroke="var(--h-cherry)"/>
<text x="616" y="52" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">test</text>
<text x="616" y="70" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.9">opened once</text>
<line x1="5" y1="98" x2="672" y2="98" stroke="var(--h-steel)" stroke-width="1.4" marker-end="url(#p4b)"/>
<text x="5" y="116" font-size="10" fill="var(--h-space)">earlier</text>
<text x="668" y="116" text-anchor="end" font-size="10" fill="var(--h-space)">later</text>
<text x="340" y="146" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Split by date, not by shuffling rows. A random split lets the model see the days around the ones it is tested on.</text>
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
<text x="340" y="55" text-anchor="middle" font-size="10" fill="var(--md-default-bg-color)" opacity="0.75">group members · dataset · date</text>
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
<text x="332" y="146" text-anchor="middle" font-size="24" font-weight="700" fill="#ffffff">MAE 1.6 °C</text>
<text x="332" y="170" text-anchor="middle" font-size="11" fill="#ffffff" opacity="0.9">give or take 0.1</text>
<text x="332" y="208" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.9">baseline: 2.3 °C, give or take 0.1</text>
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

Hand both in from your group repository:

```text
project-4/
├── data/raw/        # cached source files, committed
├── notebook.ipynb
└── poster.pdf
```

## Same pipeline, different scoreboard

The pipeline does not change between the two kinds of problem. What changes is
how you keep score.

| | Regression | Classification |
|---|---|---|
| You predict | a number | a class, and the probability behind it |
| Baseline | a simple rule given in your tab | the majority class, then a simple rule |
| Main metric | MAE, in the target's own units | precision, recall and F1 on the class you care about |
| Always show | residuals against predictions | the confusion matrix |
| The range | bootstrap the test set | bootstrap the test set |

## Regression: predict a number

=== "Group A: what will tomorrow's high be?"

    **The question.** A forecast that says "tomorrow will be like today" is
    right more often than you would think. Can a model do meaningfully better
    for Tirana?

    **The data.** Daily weather for Tirana from the
    [Open-Meteo archive API](https://open-meteo.com/en/docs/historical-weather-api),
    1991 to 2025. You collected this kind of data in Project 3. Target:
    tomorrow's maximum temperature.

    **Baseline.** Persistence: tomorrow's high equals today's.

    **Must do.** Beat persistence on MAE and check whether the two ranges
    overlap. If they do, say so: that is a finding too. Then report your error separately for each season, and say which season is
    hardest and why.

    **Watch out for.** Shifting the target the wrong way, so that tomorrow
    quietly ends up among your features. Day of year is circular: 31 December
    sits next to 1 January, not 364 days away from it. And ERA5 is a grid
    reconstruction, as Group C found in Project 3.

    **If you finish early.** Predict three days ahead instead of one, and watch
    what happens to the gap between your model and persistence.

=== "Group B: how busy will the bikes be?"

    **The question.** A bike-sharing system has to move bikes to where people
    will want them before they want them. How well can you predict demand an
    hour at a time?

    **The data.** The
    [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset):
    hourly rentals in Washington, D.C. for 2011 and 2012, with weather and
    calendar columns. One download, cached. Target: `cnt`, rentals in the hour.

    **Baseline.** The average count for that hour of the day, from the
    training data.

    **Must do.** Beat the hourly average on MAE, with a range. Then compare
    your errors on working days against weekends and say what the model finds
    harder.

    **Watch out for.** `casual` plus `registered` equals `cnt`, so either one
    as a feature is the answer in disguise. Counts are skewed, so try a log
    target, but report MAE back in rentals. Demand grew between 2011 and 2012,
    which your test set will notice even if your training set does not.

    **If you finish early.** Model `casual` and `registered` separately and
    find out which kind of rider the weather affects more.

## Classification: predict a class

=== "Group C: will it rain tomorrow?"

    **The question.** The simplest useful forecast there is. Can a model say
    whether tomorrow is a rain day better than a rule anyone could apply?

    **The data.** Daily weather for one Albanian city from the
    [Open-Meteo archive API](https://open-meteo.com/en/docs/historical-weather-api),
    1991 to 2025. Target: whether tomorrow is a rain day. You choose the
    millimetre threshold and defend it, as Group D did in Project 3.

    **Baseline.** Two of them. Always predict the majority class, then
    persistence: rain tomorrow if it rained today.

    **Must do.** Beat persistence on F1 for rain days, with a range. Show the
    confusion matrix for summer and for winter separately.

    **Watch out for.** Two thresholds that are easy to confuse: the millimetres
    that make a rain day, and the probability above which you predict rain. In
    July almost every day is dry, so a model that never predicts rain scores
    high accuracy and is useless. Look at recall before you celebrate.

    **If you finish early.** Move the probability threshold and plot precision
    against recall. Pick the threshold you would use to plan an outdoor
    wedding, and the one you would use to plan a harvest.

=== "Group D: podium or not?"

    **The question.** Group B in Project 3 found that the grid explains a lot.
    Can a model that knows more than the grid position say who finishes on the
    podium?

    **The data.** Race results and qualifying from the
    [Jolpica-F1 API](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md),
    2000 to 2025, cached under the same rules as Project 3. Target: whether a
    driver finishes in the top three.

    **Baseline.** Always predict "no podium", then: podium if the driver starts
    in the top three.

    **Must do.** Beat the grid rule on F1 for podium finishes, with a range.
    Then name the kind of race where your model and the grid rule disagree most.

    **Watch out for.** Only features known before the lights go out. Laps led,
    finishing status or anything else from the race itself is leakage. Rolling
    form, such as points over the last three races, must use previous races
    only. Split by season, never by row, so that the twenty drivers of one race
    stay on the same side of the split. Only 3 in 20 rows are podiums.

    **If you finish early.** Train on 2000 to 2013 and test on the hybrid era
    from 2014, and see how much of what the model learned survives a change of
    rules.

## What you will be asked

Every group gets these questions on presentation day. Prepare the answers in
the notebook, not on the morning.

1. What does your baseline score, and how much did you beat it by, give or take?
2. How did you split, and how do you know nothing from the test set leaked in?
3. Show us one case where your model is badly wrong. Why is it wrong there?
4. What did you try that did not work, and what did it tell you?
5. If someone used this model tomorrow, what should worry them?

!!! tip "Before you train anything"

    Each group writes down two numbers on paper: what you expect the baseline
    to score, and what you expect your best model to score. Fold it up and give
    it to me.

    Most groups overestimate the model and underestimate the baseline. Finding
    out which way you were wrong is half of what this project is for.

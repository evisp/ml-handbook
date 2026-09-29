# Project 2: One page, one finding

<span class="badge badge--time">Storytelling</span>
<span class="badge badge--level">No models</span>
<span class="badge">Groups of 3 or 4</span>
<span class="badge">Presented 19 October</span>

Your group gets a dataset and a question. You answer it with pandas and charts,
and then you fit the whole answer onto a single page.

## What you are doing and why

You can already make a chart. What you cannot yet do is decide which three
charts matter and throw the rest away.

That is the skill this project is built around. A one-page limit is the only
reliable way to learn it, because six charts is always easier than three. The
same constraint runs through professional life: nobody reads your notebook,
they read the one slide you put in front of them, and what you chose to leave
off it is most of the work.

There are no models in this project. No training, no predictions, no accuracy.
If you find yourself importing scikit-learn you have misread the brief.

<svg viewBox="0 0 680 266" role="img" aria-labelledby="d1-title d1-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="d1-title">The two deliverables and what each one rewards</title>
<desc id="d1-desc">The notebook rewards thoroughness and shows everything you tried. The poster rewards ruthlessness and shows only what survived.</desc>
<rect x="10" y="30" width="320" height="176" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="170" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">the notebook</text>
<text x="170" y="62" text-anchor="middle" font-size="15" font-weight="700" fill="var(--md-default-fg-color)">show everything</text>
<text x="170" y="92" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">every hypothesis, including the ones</text>
<text x="170" y="110" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">that did not survive</text>
<text x="170" y="132" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">every cleaning decision and why</text>
<text x="170" y="154" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">as many charts as you need</text>
<text x="170" y="184" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-graphite)">rewards being thorough</text>
<rect x="350" y="30" width="320" height="176" rx="12" fill="var(--h-cherry)"/>
<text x="510" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="var(--h-cherry)">the poster</text>
<text x="510" y="62" text-anchor="middle" font-size="15" font-weight="700" fill="#ffffff">show almost nothing</text>
<text x="510" y="92" text-anchor="middle" font-size="11.5" fill="#ffffff" opacity="0.92">one question, one answer</text>
<text x="510" y="114" text-anchor="middle" font-size="11.5" fill="#ffffff" opacity="0.92">three charts, and not a fourth</text>
<text x="510" y="136" text-anchor="middle" font-size="11.5" fill="#ffffff" opacity="0.92">one page, no scrolling, no excuses</text>
<text x="510" y="184" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">rewards being ruthless</text>
<text x="340" y="238" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Most people are naturally good at one of these. The project exists to force you into the other one.</text>
<text x="340" y="258" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Deciding what to leave out is the harder skill.</text>
</svg>

## Your group and your question

Each group has a different dataset and a different question. The questions are
deliberately specific: none of them is "explore this data".

<div class="grid cards" markdown>

-   :material-ferry: **Group 1: Titanic**

    ---

    **Could Jack have been saved?**

    What would have had to be different about him? Answer with the data, not
    with the film.

    [Dataset](https://www.kaggle.com/c/titanic/data) · 891 passengers, 12 columns

-   :material-music: **Group 2: Spotify**

    ---

    **What makes a song popular?**

    You have every audio feature Spotify measures. Find out how much of
    popularity they actually explain.

    [Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) · 114,000 tracks, 114 genres

-   :material-television-play: **Group 3: Netflix**

    ---

    **What kind of company is Netflix now, and when did it change?**

    The catalogue has a history in it. Find the turning point and prove it.

    [Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows) · 8,807 titles, 12 columns

-   :material-school: **Group 4: Student performance**

    ---

    **What predicts a final grade, and which of those would you refuse to use?**

    Two questions. The second one is the harder half and carries equal weight.

    [Dataset](https://archive.ics.uci.edu/dataset/320/student+performance) · 395 and 649 students, 33 columns

</div>

!!! warning "Every one of these datasets has a trap in it"

    Not a mistake in the file. A way of looking at it that produces a confident,
    wrong answer, and that plenty of published analyses of that exact dataset
    have fallen into.

    You are not being told which. Finding it is part of the work, and the
    section on things that are wrong with your data is where you report it.

## Deliverable one: the poster

One page, landscape, exported as PDF. It holds five things and nothing else.

<svg viewBox="0 0 680 350" role="img" aria-labelledby="d2-title d2-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="d2-title">What goes on the poster</title>
<desc id="d2-desc">A single landscape page holding the question as a headline, the answer in one large sentence, three charts each with a finding as its title and one line of explanation, and a note on what the data could not answer.</desc>
<rect x="60" y="20" width="560" height="290" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry)" stroke-width="2"/>
<rect x="80" y="38" width="520" height="40" rx="6" fill="var(--h-cherry)"/>
<text x="340" y="58" dy="0.36em" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff">the question, as a headline</text>
<rect x="80" y="86" width="520" height="44" rx="6" fill="var(--md-default-bg-color)" stroke="var(--h-cherry-line)" stroke-width="2"/>
<text x="340" y="102" text-anchor="middle" font-size="13" font-weight="700" fill="var(--h-cherry)">the answer, in one sentence, large</text>
<text x="340" y="120" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">if you cannot fit it in one sentence you have not finished thinking</text>
<rect x="80" y="140" width="168" height="120" rx="6" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="164" y="160" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--md-default-fg-color)">a title that states</text>
<text x="164" y="174" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--md-default-fg-color)">the finding</text>
<rect x="96" y="182" width="136" height="48" rx="4" fill="var(--h-cherry)" opacity="0.16"/>
<text x="164" y="210" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-cherry)">chart 1</text>
<text x="164" y="246" text-anchor="middle" font-size="9.5" fill="var(--h-space)">one line of explanation</text>
<rect x="256" y="140" width="168" height="120" rx="6" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="340" y="160" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--md-default-fg-color)">a title that states</text>
<text x="340" y="174" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--md-default-fg-color)">the finding</text>
<rect x="272" y="182" width="136" height="48" rx="4" fill="var(--h-cherry)" opacity="0.16"/>
<text x="340" y="210" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-cherry)">chart 2</text>
<text x="340" y="246" text-anchor="middle" font-size="9.5" fill="var(--h-space)">one line of explanation</text>
<rect x="432" y="140" width="168" height="120" rx="6" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="516" y="160" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--md-default-fg-color)">a title that states</text>
<text x="516" y="174" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--md-default-fg-color)">the finding</text>
<rect x="448" y="182" width="136" height="48" rx="4" fill="var(--h-cherry)" opacity="0.16"/>
<text x="516" y="210" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-cherry)">chart 3</text>
<text x="516" y="246" text-anchor="middle" font-size="9.5" fill="var(--h-space)">one line of explanation</text>
<rect x="80" y="268" width="520" height="30" rx="6" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)" stroke-dasharray="5 4"/>
<text x="340" y="283" dy="0.36em" text-anchor="middle" font-size="11" font-weight="700" fill="var(--h-graphite)">what this data could not tell you</text>
<text x="340" y="332" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Five elements. A fourth chart is not an improvement, it is a failure to choose.</text>
</svg>

1. **The question**, as a headline.
2. **The answer**, in one sentence, large enough to read from two metres away.
3. **Three charts. Not four.** Each with a title that states a finding rather
   than naming the variables. "Third class men had almost no chance" beats
   "Survival by class and sex".
4. **One line under each chart** saying what it shows.
5. **One line on what this data could not tell you.**

Rules:

- **Every chart must come from your own code.** Build the layout wherever you
  like, in slides, in Canva, or with `plt.subplots`, but the charts are yours.
- Readable when projected. If the axis labels vanish on the screen, the chart
  has failed at its only job.
- Your group's names on it.

The hardest rule is the third one. You will want six charts. Cutting to three is
where the learning is, and it is also where the arguments in your group will
happen.

## Deliverable two: the notebook

One notebook, in a group GitHub repository, with sections in this order. It is
the first-hour workflow from the handbook, done properly.

1. **Load and profile.** Shape, dtypes, missing values, and what the columns
   mean. Two commands and a short paragraph.

2. **Three hypotheses, written before you explore.** This section must come
   first in the notebook. If you write them afterwards you will write ones you
   already know are true, and you will have proved nothing.

3. **Cleaning, with a reason per decision.** One line each: what you did and
   why. Types, missing values, duplicates, anything you dropped.

4. **Exploration.** Distributions first, then relationships. This is where the
   charts that did not make the poster live.

5. **Each hypothesis tested,** marked survived or rejected. A rejected
   hypothesis is a result. Do not quietly delete it.

6. **Three things wrong with this data,** with evidence for each. See below.

7. **Conclusion and limitations.** The answer to your question, and an honest
   paragraph on what would change it.

## Three things wrong with this data

Every one of these four datasets has documented problems. Your job is to find
three of them in yours and prove each with a line of code or a chart.

The kinds of thing that count:

- A column that says one thing and means another.
- Missing values that are not missing at random.
- A group that is far too small to conclude anything about.
- A number that cannot be true.
- A column that would not have existed at the moment you would need to use it.
- Whole populations absent from the data entirely.

The kind of thing that does not count: "there are some missing values". Say
which, how many, and why it matters for your question.

This section is the difference between reading a dataset and interrogating one.
It is also the habit that makes you useful in a job, because the first thing a
real dataset does is lie to you.

## How it is marked

| Weight | What is assessed |
|---|---|
| 50 | The notebook: hypotheses written first, cleaning justified, exploration that goes somewhere, honest results |
| 30 | The poster's finding: is the answer supported by what you showed? |
| 20 | The poster's craft: is it readable, edited, and does it respect the five-element limit? |

The ordering matters. A beautiful poster with a weak finding scores lower than a
plain one with a strong finding.

**What earns marks fastest**: rates where rates belong rather than raw counts,
denominators stated, axes that start at zero when they should, and a limitations
section that admits something real.

**What loses them fastest**: a chart with no title, a claim with no chart, a
notebook where the hypotheses were obviously written last, and six charts on the
poster.

## Timeline

<svg viewBox="0 0 680 216" role="img" aria-labelledby="d3-title d3-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="d3-title">The three weeks to presentation day</title>
<desc id="d3-desc">Week one, write the hypotheses and clean the data. Week two, explore and test them. Week three, cut to three charts and build the poster. Presentation on the nineteenth of October.</desc>
<defs><marker id="tl" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<line x1="70" y1="96" x2="600" y2="96" stroke="var(--h-steel)" stroke-width="2"/>
<circle cx="130" cy="96" r="13" fill="var(--h-cherry)"/>
<text x="130" y="96" dy="0.36em" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">1</text>
<text x="130" y="64" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">week 1</text>
<text x="130" y="128" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">hypotheses first,</text>
<text x="130" y="143" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">then clean</text>
<circle cx="290" cy="96" r="13" fill="var(--h-cherry)"/>
<text x="290" y="96" dy="0.36em" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">2</text>
<text x="290" y="64" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">week 2</text>
<text x="290" y="128" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">explore and</text>
<text x="290" y="143" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">test them</text>
<circle cx="450" cy="96" r="13" fill="var(--h-cherry)"/>
<text x="450" y="96" dy="0.36em" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">3</text>
<text x="450" y="64" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">week 3</text>
<text x="450" y="128" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">cut to three,</text>
<text x="450" y="143" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">build the poster</text>
<rect x="466" y="74" width="160" height="44" rx="10" fill="var(--h-cherry)"/>
<text x="546" y="90" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">19 October</text>
<text x="546" y="108" text-anchor="middle" font-size="10.5" fill="#ffffff" opacity="0.9">gallery walk</text>
<text x="340" y="184" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Groups that leave the poster to the last evening produce six charts and no argument.</text>
<text x="340" y="204" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Cutting takes longer than plotting.</text>
</svg>

A suggestion on the order of work, from groups who have done this before: spend
week one on the data and resist the urge to plot anything pretty. The good
charts come from understanding, and understanding comes from cleaning.

## Before you present, check

1. The poster has exactly three charts.
2. Somebody outside your group can read the answer in under thirty seconds.
3. Every chart title states a finding, not a pair of variable names.
4. Your notebook's hypotheses section is the first thing after the profiling,
   and you did not edit it after exploring.
5. Every cleaning decision has a reason next to it.
6. You have three genuine problems with the data, each with evidence.
7. Your limitations paragraph says something that costs you.
8. The repository link works from a private browser window.

## Questions to think about

These are worth arguing about in your group, and some of them will come back at
you on the nineteenth.

1. You cut three charts to get to three. What did the cut ones say, and are you
   confident none of them contradicted your answer?
2. Your question has a yes or no shape. Did the data actually give you one, or
   did you round an unclear answer into a clean one?
3. If another group had your dataset and your question, would they reach your
   conclusion? Which of your decisions would they most likely have made
   differently?
4. Somebody reads only your poster and acts on it. What is the worst decision
   they could make that your poster did not warn them about?
5. You found three things wrong with the data. Would you still publish your
   finding? What would you say alongside it?

## A note on tools

pandas and a plotting library are required. NumPy where it helps, and there is
no prize for using it if it does not.

The poster layout can be built anywhere. If you want it entirely in Python,
`plt.subplots` with `fig.savefig("poster.pdf", bbox_inches="tight")` produces a
single-page PDF and gives you complete control. If you would rather assemble it
in slides or Canva, export your charts as PNGs at `dpi=200` and place them.
Neither route scores higher.

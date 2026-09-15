# Working with data

<span class="badge badge--time">25 min</span>
<span class="badge badge--level">Concepts</span>
<span class="badge">Read before you collect anything</span>

You will spend far more of your career on data than on models. This page covers
the first two boxes of the lifecycle, which is where projects are usually won
or lost.

## Zooming in

The five stage loop hides how much lives inside the first two boxes.

<svg viewBox="0 0 680 244" role="img" aria-labelledby="zoom-title zoom-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="zoom-title">Inside collect and prepare</title>
<desc id="zoom-desc">The collect and prepare stages of the lifecycle expand into five steps: find sources, check consent, label, clean, and split the data.</desc>
<rect x="5" y="10" width="118" height="40" rx="8" fill="var(--h-cherry)"/>
<text x="64" y="30" dy="0.36em" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">Collect</text>
<rect x="143" y="10" width="118" height="40" rx="8" fill="var(--h-cherry)"/>
<text x="202" y="30" dy="0.36em" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">Prepare</text>
<rect x="281" y="10" width="118" height="40" rx="8" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="340" y="30" dy="0.36em" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Train</text>
<rect x="419" y="10" width="118" height="40" rx="8" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="478" y="30" dy="0.36em" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Evaluate</text>
<rect x="557" y="10" width="118" height="40" rx="8" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="616" y="30" dy="0.36em" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Deploy</text>
<polygon points="5,58 261,58 675,112 5,112" fill="var(--h-cherry-wash)"/>
<rect x="5" y="126" width="126" height="70" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="68" y="154" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">Find sources</text>
<text x="68" y="174" text-anchor="middle" font-size="10" fill="var(--h-graphite)">where examples</text>
<text x="68" y="187" text-anchor="middle" font-size="10" fill="var(--h-graphite)">come from</text>
<rect x="141" y="126" width="126" height="70" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="204" y="154" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">Check consent</text>
<text x="204" y="174" text-anchor="middle" font-size="10" fill="var(--h-graphite)">may you actually</text>
<text x="204" y="187" text-anchor="middle" font-size="10" fill="var(--h-graphite)">use this</text>
<rect x="277" y="126" width="126" height="70" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="340" y="154" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">Label</text>
<text x="340" y="174" text-anchor="middle" font-size="10" fill="var(--h-graphite)">who decides</text>
<text x="340" y="187" text-anchor="middle" font-size="10" fill="var(--h-graphite)">the answer</text>
<rect x="413" y="126" width="126" height="70" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="476" y="154" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">Clean</text>
<text x="476" y="174" text-anchor="middle" font-size="10" fill="var(--h-graphite)">fix errors, drop</text>
<text x="476" y="187" text-anchor="middle" font-size="10" fill="var(--h-graphite)">duplicates</text>
<rect x="549" y="126" width="126" height="70" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry)"/>
<text x="612" y="154" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">Split</text>
<text x="612" y="174" text-anchor="middle" font-size="10" fill="var(--h-graphite)">train, validation,</text>
<text x="612" y="187" text-anchor="middle" font-size="10" fill="var(--h-graphite)">test</text>
<text x="340" y="222" text-anchor="middle" font-size="12" font-weight="600" fill="var(--h-graphite)">none of this is glamorous, and all of it decides whether the project works</text>
</svg>

## What data looks like

Two distinctions, and you need both to describe a dataset.

**Structured or unstructured.** Structured data arrives in rows and columns
with a known meaning per column: a spreadsheet of sales, sensor readings over
time, a database table. Unstructured data does not: text, images, audio,
video, free-form notes. Most of the world's data is unstructured, and it is the
reason deep learning mattered.

**Labelled or unlabelled.** Labelled means each example carries the answer you
want predicted. Unlabelled means it does not. This is the distinction that
decides your learning family, and labelling is usually the expensive part.
Collecting ten thousand photos is easy. Getting ten thousand correct labels is
a project.

!!! tip "Labels are opinions written down"

    Somebody decided that this email is spam and that face is smiling. Ask two
    people to label the same ambiguous examples and they will disagree on some
    of them. The rate at which labellers agree is a real measurement, and if
    humans cannot agree on an example, no model will learn it reliably.

## Three splits, three jobs

Before training anything, you divide your data into three parts with different
jobs. The analogy is exams, and it is exact.

<svg viewBox="0 0 680 210" role="img" aria-labelledby="split-title split-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="split-title">Training, validation and test data</title>
<desc id="split-desc">Training data is the study material the model learns from. Validation data is a practice quiz used to tune choices. Test data is the final exam, touched once at the very end.</desc>
<rect x="10" y="30" width="210" height="112" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="30" y="56" font-size="11" font-weight="700" fill="var(--h-space)">about 60 percent</text>
<text x="30" y="82" font-size="15" font-weight="700" fill="var(--md-default-fg-color)">Training data</text>
<text x="30" y="104" font-size="12" font-style="italic" fill="var(--h-cherry)">study the material</text>
<text x="30" y="126" font-size="11.5" fill="var(--h-graphite)">the model learns from this</text>
<rect x="235" y="30" width="210" height="112" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="255" y="56" font-size="11" font-weight="700" fill="var(--h-space)">about 20 percent</text>
<text x="255" y="82" font-size="15" font-weight="700" fill="var(--md-default-fg-color)">Validation data</text>
<text x="255" y="104" font-size="12" font-style="italic" fill="var(--h-cherry)">the practice quiz</text>
<text x="255" y="126" font-size="11.5" fill="var(--h-graphite)">you tune your choices here</text>
<rect x="460" y="30" width="210" height="112" rx="12" fill="var(--h-cherry)"/>
<text x="480" y="56" font-size="11" font-weight="700" fill="#ffffff" opacity="0.85">about 20 percent</text>
<text x="480" y="82" font-size="15" font-weight="700" fill="#ffffff">Test data</text>
<text x="480" y="104" font-size="12" font-style="italic" fill="#ffffff" opacity="0.92">the final exam</text>
<text x="480" y="126" font-size="11.5" fill="#ffffff" opacity="0.88">touched once, at the end</text>
<text x="340" y="176" text-anchor="middle" font-size="12.5" font-weight="600" fill="var(--md-default-fg-color)">an example that appears in more than one split turns a test into a memory check</text>
<text x="340" y="196" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">the percentages are a starting point, not a rule</text>
</svg>

The one that needs explaining is validation. If you only had training and test
data, you would train, check the test score, change something, train again,
check again. After twenty rounds of that you have tuned your model to the test
set through your own decisions. The score is no longer honest, and nobody
cheated on purpose.

The validation set absorbs that. You tune against it as much as you like. The
test set stays sealed until you are finished, which is what keeps its number
meaningful.

!!! warning "Split before you clean, not after"

    If you remove duplicates or fill in missing values across the whole dataset
    and split afterwards, information from your test set has leaked into your
    training. This is called leakage, it is common, and it produces excellent
    scores that collapse the moment the model meets real data.

## Variety beats volume

Here is a question worth answering before you read on. If you duplicate every
image in your dataset ten times, do you have more data?

No. You have the same data, ten times. Nothing new has been shown to the model.
What you have actually done is made the model more confident about exactly the
examples you already had, which is the opposite of what you want.

The reason is the same as the reason you revise for an exam by doing varied
problems rather than reading one answer forty times. What makes data useful is
coverage of the ways the real input can differ: lighting, angles, phrasings,
handwriting, accents, the unusual cases.

Which is why fifty varied photographs beat five hundred near-identical frames
from one webcam burst. In your first project this is the single decision that
separates a good result from a meaningless one.

## Where data comes from, and what to check

Four common sources, each with its own problem.

<div class="grid cards" markdown>

-   :material-camera-outline: **You collect it**

    ---

    Full control over variety and consent. Slow, small, and it will carry the
    habits of whoever collected it.

-   :material-database-outline: **It already exists in the organisation**

    ---

    Logs, records, past transactions. Plentiful and free, but it was recorded
    for another purpose and reflects how things used to be done.

-   :material-earth: **Public datasets**

    ---

    Fast to start and comparable with others' results. Check the licence, and
    check who is represented before trusting it.

-   :material-robot-outline: **Generated or synthetic**

    ---

    Cheap and unlimited. It carries the biases of whatever generated it, so it
    is a supplement rather than a foundation.

</div>

Whatever the source, run this checklist before you build anything on it.

| Check | The question to ask |
|---|---|
| **Coverage** | Who or what is represented, and who is missing entirely? |
| **Balance** | How many examples per class? Is one class rare? |
| **Label quality** | Who labelled it, by what rule, and would a second person agree? |
| **Duplicates** | Are there near-identical examples, and could they land in two splits? |
| **Consent and licence** | Are you allowed to use this, for this purpose? |
| **Age** | When was it collected, and has the world moved since? |
| **Missing values** | What is absent, and is it absent at random or for a reason? |

The last one is subtler than it sounds. Missing data is often informative.
Income is missing more often for people who declined to answer, and declining
is not random.

## Three ways real data is broken

**Noisy.** Wrong labels, blurry images, typos, sensor errors. The real pattern
is still there, buried under interference. When Apple launched its own Maps app
in 2012, the underlying geographic data was merged from several sources without
enough verification. Towns appeared in the wrong place, landmarks were
mislabelled, and the company ended up issuing a public apology. The algorithms
were not the problem.

**Biased.** The data reflects an unfair or incomplete world. Amazon spent
several years building a system to rank job applicants, and found it
systematically downgraded CVs from women for technical roles. It had learned
from a decade of the company's own hiring, in which most technical hires were
men. The system worked exactly as designed. The design was to reproduce the
past, and the past was the problem. The project was abandoned.

**Imbalanced.** One class dominates. If 95 percent of emails are not spam, a
model that says "not spam" every time scores 95 percent and is useless.
Imbalance also causes quieter failures: a widely reported study of pedestrian
detection models found they performed worse on darker skinned pedestrians,
consistent with training sets that contained far more lighter skinned examples.
The model was not asked to treat groups differently. It was just shown more of
some than others.

Three different problems, one shared lesson. None of them is fixed by a better
algorithm.

## Questions to sit with

1. You have a hundred photographs of yourself and none of anyone else. What
   will your model learn, and what would you predict happens when a stranger
   uses it?
2. Duplicating data does not add information. Is a slightly rotated copy of an
   image a duplicate or a new example? What is your reasoning?
3. If an organisation's records reflect how it used to behave, can a model
   trained on them ever help it change? What would you have to do differently?
4. Two labellers disagree on 15 percent of your examples. Is that a labelling
   problem, a definition problem, or a sign the task itself is not well posed?
5. The spam filter that always says "not spam" scores 95 percent. It is also
   the model nobody would ship. What does that tell you about accuracy as a
   target?

## Next

Data is where most harm enters a system, which makes the next page the
practical continuation of this one rather than a change of subject.

[Ethics and responsibility](ethics-and-responsibility.md){ .h-button }

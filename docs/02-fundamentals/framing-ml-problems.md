# Framing a machine learning problem

<span class="badge badge--time">25 min</span>
<span class="badge badge--level">Concepts</span>
<span class="badge">Read before session two</span>

Nobody will hand you a well posed machine learning problem. They will say "can
we use AI to reduce customer complaints". Turning that into something you can
build is the first skill of the job, and this page is the method.

## Why this matters

Most failed machine learning projects were not failures of modelling. They were
projects where nobody agreed on what was being predicted, or on how anyone
would know it had worked, until months of effort had already gone in.

Ten minutes of framing at the start prevents that. It is also the part of the
job that never gets automated, because it is the part that requires knowing
what people actually want.

## The formal version

Tom Mitchell gave machine learning its standard definition in 1997, and it
holds up because it is really a checklist. A system learns if its performance
improves with experience, where three things are named:

| | | |
|---|---|---|
| **T** | Task | What it is supposed to do |
| **E** | Experience | What it learns from |
| **P** | Performance | How improvement is measured |

If you cannot fill in all three, you do not yet have a machine learning
problem. You have a wish.

## Five questions

Mitchell's three, plus two the textbook leaves out and the real world does not.

<svg viewBox="0 0 680 328" role="img" aria-labelledby="fq-title fq-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="fq-title">Five questions for framing a problem</title>
<desc id="fq-desc">One, what exactly is the task. Two, what experience will it learn from. Three, how will success be measured. Four, which learning family is this. Five, should this exist at all.</desc>
<circle cx="40" cy="34" r="18" fill="var(--h-surface)" stroke="var(--h-surface-line)" stroke-width="2"/>
<text x="40" y="34" dy="0.36em" text-anchor="middle" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">1</text>
<text x="76" y="30" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">What exactly is the task?</text>
<text x="76" y="50" font-size="12" fill="var(--h-graphite)">For one example: what goes in, and what comes out</text>
<circle cx="40" cy="94" r="18" fill="var(--h-surface)" stroke="var(--h-surface-line)" stroke-width="2"/>
<text x="40" y="94" dy="0.36em" text-anchor="middle" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">2</text>
<text x="76" y="90" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">What will it learn from?</text>
<text x="76" y="110" font-size="12" fill="var(--h-graphite)">Which data, from whom, how much, and who labels it</text>
<circle cx="40" cy="154" r="18" fill="var(--h-surface)" stroke="var(--h-surface-line)" stroke-width="2"/>
<text x="40" y="154" dy="0.36em" text-anchor="middle" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">3</text>
<text x="76" y="150" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">How will you know it works?</text>
<text x="76" y="170" font-size="12" fill="var(--h-graphite)">One measure, agreed before you start, not after</text>
<circle cx="40" cy="214" r="18" fill="var(--h-surface)" stroke="var(--h-surface-line)" stroke-width="2"/>
<text x="40" y="214" dy="0.36em" text-anchor="middle" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">4</text>
<text x="76" y="210" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">Which learning family is this?</text>
<text x="76" y="230" font-size="12" fill="var(--h-graphite)">It decides what data you need and what can go wrong</text>
<circle cx="40" cy="274" r="18" fill="var(--h-cherry)"/>
<text x="40" y="274" dy="0.36em" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff">5</text>
<text x="76" y="270" font-size="14" font-weight="700" fill="var(--h-cherry)">Should this exist at all?</text>
<text x="76" y="290" font-size="12" fill="var(--h-graphite)">Who benefits, who is harmed, and who gets to decide</text>
<text x="76" y="316" font-size="11.5" font-style="italic" fill="var(--h-space)">questions one to three are Mitchell's. Four is engineering. Five is yours.</text>
</svg>

Question one is harder than it looks. "Reduce customer complaints" is not a
task. "Given a support message, predict whether it will become a formal
complaint" is. The test is whether you can describe a single example: one input,
one output.

## Which family

<svg viewBox="0 0 680 372" role="img" aria-labelledby="fam-title fam-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="fam-title">Choosing a learning family</title>
<desc id="fam-desc">If you have labelled examples, supervised learning, which splits into regression when the answer is a number and classification when it is a category. If you have examples with no labels, unsupervised learning, which covers clustering and dimensionality reduction. If you have an environment to act in, reinforcement learning, which produces a policy.</desc>
<rect x="240" y="8" width="200" height="42" rx="10" fill="var(--h-cherry)"/>
<text x="340" y="29" dy="0.36em" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff">What do you have?</text>
<polyline points="340,50 340,72 115,72 115,96" fill="none" stroke="var(--h-steel)" stroke-width="2"/>
<polyline points="340,50 340,96" fill="none" stroke="var(--h-steel)" stroke-width="2"/>
<polyline points="340,50 340,72 565,72 565,96" fill="none" stroke="var(--h-steel)" stroke-width="2"/>
<rect x="10" y="96" width="210" height="60" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="115" y="122" text-anchor="middle" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">Supervised</text>
<text x="115" y="142" text-anchor="middle" font-size="11" fill="var(--h-graphite)">examples with the answers</text>
<rect x="235" y="96" width="210" height="60" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="340" y="122" text-anchor="middle" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">Unsupervised</text>
<text x="340" y="142" text-anchor="middle" font-size="11" fill="var(--h-graphite)">examples, no answers</text>
<rect x="460" y="96" width="210" height="60" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="565" y="122" text-anchor="middle" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">Reinforcement</text>
<text x="565" y="142" text-anchor="middle" font-size="11" fill="var(--h-graphite)">an environment to act in</text>
<line x1="115" y1="156" x2="115" y2="178" stroke="var(--h-steel)" stroke-width="2"/>
<line x1="340" y1="156" x2="340" y2="178" stroke="var(--h-steel)" stroke-width="2"/>
<line x1="565" y1="156" x2="565" y2="178" stroke="var(--h-steel)" stroke-width="2"/>
<rect x="10" y="178" width="210" height="48" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="115" y="198" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">Regression</text>
<text x="115" y="216" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">the answer is a number</text>
<rect x="235" y="178" width="210" height="48" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="340" y="198" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">Clustering</text>
<text x="340" y="216" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">find the natural groups</text>
<rect x="460" y="178" width="210" height="48" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="565" y="198" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">A policy</text>
<text x="565" y="216" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">what to do in each situation</text>
<rect x="10" y="236" width="210" height="48" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-cherry)"/>
<text x="115" y="256" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">Classification</text>
<text x="115" y="274" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">the answer is a category</text>
<rect x="235" y="236" width="210" height="48" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="340" y="256" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">Dimensionality reduction</text>
<text x="340" y="274" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">squeeze detail into a simpler view</text>
<text x="115" y="306" text-anchor="middle" font-size="11" fill="var(--h-graphite)">binary, multi-class, multi-label</text>
<text x="340" y="348" text-anchor="middle" font-size="12" font-weight="600" fill="var(--h-cherry)">your first project is multi-class classification: one hand, one of three labels</text>
</svg>

### Supervised learning

You have examples and you have the right answers. The model learns the mapping
from one to the other. This is most of what you will build this year, and it
comes in four flavours that differ only in the shape of the answer.

| Flavour | One example gets | Example |
|---|---|---|
| **Binary classification** | One of two labels | Spam or not spam |
| **Multi-class classification** | One of many labels | Rock, paper, or scissors |
| **Multi-label classification** | Any number of labels at once | A photo tagged beach, dog, friend |
| **Regression** | A number | A flat in Tirana costs 92,000 euro |

The distinction between multi-class and multi-label catches people out. Multi
class means exactly one answer per example. Multi label means an example can
carry several at once, and they are not competing.

!!! tip "Naming the flavour tells you what to collect"

    Multi-label needs every applicable tag on every training example, not just
    the most obvious one. Regression needs examples spread across the whole
    range of values, not clustered at the average. The flavour is not
    bookkeeping, it changes the data you have to go and get.

### Unsupervised learning

You have examples and no answers. The model finds structure by itself.

**Clustering** groups similar examples. Customers who behave alike, documents
about the same topic, the transaction that looks unlike all the others. The
algorithm produces the groups. Naming them is your job, and that naming is a
judgment, not an output.

**Dimensionality reduction** compresses many measurements into a few that keep
most of the information. It is how you visualise data with fifty columns, and
how you strip out the parts that carry no signal.

There is a trap here worth knowing now. Unsupervised results always look like
something. Give any algorithm data and it will return groups. Whether those
groups mean anything is a separate question, and one the algorithm cannot
answer for you.

### Reinforcement learning

An agent acts in an environment, receives a reward or a penalty, and adjusts.
Over many attempts it learns a **policy**, which is a rule for what to do in
each situation rather than a single answer.

Four words cover it: agent, environment, action, reward. This is how systems
learn to play Go without being shown human games, and how robots learn to walk.

Being honest about scope: you will not train a reinforcement learning agent in
this programme. You need to recognise it, know when a problem is one, and know
that the hard part is the reward. Define it badly and the agent will maximise
exactly what you asked for rather than what you wanted, which is known as
reward hacking and is a live research problem.

## Two worked examples

**A well posed one.**

> **Problem as stated:** stop spam reaching the inbox.
>
> **Task:** given one email, predict spam or not spam.
> **Experience:** past emails that users marked as spam or left alone.
> **Performance:** of the emails we flag, how many really were spam. Losing a
> real email costs far more than letting one spam through.
> **Family:** supervised, binary classification.
> **Should it exist:** yes. Low stakes, visible to the user, reversible.

**One where the answer is no.**

> **Problem as stated:** predict criminal behaviour from facial photographs.
>
> **Task:** given a face, predict a label.
> **Experience:** photographs, labelled using past convictions.
> **Performance:** accuracy, presumably.
> **Family:** supervised, binary classification.
> **Should it exist:** no.

The first four rows fill in perfectly. That is the point. A problem can be
technically well posed and still be one you refuse.

Look at the experience row. Convictions are not criminality, they are the
record of who was arrested, charged and convicted, which reflects who was
policed. The model would learn the pattern of past enforcement and present it
as a property of a face. And there is no reason to think the target is visible
in a face at all, which means any accuracy you achieve is measuring something
other than what you claimed.

Question five is not a formality tacked on the end. Sometimes it is the only
one that matters.

## Try it yourself

Frame each of these with the five questions before the next session. Two of
them are traps.

1. Suggest which film a user should watch next.
2. Predict how many minutes late the city bus will arrive.
3. Sort a photo library into people, places, food and documents.
4. Decide which job applicants get a first interview.

For each, say what one example looks like going in and coming out, where the
data would come from, what you would measure, which family and flavour it is,
and whether you would build it.

## Questions to sit with

1. Question three says agree the measure before you start. Why does the order
   matter, given that the measure is the same either way?
2. A clustering algorithm always returns groups, even from random data. How
   would you tell a real pattern from an invented one?
3. In the facial criminality example, the experience is the problem. Name
   another task where the data you can get is a poor stand-in for the thing you
   actually care about.
4. If reward hacking means the agent maximises what you asked for rather than
   what you wanted, is that a failure of the agent or of the person who wrote
   the reward?

## Next

You have a problem. Now you need the data, which is where most of the work and
most of the mistakes live.

[Working with data](working-with-data.md){ .h-button }

# Ethics and responsibility

<span class="badge badge--time">25 min</span>
<span class="badge badge--level">Concepts</span>
<span class="badge">Read before the second project</span>

This is not a separate subject that gets bolted on at the end. Almost every
ethical problem in machine learning is a technical decision that somebody made
early and casually, usually while collecting data.

## Why this sits in an engineering course

In your first project you photographed hands. Whose hands, in what light,
against what wall. Those were technical choices about data collection. They
were also the choices that decided whose gestures the model would recognise and
whose it would not.

There was no separate moment where ethics happened. It happened while you were
setting up the webcam.

Scale that up. A hiring model trained on ten years of a company's hires learns
who that company promoted. A model that predicts which patients need extra care
and uses past spending as a proxy for need learns that less money was spent on
people who got less care. Neither system was built by anyone who wanted that
outcome. Both were built by people who did not ask early enough.

## Harm does not wait for deployment

Here is the same lifecycle from the last page, with what can go wrong at each
step.

<svg viewBox="0 0 680 230" role="img" aria-labelledby="harm-title harm-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="harm-title">Where harm enters the machine learning lifecycle</title>
<desc id="harm-desc">At collection, consent and who is missing. At preparation, the judgments hidden in labels. At training, the past being repeated. At evaluation, averages hiding the worst case. At deployment, whether anyone can appeal.</desc>
<defs><marker id="hm-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<text x="340" y="24" text-anchor="middle" font-size="12" fill="var(--h-space)">harm enters at every stage, not only the last one</text>
<rect x="5" y="40" width="118" height="54" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="64" y="72" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">Collect</text>
<rect x="143" y="40" width="118" height="54" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="202" y="72" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">Prepare</text>
<rect x="281" y="40" width="118" height="54" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="340" y="72" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">Train</text>
<rect x="419" y="40" width="118" height="54" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="478" y="72" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">Evaluate</text>
<rect x="557" y="40" width="118" height="54" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="616" y="72" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">Deploy</text>
<line x1="127" y1="67" x2="139" y2="67" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#hm-arrow)"/>
<line x1="265" y1="67" x2="277" y2="67" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#hm-arrow)"/>
<line x1="403" y1="67" x2="415" y2="67" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#hm-arrow)"/>
<line x1="541" y1="67" x2="553" y2="67" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#hm-arrow)"/>
<line x1="64" y1="98" x2="64" y2="118" stroke="var(--h-cherry-line)" stroke-width="2"/>
<line x1="202" y1="98" x2="202" y2="118" stroke="var(--h-cherry-line)" stroke-width="2"/>
<line x1="340" y1="98" x2="340" y2="118" stroke="var(--h-cherry-line)" stroke-width="2"/>
<line x1="478" y1="98" x2="478" y2="118" stroke="var(--h-cherry-line)" stroke-width="2"/>
<line x1="616" y1="98" x2="616" y2="118" stroke="var(--h-cherry-line)" stroke-width="2"/>
<rect x="5" y="122" width="118" height="70" rx="10" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="64" y="144" text-anchor="middle" font-size="11" font-weight="600" fill="var(--h-cherry)">Did they agree?</text>
<text x="64" y="162" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">And who is</text>
<text x="64" y="176" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">not in the data?</text>
<rect x="143" y="122" width="118" height="70" rx="10" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="202" y="144" text-anchor="middle" font-size="11" font-weight="600" fill="var(--h-cherry)">Who labelled it?</text>
<text x="202" y="162" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">Every label is</text>
<text x="202" y="176" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">a judgment</text>
<rect x="281" y="122" width="118" height="70" rx="10" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="340" y="144" text-anchor="middle" font-size="11" font-weight="600" fill="var(--h-cherry)">Whose past?</text>
<text x="340" y="162" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">The model will</text>
<text x="340" y="176" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">repeat it</text>
<rect x="419" y="122" width="118" height="70" rx="10" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="478" y="144" text-anchor="middle" font-size="11" font-weight="600" fill="var(--h-cherry)">Good for whom?</text>
<text x="478" y="162" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">An average hides</text>
<text x="478" y="176" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">the worst group</text>
<rect x="557" y="122" width="118" height="70" rx="10" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="616" y="144" text-anchor="middle" font-size="11" font-weight="600" fill="var(--h-cherry)">Can they appeal?</text>
<text x="616" y="162" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">Someone must</text>
<text x="616" y="176" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">be answerable</text>
</svg>

Notice that four of the five risks appear before the model exists. By the time
something is deployed, most of the damage has already been designed in.

## Four questions to ask before you build

Ask them out loud, in the group, before any data is collected. They take ten
minutes and they catch most of what goes wrong.

<div class="grid cards" markdown>

-   :material-account-group-outline: **Who is in the data, and who is not?**

    ---

    List the people your examples come from. Then list who is missing. The
    second list is where your model will fail, and you will not notice, because
    your tests come from the first list.

-   :material-scale-balance: **What does a mistake cost, and to whom?**

    ---

    A wrong film recommendation costs nothing. A wrong fraud flag freezes
    somebody's money. The same accuracy figure means completely different
    things in those two systems.

-   :material-gavel: **Can the person affected find out and object?**

    ---

    If a system says no, can anyone explain why, and is there a route to a
    human? A system nobody can question is not a technical achievement.

-   :material-eye-outline: **Would you be comfortable if this were public?**

    ---

    The data, the labels, the error rates. If the honest answer is no, that
    discomfort is information. Work out what is causing it.

</div>

## Personal data

Most rules about personal data across Europe follow the same few principles,
and Albanian law is built on that model. You do not need to be a lawyer, but
you should know the shape of it, because it changes how you design systems.

**Purpose.** You collect data for a stated reason. Using it later for something
else is a new decision, not a free extra.

**Minimisation.** Collect what you need and no more. Extra fields you might use
one day are a liability, not an asset.

**Consent that means something.** Freely given, informed, and refusable. A
pre-ticked box, or consent as the price of a service someone cannot avoid, is
not consent.

**Deletion.** People can ask you to remove their data. Build so that this is
possible. Data scattered across notebooks, laptops and three cloud buckets
cannot be deleted, only forgotten about.

**Special categories.** Health, biometrics, ethnicity, religion, political
views and sexual orientation carry stricter rules almost everywhere. Face and
fingerprint data are biometric. Think carefully before a student project
touches any of these.

There is also the EU AI Act, which sorts systems by risk. Some uses are banned,
some are high risk and carry obligations, most are neither. If you build for a
European market, somebody on your team will need to know which bucket you are
in.

!!! warning "This page is orientation, not legal advice"

    Rules change, and the details differ by country and by sector. When
    something real is at stake, check the current law or ask somebody whose job
    it is. Knowing that a question exists is the part you are responsible for.

## Bias is not a bug you patch at the end

There is a tempting story where the model is neutral and bias is a defect that
gets fixed before release. It does not work that way.

A model has no preferences. It reproduces the patterns in what it was shown.
If the data reflects a world that treated people differently, the model
reproduces that treatment, efficiently and at scale, while looking objective
because it is a computer.

Three ways it gets in:

**Who is represented.** Face recognition that works far better on some skin
tones than others, because of who was in the training photographs.

**What you chose to measure.** A model predicting "good employee" using past
promotions is predicting who got promoted, which is not the same thing.

**What the system does once it runs.** Predictive policing sends officers where
arrests happened before, which produces more arrests there, which confirms the
prediction. The model helps create the evidence for itself.

The last one is the hardest, because the system looks accurate the whole time.

## Write things down

Two documents cost very little and change how teams behave.

A **data statement** records where examples came from, who is represented, how
consent was handled, and what is known to be missing. You wrote one in the
first project. Keep doing it.

A **model card** records what the model is for, how well it performs, and
importantly how well it performs on different groups rather than only on
average. It also records what the model should not be used for.

Both exist so that the person who inherits your system in two years knows what
they have. Often that person is you.

## Who is responsible

When a model causes harm, there is a familiar shrug. The engineer says they
only built what was specified. The manager says they trusted the technical
team. The company points at the algorithm as if it arrived from somewhere else.

Someone chose the training data. Someone chose the threshold. Someone decided
it was ready. A model is a product of decisions by people, and responsibility
does not evaporate because the final step was automatic.

You will at some point be asked to build something you are not comfortable
with. Two things help. Raise it early, when changing course is cheap. And put
it in writing, not because of blame, but because a concern in an email gets
discussed while a concern in a corridor does not.

## Questions to sit with

1. Your face recognition model is 99 percent accurate overall and 80 percent
   accurate for one group of users. The average is excellent. Ship it or not,
   and what would change your answer?
2. You can remove a sensitive field from your data, but other fields predict it
   almost perfectly. Postcode often predicts ethnicity. Have you removed the
   bias or only hidden it?
3. A model refuses someone a loan. What is the minimum you owe that person?
   A reason, a human to talk to, a right to a second look, or nothing at all?
4. Your employer asks for a system you think is harmful but legal. What do you
   do, concretely, on Monday morning?
5. Who should decide whether a system is safe enough to deploy? The engineers
   who understand it, the managers who own it, or the people it will be used
   on, who understand it least and are affected most?

## Next

The last piece of the foundation is what actually happens when a model trains.

[What training actually does](how-training-works.md){ .h-button }

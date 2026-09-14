# Project 1: Rock, paper, scissors

<span class="badge badge--time">Two sessions</span>
<span class="badge badge--level">No code required</span>
<span class="badge">Groups of 3</span>

Build a model that recognises a hand showing rock, paper, or scissors. Then
find out what it really learned, which is usually not what you intended.

## What you are doing and why

You will use [Teachable Machine](https://teachablemachine.withgoogle.com/), a
browser tool that trains an image classifier without any code. In about forty
minutes you go through the entire machine learning lifecycle: collect data,
train, evaluate, and see it used. Every module for the next nine months is a
more rigorous version of one of those steps.

Rock, paper, scissors was chosen because it is hard in a useful way. The three
gestures are made by the same hand, in the same position, against the same
background. Scissors and paper share most of their outline. A model that scores
perfectly here has almost certainly learned something other than the gesture,
and finding out what is the actual assignment.

<svg viewBox="0 0 680 170" role="img" aria-labelledby="flow-title flow-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="flow-title">The five steps of the project</title>
<desc id="flow-desc">Collect images, train the model, export it, swap with another group and test their model, then report your results.</desc>
<defs><marker id="pf-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<rect x="5" y="40" width="118" height="72" rx="10" fill="var(--h-cherry)"/>
<text x="64" y="70" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">Collect</text>
<text x="64" y="90" text-anchor="middle" font-size="10.5" fill="#ffffff" opacity="0.9">your own images</text>
<rect x="143" y="40" width="118" height="72" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="202" y="70" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">Train</text>
<text x="202" y="90" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">one click</text>
<rect x="281" y="40" width="118" height="72" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="340" y="70" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">Export</text>
<text x="340" y="90" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">save the model</text>
<rect x="419" y="40" width="118" height="72" rx="10" fill="var(--h-cherry)"/>
<text x="478" y="70" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">Swap</text>
<text x="478" y="90" text-anchor="middle" font-size="10.5" fill="#ffffff" opacity="0.9">test each other</text>
<rect x="557" y="40" width="118" height="72" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="616" y="70" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">Report</text>
<text x="616" y="90" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">explain the gap</text>
<line x1="127" y1="76" x2="139" y2="76" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#pf-arrow)"/>
<line x1="265" y1="76" x2="277" y2="76" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#pf-arrow)"/>
<line x1="403" y1="76" x2="415" y2="76" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#pf-arrow)"/>
<line x1="541" y1="76" x2="553" y2="76" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#pf-arrow)"/>
<text x="340" y="24" text-anchor="middle" font-size="12" fill="var(--h-space)">the two red steps are where the learning happens</text>
<text x="340" y="140" text-anchor="middle" font-size="12" font-weight="600" fill="var(--h-graphite)">most groups rush Collect and then cannot explain Report</text>
</svg>

## Groups

Five groups of three. Inside your group, agree who holds each role. Everybody
does everything, but one person is responsible for each part being done
properly.

| Role | Responsible for |
|---|---|
| Data lead | The data statement, the collection plan, making sure the images vary |
| Training lead | Running the training, the exported model, the screenshots |
| Evaluation lead | The results grid, the failure examples, the numbers being honest |

Roles rotate on the next project, so do not give the writing to whoever writes
best.

## Before you collect anything: the data statement

Write this first. It takes five minutes and it goes at the top of your README.

> **What we are collecting:** photographs of hands making three gestures.
> **Whose:** our own, and anyone else who agrees in advance.
> **Where it is stored:** our group repository, which is public.
> **Would we be comfortable if this were published?** Yes or no, and why.

If you decide to photograph someone outside the group, ask first and record
that you asked. If a face appears in any frame, either crop it or drop the
image. You are about to publish this to GitHub, which means the internet.

!!! warning "This is not a formality"

    Most of the ethical problems in real machine learning start at exactly this
    step, before anyone has trained anything. Half of next week's discussion
    comes back to what you wrote here.

## Step 1: Collect your training images

Open Teachable Machine, choose **Image Project**, then **Standard image model**.
Create three classes and name them `rock`, `paper`, `scissors`.

Collect at least **50 images per class**, more if you have time. You can use
the webcam to capture bursts, or upload photos and frames taken from video.

Here is the part that decides whether your project is any good. **Vary
everything you can.**

<div class="grid cards" markdown>

-   :material-hand-clap: **Hands**

    ---

    All three group members, not just one. Left hands and right hands.

-   :material-rotate-3d-variant: **Angles**

    ---

    Straight on, tilted, from above, rotated. Close and further away.

-   :material-lightbulb-outline: **Light**

    ---

    Near a window, under the room lights, in a darker corner.

-   :material-image-filter-hdr: **Backgrounds**

    ---

    Different walls, a desk, a jacket, a busy background and a plain one.

</div>

If all fifty of your rock images are the same hand in the same chair, your
model will learn that hand and that chair. It will still score beautifully in
that chair, which is exactly the trap.

## Step 2: Train and export

Click **Train Model** and leave the tab open while it runs.

When it finishes, use the preview panel. Show it each gesture and watch the
confidence bars. Try a few things on purpose:

- Move to a different part of the room.
- Have someone whose hands were not photographed try it.
- Show it something that is none of the three, a pen or a coffee cup, and note
  what it claims and how confident it is.

That last one matters. The model has no option to say "I do not know". It will
give you a confident answer for an object it has never conceived of. Write down
what happened, you will be asked about it.

Then click **Export Model** and save the files into your repository.

!!! tip "Take screenshots as you go"

    You need three for the README and it is much easier to capture them now
    than to recreate them later: the class list with image counts, the model
    working correctly, and the model confidently wrong.

## Step 3: Swap and test

This is the heart of the project.

Each group collects a test set for the next group around the ring, and tests
that group's model with it.

<svg viewBox="0 0 680 300" role="img" aria-labelledby="swap-title swap-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="swap-title">The testing swap between groups</title>
<desc id="swap-desc">Five groups in a ring. Each group collects a test set for the next group and runs it against that group's model. Group one tests group two, group two tests group three, and so on back around to group one.</desc>
<defs><marker id="sw-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-cherry)"/></marker></defs>
<circle cx="340" cy="40" r="34" fill="var(--h-surface)" stroke="var(--h-surface-line)" stroke-width="2"/>
<text x="340" y="40" dy="0.36em" text-anchor="middle" font-size="15" font-weight="700" fill="var(--md-default-fg-color)">G1</text>
<circle cx="445" cy="116" r="34" fill="var(--h-surface)" stroke="var(--h-surface-line)" stroke-width="2"/>
<text x="445" y="116" dy="0.36em" text-anchor="middle" font-size="15" font-weight="700" fill="var(--md-default-fg-color)">G2</text>
<circle cx="405" cy="239" r="34" fill="var(--h-surface)" stroke="var(--h-surface-line)" stroke-width="2"/>
<text x="405" y="239" dy="0.36em" text-anchor="middle" font-size="15" font-weight="700" fill="var(--md-default-fg-color)">G3</text>
<circle cx="275" cy="239" r="34" fill="var(--h-surface)" stroke="var(--h-surface-line)" stroke-width="2"/>
<text x="275" y="239" dy="0.36em" text-anchor="middle" font-size="15" font-weight="700" fill="var(--md-default-fg-color)">G4</text>
<circle cx="235" cy="116" r="34" fill="var(--h-surface)" stroke="var(--h-surface-line)" stroke-width="2"/>
<text x="235" y="116" dy="0.36em" text-anchor="middle" font-size="15" font-weight="700" fill="var(--md-default-fg-color)">G5</text>
<line x1="367" y1="60" x2="414" y2="94" stroke="var(--h-cherry)" stroke-width="2" marker-end="url(#sw-arrow)"/>
<line x1="434" y1="148" x2="416" y2="203" stroke="var(--h-cherry)" stroke-width="2" marker-end="url(#sw-arrow)"/>
<line x1="371" y1="239" x2="313" y2="239" stroke="var(--h-cherry)" stroke-width="2" marker-end="url(#sw-arrow)"/>
<line x1="265" y1="207" x2="247" y2="152" stroke="var(--h-cherry)" stroke-width="2" marker-end="url(#sw-arrow)"/>
<line x1="263" y1="96" x2="309" y2="62" stroke="var(--h-cherry)" stroke-width="2" marker-end="url(#sw-arrow)"/>
<text x="340" y="146" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">each group collects</text>
<text x="340" y="166" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">the test set for the next</text>
<text x="340" y="188" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">nobody tests their own model</text>
</svg>

Rules for the test set you collect:

- **30 images: 10 rock, 10 paper, 10 scissors.**
- Taken somewhere other than where the model was trained, or on a different
  day, or both.
- Ordinary gestures. You are testing the model, not playing tricks on it. Save
  the tricks for the stretch task.
- Hands from your group, since the point is that they are hands the model has
  never seen.

Then run those 30 images through the other group's model, one at a time, and
record what it says. Fill in this grid.

| | said rock | said paper | said scissors |
|---|---|---|---|
| **actually rock** | | | |
| **actually paper** | | | |
| **actually scissors** | | | |

Each row adds up to 10. The diagonal is what it got right. Everything off the
diagonal is a mistake with a story behind it.

Expect the accuracy to drop, often sharply. That drop is the finding. A group
whose model falls from 98 percent to 55 percent has a far more interesting
project than one that stays high, as long as they can explain the fall.

## Step 4: Write it up

One repository per group, public, on GitHub. The README contains:

1. The data statement from before you started.
2. How many images per class, and what you varied.
3. The results grid from the group who tested you, with the accuracy figure.
4. Three screenshots: the class list, a correct prediction, a confident
   mistake.
5. **The single worst failure.** The image where the model was most confident
   and most wrong, with its confidence percentage. Say what you think it
   latched onto.
6. Three to five sentences on what your model actually learned, as opposed to
   what you were trying to teach it.

Point six is the whole project. Everything above it is evidence for it.

Commit and push as you go, not in one dump at the end. This is your Git
tutorial being used for real.

## Step 5: The review session

Each group presents for about eight minutes, then takes questions. Questions
come from these four areas, so prepare for all of them.

<div class="grid cards" markdown>

-   :material-camera-outline: **Where the data came from**

    ---

    Who is in it, who is not, and what you varied. What would you collect
    differently with another two hours?

-   :material-chart-box-outline: **How you know it works**

    ---

    What the number means, where the mistakes clustered, and why that pattern
    and not another.

-   :material-shield-account-outline: **Privacy and ethics**

    ---

    Consent, storage, publication. Who gets hurt if this is wrong, and does the
    answer change with what it is used for?

-   :material-help-circle-outline: **What it really learned**

    ---

    The gesture, the hand, the room, or the lighting. How would you prove which
    one?

</div>

The last area is where the marks are. If your answer to "what did it learn" is
"rock, paper and scissors", you have not looked hard enough.

## What good looks like

Two groups, both honest, very different outcomes.

**Weak:** 97 percent on the swapped test set. Trained on one person's hand
against a white wall, tested against the same white wall because nobody read
the rule. Report says the model works well. Nothing was learned about machine
learning.

**Strong:** 61 percent on the swapped test set. Scissors read as paper nine
times out of ten. The group noticed that their scissors images were all shot
from the side, where two fingers overlap into a shape like an open hand, and
that the other group shot from the front. They name the fix: collect scissors
from more angles. That is a real diagnosis, and it is the same reasoning a
working engineer does on a production model.

## If you finish early

Try to break another group's model on purpose. Find an input that fools it
badly and record the confidence score. A sleeve, a shadow, a hand at the edge
of frame, a gesture held halfway between two classes.

This is not mischief. Deliberately attacking your own systems is a real job in
this industry, and the images you find make the best slides in the review.

## Before you present, check

1. The README has all six items, and the data statement was written first.
2. Your results grid came from another group, not from your own testing.
3. You can point at one specific image and say what the model latched onto.
4. Everyone in the group can answer questions about every part of it, not only
   their own role.
5. The repository link works from a private browser window.

## Questions to think about

These go beyond the marking and are worth an argument in the room.

1. Your model works on your hands and fails on somebody else's. In a product
   shipped to thousands of people, whose hands would it have failed on, and
   would anyone have noticed before release?
2. You never told the model what a gesture is. It found something in the pixels
   that separates your three piles. Is that learning, or is it sorting?
3. The model gave a confident answer for a coffee cup. What should a system do
   when it meets something it has never seen, and who decides what it does?
4. You could push accuracy up by testing in the same room you trained in. It
   would not be a lie, the number would be real. Why is it still dishonest?
5. If this model were deployed to judge a rock, paper, scissors tournament with
   prize money, what would you need to add before you would be comfortable
   signing off on it?

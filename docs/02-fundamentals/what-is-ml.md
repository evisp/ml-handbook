# How machines learn

<span class="badge badge--time">30 min</span>
<span class="badge badge--level">Concepts</span>
<span class="badge">Read before the project review</span>

Machine learning is one idea. Instead of writing the rules, you show the
machine examples and let it work the rules out. Everything else on this page is
detail.

## The switch

For fifty years, getting a computer to do something meant writing down how.
Rules in, data in, answers out. This works beautifully when you can state the
rule. Payroll, tax, sorting a list.

Now try to write the rule for telling a photo of rock from a photo of paper.
Not the idea of it, the actual rule, in terms of pixel values. You cannot. You
know the answer instantly and you cannot say how you know.

That is the gap machine learning fills. You supply the examples and the
answers, and the machine produces the rule.

<svg viewBox="0 0 680 240" role="img" aria-labelledby="prog-title prog-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="prog-title">Traditional programming compared with machine learning</title>
<desc id="prog-desc">In traditional programming, rules and data go in and answers come out. In machine learning, data and answers go in and the rules, called a model, come out.</desc>
<defs><marker id="pr-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<text x="8" y="26" font-size="13" font-weight="700" fill="var(--h-graphite)">Traditional programming</text>
<rect x="8" y="40" width="130" height="52" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="73" y="72" text-anchor="middle" font-size="13.5" font-weight="600" fill="var(--md-default-fg-color)">Rules</text>
<text x="158" y="72" text-anchor="middle" font-size="18" fill="var(--h-space)">+</text>
<rect x="178" y="40" width="130" height="52" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="243" y="72" text-anchor="middle" font-size="13.5" font-weight="600" fill="var(--md-default-fg-color)">Data</text>
<line x1="316" y1="66" x2="386" y2="66" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#pr-arrow)"/>
<rect x="394" y="40" width="150" height="52" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="469" y="72" text-anchor="middle" font-size="13.5" font-weight="600" fill="var(--md-default-fg-color)">Answers</text>
<text x="8" y="148" font-size="13" font-weight="700" fill="var(--h-cherry)">Machine learning</text>
<rect x="8" y="162" width="130" height="52" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="73" y="194" text-anchor="middle" font-size="13.5" font-weight="600" fill="var(--md-default-fg-color)">Data</text>
<text x="158" y="194" text-anchor="middle" font-size="18" fill="var(--h-space)">+</text>
<rect x="178" y="162" width="130" height="52" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="243" y="194" text-anchor="middle" font-size="13.5" font-weight="600" fill="var(--md-default-fg-color)">Answers</text>
<line x1="316" y1="188" x2="386" y2="188" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#pr-arrow)"/>
<rect x="394" y="162" width="150" height="52" rx="10" fill="var(--h-cherry)"/>
<text x="469" y="188" text-anchor="middle" font-size="13.5" font-weight="700" fill="#ffffff">Rules</text>
<text x="469" y="205" text-anchor="middle" font-size="11" fill="#ffffff" opacity="0.9">we call this a model</text>
</svg>

Read the second row again. The answers are an input. This is why data work is
most of the job, and why a model is only ever as good as the examples it was
shown.

## The words you need

Every term below is something you will physically do in the first project. Keep
this section open while you work.

| Term | What it means | In the project |
|---|---|---|
| **Class** | One of the categories you are sorting into | Rock, paper, scissors |
| **Example** | One item of data | One photo of a hand |
| **Label** | The correct answer for that example | "This one is paper" |
| **Feature** | Something measurable about an example | Shape, edges, colour |
| **Training** | Adjusting the model until it fits the examples | Clicking Train Model |
| **Model** | The rules the machine ended up with | The file you export |
| **Prediction** | The model's answer on something new | "Scissors" |
| **Confidence** | How sure the model is, from 0 to 100 | The bar under each class |
| **Training set** | The examples used to build the model | Your collected photos |
| **Test set** | Held back examples, never used in training | The 20 photos you test with |
| **Generalisation** | Working on data it has never seen | The whole point |
| **Overfitting** | Memorising the training set instead of learning the pattern | 100 percent in the room, 50 percent elsewhere |

Two of those deserve more than a table row.

**Confidence is not correctness.** A model gives you a number for how well the
input matches what it learned. It has no way to say "I have never seen anything
like this". Show a rock, paper, scissors model a photograph of a dog and it
will confidently tell you it is paper. This trips up people building real
systems, not just students.

**Generalisation is the only thing that matters.** A model that is perfect on
its training data and useless on anything else has learned nothing. It is a
lookup table. The whole discipline exists to close the gap between those two
numbers.

## The loop

This is the diagram you will see for the rest of the year. Every module we do
is a deeper pass at one of these five stages.

<svg viewBox="0 0 680 230" role="img" aria-labelledby="loop-title loop-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="loop-title">The machine learning lifecycle</title>
<desc id="loop-desc">Collect data, prepare it, train a model, evaluate it, then deploy it. What you learn from evaluation and from use sends you back to the start.</desc>
<defs><marker id="lp-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<rect x="5" y="56" width="118" height="72" rx="10" fill="var(--h-cherry)"/>
<text x="64" y="86" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">Collect</text>
<text x="64" y="106" text-anchor="middle" font-size="10.5" fill="#ffffff" opacity="0.9">get examples</text>
<rect x="143" y="56" width="118" height="72" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="202" y="86" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">Prepare</text>
<text x="202" y="106" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">clean and label</text>
<rect x="281" y="56" width="118" height="72" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="340" y="86" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">Train</text>
<text x="340" y="106" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">fit the model</text>
<rect x="419" y="56" width="118" height="72" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="478" y="86" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">Evaluate</text>
<text x="478" y="106" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">test on new data</text>
<rect x="557" y="56" width="118" height="72" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="616" y="86" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">Deploy</text>
<text x="616" y="106" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">let people use it</text>
<line x1="127" y1="92" x2="139" y2="92" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#lp-arrow)"/>
<line x1="265" y1="92" x2="277" y2="92" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#lp-arrow)"/>
<line x1="403" y1="92" x2="415" y2="92" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#lp-arrow)"/>
<line x1="541" y1="92" x2="553" y2="92" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#lp-arrow)"/>
<polyline points="616,134 616,176 64,176 64,136" fill="none" stroke="var(--h-steel)" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#lp-arrow)"/>
<text x="340" y="200" text-anchor="middle" font-size="12" font-weight="600" fill="var(--h-graphite)">what you learn sends you back to the data</text>
<text x="340" y="30" text-anchor="middle" font-size="12" fill="var(--h-space)">most of the work happens in the first two boxes</text>
</svg>

Beginners expect the interesting work to be in Train. In practice that is the
step that takes the least of your time and the least of your judgement. Collect
and Prepare decide whether the project succeeds.

## Three ways to learn

Almost everything you will meet falls into one of three setups.

<div class="grid cards" markdown>

-   :material-tag-multiple-outline: **Supervised**

    ---

    You give examples and the right answers. The model learns the mapping.
    Spam or not spam, price of a flat, rock or paper or scissors.

    This is most of what you will do this year, and it is what the first
    project uses.

-   :material-shape-outline: **Unsupervised**

    ---

    You give examples and no answers. The model finds structure on its own,
    usually groups.

    Customer segments, topics in a pile of documents, spotting the transaction
    that looks unlike the rest.

-   :material-trophy-outline: **Reinforcement**

    ---

    No examples. The model tries things, gets a reward or a penalty, and
    adjusts.

    Game playing, robot control. This is how a system learns to beat humans at
    Go without being shown how a human plays.

</div>

## Knowing whether it works

Accuracy is the number everyone quotes and the number that hides the most.

Say your model is right on 27 of 30 test images. That is 90 percent. Now ask a
better question: what were the three mistakes? A grid of actual against
predicted answers it, and it takes two minutes to fill in by hand.

<svg viewBox="0 0 680 320" role="img" aria-labelledby="cm-title cm-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="cm-title">A results grid for a rock paper scissors model</title>
<desc id="cm-desc">Rows are the true class and columns are what the model predicted. The diagonal holds correct answers. Everything off the diagonal is a mistake. In this example, paper is confused with rock twice and scissors is called paper three times.</desc>
<text x="365" y="34" text-anchor="middle" font-size="13" font-weight="700" fill="var(--h-graphite)">what the model said</text>
<text x="255" y="72" text-anchor="middle" font-size="12.5" font-weight="600" fill="var(--md-default-fg-color)">rock</text>
<text x="365" y="72" text-anchor="middle" font-size="12.5" font-weight="600" fill="var(--md-default-fg-color)">paper</text>
<text x="475" y="72" text-anchor="middle" font-size="12.5" font-weight="600" fill="var(--md-default-fg-color)">scissors</text>
<text x="188" y="52" text-anchor="end" font-size="13" font-weight="700" fill="var(--h-graphite)">truth</text>
<text x="188" y="126" text-anchor="end" font-size="12.5" font-weight="600" fill="var(--md-default-fg-color)">rock</text>
<text x="188" y="186" text-anchor="end" font-size="12.5" font-weight="600" fill="var(--md-default-fg-color)">paper</text>
<text x="188" y="246" text-anchor="end" font-size="12.5" font-weight="600" fill="var(--md-default-fg-color)">scissors</text>
<rect x="200" y="90" width="110" height="60" rx="6" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="255" y="126" text-anchor="middle" font-size="17" font-weight="700" fill="var(--h-cherry)">9</text>
<rect x="312" y="90" width="110" height="60" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="367" y="126" text-anchor="middle" font-size="17" fill="var(--h-graphite)">1</text>
<rect x="424" y="90" width="110" height="60" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="479" y="126" text-anchor="middle" font-size="17" fill="var(--h-graphite)">0</text>
<rect x="200" y="152" width="110" height="60" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="255" y="188" text-anchor="middle" font-size="17" fill="var(--h-graphite)">2</text>
<rect x="312" y="152" width="110" height="60" rx="6" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="367" y="188" text-anchor="middle" font-size="17" font-weight="700" fill="var(--h-cherry)">7</text>
<rect x="424" y="152" width="110" height="60" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="479" y="188" text-anchor="middle" font-size="17" fill="var(--h-graphite)">1</text>
<rect x="200" y="214" width="110" height="60" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="255" y="250" text-anchor="middle" font-size="17" fill="var(--h-graphite)">0</text>
<rect x="312" y="214" width="110" height="60" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="367" y="250" text-anchor="middle" font-size="17" font-weight="700" fill="var(--h-cherry)">3</text>
<rect x="424" y="214" width="110" height="60" rx="6" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="479" y="250" text-anchor="middle" font-size="17" font-weight="700" fill="var(--h-cherry)">7</text>
<text x="558" y="126" font-size="11.5" fill="var(--h-space)">10 rocks</text>
<text x="558" y="188" font-size="11.5" fill="var(--h-space)">10 papers</text>
<text x="558" y="250" font-size="11.5" fill="var(--h-space)">10 scissors</text>
<text x="365" y="302" text-anchor="middle" font-size="12.5" font-weight="600" fill="var(--h-graphite)">the diagonal is what it got right, everything else is a story worth telling</text>
</svg>

This model is 77 percent accurate overall. The average hides the interesting
part: scissors gets called paper three times out of ten, and nothing else is
badly broken. Now you have a specific problem to fix rather than a vague wish
for a better number.

Ask yourself why that confusion exists. Probably because a hand showing
scissors and a hand showing paper share most of their outline, and because
scissors was photographed at an angle where two fingers looked like four. That
sentence is machine learning. Not the training, the explaining.

!!! warning "Accuracy also lies when your classes are uneven"

    Suppose one in a hundred transactions is fraud. A model that says "not
    fraud" every single time scores 99 percent accuracy and catches nothing.
    Whenever someone quotes an accuracy figure, ask what the split of the data
    was.

## What goes wrong

Three failures cause most of the trouble, and you will probably produce at
least two of them this week.

**Overfitting.** The model learns your specific examples rather than the
pattern. Signs: near perfect on training data, much worse on anything new. If
every photo was taken in the same room, the model may have learned the room.

**Biased data.** The model learns exactly what you showed it, including what
you did not intend. If every hand in your training photos belongs to one
person, the model has partly learned that hand. Hiring tools trained on past
hires repeat the past. A model cannot want to be fair. It can only reflect the
examples.

**Distribution shift.** The world stops matching the training data. Lighting
changes, cameras change, people change. A model that worked in March quietly
stops working in September, and nobody notices because it still returns
confident answers.

Notice what all three have in common. None of them is a problem with the
algorithm. They are all problems with data.

## Questions to sit with

Bring answers to the project review.

1. Your model is 95 percent accurate on your own photos and 60 percent on
   another group's. Nothing changed but the photographer. What did it actually
   learn?
2. Would you rather deploy a model that is 95 percent accurate and impossible
   to explain, or 80 percent accurate and fully explainable? Does your answer
   change if it is sorting photos, approving loans, or reading X-rays?
3. Your model is confidently wrong on one image. Confidence 92 percent, answer
   incorrect. What does that tell you about what confidence measures?
4. The examples you collected encode a set of choices: who was in frame, what
   the light was, what counted as a valid gesture. Who made those choices, and
   who was left out by them?
5. If a model trained on the past predicts the future, what happens when the
   past contains something we want to change?

## Next

You now have the vocabulary. Two things follow from it.

Before the second project, read the page on the parts of this that are not
technical, because every failure above lands on somebody.

[Ethics and responsibility](ethics-and-responsibility.md){ .h-button }

And when the maths module opens, the page on what training actually does will
turn "adjust the model until it fits" into something you can picture.

[What training actually does](how-training-works.md)

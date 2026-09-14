# What AI is

<span class="badge badge--time">25 min</span>
<span class="badge badge--level">Concepts</span>
<span class="badge">No prerequisites</span>

People have been arguing about what counts as artificial intelligence for
seventy years, and they have not stopped. This page gives you the arguments, so
that when someone says "that is not real AI" you know which position they are
taking.

## Why this matters

You are about to spend nine months building these systems. You will be asked
what you do, by recruiters, by relatives, by clients. The answer "I work in AI"
is close to meaningless right now, because the term covers a spam filter and a
research programme aimed at machines that think.

There is a second reason, and it is less obvious. A definition decides things.
What gets called AI gets funded, taught, regulated, and built. What falls
outside the definition gets ignored. When people argue about the meaning of the
word they are usually arguing about something else, which is what should be
built and who decides.

## The most useful definition

The one that has survived longest comes from Russell and Norvig, whose textbook
has trained most of the field since the 1990s. They define AI as the study of
**agents**: systems that perceive an environment and act on it.

<svg viewBox="0 0 680 200" role="img" aria-labelledby="agent-title agent-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="agent-title">The agent loop</title>
<desc id="agent-desc">An agent receives percepts from the world and takes actions that change the world. A goal, or performance measure, defines what counts as success.</desc>
<defs><marker id="ag-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<text x="180" y="26" text-anchor="middle" font-size="12" font-weight="600" fill="var(--h-graphite)">a goal that defines success</text>
<line x1="180" y1="36" x2="180" y2="64" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#ag-arrow)"/>
<rect x="90" y="70" width="180" height="80" rx="12" fill="var(--h-cherry)"/>
<text x="180" y="104" text-anchor="middle" font-size="16" font-weight="700" fill="#ffffff">Agent</text>
<text x="180" y="126" text-anchor="middle" font-size="11.5" fill="#ffffff" opacity="0.88">decides what to do next</text>
<rect x="410" y="70" width="180" height="80" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="500" y="104" text-anchor="middle" font-size="16" font-weight="700" fill="var(--md-default-fg-color)">The world</text>
<text x="500" y="126" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">changes when the agent acts</text>
<line x1="406" y1="95" x2="274" y2="95" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#ag-arrow)"/>
<text x="340" y="86" text-anchor="middle" font-size="12" font-weight="600" fill="var(--h-cherry)">percepts</text>
<line x1="274" y1="128" x2="406" y2="128" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#ag-arrow)"/>
<text x="340" y="150" text-anchor="middle" font-size="12" font-weight="600" fill="var(--h-cherry)">actions</text>
</svg>

This definition is useful because it is testable. Point it at any system and
ask three questions. What does it perceive? What can it do? What is it trying
to achieve?

A thermostat perceives temperature, acts by switching a heater, and aims at a
set point. By this definition it is an agent, just an extremely simple one.
That is not a flaw in the definition. Intelligence is a spectrum, not a club
with a door.

!!! question "Try it on ChatGPT"

    What does it perceive? Text that a person typed. What does it do? Produce
    text. What is it trying to achieve? Something set by whoever trained it,
    not by itself.

    So it is an agent, but a limited one. It cannot set its own goals, it waits
    to be prompted, and in its basic form it only touches the world through
    words on a screen. This is why people call systems like it weak AI, or a
    co-pilot. Newer versions that can browse, run code, and call other tools
    move further along the spectrum, which is exactly why the word agent has
    come back into fashion.

## Four things people mean by intelligence

Underneath the arguing, most definitions of AI land in one of four boxes. They
differ on two questions. Is intelligence about thinking or about doing? And is
the standard human behaviour, or is it success?

<svg viewBox="0 0 680 270" role="img" aria-labelledby="quad-title quad-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="quad-title">Four aspirations for AI</title>
<desc id="quad-desc">A two by two grid. Rows are thinking and acting. Columns are like humans and rationally. Thinking like humans is cognitive modelling. Thinking rationally is logic and proof. Acting like humans is imitation, including the Turing test. Acting rationally is the rational agent view, where most of the field sits today.</desc>
<text x="290" y="40" text-anchor="middle" font-size="13" font-weight="700" fill="var(--h-graphite)">like humans</text>
<text x="545" y="40" text-anchor="middle" font-size="13" font-weight="700" fill="var(--h-graphite)">rationally</text>
<text x="158" y="118" text-anchor="end" font-size="13" font-weight="700" fill="var(--h-graphite)">THINK</text>
<text x="158" y="213" text-anchor="end" font-size="13" font-weight="700" fill="var(--h-graphite)">ACT</text>
<rect x="170" y="60" width="240" height="80" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="190" y="90" font-size="13.5" font-weight="700" fill="var(--md-default-fg-color)">Cognitive modelling</text>
<text x="190" y="112" font-size="11.5" fill="var(--h-graphite)">copy how people reason</text>
<text x="190" y="128" font-size="11.5" fill="var(--h-graphite)">Newell and Simon, 1960s</text>
<rect x="425" y="60" width="240" height="80" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="445" y="90" font-size="13.5" font-weight="700" fill="var(--md-default-fg-color)">Logic and proof</text>
<text x="445" y="112" font-size="11.5" fill="var(--h-graphite)">follow the laws of thought</text>
<text x="445" y="128" font-size="11.5" fill="var(--h-graphite)">Aristotle to expert systems</text>
<rect x="170" y="155" width="240" height="80" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="190" y="185" font-size="13.5" font-weight="700" fill="var(--md-default-fg-color)">Imitation</text>
<text x="190" y="207" font-size="11.5" fill="var(--h-graphite)">be indistinguishable from a person</text>
<text x="190" y="223" font-size="11.5" fill="var(--h-graphite)">the Turing test, chatbots</text>
<rect x="425" y="155" width="240" height="80" rx="10" fill="var(--h-cherry)"/>
<text x="445" y="185" font-size="13.5" font-weight="700" fill="#ffffff">Rational agents</text>
<text x="445" y="207" font-size="11.5" fill="#ffffff" opacity="0.9">do whatever achieves the goal</text>
<text x="445" y="223" font-size="11.5" fill="#ffffff" opacity="0.9">where most of the field sits now</text>
</svg>

The bottom right box won, for a practical reason. Acting rationally is the only
one of the four you can measure. You can count how often a system reaches its
goal. You cannot easily count how human its reasoning felt.

This is worth holding on to, because it explains a recurring fight. When a
system succeeds by a method no human would use, some people say that is
intelligence and others say it is a trick. Both are being consistent, they are
just standing in different boxes.

## Seventy years in one screen

<svg viewBox="0 0 680 400" role="img" aria-labelledby="tl-title tl-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="tl-title">A short timeline of AI</title>
<desc id="tl-desc">1950 Turing proposes the imitation game. 1956 the Dartmouth workshop names the field. 1970s and 1980s expert systems encode human rules. 1997 Deep Blue beats Kasparov by search. 2012 AlexNet makes deep learning the default for vision. 2017 onward, transformers lead to modern language models.</desc>
<line x1="56" y1="40" x2="56" y2="356" stroke="var(--h-steel)" stroke-width="2"/>
<circle cx="56" cy="40" r="9" fill="var(--h-cherry)"/>
<text x="84" y="36" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">1950</text>
<text x="84" y="56" font-size="12.5" fill="var(--h-graphite)">Turing asks whether a machine could imitate a person well enough to fool one</text>
<circle cx="56" cy="103" r="9" fill="var(--h-cherry)"/>
<text x="84" y="99" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">1956</text>
<text x="84" y="119" font-size="12.5" fill="var(--h-graphite)">A summer workshop at Dartmouth gives the field its name and its agenda</text>
<circle cx="56" cy="166" r="9" fill="var(--h-cherry)"/>
<text x="84" y="162" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">1970s and 1980s</text>
<text x="84" y="182" font-size="12.5" fill="var(--h-graphite)">Expert systems encode human knowledge as rules, then hit their ceiling</text>
<circle cx="56" cy="229" r="9" fill="var(--h-cherry)"/>
<text x="84" y="225" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">1997</text>
<text x="84" y="245" font-size="12.5" fill="var(--h-graphite)">Deep Blue beats Kasparov at chess, by searching rather than understanding</text>
<circle cx="56" cy="292" r="9" fill="var(--h-cherry)"/>
<text x="84" y="288" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">2012</text>
<text x="84" y="308" font-size="12.5" fill="var(--h-graphite)">AlexNet wins an image contest by a wide margin and deep learning takes over</text>
<circle cx="56" cy="356" r="9" fill="var(--h-cherry)"/>
<text x="84" y="352" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">2017 onward</text>
<text x="84" y="372" font-size="12.5" fill="var(--h-graphite)">The transformer architecture leads to the language models everyone now uses</text>
</svg>

Two things in that list are worth pausing on.

**The field has had winters.** Twice, expectations ran far ahead of results,
funding collapsed, and researchers avoided the term artificial intelligence on
their grant applications. People who lived through those are more careful with
predictions than people who did not.

**Deep Blue started an argument we are still having.** After it won, John
McCarthy, the man who coined the term artificial intelligence, complained that
it had not advanced the science at all. It played well by brute force, not by
thinking. If you only care about results, Deep Blue was a triumph. If you care
about understanding intelligence, it was a detour. Every discussion about
whether large language models understand anything is a rerun of this argument.

## Why this moment happened

Nothing in the last decade required a conceptual breakthrough that the 1980s
lacked. Neural networks are from the 1950s. Backpropagation was working in the
1980s. What changed is that three things arrived together.

<svg viewBox="0 0 680 210" role="img" aria-labelledby="why-title why-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="why-title">Why AI advanced now</title>
<desc id="why-desc">Data, computing power and better algorithms arrived at the same time, which produced the current generation of systems.</desc>
<defs><marker id="why-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<rect x="8" y="14" width="200" height="74" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="108" y="44" text-anchor="middle" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">Data</text>
<text x="108" y="66" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">the internet made examples cheap</text>
<rect x="240" y="14" width="200" height="74" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="340" y="44" text-anchor="middle" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">Compute</text>
<text x="340" y="66" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">graphics cards made training fast</text>
<rect x="472" y="14" width="200" height="74" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="572" y="44" text-anchor="middle" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">Algorithms</text>
<text x="572" y="66" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">architectures that scale with both</text>
<line x1="108" y1="92" x2="200" y2="128" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#why-arrow)"/>
<line x1="340" y1="92" x2="340" y2="128" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#why-arrow)"/>
<line x1="572" y1="92" x2="480" y2="128" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#why-arrow)"/>
<rect x="120" y="134" width="440" height="60" rx="10" fill="var(--h-cherry)"/>
<text x="340" y="160" text-anchor="middle" font-size="15" font-weight="700" fill="#ffffff">Systems that learn from examples at scale</text>
<text x="340" y="180" text-anchor="middle" font-size="11.5" fill="#ffffff" opacity="0.9">the same old ideas, finally given enough of everything</text>
</svg>

Remember this when you read that some new result is a breakthrough. Sometimes
it is. Often it is an old idea that finally had enough data and enough hardware.

## Where language models sit

A large language model is trained to predict the next piece of text, over and
over, across an enormous amount of writing. That is the whole training
objective. Everything else it appears to do comes out of doing that well.

That description is accurate and it is also unsatisfying, because the results
are hard to square with how simple the objective is. Which is precisely the
open question. Does predicting text well require something worth calling
understanding, or is fluent output just fluent output? Serious people disagree,
and you should be suspicious of anyone on either side who sounds completely
certain.

What you can say with confidence is narrower and more useful:

- It is trained on human writing, so it reflects what is in that writing,
  including the parts we would rather it did not.
- It has no way to check its output against the world. It produces text that
  fits the pattern, whether or not the content is true.
- On its own it acts only by producing text. Connected to tools, search, and
  code execution, it acts on much more.

## Definitions are decisions

Here is the idea worth taking away from this page, and it comes from Amarda
Shehu, whose framing this section follows.

A definition is never neutral. It is an act of boundary drawing. When we decide
what counts as AI, we are not only describing the world, we are deciding what
gets built, what gets funded, what gets taught, and what gets regulated.

Define AI narrowly as pattern recognition at scale and the conversation becomes
technical. Define it as systems that make decisions affecting people and the
conversation becomes about accountability. Same systems. Different boundary.
Different set of questions you are allowed to ask.

You will be inside these systems for the rest of your career. Knowing that the
boundary is a choice, made by people, is the difference between following the
field and thinking about it.

## Questions to sit with

Bring your answers to the next session. There is no key at the back of the book.

1. A system that reaches a goal by a method no human would use. Is it
   intelligent, or is it a trick? Does your answer change if the goal is
   winning chess, diagnosing a patient, or approving a loan?
2. Which of the four boxes above would you build in, if the choice were yours?
   What does your choice make easy, and what does it make impossible?
3. If a definition decides what gets funded and taught, who should get to write
   it? Researchers, companies, governments, or the people the systems act on?
4. ChatGPT passes the Turing test on most days. Turing proposed that test as
   the bar for machine intelligence. Has the bar been cleared, or was it the
   wrong bar?
5. What is a task you would refuse to let an AI system do, no matter how
   accurate it became? Try to say why in one sentence.

## Next

Now the part you will use every day: how a machine learns anything at all.

[How machines learn](what-is-ml.md){ .h-button }

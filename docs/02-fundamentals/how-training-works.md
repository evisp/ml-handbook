# What training actually does

<span class="badge badge--time">30 min</span>
<span class="badge badge--level">Concepts</span>
<span class="badge">Read alongside the maths module</span>

"Training adjusts the model until it fits the examples" is true and it explains
nothing. This page replaces it with a picture you can hold in your head. No
equations, just the shape of the idea.

## A model is a pile of numbers

Whatever else you hear, a trained model is a large collection of numbers called
weights, plus the arithmetic that combines them. Training means finding good
values for those numbers. That is the whole of it.

Start with one unit, historically called a neuron.

<svg viewBox="0 0 680 220" role="img" aria-labelledby="neuron-title neuron-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="neuron-title">A single unit</title>
<desc id="neuron-desc">Three inputs each multiplied by a weight, added together, passed through a decision step, producing one output. Training means changing the weights.</desc>
<defs><marker id="nu-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<circle cx="52" cy="50" r="22" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="52" y="50" dy="0.36em" text-anchor="middle" font-size="13" font-weight="600" fill="var(--md-default-fg-color)">x1</text>
<circle cx="52" cy="110" r="22" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="52" y="110" dy="0.36em" text-anchor="middle" font-size="13" font-weight="600" fill="var(--md-default-fg-color)">x2</text>
<circle cx="52" cy="170" r="22" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="52" y="170" dy="0.36em" text-anchor="middle" font-size="13" font-weight="600" fill="var(--md-default-fg-color)">x3</text>
<text x="52" y="204" text-anchor="middle" font-size="11" fill="var(--h-graphite)">inputs</text>
<line x1="76" y1="56" x2="246" y2="96" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#nu-arrow)"/>
<line x1="76" y1="110" x2="246" y2="110" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#nu-arrow)"/>
<line x1="76" y1="164" x2="246" y2="124" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#nu-arrow)"/>
<text x="150" y="66" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-cherry)">w1</text>
<text x="150" y="102" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-cherry)">w2</text>
<text x="150" y="162" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-cherry)">w3</text>
<rect x="252" y="76" width="190" height="68" rx="12" fill="var(--h-surface)" stroke="var(--h-cherry)"/>
<text x="347" y="104" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">add it all up,</text>
<text x="347" y="124" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">then squash the result</text>
<line x1="446" y1="110" x2="516" y2="110" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#nu-arrow)"/>
<rect x="524" y="86" width="150" height="48" rx="10" fill="var(--h-cherry)"/>
<text x="599" y="110" dy="0.36em" text-anchor="middle" font-size="13.5" font-weight="700" fill="#ffffff">one number out</text>
<text x="347" y="176" text-anchor="middle" font-size="12" font-weight="600" fill="var(--h-graphite)">training changes w1, w2 and w3, nothing else</text>
</svg>

Each input is multiplied by its weight and the results are added. A large
weight means that input matters a lot. A weight near zero means it is being
ignored. Then the sum is squashed by a simple function, which is what lets the
network represent curves rather than only straight lines.

One unit is not much. Stack hundreds of them into a layer, then stack layers on
top of each other, and the early layers end up detecting simple things like
edges while later layers combine those into shapes and eventually into "this is
a hand showing scissors". Nobody programmed that progression. It falls out of
training.

That stacking is what the word **deep** means in deep learning. Not deep as in
profound. Deep as in many layers.

## The model needs to know how wrong it is

Before anything can improve, you need a single number that says how badly the
model is doing. That number is the **loss**.

Show the model an image of paper. It outputs confidence for each class, say 20
percent rock, 50 percent paper, 30 percent scissors. The right answer was
paper, so it was partly right. The loss function turns that gap into one
number. Confident and correct gives a small loss. Confident and wrong gives a
large one.

This matters more than it looks. Choosing the loss function is choosing what
the model tries to be good at. It optimises exactly what you measure, and
nothing else. Every story about a system that technically achieved its target
while doing something nobody wanted is a story about a loss function.

## Rolling downhill

Now you have a number to make smaller. Imagine plotting the loss against one
weight. The picture is a valley, and training means walking to the bottom.

<svg viewBox="0 0 680 260" role="img" aria-labelledby="gd-title gd-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="gd-title">Gradient descent</title>
<desc id="gd-desc">A curve shaped like a valley. Error is on the vertical axis and the value of a weight on the horizontal axis. The model starts high on one side and takes repeated small steps downhill towards the lowest point.</desc>
<line x1="40" y1="228" x2="650" y2="228" stroke="var(--h-surface-line)" stroke-width="2"/>
<line x1="40" y1="30" x2="40" y2="228" stroke="var(--h-surface-line)" stroke-width="2"/>
<text x="345" y="250" text-anchor="middle" font-size="12" fill="var(--h-graphite)">the value of one weight</text>
<text x="52" y="42" font-size="12" fill="var(--h-graphite)">how wrong the model is</text>
<path d="M60 40 Q340 320 620 40" fill="none" stroke="var(--h-steel)" stroke-width="2.5"/>
<circle cx="144" cy="111" r="8" fill="var(--h-cherry)"/>
<circle cx="228" cy="158" r="8" fill="var(--h-cherry)" opacity="0.8"/>
<circle cx="295" cy="177" r="8" fill="var(--h-cherry)" opacity="0.65"/>
<circle cx="340" cy="180" r="10" fill="var(--h-cherry)"/>
<text x="130" y="96" text-anchor="end" font-size="11.5" font-weight="600" fill="var(--h-graphite)">start here, badly wrong</text>
<text x="340" y="206" text-anchor="middle" font-size="11.5" font-weight="600" fill="var(--h-cherry)">as good as it gets</text>
<text x="500" y="120" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">each dot is one small step</text>
<text x="500" y="138" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">in the direction that reduces error</text>
</svg>

At each step the model works out which way is downhill for every weight at
once, and nudges them all a little in that direction. Repeat a few thousand
times. That is gradient descent, and it is what is happening whenever you see a
progress bar during training.

Two details you will meet constantly:

**Learning rate** is the size of the step. Too small and training takes
forever. Too large and you leap across the valley and end up higher than you
started. Most training problems that look mysterious are a learning rate
problem.

**Epochs and batches.** An epoch is one pass through all your training
examples. Because looking at every example before each step is slow, the data
is split into batches and a step is taken after each batch. More epochs means
more passes, which is not automatically better, as the next section explains.

Real models have millions of weights, so the valley has millions of dimensions
and cannot be drawn. The picture still works. The maths is the same shape.

## Knowing when to stop

Here is the most useful graph in machine learning.

<svg viewBox="0 0 680 260" role="img" aria-labelledby="ov-title ov-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="ov-title">Training error against validation error</title>
<desc id="ov-desc">Two curves over training time. Error on the training data keeps falling. Error on held back data falls at first, then starts rising. The point where it turns is where the model stops learning the pattern and starts memorising.</desc>
<line x1="44" y1="222" x2="650" y2="222" stroke="var(--h-surface-line)" stroke-width="2"/>
<line x1="44" y1="28" x2="44" y2="222" stroke="var(--h-surface-line)" stroke-width="2"/>
<text x="350" y="246" text-anchor="middle" font-size="12" fill="var(--h-graphite)">training time</text>
<text x="56" y="42" font-size="12" fill="var(--h-graphite)">error</text>
<path d="M60 70 C200 150 400 192 640 204" fill="none" stroke="var(--h-steel)" stroke-width="2.5"/>
<path d="M60 84 C190 152 330 176 400 174 C480 172 550 132 640 96" fill="none" stroke="var(--h-cherry)" stroke-width="2.5"/>
<line x1="400" y1="60" x2="400" y2="200" stroke="var(--h-cherry-line)" stroke-width="2" stroke-dasharray="5 4"/>
<text x="400" y="50" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">stop here</text>
<text x="600" y="220" text-anchor="end" font-size="11.5" font-weight="600" fill="var(--h-graphite)">on data it trained on</text>
<text x="600" y="86" text-anchor="end" font-size="11.5" font-weight="600" fill="var(--h-cherry)">on data it has never seen</text>
<text x="215" y="108" font-size="11.5" fill="var(--h-graphite)">both improving: it is learning</text>
<text x="470" y="150" font-size="11.5" fill="var(--h-cherry)">one improving, one worse:</text>
<text x="470" y="166" font-size="11.5" fill="var(--h-cherry)">it is memorising</text>
</svg>

The grey line always falls. Give a model enough time and it can memorise its
training data completely, which is why the grey line on its own tells you
nothing.

The red line is the one that matters. It is the error on the validation set,
the held back data you met earlier in this module. While both lines fall, the
model is learning something general. When the red line turns upward, the model has
started memorising particulars that do not transfer. Everything after that
point makes it worse at the only job it has.

Stopping at the turn is called early stopping, and it is the most common cure
for overfitting. This is also the answer to why you hold data back at all.
Without the red line you cannot see the turn, and you will train confidently
past it.

This is the same thing you saw in the first project. Your model was excellent
on your own photographs and much worse on another group's. That gap is the two
lines separating, measured with hands instead of a graph.

## Why Teachable Machine trained in thirty seconds

Training an image model from nothing takes many hours and a lot of data. Your
project took under a minute on a laptop. Something else is going on.

Teachable Machine starts from a network already trained on millions of
photographs. That network has learned generally useful things: edges, textures,
curves, the parts that make up objects. It knows nothing about your gestures,
but it already knows how to look at images.

Training then only adjusts the last small part, the part that maps those
general features onto your three classes. Almost all the weights stay as they
were. This is called **transfer learning**, and it is why a group of students
can build a working classifier in an afternoon.

It also explains a failure you may have seen. The features came from photographs
of everyday scenes, mostly taken in good light by people in certain parts of
the world. When your test images look unlike that, the borrowed features fit
less well, and accuracy drops for reasons that have nothing to do with your
fifty photographs.

## Questions to sit with

1. A model optimises exactly the number you give it. Name a task where the
   thing you can measure is not quite the thing you want, and say what the
   model would learn to do instead.
2. Training error keeps falling while validation error rises. Your manager sees
   only the first graph and is delighted. What do you say?
3. If almost all the weights in your project's model came from somebody else's
   training on somebody else's photographs, whose model is it, and who is
   responsible for what it inherited?
4. A model with millions of weights can memorise its training set completely.
   Why does adding more training data make that harder, and is more data always
   the right answer?
5. Nobody programmed the early layers to detect edges. It emerged from
   minimising a number. Does that make the model's behaviour explainable, and
   to whom?

## Next

That completes the foundations. Everything from here is these ideas applied
with more rigour, starting with the maths that makes the downhill step
possible.

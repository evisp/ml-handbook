# Plotting

<span class="badge badge--time">70 min</span>
<span class="badge badge--level">Foundations</span>
<span class="badge">Needs: Pandas</span>

A chart is how you find the problems that `info()` and `describe()` cannot show
you. This page is about choosing the right one and reading it honestly, and then
about the two libraries you will draw it with.

## Why this matters

Summary statistics hide shape. Two columns can share a mean, a standard
deviation and a correlation and still look nothing alike when you plot them.
This is the point of Anscombe's quartet: four datasets with nearly identical
statistics, one a clean line, one a curve, one a line dragged by a single
outlier. Nothing in the numbers tells you which you have.

There is a second reason, which arrives later and matters more. Somebody will
eventually decide something because of a chart you made. Making that chart
accurate is a technical skill. Making it honest is a separate one, and this page
covers both.

## How people read a chart

Not all visual channels are read equally well. Human beings judge position
almost perfectly, length nearly as well, and everything after that badly.

<svg viewBox="0 0 680 236" role="img" aria-labelledby="enc-title enc-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="enc-title">How accurately people read each visual encoding</title>
<desc id="enc-desc">The same pair of values shown as position, length, angle, area and colour. Position and length are read most accurately. Angle, area and colour are read least accurately.</desc>
<rect x="2" y="38" width="124" height="130" rx="10" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="64" y="30" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">position</text>
<rect x="140" y="38" width="124" height="130" rx="10" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="202" y="30" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">length</text>
<rect x="278" y="38" width="124" height="130" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="340" y="30" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">angle</text>
<rect x="416" y="38" width="124" height="130" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="478" y="30" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">area</text>
<rect x="554" y="38" width="124" height="130" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="616" y="30" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">colour</text>
<line x1="24" y1="156" x2="104" y2="156" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="24" y1="156" x2="24" y2="54" stroke="var(--h-steel)" stroke-width="1.5"/>
<circle cx="50" cy="120" r="6" fill="var(--h-cherry)"/>
<circle cx="86" cy="82" r="6" fill="var(--h-cherry)"/>
<line x1="160" y1="156" x2="244" y2="156" stroke="var(--h-steel)" stroke-width="1.5"/>
<rect x="172" y="106" width="24" height="50" fill="var(--h-cherry)"/>
<rect x="210" y="76" width="24" height="80" fill="var(--h-cherry)"/>
<circle cx="312" cy="108" r="24" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<path d="M 312,108 L 312,84 A 24,24 0 0 1 335.6,112.2 Z" fill="var(--h-cherry)"/>
<circle cx="368" cy="108" r="24" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<path d="M 368,108 L 368,84 A 24,24 0 0 1 380.0,128.8 Z" fill="var(--h-cherry)"/>
<circle cx="452" cy="112" r="15" fill="var(--h-cherry)"/>
<circle cx="504" cy="112" r="24" fill="var(--h-cherry)"/>
<rect x="570" y="88" width="42" height="42" rx="6" fill="var(--h-cherry)" opacity="0.35"/>
<rect x="622" y="88" width="42" height="42" rx="6" fill="var(--h-cherry)" opacity="0.9"/>
<line x1="20" y1="192" x2="660" y2="192" stroke="var(--h-surface-line)" stroke-width="2"/>
<text x="24" y="212" font-size="12" font-weight="700" fill="var(--h-cherry)">read most accurately</text>
<text x="656" y="212" text-anchor="end" font-size="12" font-weight="700" fill="var(--h-space)">read least accurately</text>
</svg>

This one ranking explains a surprising number of chart rules. It is why a bar
chart beats a pie chart, because comparing bar heights is comparing lengths
while comparing pie slices is comparing angles. It is why bubble sizes mislead,
because area is read poorly and doubling a radius quadruples the area. And it is
why colour belongs to categories or to a rough sense of high and low, never to a
value somebody needs to read off precisely.

The rule that follows: **put what matters most on a position or length channel.**
Push the rest to colour and shape.

## Choosing a chart

Start from what you have and what you are asking, never from which chart looks
impressive.

<svg viewBox="0 0 680 236" role="img" aria-labelledby="ch-title ch-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="ch-title">Choosing a chart from what you have</title>
<desc id="ch-desc">One numeric column gives a histogram or box plot. One categorical column gives a bar chart. Two numeric columns give a scatter plot. A numeric column split by a category gives a box plot or grouped bars. Many numeric columns give a correlation heatmap.</desc>
<rect x="230" y="8" width="220" height="40" rx="10" fill="var(--h-cherry)"/>
<text x="340" y="28" dy="0.36em" text-anchor="middle" font-size="13.5" font-weight="700" fill="#ffffff">what do you have?</text>
<polyline points="340,48 340,68 68,68 68,92" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<polyline points="340,48 340,68 204,68 204,92" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<polyline points="340,48 340,92" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<polyline points="340,48 340,68 476,68 476,92" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<polyline points="340,48 340,68 612,68 612,92" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<rect x="5" y="92" width="126" height="90" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="68" y="112" text-anchor="middle" font-size="11" fill="var(--h-graphite)">one numeric</text>
<text x="68" y="126" text-anchor="middle" font-size="11" fill="var(--h-graphite)">column</text>
<text x="68" y="150" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">Histogram</text>
<text x="68" y="168" text-anchor="middle" font-size="10" fill="var(--h-space)">or box plot</text>
<rect x="141" y="92" width="126" height="90" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="204" y="112" text-anchor="middle" font-size="11" fill="var(--h-graphite)">one category</text>
<text x="204" y="126" text-anchor="middle" font-size="11" fill="var(--h-graphite)">column</text>
<text x="204" y="150" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">Bar chart</text>
<text x="204" y="168" text-anchor="middle" font-size="10" fill="var(--h-space)">counts per group</text>
<rect x="277" y="92" width="126" height="90" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="340" y="112" text-anchor="middle" font-size="11" fill="var(--h-graphite)">two numeric</text>
<text x="340" y="126" text-anchor="middle" font-size="11" fill="var(--h-graphite)">columns</text>
<text x="340" y="150" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">Scatter</text>
<text x="340" y="168" text-anchor="middle" font-size="10" fill="var(--h-space)">line if ordered in time</text>
<rect x="413" y="92" width="126" height="90" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="476" y="112" text-anchor="middle" font-size="11" fill="var(--h-graphite)">numeric split</text>
<text x="476" y="126" text-anchor="middle" font-size="11" fill="var(--h-graphite)">by category</text>
<text x="476" y="150" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">Box plot</text>
<text x="476" y="168" text-anchor="middle" font-size="10" fill="var(--h-space)">or grouped bars</text>
<rect x="549" y="92" width="126" height="90" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="612" y="112" text-anchor="middle" font-size="11" fill="var(--h-graphite)">many numeric</text>
<text x="612" y="126" text-anchor="middle" font-size="11" fill="var(--h-graphite)">columns</text>
<text x="612" y="150" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">Heatmap</text>
<text x="612" y="168" text-anchor="middle" font-size="10" fill="var(--h-space)">of correlations</text>
<text x="340" y="214" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Start from the data you hold and the question you are asking. Never from the chart that looks impressive.</text>
</svg>

## The six charts you actually need

Almost all exploratory work uses these. Learn them properly and the exotic ones
can wait until you need them.

<svg viewBox="0 0 680 430" role="img" aria-labelledby="gal-title gal-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="gal-title">Six chart types and the question each answers</title>
<desc id="gal-desc">Histogram for the shape of one numeric variable. Bar chart for comparing categories. Scatter for the relationship between two numeric variables. Line for change over time. Box plot for comparing distributions across groups. Heatmap for many pairs at once.</desc>
<rect x="5" y="14" width="214" height="192" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="112" y="40" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">Histogram</text>
<text x="112" y="174" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">what shape is this</text>
<text x="112" y="191" text-anchor="middle" font-size="10.5" fill="var(--h-space)">one numeric column</text>
<rect x="233" y="14" width="214" height="192" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="340" y="40" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">Bar chart</text>
<text x="340" y="174" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">which category is bigger</text>
<text x="340" y="191" text-anchor="middle" font-size="10.5" fill="var(--h-space)">counts or totals</text>
<rect x="461" y="14" width="214" height="192" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="568" y="40" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">Scatter</text>
<text x="568" y="174" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">do these two move together</text>
<text x="568" y="191" text-anchor="middle" font-size="10.5" fill="var(--h-space)">two numeric columns</text>
<rect x="5" y="224" width="214" height="192" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="112" y="250" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">Line</text>
<text x="112" y="384" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">how did it change over time</text>
<text x="112" y="401" text-anchor="middle" font-size="10.5" fill="var(--h-space)">a value in order</text>
<rect x="233" y="224" width="214" height="192" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="340" y="250" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">Box plot</text>
<text x="340" y="384" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">how do groups compare</text>
<text x="340" y="401" text-anchor="middle" font-size="10.5" fill="var(--h-space)">numeric split by category</text>
<rect x="461" y="224" width="214" height="192" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="568" y="250" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">Heatmap</text>
<text x="568" y="384" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">which pairs relate</text>
<text x="568" y="401" text-anchor="middle" font-size="10.5" fill="var(--h-space)">many numeric columns</text>
<line x1="35" y1="144" x2="195" y2="144" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="35" y1="144" x2="35" y2="60" stroke="var(--h-steel)" stroke-width="1.5"/>
<rect x="39" y="130" width="19" height="14" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="61" y="114" width="19" height="30" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="83" y="92" width="19" height="52" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="105" y="72" width="19" height="72" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="127" y="84" width="19" height="60" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="149" y="110" width="19" height="34" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="171" y="126" width="19" height="18" fill="var(--h-cherry)" opacity="0.85"/>
<line x1="263" y1="144" x2="423" y2="144" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="263" y1="144" x2="263" y2="60" stroke="var(--h-steel)" stroke-width="1.5"/>
<rect x="275" y="78" width="26" height="66" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="313" y="104" width="26" height="40" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="351" y="66" width="26" height="78" fill="var(--h-cherry)" opacity="0.85"/>
<rect x="389" y="118" width="26" height="26" fill="var(--h-cherry)" opacity="0.85"/>
<line x1="491" y1="144" x2="651" y2="144" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="491" y1="144" x2="491" y2="60" stroke="var(--h-steel)" stroke-width="1.5"/>
<circle cx="525.8" cy="117.9" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="581.7" cy="93.8" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="503.7" cy="129.2" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="539.7" cy="120.3" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="606.2" cy="82.7" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="510.8" cy="112.5" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="571.7" cy="88.1" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="508.4" cy="119.8" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="554.9" cy="119.6" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="562.4" cy="86.7" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="605.2" cy="84.6" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="595.7" cy="90.3" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="625.1" cy="80.8" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="536.7" cy="119.6" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="584.1" cy="104.8" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="547.5" cy="120.2" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="589.5" cy="89.7" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="642.4" cy="64.0" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="600.2" cy="107.0" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="641.6" cy="65.8" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="602.1" cy="105.4" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="578.7" cy="101.3" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="621.6" cy="79.4" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="607.7" cy="85.9" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="541.8" cy="98.5" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<circle cx="503.6" cy="131.3" r="3.2" fill="var(--h-cherry)" opacity="0.75"/>
<line x1="35" y1="354" x2="195" y2="354" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="35" y1="354" x2="35" y2="270" stroke="var(--h-steel)" stroke-width="1.5"/>
<polyline points="41,336 62,328 83,332 104,314 125,302 146,308 167,290 188,280" fill="none" stroke="var(--h-cherry)" stroke-width="2.5"/>
<line x1="263" y1="354" x2="423" y2="354" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="263" y1="354" x2="263" y2="270" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="294" y1="344" x2="294" y2="280" stroke="var(--h-steel)" stroke-width="1.5"/>
<rect x="281" y="298" width="26" height="30" fill="var(--h-cherry)" opacity="0.35" stroke="var(--h-cherry)"/>
<line x1="281" y1="314" x2="307" y2="314" stroke="var(--h-cherry)" stroke-width="2.5"/>
<line x1="340" y1="348" x2="340" y2="296" stroke="var(--h-steel)" stroke-width="1.5"/>
<rect x="327" y="312" width="26" height="24" fill="var(--h-cherry)" opacity="0.35" stroke="var(--h-cherry)"/>
<line x1="327" y1="326" x2="353" y2="326" stroke="var(--h-cherry)" stroke-width="2.5"/>
<line x1="386" y1="338" x2="386" y2="276" stroke="var(--h-steel)" stroke-width="1.5"/>
<rect x="373" y="294" width="26" height="26" fill="var(--h-cherry)" opacity="0.35" stroke="var(--h-cherry)"/>
<line x1="373" y1="306" x2="399" y2="306" stroke="var(--h-cherry)" stroke-width="2.5"/>
<rect x="501" y="272" width="32" height="20" fill="var(--h-cherry)" opacity="1.00"/>
<rect x="535" y="272" width="32" height="20" fill="var(--h-cherry)" opacity="0.62"/>
<rect x="569" y="272" width="32" height="20" fill="var(--h-cherry)" opacity="0.18"/>
<rect x="603" y="272" width="32" height="20" fill="var(--h-cherry)" opacity="0.05"/>
<rect x="501" y="294" width="32" height="20" fill="var(--h-cherry)" opacity="0.62"/>
<rect x="535" y="294" width="32" height="20" fill="var(--h-cherry)" opacity="1.00"/>
<rect x="569" y="294" width="32" height="20" fill="var(--h-cherry)" opacity="0.44"/>
<rect x="603" y="294" width="32" height="20" fill="var(--h-cherry)" opacity="0.12"/>
<rect x="501" y="316" width="32" height="20" fill="var(--h-cherry)" opacity="0.18"/>
<rect x="535" y="316" width="32" height="20" fill="var(--h-cherry)" opacity="0.44"/>
<rect x="569" y="316" width="32" height="20" fill="var(--h-cherry)" opacity="1.00"/>
<rect x="603" y="316" width="32" height="20" fill="var(--h-cherry)" opacity="0.70"/>
<rect x="501" y="338" width="32" height="20" fill="var(--h-cherry)" opacity="0.05"/>
<rect x="535" y="338" width="32" height="20" fill="var(--h-cherry)" opacity="0.12"/>
<rect x="569" y="338" width="32" height="20" fill="var(--h-cherry)" opacity="0.70"/>
<rect x="603" y="338" width="32" height="20" fill="var(--h-cherry)" opacity="1.00"/>
</svg>

A few notes that save time later.

**Histogram** answers "what shape is this column". Change the number of bins and
look again. Too few bins hide structure, too many turn the chart into noise, and
the default is a guess rather than an answer.

**Bar chart** compares categories. Start the axis at zero, always, for the reason
in the next section. Sort the bars by value unless the categories have a natural
order, because a sorted bar chart answers "which is biggest" instantly.

**Scatter** is the workhorse for two numeric columns, and the first thing to
reach for when you want to know whether a feature has any relationship with your
target.

**Line** is for a value in order, usually time. If your x axis is a category
rather than a sequence, you want bars. A line between unordered categories
implies a trend that does not exist.

**Box plot** compares distributions across groups, and shows the median,
quartiles and outliers at once. It is the fastest way to see whether a category
actually separates your data.

**Heatmap** shows many pairs at once, most often correlations between every pair
of numeric columns. Use a diverging colour scale centred at zero, so that
positive and negative are visually opposite.

And the one to avoid: **pie charts**. They ask you to compare angles, which is
near the bottom of the ladder. Two or three slices is survivable. More than that
and a sorted bar chart is better in every way.

## Charts that lie

Most misleading charts are not fraud. They are somebody reaching for a default
without thinking about it.

<svg viewBox="0 0 680 268" role="img" aria-labelledby="lie-title lie-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="lie-title">The same numbers on two different axes</title>
<desc id="lie-desc">Four values, 98, 99, 100 and 102, drawn with the vertical axis starting at zero and again with it starting at 95. The second chart makes a two percent difference look enormous.</desc>
<text x="170" y="26" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">axis starts at 0</text>
<line x1="60" y1="200" x2="300" y2="200" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="60" y1="200" x2="60" y2="50" stroke="var(--h-steel)" stroke-width="1.5"/>
<text x="52" y="204" text-anchor="end" font-size="10" fill="var(--h-space)">0</text>
<text x="52" y="56" text-anchor="end" font-size="10" fill="var(--h-space)">110</text>
<rect x="74" y="66.4" width="38" height="133.6" fill="var(--h-cherry)" opacity="0.85"/>
<text x="93" y="216" text-anchor="middle" font-size="10" fill="var(--h-space)">98</text>
<rect x="128" y="65.0" width="38" height="135.0" fill="var(--h-cherry)" opacity="0.85"/>
<text x="147" y="216" text-anchor="middle" font-size="10" fill="var(--h-space)">99</text>
<rect x="182" y="63.6" width="38" height="136.4" fill="var(--h-cherry)" opacity="0.85"/>
<text x="201" y="216" text-anchor="middle" font-size="10" fill="var(--h-space)">100</text>
<rect x="236" y="60.9" width="38" height="139.1" fill="var(--h-cherry)" opacity="0.85"/>
<text x="255" y="216" text-anchor="middle" font-size="10" fill="var(--h-space)">102</text>
<text x="510" y="26" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">axis starts at 95</text>
<line x1="400" y1="200" x2="640" y2="200" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="400" y1="200" x2="400" y2="50" stroke="var(--h-steel)" stroke-width="1.5"/>
<text x="392" y="204" text-anchor="end" font-size="10" fill="var(--h-space)">95</text>
<text x="392" y="56" text-anchor="end" font-size="10" fill="var(--h-space)">103</text>
<rect x="414" y="143.8" width="38" height="56.2" fill="var(--h-cherry)" opacity="0.85"/>
<text x="433" y="216" text-anchor="middle" font-size="10" fill="var(--h-space)">98</text>
<rect x="468" y="125.0" width="38" height="75.0" fill="var(--h-cherry)" opacity="0.85"/>
<text x="487" y="216" text-anchor="middle" font-size="10" fill="var(--h-space)">99</text>
<rect x="522" y="106.2" width="38" height="93.8" fill="var(--h-cherry)" opacity="0.85"/>
<text x="541" y="216" text-anchor="middle" font-size="10" fill="var(--h-space)">100</text>
<rect x="576" y="68.8" width="38" height="131.2" fill="var(--h-cherry)" opacity="0.85"/>
<text x="595" y="216" text-anchor="middle" font-size="10" fill="var(--h-space)">102</text>
<text x="340" y="248" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Same four numbers. The chart on the right is not a chart, it is an argument.</text>
</svg>

**The truncated axis.** Bar charts encode value as length, so cutting the axis
breaks the encoding: a bar twice as tall no longer means twice as much. Start
bar charts at zero. Line charts are different, because they encode change rather
than magnitude, so a truncated axis there is often the honest choice. Label it
clearly either way.

**Dual axes.** Two lines with two different y scales can be made to cross,
diverge or agree simply by choosing the scales. If you need two units on one
chart, stack two charts instead.

**Cherry-picked ranges.** The same series over five years and over five months
can tell opposite stories. Show the range that answers the question, and say
what range you chose.

**Colour that carries meaning it should not.** Red for one group and green for
another implies bad and good. Around one man in twelve cannot reliably tell
those two apart anyway. Use a colourblind-safe palette and keep semantic colours
for things that really are semantic.

!!! ml "ML connection"

    You will present model results to people who cannot read your code and will
    trust the picture. An accuracy of 94 percent on a chart with no baseline
    looks excellent until somebody notices that always guessing the majority
    class gives 93. Whether that baseline appears on the chart is a choice you
    make, and it is the same choice as where the axis starts.

## When there is too much data

<svg viewBox="0 0 680 246" role="img" aria-labelledby="op-title op-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="op-title">Overplotting and the fix</title>
<desc id="op-desc">The same scatter plot drawn twice. With large opaque points the data is a solid blob. With small semi transparent points the shape of the relationship and the density become visible.</desc>
<text x="155" y="26" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-graphite)">large opaque points</text>
<rect x="20" y="34" width="270" height="164" rx="8" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<circle cx="126.3" cy="163.2" r="5" fill="var(--h-cherry)"/>
<circle cx="43.6" cy="168.6" r="5" fill="var(--h-cherry)"/>
<circle cx="49.5" cy="173.4" r="5" fill="var(--h-cherry)"/>
<circle cx="145.3" cy="135.3" r="5" fill="var(--h-cherry)"/>
<circle cx="141.0" cy="126.0" r="5" fill="var(--h-cherry)"/>
<circle cx="67.7" cy="127.5" r="5" fill="var(--h-cherry)"/>
<circle cx="142.2" cy="143.0" r="5" fill="var(--h-cherry)"/>
<circle cx="58.0" cy="164.6" r="5" fill="var(--h-cherry)"/>
<circle cx="82.1" cy="173.4" r="5" fill="var(--h-cherry)"/>
<circle cx="235.5" cy="109.4" r="5" fill="var(--h-cherry)"/>
<circle cx="174.7" cy="124.2" r="5" fill="var(--h-cherry)"/>
<circle cx="63.3" cy="177.1" r="5" fill="var(--h-cherry)"/>
<circle cx="185.6" cy="111.3" r="5" fill="var(--h-cherry)"/>
<circle cx="163.4" cy="119.5" r="5" fill="var(--h-cherry)"/>
<circle cx="219.4" cy="81.3" r="5" fill="var(--h-cherry)"/>
<circle cx="143.9" cy="111.7" r="5" fill="var(--h-cherry)"/>
<circle cx="198.5" cy="143.3" r="5" fill="var(--h-cherry)"/>
<circle cx="46.6" cy="180.3" r="5" fill="var(--h-cherry)"/>
<circle cx="71.0" cy="163.6" r="5" fill="var(--h-cherry)"/>
<circle cx="195.4" cy="85.7" r="5" fill="var(--h-cherry)"/>
<circle cx="232.9" cy="106.4" r="5" fill="var(--h-cherry)"/>
<circle cx="163.2" cy="106.1" r="5" fill="var(--h-cherry)"/>
<circle cx="248.1" cy="81.4" r="5" fill="var(--h-cherry)"/>
<circle cx="213.0" cy="116.2" r="5" fill="var(--h-cherry)"/>
<circle cx="214.8" cy="107.5" r="5" fill="var(--h-cherry)"/>
<circle cx="89.8" cy="164.3" r="5" fill="var(--h-cherry)"/>
<circle cx="36.9" cy="179.8" r="5" fill="var(--h-cherry)"/>
<circle cx="89.2" cy="177.0" r="5" fill="var(--h-cherry)"/>
<circle cx="70.0" cy="181.7" r="5" fill="var(--h-cherry)"/>
<circle cx="253.8" cy="92.6" r="5" fill="var(--h-cherry)"/>
<circle cx="185.6" cy="99.8" r="5" fill="var(--h-cherry)"/>
<circle cx="236.1" cy="101.3" r="5" fill="var(--h-cherry)"/>
<circle cx="148.8" cy="113.8" r="5" fill="var(--h-cherry)"/>
<circle cx="79.3" cy="173.8" r="5" fill="var(--h-cherry)"/>
<circle cx="75.1" cy="156.3" r="5" fill="var(--h-cherry)"/>
<circle cx="115.0" cy="151.5" r="5" fill="var(--h-cherry)"/>
<circle cx="93.3" cy="123.4" r="5" fill="var(--h-cherry)"/>
<circle cx="179.5" cy="101.9" r="5" fill="var(--h-cherry)"/>
<circle cx="223.8" cy="117.0" r="5" fill="var(--h-cherry)"/>
<circle cx="237.0" cy="73.9" r="5" fill="var(--h-cherry)"/>
<circle cx="125.0" cy="141.3" r="5" fill="var(--h-cherry)"/>
<circle cx="190.7" cy="112.5" r="5" fill="var(--h-cherry)"/>
<circle cx="94.7" cy="169.5" r="5" fill="var(--h-cherry)"/>
<circle cx="60.1" cy="174.2" r="5" fill="var(--h-cherry)"/>
<circle cx="61.3" cy="171.6" r="5" fill="var(--h-cherry)"/>
<circle cx="235.1" cy="77.9" r="5" fill="var(--h-cherry)"/>
<circle cx="90.4" cy="164.7" r="5" fill="var(--h-cherry)"/>
<circle cx="94.0" cy="125.5" r="5" fill="var(--h-cherry)"/>
<circle cx="141.3" cy="129.4" r="5" fill="var(--h-cherry)"/>
<circle cx="57.5" cy="179.2" r="5" fill="var(--h-cherry)"/>
<circle cx="232.2" cy="91.8" r="5" fill="var(--h-cherry)"/>
<circle cx="251.0" cy="73.8" r="5" fill="var(--h-cherry)"/>
<circle cx="181.8" cy="123.6" r="5" fill="var(--h-cherry)"/>
<circle cx="278" cy="53.7" r="5" fill="var(--h-cherry)"/>
<circle cx="94.4" cy="158.5" r="5" fill="var(--h-cherry)"/>
<circle cx="193.7" cy="89.4" r="5" fill="var(--h-cherry)"/>
<circle cx="120.1" cy="172.5" r="5" fill="var(--h-cherry)"/>
<circle cx="278" cy="48.5" r="5" fill="var(--h-cherry)"/>
<circle cx="227.6" cy="78.5" r="5" fill="var(--h-cherry)"/>
<circle cx="157.0" cy="126.1" r="5" fill="var(--h-cherry)"/>
<circle cx="44.4" cy="189.1" r="5" fill="var(--h-cherry)"/>
<circle cx="214.0" cy="99.1" r="5" fill="var(--h-cherry)"/>
<circle cx="278" cy="73.9" r="5" fill="var(--h-cherry)"/>
<circle cx="125.7" cy="151.2" r="5" fill="var(--h-cherry)"/>
<circle cx="90.8" cy="179.8" r="5" fill="var(--h-cherry)"/>
<circle cx="255.7" cy="65.6" r="5" fill="var(--h-cherry)"/>
<circle cx="192.0" cy="101.8" r="5" fill="var(--h-cherry)"/>
<circle cx="212.6" cy="92.3" r="5" fill="var(--h-cherry)"/>
<circle cx="203.8" cy="98.9" r="5" fill="var(--h-cherry)"/>
<circle cx="209.0" cy="118.2" r="5" fill="var(--h-cherry)"/>
<circle cx="252.2" cy="83.0" r="5" fill="var(--h-cherry)"/>
<circle cx="256.4" cy="66.2" r="5" fill="var(--h-cherry)"/>
<circle cx="86.9" cy="190" r="5" fill="var(--h-cherry)"/>
<circle cx="241.4" cy="115.1" r="5" fill="var(--h-cherry)"/>
<circle cx="258.3" cy="59.8" r="5" fill="var(--h-cherry)"/>
<circle cx="167.8" cy="121.6" r="5" fill="var(--h-cherry)"/>
<circle cx="253.2" cy="57.4" r="5" fill="var(--h-cherry)"/>
<circle cx="228.8" cy="90.4" r="5" fill="var(--h-cherry)"/>
<circle cx="232.6" cy="100.9" r="5" fill="var(--h-cherry)"/>
<circle cx="108.5" cy="169.0" r="5" fill="var(--h-cherry)"/>
<circle cx="93.2" cy="155.6" r="5" fill="var(--h-cherry)"/>
<circle cx="239.9" cy="94.0" r="5" fill="var(--h-cherry)"/>
<circle cx="186.2" cy="106.4" r="5" fill="var(--h-cherry)"/>
<circle cx="233.8" cy="78.8" r="5" fill="var(--h-cherry)"/>
<circle cx="175.4" cy="124.4" r="5" fill="var(--h-cherry)"/>
<circle cx="107.2" cy="160.6" r="5" fill="var(--h-cherry)"/>
<circle cx="57.4" cy="165.3" r="5" fill="var(--h-cherry)"/>
<circle cx="160.2" cy="136.0" r="5" fill="var(--h-cherry)"/>
<circle cx="169.2" cy="111.5" r="5" fill="var(--h-cherry)"/>
<circle cx="169.0" cy="131.3" r="5" fill="var(--h-cherry)"/>
<circle cx="199.7" cy="94.1" r="5" fill="var(--h-cherry)"/>
<circle cx="227.7" cy="87.4" r="5" fill="var(--h-cherry)"/>
<circle cx="164.1" cy="112.0" r="5" fill="var(--h-cherry)"/>
<circle cx="182.8" cy="109.6" r="5" fill="var(--h-cherry)"/>
<circle cx="170.2" cy="118.5" r="5" fill="var(--h-cherry)"/>
<circle cx="251.7" cy="79.2" r="5" fill="var(--h-cherry)"/>
<circle cx="193.8" cy="107.8" r="5" fill="var(--h-cherry)"/>
<circle cx="82.5" cy="176.9" r="5" fill="var(--h-cherry)"/>
<circle cx="57.0" cy="178.2" r="5" fill="var(--h-cherry)"/>
<circle cx="200.3" cy="73.0" r="5" fill="var(--h-cherry)"/>
<circle cx="71.2" cy="140.0" r="5" fill="var(--h-cherry)"/>
<circle cx="100.0" cy="136.1" r="5" fill="var(--h-cherry)"/>
<circle cx="104.0" cy="151.1" r="5" fill="var(--h-cherry)"/>
<circle cx="178.5" cy="124.5" r="5" fill="var(--h-cherry)"/>
<circle cx="61.8" cy="170.3" r="5" fill="var(--h-cherry)"/>
<circle cx="122.1" cy="155.9" r="5" fill="var(--h-cherry)"/>
<circle cx="223.8" cy="103.0" r="5" fill="var(--h-cherry)"/>
<circle cx="153.8" cy="133.2" r="5" fill="var(--h-cherry)"/>
<circle cx="178.4" cy="110.9" r="5" fill="var(--h-cherry)"/>
<circle cx="275.5" cy="42" r="5" fill="var(--h-cherry)"/>
<circle cx="63.7" cy="173.0" r="5" fill="var(--h-cherry)"/>
<circle cx="218.2" cy="102.7" r="5" fill="var(--h-cherry)"/>
<circle cx="159.1" cy="117.9" r="5" fill="var(--h-cherry)"/>
<circle cx="118.0" cy="180.5" r="5" fill="var(--h-cherry)"/>
<circle cx="169.4" cy="110.6" r="5" fill="var(--h-cherry)"/>
<circle cx="47.7" cy="158.1" r="5" fill="var(--h-cherry)"/>
<circle cx="75.0" cy="163.5" r="5" fill="var(--h-cherry)"/>
<circle cx="248.2" cy="107.6" r="5" fill="var(--h-cherry)"/>
<circle cx="65.3" cy="159.3" r="5" fill="var(--h-cherry)"/>
<circle cx="87.8" cy="130.7" r="5" fill="var(--h-cherry)"/>
<circle cx="113.4" cy="164.7" r="5" fill="var(--h-cherry)"/>
<circle cx="101.3" cy="159.8" r="5" fill="var(--h-cherry)"/>
<circle cx="55.2" cy="187.7" r="5" fill="var(--h-cherry)"/>
<circle cx="110.8" cy="133.2" r="5" fill="var(--h-cherry)"/>
<circle cx="160.7" cy="138.3" r="5" fill="var(--h-cherry)"/>
<circle cx="44.2" cy="180.8" r="5" fill="var(--h-cherry)"/>
<circle cx="200.0" cy="96.1" r="5" fill="var(--h-cherry)"/>
<circle cx="155.3" cy="124.7" r="5" fill="var(--h-cherry)"/>
<circle cx="213.5" cy="97.6" r="5" fill="var(--h-cherry)"/>
<circle cx="218.9" cy="100.0" r="5" fill="var(--h-cherry)"/>
<circle cx="210.9" cy="102.7" r="5" fill="var(--h-cherry)"/>
<circle cx="226.1" cy="66.5" r="5" fill="var(--h-cherry)"/>
<circle cx="130.4" cy="139.9" r="5" fill="var(--h-cherry)"/>
<circle cx="90.6" cy="177.0" r="5" fill="var(--h-cherry)"/>
<circle cx="101.8" cy="157.6" r="5" fill="var(--h-cherry)"/>
<circle cx="247.8" cy="70.1" r="5" fill="var(--h-cherry)"/>
<circle cx="105.4" cy="162.3" r="5" fill="var(--h-cherry)"/>
<circle cx="154.0" cy="144.0" r="5" fill="var(--h-cherry)"/>
<circle cx="137.0" cy="140.8" r="5" fill="var(--h-cherry)"/>
<circle cx="167.1" cy="161.3" r="5" fill="var(--h-cherry)"/>
<circle cx="110.8" cy="146.5" r="5" fill="var(--h-cherry)"/>
<circle cx="111.4" cy="141.0" r="5" fill="var(--h-cherry)"/>
<circle cx="84.8" cy="157.8" r="5" fill="var(--h-cherry)"/>
<circle cx="112.7" cy="159.6" r="5" fill="var(--h-cherry)"/>
<circle cx="61.4" cy="177.3" r="5" fill="var(--h-cherry)"/>
<circle cx="78.8" cy="144.3" r="5" fill="var(--h-cherry)"/>
<circle cx="200.4" cy="76.2" r="5" fill="var(--h-cherry)"/>
<circle cx="232.6" cy="92.4" r="5" fill="var(--h-cherry)"/>
<circle cx="278" cy="92.4" r="5" fill="var(--h-cherry)"/>
<circle cx="213.5" cy="117.5" r="5" fill="var(--h-cherry)"/>
<circle cx="229.3" cy="63.2" r="5" fill="var(--h-cherry)"/>
<circle cx="237.7" cy="105.6" r="5" fill="var(--h-cherry)"/>
<circle cx="168.9" cy="99.6" r="5" fill="var(--h-cherry)"/>
<circle cx="204.5" cy="72.1" r="5" fill="var(--h-cherry)"/>
<circle cx="193.5" cy="94.0" r="5" fill="var(--h-cherry)"/>
<circle cx="56.0" cy="187.8" r="5" fill="var(--h-cherry)"/>
<circle cx="73.3" cy="150.9" r="5" fill="var(--h-cherry)"/>
<circle cx="169.5" cy="93.7" r="5" fill="var(--h-cherry)"/>
<circle cx="177.6" cy="126.8" r="5" fill="var(--h-cherry)"/>
<circle cx="194.8" cy="97.3" r="5" fill="var(--h-cherry)"/>
<circle cx="212.6" cy="118.0" r="5" fill="var(--h-cherry)"/>
<circle cx="107.8" cy="157.9" r="5" fill="var(--h-cherry)"/>
<circle cx="214.1" cy="125.0" r="5" fill="var(--h-cherry)"/>
<circle cx="250.7" cy="73.3" r="5" fill="var(--h-cherry)"/>
<circle cx="140.5" cy="102.3" r="5" fill="var(--h-cherry)"/>
<circle cx="178.4" cy="107.1" r="5" fill="var(--h-cherry)"/>
<circle cx="73.3" cy="190" r="5" fill="var(--h-cherry)"/>
<circle cx="108.0" cy="145.5" r="5" fill="var(--h-cherry)"/>
<circle cx="51.5" cy="190" r="5" fill="var(--h-cherry)"/>
<circle cx="194.0" cy="92.0" r="5" fill="var(--h-cherry)"/>
<circle cx="143.5" cy="127.1" r="5" fill="var(--h-cherry)"/>
<circle cx="74.6" cy="160.4" r="5" fill="var(--h-cherry)"/>
<circle cx="267.4" cy="71.2" r="5" fill="var(--h-cherry)"/>
<circle cx="161.2" cy="91.5" r="5" fill="var(--h-cherry)"/>
<circle cx="142.2" cy="141.5" r="5" fill="var(--h-cherry)"/>
<circle cx="262.0" cy="96.5" r="5" fill="var(--h-cherry)"/>
<circle cx="38.4" cy="158.5" r="5" fill="var(--h-cherry)"/>
<circle cx="77.6" cy="148.2" r="5" fill="var(--h-cherry)"/>
<circle cx="241.0" cy="71.3" r="5" fill="var(--h-cherry)"/>
<circle cx="243.3" cy="81.6" r="5" fill="var(--h-cherry)"/>
<text x="510" y="26" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-cherry)">small, semi transparent</text>
<rect x="375" y="34" width="270" height="164" rx="8" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<circle cx="481.3" cy="163.2" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="398.6" cy="168.6" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="404.5" cy="173.4" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="500.3" cy="135.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="496.0" cy="126.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="422.7" cy="127.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="497.2" cy="143.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="413.0" cy="164.6" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="437.1" cy="173.4" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="590.5" cy="109.4" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="529.7" cy="124.2" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="418.3" cy="177.1" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="540.6" cy="111.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="518.4" cy="119.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="574.4" cy="81.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="498.9" cy="111.7" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="553.5" cy="143.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="401.6" cy="180.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="426.0" cy="163.6" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="550.4" cy="85.7" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="587.9" cy="106.4" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="518.2" cy="106.1" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="603.1" cy="81.4" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="568.0" cy="116.2" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="569.8" cy="107.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="444.8" cy="164.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="391.9" cy="179.8" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="444.2" cy="177.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="425.0" cy="181.7" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="608.8" cy="92.6" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="540.6" cy="99.8" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="591.1" cy="101.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="503.8" cy="113.8" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="434.3" cy="173.8" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="430.1" cy="156.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="470.0" cy="151.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="448.3" cy="123.4" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="534.5" cy="101.9" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="578.8" cy="117.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="592.0" cy="73.9" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="480.0" cy="141.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="545.7" cy="112.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="449.7" cy="169.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="415.1" cy="174.2" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="416.3" cy="171.6" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="590.1" cy="77.9" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="445.4" cy="164.7" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="449.0" cy="125.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="496.3" cy="129.4" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="412.5" cy="179.2" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="587.2" cy="91.8" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="606.0" cy="73.8" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="536.8" cy="123.6" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="633" cy="53.7" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="449.4" cy="158.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="548.7" cy="89.4" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="475.1" cy="172.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="633" cy="48.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="582.6" cy="78.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="512.0" cy="126.1" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="399.4" cy="189.1" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="569.0" cy="99.1" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="633" cy="73.9" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="480.7" cy="151.2" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="445.8" cy="179.8" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="610.7" cy="65.6" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="547.0" cy="101.8" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="567.6" cy="92.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="558.8" cy="98.9" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="564.0" cy="118.2" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="607.2" cy="83.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="611.4" cy="66.2" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="441.9" cy="190" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="596.4" cy="115.1" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="613.3" cy="59.8" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="522.8" cy="121.6" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="608.2" cy="57.4" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="583.8" cy="90.4" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="587.6" cy="100.9" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="463.5" cy="169.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="448.2" cy="155.6" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="594.9" cy="94.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="541.2" cy="106.4" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="588.8" cy="78.8" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="530.4" cy="124.4" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="462.2" cy="160.6" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="412.4" cy="165.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="515.2" cy="136.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="524.2" cy="111.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="524.0" cy="131.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="554.7" cy="94.1" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="582.7" cy="87.4" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="519.1" cy="112.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="537.8" cy="109.6" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="525.2" cy="118.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="606.7" cy="79.2" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="548.8" cy="107.8" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="437.5" cy="176.9" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="412.0" cy="178.2" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="555.3" cy="73.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="426.2" cy="140.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="455.0" cy="136.1" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="459.0" cy="151.1" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="533.5" cy="124.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="416.8" cy="170.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="477.1" cy="155.9" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="578.8" cy="103.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="508.8" cy="133.2" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="533.4" cy="110.9" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="630.5" cy="42" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="418.7" cy="173.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="573.2" cy="102.7" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="514.1" cy="117.9" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="473.0" cy="180.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="524.4" cy="110.6" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="402.7" cy="158.1" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="430.0" cy="163.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="603.2" cy="107.6" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="420.3" cy="159.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="442.8" cy="130.7" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="468.4" cy="164.7" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="456.3" cy="159.8" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="410.2" cy="187.7" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="465.8" cy="133.2" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="515.7" cy="138.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="399.2" cy="180.8" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="555.0" cy="96.1" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="510.3" cy="124.7" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="568.5" cy="97.6" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="573.9" cy="100.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="565.9" cy="102.7" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="581.1" cy="66.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="485.4" cy="139.9" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="445.6" cy="177.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="456.8" cy="157.6" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="602.8" cy="70.1" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="460.4" cy="162.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="509.0" cy="144.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="492.0" cy="140.8" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="522.1" cy="161.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="465.8" cy="146.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="466.4" cy="141.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="439.8" cy="157.8" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="467.7" cy="159.6" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="416.4" cy="177.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="433.8" cy="144.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="555.4" cy="76.2" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="587.6" cy="92.4" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="633" cy="92.4" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="568.5" cy="117.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="584.3" cy="63.2" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="592.7" cy="105.6" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="523.9" cy="99.6" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="559.5" cy="72.1" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="548.5" cy="94.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="411.0" cy="187.8" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="428.3" cy="150.9" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="524.5" cy="93.7" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="532.6" cy="126.8" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="549.8" cy="97.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="567.6" cy="118.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="462.8" cy="157.9" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="569.1" cy="125.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="605.7" cy="73.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="495.5" cy="102.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="533.4" cy="107.1" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="428.3" cy="190" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="463.0" cy="145.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="406.5" cy="190" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="549.0" cy="92.0" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="498.5" cy="127.1" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="429.6" cy="160.4" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="622.4" cy="71.2" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="516.2" cy="91.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="497.2" cy="141.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="617.0" cy="96.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="393.4" cy="158.5" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="432.6" cy="148.2" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="596.0" cy="71.3" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<circle cx="598.3" cy="81.6" r="2.6" fill="var(--h-cherry)" opacity="0.32"/>
<text x="340" y="228" text-anchor="middle" font-size="12" fill="var(--h-graphite)">same data, same chart type. Only the point size and alpha changed.</text>
</svg>

With enough points a scatter plot becomes a solid shape and stops carrying
information. Three fixes, in order of how often they work:

- Make points smaller and semi transparent, so density shows as darkness.
- Sample a few thousand rows. You are looking for shape, not for every record.
- Switch to a chart that bins for you, such as a hexbin or a 2D histogram.

## Part two: drawing it

Two libraries, and they are not rivals. **Matplotlib** is the engine and gives
you control over every element. **Seaborn** sits on top of it, produces
statistical charts in one line and has much better defaults.

The practical division: explore with seaborn, finish with matplotlib. Seaborn
returns matplotlib objects, so you can always drop down a level.

### Figure and axes

This is the concept that confuses every matplotlib beginner, and it takes one
picture.

<svg viewBox="0 0 680 250" role="img" aria-labelledby="fa-title fa-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="fa-title">A matplotlib figure containing two axes</title>
<desc id="fa-desc">The figure is the whole canvas. Each axes is one plot inside it, with its own x and y axis, title and labels.</desc>
<rect x="30" y="34" width="620" height="170" rx="12" fill="var(--h-surface)" stroke="var(--h-cherry)" stroke-width="2" stroke-dasharray="6 4"/>
<text x="44" y="26" font-size="12.5" font-weight="700" fill="var(--h-cherry)">fig, the whole canvas</text>
<rect x="60" y="58" width="260" height="124" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<line x1="100" y1="156" x2="296" y2="156" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="100" y1="156" x2="100" y2="80" stroke="var(--h-steel)" stroke-width="1.5"/>
<polyline points="108,146 148,120 188,128 228,96 268,86" fill="none" stroke="var(--h-cherry)" stroke-width="2.5"/>
<text x="190" y="74" text-anchor="middle" font-size="11" font-weight="700" fill="var(--md-default-fg-color)">ax[0]</text>
<rect x="360" y="58" width="260" height="124" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<line x1="400" y1="156" x2="596" y2="156" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="400" y1="156" x2="400" y2="80" stroke="var(--h-steel)" stroke-width="1.5"/>
<rect x="416" y="128" width="30" height="28" fill="var(--h-cherry)" opacity="0.8"/>
<rect x="460" y="102" width="30" height="54" fill="var(--h-cherry)" opacity="0.8"/>
<rect x="504" y="116" width="30" height="40" fill="var(--h-cherry)" opacity="0.8"/>
<rect x="548" y="90" width="30" height="66" fill="var(--h-cherry)" opacity="0.8"/>
<text x="490" y="74" text-anchor="middle" font-size="11" font-weight="700" fill="var(--md-default-fg-color)">ax[1]</text>
<text x="340" y="228" text-anchor="middle" font-size="12" fill="var(--h-graphite)">fig, ax = plt.subplots(1, 2) gives you one figure and two axes. You draw on the axes, you save the figure.</text>
</svg>

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["month"], df["sales"])
ax.set_title("Monthly sales")
ax.set_xlabel("Month")
ax.set_ylabel("Sales (EUR)")
fig.savefig("sales.png", dpi=150, bbox_inches="tight")
```

You will also see the shorter style, `plt.plot(...)` followed by `plt.title(...)`,
which draws on whichever figure happens to be current. It is fine for a quick
look and it falls apart the moment you have two charts. Learn the version above
and you never have to unlearn anything.

Several charts side by side:

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df["salary"], bins=30)
axes[0].set_title("Salary distribution")
axes[1].scatter(df["age"], df["salary"], s=8, alpha=0.3)
axes[1].set_title("Age against salary")
fig.tight_layout()
```

`tight_layout` stops the labels of one chart overlapping the next one. Call it
before saving, every time.

!!! tip "Matplotlib goes much deeper than this page"

    Everything above is enough for the whole of trimester one. When you need
    custom axes, annotations or precise styling, the official
    [quick start guide](https://matplotlib.org/stable/users/explain/quick_start.html)
    and the
    [anatomy of a figure](https://matplotlib.org/stable/gallery/showcase/anatomy.html)
    page are the two references worth bookmarking. Do not read them now. Read
    them the first time a chart will not do what you want.

### Seaborn, one line each

```python
import seaborn as sns

sns.histplot(data=df, x="salary", bins=30)
sns.countplot(data=df, x="city")
sns.scatterplot(data=df, x="age", y="salary", hue="city", alpha=0.4)
sns.lineplot(data=df, x="month", y="sales")
sns.boxplot(data=df, x="city", y="salary")
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", center=0)
```

Three arguments do most of the work. `data` takes the DataFrame, `x` and `y` take
column names as strings, and `hue` splits by a category and adds the legend for
you. That pattern is the same across every seaborn function, which is most of
why it is worth learning.

Set sensible defaults once, at the top of a notebook:

```python
sns.set_theme(style="whitegrid", palette="colorblind")
```

### Finishing a chart

A chart is finished when somebody who did not make it can read it without
asking you anything.

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="city", y="salary", ax=ax)
ax.set_title("Salary by city, 2026 survey")
ax.set_xlabel("City")
ax.set_ylabel("Monthly salary (EUR)")
fig.savefig("salary_by_city.png", dpi=150, bbox_inches="tight")
```

Note `ax=ax`, which tells seaborn to draw on the axes you created rather than
making its own. That is how the two libraries combine.

The checklist: a title that states the finding rather than the variables, axis
labels with units, a legend when there is more than one series, and a source or
date if the chart leaves your machine.

## The charts you will make in every project

Five of them, from here to the end of the programme.

| Chart | What it tells you | When |
|---|---|---|
| Distribution of the target | Is it skewed, is it balanced | Before any modelling |
| Correlation heatmap | Which features move together | Feature selection |
| Missing value map | Whether gaps cluster in one group | Cleaning |
| Learning curve | Whether more data would help | Trimester two |
| Confusion matrix | Which classes get confused with which | Evaluation |

The first three you can draw today:

```python
sns.histplot(data=df, x="target", bins=30)
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)
sns.heatmap(df.isna(), cbar=False)
```

The missing value map is the one people skip. Gaps arranged in horizontal bands
mean whole rows failed, while a solid block in one column means a field stopped
being collected on a certain date. Both change what you should do about them,
and `isna().sum()` cannot tell them apart.

## When it goes wrong

??? note "Nothing appears in my notebook"

    In a script you need `plt.show()` at the end. In Jupyter, put the plotting
    call last in the cell, or add `plt.show()`, otherwise you may see only the
    object's text representation.

??? note "My labels are cut off when I save the figure"

    Use `fig.savefig("name.png", bbox_inches="tight")`, and call
    `fig.tight_layout()` before saving when you have several subplots.

??? note "Every chart appears on top of the previous one"

    You are using the `plt.` style and drawing into whichever figure is current.
    Create a new one explicitly with `fig, ax = plt.subplots()` for each chart.

??? note "The x axis labels overlap and are unreadable"

    Either rotate them with `ax.tick_params(axis="x", rotation=45)` or swap the
    axes so the categories run down the side, which is usually the better fix
    when the names are long.

??? note "My scatter plot is a solid blob"

    Overplotting. Reduce `s`, set `alpha=0.2`, or sample the rows. See the
    section above.

??? note "The correlation heatmap is all one colour"

    You are using a sequential colour map. Use a diverging one centred at zero,
    such as `cmap="coolwarm", center=0`, so that positive and negative read as
    opposites.

## Check yourself

Use any dataset you have to hand.

1. Plot one numeric column as a histogram with three different bin counts. Say
   what each one hides and what it reveals.
2. Draw the same comparison as a bar chart and as a pie chart. Which one lets
   you rank the categories faster, and why does the encoding ladder predict
   that?
3. Take a bar chart and redraw it with a truncated axis. Write one sentence
   describing the story each version tells.
4. Make a scatter plot of two columns with at least a few thousand rows. Fix the
   overplotting and describe what became visible.
5. Build a two panel figure with `plt.subplots`, label both panels fully, and
   save it at 150 dpi with nothing cut off.
6. Draw the missing value heatmap for a real file. Say whether the gaps look
   random, and what you would do next.

## Quick reference

| Task | Code |
|---|---|
| New figure | `fig, ax = plt.subplots(figsize=(8, 5))` |
| Several panels | `fig, axes = plt.subplots(1, 2, figsize=(12, 5))` |
| Histogram | `sns.histplot(data=df, x="col", bins=30)` |
| Counts per category | `sns.countplot(data=df, x="col")` |
| Bar of a value | `sns.barplot(data=df, x="cat", y="val")` |
| Scatter | `sns.scatterplot(data=df, x="a", y="b", hue="c", alpha=0.3)` |
| Line | `sns.lineplot(data=df, x="t", y="v")` |
| Box by group | `sns.boxplot(data=df, x="cat", y="val")` |
| Correlation heatmap | `sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)` |
| Missing values | `sns.heatmap(df.isna(), cbar=False)` |
| Draw into your axes | `sns.boxplot(..., ax=ax)` |
| Labels | `ax.set_title(...)`, `ax.set_xlabel(...)`, `ax.set_ylabel(...)` |
| Rotate ticks | `ax.tick_params(axis="x", rotation=45)` |
| Defaults | `sns.set_theme(style="whitegrid", palette="colorblind")` |
| Save | `fig.savefig("f.png", dpi=150, bbox_inches="tight")` |

## Questions to sit with

1. The encoding ladder says position and length are read most accurately. Name a
   chart you have seen recently that put the most important number on a weak
   channel. What was gained by doing that, and by whom?
2. A truncated axis on a bar chart is misleading, but on a line chart it is often
   correct. What is the difference, in terms of what each chart encodes?
3. You draw a chart, dislike the story it tells, and redraw it differently. When
   is that clarification and when is it manipulation? What would you record so
   that somebody else could tell?
4. Anscombe's quartet exists because summary statistics hide shape. What is the
   equivalent risk when you report a single accuracy number for a model?

## Next

You can see your data. Now the language for describing what you are seeing, and
for saying how confident you should be about it.

[Probability](probability.md){ .h-button }

Worth your time: the
[seaborn tutorial](https://seaborn.pydata.org/tutorial.html) is short and well
written, and the
[Data Visualisation Catalogue](https://datavizcatalogue.com/) is a good browse
when you are unsure which chart fits.

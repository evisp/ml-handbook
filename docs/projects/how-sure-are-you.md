# Project 3: How sure are you?

<span class="badge badge--time">One week</span>
<span class="badge badge--level">No models</span>
<span class="badge">Groups of 3 or 4</span>
<span class="badge">Presented 16 November</span>

Nobody hands you a CSV this time. You go and get your data from two places, make
it into one table, and then answer a question in a way that says how sure you
are of the answer.

## What you are doing

Three things, and the third is the one that makes this project different from
the last one.

**You collect it yourself.** One API and one Wikipedia table. You have done both
of these separately. Now you do them in the same week and join the results,
which is where it gets interesting.

**You write down what you built.** A data contract: every column, its type, its
unit, where it came from, and what is missing from it. Somebody else should be
able to read that table and know what they are holding.

**Every number comes with a range.** Not "62 percent". Something closer to "62
percent, give or take 4, and here is why that is not just luck". A number with
no range is an opinion with a decimal point.

No models. No training, no predictions, no accuracy. If you find yourself
importing scikit-learn, you have misread the brief.

<svg viewBox="0 0 680 246" role="img" aria-labelledby="f1-title f1-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="f1-title">The shape of the week</title>
<desc id="f1-desc">Collect from two sources, clean and join them into one table with a data contract, explore it, quantify the answer with probability, and present. Politeness applies to every single request.</desc>
<defs><marker id="f1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<rect x="5" y="40" width="122" height="86" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="66" y="68" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">collect</text>
<text x="66" y="92" text-anchor="middle" font-size="10" fill="var(--h-graphite)" opacity="0.9">an API and a</text>
<text x="66" y="106" text-anchor="middle" font-size="10" fill="var(--h-graphite)" opacity="0.9">Wikipedia table</text>
<line x1="133" y1="83" x2="141" y2="83" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#f1)"/>
<rect x="141" y="40" width="122" height="86" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="202" y="68" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">clean and join</text>
<text x="202" y="92" text-anchor="middle" font-size="10" fill="var(--h-graphite)" opacity="0.9">one table,</text>
<text x="202" y="106" text-anchor="middle" font-size="10" fill="var(--h-graphite)" opacity="0.9">one contract</text>
<line x1="269" y1="83" x2="277" y2="83" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#f1)"/>
<rect x="277" y="40" width="122" height="86" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="338" y="68" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">explore</text>
<text x="338" y="92" text-anchor="middle" font-size="10" fill="var(--h-graphite)" opacity="0.9">what is in</text>
<text x="338" y="106" text-anchor="middle" font-size="10" fill="var(--h-graphite)" opacity="0.9">there at all</text>
<line x1="405" y1="83" x2="413" y2="83" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#f1)"/>
<rect x="413" y="40" width="122" height="86" rx="10" fill="var(--h-cherry)" stroke="var(--h-cherry)"/>
<text x="474" y="68" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">quantify</text>
<text x="474" y="92" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.9">how sure</text>
<text x="474" y="106" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.9">are you</text>
<line x1="541" y1="83" x2="549" y2="83" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#f1)"/>
<rect x="549" y="40" width="122" height="86" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="610" y="68" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">present</text>
<text x="610" y="92" text-anchor="middle" font-size="10" fill="var(--h-graphite)" opacity="0.9">six minutes,</text>
<text x="610" y="106" text-anchor="middle" font-size="10" fill="var(--h-graphite)" opacity="0.9">two groups</text>
<rect x="5" y="142" width="670" height="38" rx="8" fill="var(--h-cherry-wash)" stroke="var(--h-cherry-line)"/>
<text x="340" y="161" dy="0.36em" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">polite on every single request: say who you are, go slowly, cache everything</text>
<text x="340" y="206" text-anchor="middle" font-size="12" fill="var(--h-graphite)">The red box is the one that makes this project different from the last. Every number you report says how sure you are of it.</text>
<text x="340" y="226" text-anchor="middle" font-size="12" fill="var(--h-graphite)">No models. If you import scikit-learn, you have misread the brief.</text>
</svg>

## The week

Six days of work and a presentation on Monday. That is short, and the shape of
the week matters more than it did last time.

<svg viewBox="0 0 680 264" role="img" aria-labelledby="f2-title f2-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="f2-title">The seven days</title>
<desc id="f2-desc">Monday the access check, Tuesday collect from the API, Wednesday scrape Wikipedia, Thursday join clean and write the contract, Friday explore and the first probability question, Saturday the second, Sunday build the talk. Presenting Monday the sixteenth.</desc>
<rect x="5" y="46" width="88" height="112" rx="9" fill="var(--h-cherry)" stroke="var(--h-cherry)"/>
<text x="49" y="38" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-cherry)">Mon 9</text>
<text x="49" y="86" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">access check</text>
<text x="49" y="114" text-anchor="middle" font-size="9" fill="#ffffff" opacity="0.9">both</text>
<text x="49" y="126" text-anchor="middle" font-size="9" fill="#ffffff" opacity="0.9">sources, saved</text>
<rect x="101" y="46" width="88" height="112" rx="9" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="145" y="38" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-graphite)">Tue 10</text>
<text x="145" y="86" text-anchor="middle" font-size="11" font-weight="700" fill="var(--md-default-fg-color)">collect</text>
<text x="145" y="114" text-anchor="middle" font-size="9" fill="var(--h-graphite)" opacity="0.9">the</text>
<text x="145" y="126" text-anchor="middle" font-size="9" fill="var(--h-graphite)" opacity="0.9">API, cached</text>
<rect x="197" y="46" width="88" height="112" rx="9" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="241" y="38" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-graphite)">Wed 11</text>
<text x="241" y="86" text-anchor="middle" font-size="11" font-weight="700" fill="var(--md-default-fg-color)">scrape</text>
<text x="241" y="114" text-anchor="middle" font-size="9" fill="var(--h-graphite)" opacity="0.9">the</text>
<text x="241" y="126" text-anchor="middle" font-size="9" fill="var(--h-graphite)" opacity="0.9">Wikipedia table</text>
<rect x="293" y="46" width="88" height="112" rx="9" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="337" y="38" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-graphite)">Thu 12</text>
<text x="337" y="86" text-anchor="middle" font-size="11" font-weight="700" fill="var(--md-default-fg-color)">join and clean</text>
<text x="337" y="114" text-anchor="middle" font-size="9" fill="var(--h-graphite)" opacity="0.9">write</text>
<text x="337" y="126" text-anchor="middle" font-size="9" fill="var(--h-graphite)" opacity="0.9">the contract</text>
<rect x="389" y="46" width="88" height="112" rx="9" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="433" y="38" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-graphite)">Fri 13</text>
<text x="433" y="86" text-anchor="middle" font-size="11" font-weight="700" fill="var(--md-default-fg-color)">explore</text>
<text x="433" y="114" text-anchor="middle" font-size="9" fill="var(--h-graphite)" opacity="0.9">and</text>
<text x="433" y="126" text-anchor="middle" font-size="9" fill="var(--h-graphite)" opacity="0.9">question one</text>
<rect x="485" y="46" width="88" height="112" rx="9" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="529" y="38" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-graphite)">Sat 14</text>
<text x="529" y="86" text-anchor="middle" font-size="11" font-weight="700" fill="var(--md-default-fg-color)">question two</text>
<text x="529" y="114" text-anchor="middle" font-size="9" fill="var(--h-graphite)" opacity="0.9">the</text>
<text x="529" y="126" text-anchor="middle" font-size="9" fill="var(--h-graphite)" opacity="0.9">harder one</text>
<rect x="581" y="46" width="88" height="112" rx="9" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="625" y="38" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-graphite)">Sun 15</text>
<text x="625" y="86" text-anchor="middle" font-size="11" font-weight="700" fill="var(--md-default-fg-color)">build the talk</text>
<text x="625" y="114" text-anchor="middle" font-size="9" fill="var(--h-graphite)" opacity="0.9">and</text>
<text x="625" y="126" text-anchor="middle" font-size="9" fill="var(--h-graphite)" opacity="0.9">rehearse once</text>
<rect x="200" y="178" width="280" height="36" rx="9" fill="var(--md-default-fg-color)"/>
<text x="340" y="196" dy="0.36em" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-bg-color)">Monday 16 November: you present</text>
<text x="340" y="240" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Six days. Monday is a gate, not a warm-up: if you cannot reach both sources on Monday, you do not have a project yet.</text>
<text x="340" y="258" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Groups who leave collection until Wednesday discover their problem on Thursday.</text>
</svg>

!!! warning "Monday is a gate, not a warm-up"

    Before anything else, make one successful request to each of your two
    sources and save what comes back to a file. Both of them. On Monday.

    This is the rule that decides whether you have a project. A group that
    discovers on Thursday evening that the API will not answer them has lost the
    week. A group that discovers it on Monday morning has lost an hour and can
    ask for help.

## The rules

**Only your two sources.** The ones in your tab, and nothing else. If you want
to add a third, ask first.

**Be polite on every request.** A `User-Agent` header that names you and gives a
way to contact you. One request at a time. Roughly one a second, slower if the
source asks for slower. There is no prize for finishing the download first and
the cost of going fast is paid by somebody else.

**Cache everything.** The first time you fetch something, write it to disk.
Every run after that reads your copy. You will run your parsing code fifty times
before it is right and the server does not need to see any of that.

**Commit the raw files.** They are small. They are also the only proof of what
the source said on the day you asked.

**Write the data contract.** One table, like this:

| column | type | unit | source | known gaps |
|---|---|---|---|---|
| `race_date` | date | | Jolpica API | none |
| `circuit_length` | float | km | Wikipedia circuits table | 4 circuits missing |
| `grid_position` | int | position | Jolpica API | blank when the car started from the pit lane |

**Use real probability.** Across your two questions, at least two of these four:

1. A range worked out by simulation, not by a formula you found.
2. A comparison against what pure chance would produce, also by simulation.
3. A fitted distribution, with an honest look at where it does not fit.
4. Base rates or conditional probability.

**It must run with the wifi off.** Restart the kernel, disconnect, run the
notebook top to bottom from your cached files. If it fails, it is not finished.

## When the source misbehaves

It will. Plan for it rather than panicking on Thursday.

- Cache first, so that a source going down costs you nothing you had already
  collected.
- The other group on your dataset hit the same wall. Compare notes. Helping each
  other collect is allowed and sensible. Sharing analysis is not.
- A 403 or a 429 is information, not a wall. Slow down, set a proper
  `User-Agent`, and read what the response body actually says.
- If you are still stuck after an hour, message the channel. Losing a day to
  something somebody could have told you in two minutes is the one avoidable
  failure in this project.

## What a range actually buys you

This is the concept the whole project turns on, so it is worth being precise
about it.

<svg viewBox="0 0 680 274" role="img" aria-labelledby="f3-title f3-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="f3-title">The same answer, measured two ways</title>
<desc id="f3-desc">An estimate of 62 percent from 500 observations carries a narrow range that clearly excludes chance. The same 62 percent from 40 observations carries a range wide enough to include chance, so it says nothing.</desc>
<line x1="300" y1="54" x2="300" y2="206" stroke="var(--h-steel)" stroke-width="2" stroke-dasharray="5 4"/>
<text x="300" y="42" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-graphite)">what pure chance would give</text>
<line x1="60" y1="216" x2="640" y2="216" stroke="var(--h-steel)" stroke-width="1.4"/>
<text x="60" y="234" text-anchor="middle" font-size="10" fill="var(--h-space)">30%</text>
<text x="300" y="234" text-anchor="middle" font-size="10" fill="var(--h-space)">50%</text>
<text x="640" y="234" text-anchor="middle" font-size="10" fill="var(--h-space)">90%</text>
<text x="390" y="88" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-cherry)">from 500 races</text>
<line x1="352" y1="114" x2="428" y2="114" stroke="var(--h-cherry)" stroke-width="4"/>
<line x1="352" y1="105" x2="352" y2="123" stroke="var(--h-cherry)" stroke-width="3"/>
<line x1="428" y1="105" x2="428" y2="123" stroke="var(--h-cherry)" stroke-width="3"/>
<circle cx="390" cy="114" r="7" fill="var(--h-cherry)"/>
<text x="446" y="118" font-size="11" font-weight="700" fill="var(--h-cherry)">62%, give or take 4</text>
<text x="446" y="134" font-size="10.5" fill="var(--h-graphite)">the range misses 50, so this is a finding</text>
<text x="390" y="168" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-graphite)">from 40 races</text>
<line x1="248" y1="194" x2="532" y2="194" stroke="var(--h-steel)" stroke-width="4"/>
<line x1="248" y1="185" x2="248" y2="203" stroke="var(--h-steel)" stroke-width="3"/>
<line x1="532" y1="185" x2="532" y2="203" stroke="var(--h-steel)" stroke-width="3"/>
<circle cx="390" cy="194" r="7" fill="var(--h-steel)"/>
<text x="548" y="190" font-size="11" font-weight="700" fill="var(--h-graphite)">62%, give or take 15</text>
<text x="548" y="206" font-size="10.5" fill="var(--h-graphite)">the range covers 50, so it says nothing</text>
<text x="340" y="260" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Both headline numbers are 62 percent. Only one of them is a result.</text>
</svg>

You met this on the Probability page as sampling variation. Here it is the
difference between a finding and a sentence.

!!! ml "ML connection"

    In three weeks you will be comparing models and reading a difference of one
    percentage point as an improvement. The habit you build this week, asking
    how wide the range is before deciding whether the difference is real, is the
    one that stops you from doing that.

## What everyone does

Same tasks for every group. What they mean is in your tab.

| | Task |
|---|---|
| 0 | Access check: one successful request to each source, saved to disk |
| 1 | Collect from the API, with caching |
| 2 | Scrape the Wikipedia table |
| 3 | Join, clean, and write the data contract |
| 4 | Explore: what is actually in this table |
| 5 | Probability question one |
| 6 | Probability question two |
| 7 | Build the talk |

---

## Dataset one: Formula 1

Seventy-five years of race results, and a sport that has argued with itself for
all of them about whether the driver or the machinery wins races.

Two groups work on this data from different angles. You will be standing next to
each other on Monday.

| Source | What it gives you | How to behave |
|---|---|---|
| [Jolpica-F1 API](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md) | Results, qualifying, pit stops, laps, drivers, constructors, 1950 to now. Ergast-compatible JSON, no key. | 4 requests a second and 500 an hour. Pages are 30 rows by default, 100 maximum. |
| [List of Formula One circuits](https://en.wikipedia.org/wiki/List_of_Formula_One_circuits) | Circuit type, length, country, years active. | Name yourself in the `User-Agent`. Use the `/wiki/` URL with no query string. |

=== "Group A: is it the driver or the car?"

    **The question.** Teammates drive the same machinery. So when one of them
    finishes ahead of the other, that is as close to a fair fight as this sport
    offers. How much of a driver's result is the driver?

    **Question one.** Take teammate pairs across a season. How often does the
    stronger one finish ahead? Simulate what a coin flip between two equal
    drivers would produce over the same number of races, and compare.

    **Question two.** Does the size of the teammate gap depend on the kind of
    circuit? Street circuits against permanent ones, or short against long.
    Report a range, not a number.

    **Watch out for.** A head-to-head only means something when both cars
    finished. Drivers change teams mid-season. And a car that broke is not a
    driver who was slower, so decide what to do about retirements before you
    count anything, and write the decision down.

    **If you finish early.** Do the same comparison on qualifying times instead
    of race results, where mechanical retirements cannot interfere, and see
    whether the answer changes.

=== "Group B: is the race decided on Saturday?"

    **The question.** Everybody says qualifying matters. Put a number on it, and
    then find out whether that number is the same everywhere.

    **Question one.** Given pole position, what is the chance of winning? Work
    out a range by simulation, and say plainly whether the range is narrow enough
    to be useful.

    **Question two.** What does each grid position buy you? Chance of a podium
    from P1, P2, P5, P10. Then pick one circuit where the answer is unusual and
    explain why.

    **Watch out for.** Grid penalties mean the starting grid is not the
    qualifying order. The number of races at any one circuit is small, so your
    per-circuit ranges will be wide and you should say so rather than hide it.
    And the sport in 1955 is not the sport in 2025: cars finish now, and they
    did not then.

    **If you finish early.** Split your answer by era. Has the value of pole
    position gone up or down since the 1980s?

---

## Dataset two: Albanian weather

Daily weather for anywhere in Albania, back to 1940, and a set of published
"normals" on Wikipedia claiming what the weather is supposed to be.

Two groups work on this data from different angles. You will be standing next to
each other on Monday.

| Source | What it gives you | How to behave |
|---|---|---|
| [Open-Meteo archive API](https://open-meteo.com/en/docs/historical-weather-api) | Daily maximum and minimum temperature, precipitation, wind, for any coordinate, 1940 to now. ERA5 reanalysis, no key. | Free for non-commercial use. Data is CC BY 4.0, so credit it. Ask for long date ranges in one request rather than looping over days. |
| Wikipedia city pages, climate boxes: [Tirana](https://en.wikipedia.org/wiki/Tirana), [Shkodër](https://en.wikipedia.org/wiki/Shkod%C3%ABr), [Vlorë](https://en.wikipedia.org/wiki/Vlor%C3%AB), [Korçë](https://en.wikipedia.org/wiki/Kor%C3%A7%C3%AB) | The published monthly normals: average high, average low, rainfall, rainy days. | Name yourself in the `User-Agent`. One request per city, cached. |

=== "Group C: has summer actually got hotter?"

    **The question.** Everybody says the summers are worse than they used to be.
    Everybody also remembers selectively. Settle it for one city.

    **Question one.** The chance of a day above 35 degrees in June, July and
    August, for 1961 to 1990 and for 1991 to 2020. Both with simulated ranges.
    Do the ranges overlap?

    **Question two.** Did the average move, or did the tails move? Those are
    different claims. Compare the shift in the mean against the shift in the
    95th percentile, and say which one actually changed.

    **Watch out for.** ERA5 is a model reconstruction on a grid of roughly 25
    kilometres, not a thermometer standing in Skanderbeg Square. If you compare
    your numbers against a published normal, check what years that normal
    covers before you treat the difference as a finding.

    **If you finish early.** Run the same analysis for a second city and see
    whether your conclusion travels.

=== "Group D: can you still plan a summer around the weather?"

    **The question.** The climate box on Wikipedia says August has two rainy
    days. Can you plan a wedding on that?

    **Question one.** The chance of a rain day by month, for one city, with a
    range. You will have to choose what counts as a rain day. Defend the
    threshold you pick.

    **Question two.** Streaks. How long is the longest dry spell you should
    expect in July? How often does a week of rain happen in September? At least
    one of these should be answered by simulation.

    **Watch out for.** Consecutive days are not independent: rain tomorrow is
    more likely if it rained today, so anything you calculate by multiplying a
    daily probability together will be wrong. And when you compare against the
    Wikipedia normals, look carefully at the small print under each climate box.
    They are not all measuring the same thing.

    **If you finish early.** Compare a coastal city with an inland one and say
    which is easier to plan around.

---

## Before you touch any data

Each group writes down, on paper, what you expect to find. One or two sentences.
Fold it up and give it to me.

It gets opened on Monday, next to your answer.

!!! tip "Why this is worth doing"

    If you form your expectations after exploring, you will form ones you
    already know are true, and you will have proved nothing to anybody including
    yourself.

    Being wrong on Monday is the best outcome available to you. It means the
    data told you something you did not already believe, which is the only
    reason to go and get data in the first place.

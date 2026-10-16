# Data Collection: APIs

<span class="badge badge--time">65 min</span>
<span class="badge badge--level">Real data</span>
<span class="badge">Needs: Web Scraping</span>

An API is a service that agrees to give you data in a format you can rely on.
After the last page, this will feel like a holiday.

## Why this matters

Scraping is what you do when nobody provided a way in. An API is that way in.
The data arrives as data rather than as layout, the fields have names, and the
thing does not break because somebody changed a CSS class.

Most of your working life will involve calling APIs: a model provider, a payment
system, an internal service at your own company. The three things this page
covers, pagination, rate limits and messy fields, are the three things that turn
a five line example into real code.

## What an API gives you

Someone published an interface and promised it will behave a certain way. You
send a request to an address, you get back JSON, and the shape of that JSON is
documented rather than guessed.

Read the documentation first. Ten minutes there saves an hour of experiments,
and it will tell you the things you cannot discover by poking: how pagination
works, what the limits are, whether you need a key.

<svg viewBox="0 0 680 306" role="img" aria-labelledby="ra-title ra-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="ra-title">The parts of a request and a response</title>
<desc id="ra-desc">A URL is made of a scheme, a host, a path and a query string. The request carries a method and headers. The response carries a status code, headers and a JSON body.</desc>
<defs><marker id="ra" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<rect x="40" y="34" width="102" height="38" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="91" y="53" dy="0.36em" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">https://</text>
<text x="91" y="90" text-anchor="middle" font-size="11" fill="var(--h-graphite)">scheme</text>
<rect x="142" y="34" width="217" height="38" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="251" y="53" dy="0.36em" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">swapi-api.hbtn.io</text>
<text x="251" y="90" text-anchor="middle" font-size="11" fill="var(--h-graphite)">host</text>
<rect x="359" y="34" width="191" height="38" rx="6" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="455" y="53" dy="0.36em" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">/api/starships/</text>
<text x="455" y="90" text-anchor="middle" font-size="11" fill="var(--h-graphite)">path</text>
<rect x="551" y="34" width="89" height="38" rx="6" fill="var(--h-cherry)" stroke="var(--h-cherry)"/>
<text x="595" y="53" dy="0.36em" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">?page=2</text>
<text x="595" y="90" text-anchor="middle" font-size="11" fill="var(--h-graphite)">query</text>
<rect x="40" y="118" width="250" height="122" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="60" y="144" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">the request</text>
<text x="60" y="170" font-size="11.5" fill="var(--h-graphite)">GET, the only verb you need here</text>
<text x="60" y="192" font-size="11.5" fill="var(--h-graphite)">params, your query string</text>
<text x="60" y="214" font-size="11.5" fill="var(--h-graphite)">headers, who you are and what</text>
<text x="60" y="230" font-size="11.5" fill="var(--h-graphite)">format you want back</text>
<line x1="300" y1="179" x2="378" y2="179" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#ra)"/>
<rect x="390" y="118" width="250" height="122" rx="10" fill="var(--h-cherry)"/>
<text x="410" y="144" font-size="13" font-weight="700" fill="#ffffff">the response</text>
<text x="410" y="170" font-size="11.5" fill="#ffffff" opacity="0.92">status code, did it work</text>
<text x="410" y="192" font-size="11.5" fill="#ffffff" opacity="0.92">headers, including your rate limit</text>
<text x="410" y="214" font-size="11.5" fill="#ffffff" opacity="0.92">body, the JSON you came for</text>
<text x="340" y="276" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Beginners read only the body. The status code and the headers are where the useful information about failure lives.</text>
</svg>

A URL carries most of what you are asking for. The scheme and host say where,
the path says what kind of thing, and the query string narrows it down. In
`requests` you rarely build that string yourself.

```python
import requests

response = requests.get(
    "https://swapi-api.hbtn.io/api/starships/",
    params={"page": 2},
    timeout=10,
)
print(response.url)
print(response.status_code)
```

Passing `params` as a dictionary rather than gluing the string together handles
the encoding for you, which matters the moment a value contains a space or an
accented character.

!!! tip "Use the base URL the task gives you"

    The original Star Wars API has moved more than once. Use whichever address
    the project specifies, and define it once at the top of your file as a
    constant so that changing it is a one line edit.

## Reading the response

Two parts matter and beginners read only one of them.

<svg viewBox="0 0 680 246" role="img" aria-labelledby="sc-title sc-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="sc-title">What the four families of status code mean</title>
<desc id="sc-desc">Two hundreds mean success. Three hundreds are redirects and are followed automatically. Four hundreds mean your request was wrong. Five hundreds mean the server failed and you should retry later.</desc>
<rect x="5" y="20" width="152" height="150" rx="10" fill="var(--h-cherry)" stroke="var(--h-cherry)"/>
<text x="81" y="56" text-anchor="middle" font-size="22" font-weight="700" fill="#ffffff">2xx</text>
<text x="81" y="84" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">it worked</text>
<text x="81" y="112" text-anchor="middle" font-size="11" fill="#ffffff" opacity="0.9">use the body</text>
<text x="81" y="148" text-anchor="middle" font-size="10.5" fill="#ffffff" opacity="0.75">200 OK</text>
<rect x="175" y="20" width="152" height="150" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="251" y="56" text-anchor="middle" font-size="22" font-weight="700" fill="var(--md-default-fg-color)">3xx</text>
<text x="251" y="84" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">moved</text>
<text x="251" y="112" text-anchor="middle" font-size="11" fill="var(--md-default-fg-color)" opacity="0.9">requests follows it for you</text>
<text x="251" y="148" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color)" opacity="0.75">301, 302</text>
<rect x="345" y="20" width="152" height="150" rx="10" fill="var(--h-cherry-wash)" stroke="var(--h-cherry)"/>
<text x="421" y="56" text-anchor="middle" font-size="22" font-weight="700" fill="var(--md-default-fg-color)">4xx</text>
<text x="421" y="84" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">your mistake</text>
<text x="421" y="112" text-anchor="middle" font-size="11" fill="var(--md-default-fg-color)" opacity="0.9">fix the request</text>
<text x="421" y="148" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color)" opacity="0.75">404, 401, 403, 429</text>
<rect x="515" y="20" width="152" height="150" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="591" y="56" text-anchor="middle" font-size="22" font-weight="700" fill="var(--md-default-fg-color)">5xx</text>
<text x="591" y="84" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">their problem</text>
<text x="591" y="112" text-anchor="middle" font-size="11" fill="var(--md-default-fg-color)" opacity="0.9">wait and retry</text>
<text x="591" y="148" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color)" opacity="0.75">500, 503</text>
<text x="340" y="206" text-anchor="middle" font-size="12" fill="var(--h-graphite)">response.raise_for_status() turns everything in the last two columns into an exception.</text>
<text x="340" y="226" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Sometimes that is what you want. Sometimes you need to read the code yourself and act on it.</text>
</svg>

```python
if response.status_code == 200:
    data = response.json()
```

`.json()` parses the body into Python dictionaries and lists. There is no need
for the `json` module unless you are handed a string rather than a response.

You have two ways of dealing with failure and they suit different situations.

```python
response.raise_for_status()
```

That turns any 4xx or 5xx into an exception, which is right when you want the
script to stop. But the GitHub task needs the opposite: a 404 should print
`Not found` and a 403 should tell you when the limit resets. When the status
code is information you need to act on, read it yourself.

```python
if response.status_code == 404:
    print("Not found")
elif response.status_code == 403:
    print("rate limited")
else:
    data = response.json()
```

## Working with JSON

JSON becomes dictionaries, lists, strings, numbers, booleans and `None`. You
already know how to walk that, from the Python warm up.

```python
data = response.json()
print(data["count"])
for ship in data["results"]:
    print(ship["name"], ship["passengers"])
```

The habit that saves you: **look at one record before writing any logic.**

```python
from pprint import pprint

pprint(data["results"][0])
```

And use `.get()` when a field may be absent, because a missing key raises while a
missing value does not.

```python
homeworld = species.get("homeworld")
```

Some SWAPI species have `null` for homeworld. `species["homeworld"]` gives you
`None` there, and calling `requests.get(None)` fails in a way that takes a while
to trace back to its cause.

## Pagination

Almost no API gives you everything at once. It hands you a page and tells you
where the next one is.

<svg viewBox="0 0 680 282" role="img" aria-labelledby="pe-title pe-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="pe-title">How a paginated JSON response is shaped</title>
<desc id="pe-desc">Each page is an envelope holding a count, a link to the next page, and an array of results. You follow the next link until it is null.</desc>
<defs><marker id="pe" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<rect x="10" y="34" width="204" height="150" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)" stroke-width="1"/>
<text x="112" y="58" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">page 1</text>
<text x="26" y="84" font-size="11" fill="var(--h-graphite)">"count": 82</text>
<text x="26" y="106" font-size="11" font-weight="700" fill="var(--h-cherry)">"next": ".../?page=2"</text>
<text x="26" y="128" font-size="11" fill="var(--h-graphite)">"results": [ ... ]</text>
<rect x="26" y="140" width="172" height="28" rx="6" fill="var(--h-cherry)" opacity="0.2"/>
<text x="112" y="154" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-cherry)">10 of 82 collected</text>
<line x1="218" y1="109" x2="236" y2="109" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#pe)"/>
<rect x="238" y="34" width="204" height="150" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)" stroke-width="1"/>
<text x="340" y="58" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">page 2</text>
<text x="254" y="84" font-size="11" fill="var(--h-graphite)">"count": 82</text>
<text x="254" y="106" font-size="11" font-weight="700" fill="var(--h-cherry)">"next": ".../?page=3"</text>
<text x="254" y="128" font-size="11" fill="var(--h-graphite)">"results": [ ... ]</text>
<rect x="254" y="140" width="172" height="28" rx="6" fill="var(--h-cherry)" opacity="0.2"/>
<text x="340" y="154" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-cherry)">20 of 82 collected</text>
<line x1="446" y1="109" x2="464" y2="109" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#pe)"/>
<rect x="466" y="34" width="204" height="150" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry)" stroke-width="2"/>
<text x="568" y="58" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">page 9</text>
<text x="482" y="84" font-size="11" fill="var(--h-graphite)">"count": 82</text>
<text x="482" y="106" font-size="11" font-weight="700" fill="var(--h-cherry)">"next": null</text>
<text x="482" y="128" font-size="11" fill="var(--h-graphite)">"results": [ ... ]</text>
<rect x="482" y="140" width="172" height="28" rx="6" fill="var(--h-cherry)" opacity="0.2"/>
<text x="568" y="154" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-cherry)">82 of 82 collected</text>
<text x="352" y="26" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-graphite)">dots stand for the pages in between</text>
<text x="340" y="220" text-anchor="middle" font-size="12" fill="var(--h-graphite)">The API tells you where the next page is. Read that field rather than counting pages yourself,</text>
<text x="340" y="240" text-anchor="middle" font-size="12" fill="var(--h-graphite)">because the number of pages changes whenever the data does.</text>
<text x="340" y="266" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-cherry)">stop when next is null</text>
</svg>

```python
def fetch_all(url):
    """Follow the next links and return every result."""
    results = []
    while url:
        payload = requests.get(url, timeout=10).json()
        results.extend(payload["results"])
        url = payload.get("next")
    return results
```

Four lines, and it is the whole of pagination for this style of API. The loop
ends by itself because `next` is `null` on the last page, and `while None` is
false.

Note what is not there: no page counting, no guessing how many there are. If the
data grows tomorrow, this code still works.

!!! warning "Not every API paginates the same way"

    This one gives you a full URL for the next page. Others give you a page
    number you have to build the URL from, or a cursor token, or a `Link` header.
    The documentation tells you which. The concept is identical: ask, collect,
    ask again, stop when the API says there is no more.

## The data is messier than you expect

This is the part of the objectives that sounds innocuous and is not. Here is
what SWAPI actually returns for a starship's passenger capacity:

```python
"843"
"n/a"
"unknown"
"30-165"
"1,000"
```

Every one of those is a string. Two of them are not numbers at all, one is a
range, and one has a thousands separator that will break `int()`.

So the first task is not really about pagination. It is about deciding what to
do with each of those cases before you can compare anything to
`passengerCount`.

```python
def parse_passengers(raw):
    """Turn a passengers field into a number, or None if it is not one."""
    if raw is None:
        return None
    cleaned = raw.replace(",", "").strip().lower()
    if cleaned in ("n/a", "unknown", ""):
        return None
    if "-" in cleaned:
        cleaned = cleaned.split("-")[0]
    return int(cleaned) if cleaned.isdigit() else None
```

Returning `None` rather than zero is deliberate. A ship whose capacity is unknown
is not a ship that carries nobody, and quietly turning one into the other is how
wrong answers get produced confidently.

!!! ml "ML connection"

    You have now met this three times: missing values in pandas, missing data in
    Fundamentals, and unknown fields here. The pattern is always the same. An
    absent value is not a zero, and deciding what it means is a judgment you
    should make on purpose and write down.

Then the actual task becomes short.

```python
def availableShips(passengerCount):
    """Return the names of ships that can carry at least passengerCount."""
    ships = []
    for ship in fetch_all("https://swapi-api.hbtn.io/api/starships/"):
        capacity = parse_passengers(ship.get("passengers"))
        if capacity is not None and capacity >= passengerCount:
            ships.append(ship["name"])
    return ships
```

The second task is the same shape with a different filter. A species counts as
sentient if the word appears in either `classification` or `designation`, so
check both, and lowercase them before comparing because the data is not
consistent about case.

## One request per item adds up

The sentient planets task has a trap in it. Each species carries its homeworld
as a URL rather than a name, so you have to fetch it. Do that naively and you
send one request per species, including for planets you already looked up.

<svg viewBox="0 0 680 290" role="img" aria-labelledby="np-title np-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="np-title">Requesting the same thing over and over</title>
<desc id="np-desc">Without a cache, every species triggers its own request for a home planet, including planets already fetched. With a cache, each planet is fetched once and reused.</desc>
<defs><marker id="np" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<text x="170" y="26" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-graphite)">without a cache</text>
<text x="510" y="26" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">with a cache</text>
<rect x="20" y="44" width="96" height="26" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="68" y="57" dy="0.36em" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color)">species 1</text>
<line x1="120" y1="57" x2="196" y2="57" stroke="var(--h-steel)" stroke-width="1.5" marker-end="url(#np)"/>
<rect x="200" y="44" width="110" height="26" rx="5" fill="var(--h-cherry)" opacity="0.8"/>
<text x="255" y="57" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="#ffffff">Tatooine</text>
<rect x="20" y="78" width="96" height="26" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="68" y="91" dy="0.36em" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color)">species 2</text>
<line x1="120" y1="91" x2="196" y2="91" stroke="var(--h-steel)" stroke-width="1.5" marker-end="url(#np)"/>
<rect x="200" y="78" width="110" height="26" rx="5" fill="var(--h-cherry)" opacity="0.8"/>
<text x="255" y="91" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="#ffffff">Naboo</text>
<rect x="20" y="112" width="96" height="26" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="68" y="125" dy="0.36em" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color)">species 3</text>
<line x1="120" y1="125" x2="196" y2="125" stroke="var(--h-steel)" stroke-width="1.5" marker-end="url(#np)"/>
<rect x="200" y="112" width="110" height="26" rx="5" fill="var(--h-cherry)" opacity="0.8"/>
<text x="255" y="125" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="#ffffff">Tatooine</text>
<rect x="20" y="146" width="96" height="26" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="68" y="159" dy="0.36em" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color)">species 4</text>
<line x1="120" y1="159" x2="196" y2="159" stroke="var(--h-steel)" stroke-width="1.5" marker-end="url(#np)"/>
<rect x="200" y="146" width="110" height="26" rx="5" fill="var(--h-cherry)" opacity="0.8"/>
<text x="255" y="159" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="#ffffff">Kashyyyk</text>
<rect x="20" y="180" width="96" height="26" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="68" y="193" dy="0.36em" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color)">species 5</text>
<line x1="120" y1="193" x2="196" y2="193" stroke="var(--h-steel)" stroke-width="1.5" marker-end="url(#np)"/>
<rect x="200" y="180" width="110" height="26" rx="5" fill="var(--h-cherry)" opacity="0.8"/>
<text x="255" y="193" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="#ffffff">Naboo</text>
<rect x="20" y="214" width="96" height="26" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="68" y="227" dy="0.36em" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color)">species 6</text>
<line x1="120" y1="227" x2="196" y2="227" stroke="var(--h-steel)" stroke-width="1.5" marker-end="url(#np)"/>
<rect x="200" y="214" width="110" height="26" rx="5" fill="var(--h-cherry)" opacity="0.8"/>
<text x="255" y="227" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="#ffffff">Tatooine</text>
<rect x="360" y="44" width="96" height="26" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="408" y="57" dy="0.36em" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color)">species 1</text>
<line x1="460" y1="57" x2="536" y2="62" stroke="var(--h-cherry)" stroke-width="1.5" stroke-dasharray="none" marker-end="url(#np)"/>
<rect x="360" y="78" width="96" height="26" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="408" y="91" dy="0.36em" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color)">species 2</text>
<line x1="460" y1="91" x2="536" y2="132" stroke="var(--h-cherry)" stroke-width="1.5" stroke-dasharray="none" marker-end="url(#np)"/>
<rect x="360" y="112" width="96" height="26" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="408" y="125" dy="0.36em" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color)">species 3</text>
<line x1="460" y1="125" x2="536" y2="62" stroke="var(--h-steel)" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#np)"/>
<rect x="360" y="146" width="96" height="26" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="408" y="159" dy="0.36em" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color)">species 4</text>
<line x1="460" y1="159" x2="536" y2="202" stroke="var(--h-cherry)" stroke-width="1.5" stroke-dasharray="none" marker-end="url(#np)"/>
<rect x="360" y="180" width="96" height="26" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="408" y="193" dy="0.36em" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color)">species 5</text>
<line x1="460" y1="193" x2="536" y2="132" stroke="var(--h-steel)" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#np)"/>
<rect x="360" y="214" width="96" height="26" rx="5" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="408" y="227" dy="0.36em" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color)">species 6</text>
<line x1="460" y1="227" x2="536" y2="62" stroke="var(--h-steel)" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#np)"/>
<rect x="540" y="49" width="110" height="26" rx="5" fill="var(--h-cherry)"/>
<text x="595" y="62" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="#ffffff">Tatooine</text>
<rect x="540" y="119" width="110" height="26" rx="5" fill="var(--h-cherry)"/>
<text x="595" y="132" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="#ffffff">Naboo</text>
<rect x="540" y="189" width="110" height="26" rx="5" fill="var(--h-cherry)"/>
<text x="595" y="202" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="#ffffff">Kashyyyk</text>
<text x="170" y="268" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-graphite)">6 requests</text>
<text x="510" y="268" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-cherry)">3 requests, 3 cache hits</text>
<text x="340" y="286" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">Across 37 species the saving is far larger, and it is the difference between finishing and hitting a rate limit.</text>
</svg>

```python
def planet_name(url, cache):
    """Fetch a planet name, remembering ones already seen."""
    if url is None:
        return None
    if url not in cache:
        cache[url] = requests.get(url, timeout=10).json()["name"]
    return cache[url]
```

A dictionary is all the cache you need. Pass the same one through your loop and
every repeated homeworld becomes free.

This matters more than the time saved. Every avoided request is one you have not
spent from your rate limit, and rate limits are the subject of the next section.

## Rate limits

An API decides how many requests you may send in a period. Exceed it and you get
refused, usually with 403 or 429.

<svg viewBox="0 0 680 274" role="img" aria-labelledby="rl-title rl-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="rl-title">A rate limit budget over time</title>
<desc id="rl-desc">Each request lowers the remaining allowance. When it reaches zero the API refuses with a 403 until the reset time, when the allowance is restored.</desc>
<defs><marker id="rl" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<line x1="60" y1="190" x2="650" y2="190" stroke="var(--h-steel)" stroke-width="1.5"/>
<line x1="60" y1="190" x2="60" y2="40" stroke="var(--h-steel)" stroke-width="1.5"/>
<text x="70" y="36" font-size="11.5" font-weight="700" fill="var(--h-graphite)">X-RateLimit-Remaining</text>
<text x="52" y="54" text-anchor="end" font-size="10" fill="var(--h-space)">60</text>
<text x="52" y="194" text-anchor="end" font-size="10" fill="var(--h-space)">0</text>
<polyline points="60,50 87,50 87,62 114,62 114,73 141,73 141,85 168,85 168,97 195,97 195,109 222,109 222,120 249,120 249,132 276,132 276,144 303,144 303,155 330,155 330,167 357,167 357,179 384,179 384,190" fill="none" stroke="var(--h-cherry)" stroke-width="2.5"/>
<rect x="384" y="40" width="120" height="150" fill="var(--h-cherry)" opacity="0.12"/>
<text x="444" y="118" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-cherry)">403</text>
<text x="444" y="138" text-anchor="middle" font-size="10.5" fill="var(--h-cherry)">every request refused</text>
<line x1="504" y1="40" x2="504" y2="200" stroke="var(--md-default-fg-color)" stroke-width="2" stroke-dasharray="5 4"/>
<text x="510" y="52" font-size="11.5" font-weight="700" fill="var(--md-default-fg-color)">X-RateLimit-Reset</text>
<line x1="504" y1="50" x2="640" y2="50" stroke="var(--h-cherry)" stroke-width="2.5"/>
<text x="572" y="42" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-cherry)">allowance restored</text>
<text x="355" y="212" text-anchor="middle" font-size="11" fill="var(--h-space)">time</text>
<text x="340" y="242" text-anchor="middle" font-size="12" fill="var(--h-graphite)">The reset time arrives as a Unix timestamp in seconds. Subtract the current time to get how long you must wait.</text>
<text x="340" y="262" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Read Remaining as you go, and slow down before you hit zero rather than after.</text>
</svg>

GitHub tells you exactly where you stand, in the response headers.

```python
print(response.headers.get("X-RateLimit-Limit"))
print(response.headers.get("X-RateLimit-Remaining"))
print(response.headers.get("X-RateLimit-Reset"))
```

`Reset` is a Unix timestamp, meaning seconds since the start of 1970. To turn it
into something a person can read, subtract the current time.

```python
import time

def minutes_until_reset(response):
    """How many minutes until the rate limit is restored."""
    reset_at = int(response.headers.get("X-RateLimit-Reset", 0))
    return round((reset_at - time.time()) / 60)
```

That is the third task. When the status is 403, print `Reset in X min` using this
number.

```python
#!/usr/bin/python3
"""Print the location of a GitHub user."""
import sys
import time
import requests

if __name__ == '__main__':
    response = requests.get(sys.argv[1])
    if response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        reset_at = int(response.headers.get("X-RateLimit-Reset", 0))
        print("Reset in {} min".format(round((reset_at - time.time()) / 60)))
    else:
        print(response.json().get("location"))
```

The `if __name__ == '__main__':` guard is required by the task and you met it in
the Python warm up. It means the file can be imported without running the script,
which is what the checker relies on.

### Not hitting the limit in the first place

Three habits, in order of how much they help.

**Cache**, as above. The cheapest request is one you do not send.

**Slow down on purpose.** A short sleep between requests keeps you well inside
almost any limit.

**Back off when refused.** Do not retry immediately, and do not retry at a fixed
interval. Double the wait each time.

```python
def get_with_backoff(url, attempts=5):
    """Retry with a doubling delay when the server pushes back."""
    delay = 1
    for _ in range(attempts):
        response = requests.get(url, timeout=10)
        if response.status_code not in (429, 500, 502, 503):
            return response
        time.sleep(delay)
        delay *= 2
    return response
```

Waiting 1, 2, 4, 8 seconds gives a struggling server room to recover. Retrying
every half second gives it more of what caused the problem.

## Keys and secrets

Many APIs want a key. GitHub works without one at 60 requests an hour, and with
one at 5000, which is the difference between finishing the task and waiting.

A key goes in a header:

```python
headers = {"Authorization": "Bearer " + token}
response = requests.get(url, headers=headers, timeout=10)
```

Where the token does **not** go: in your code, in your notebook, or anywhere
inside your repository. Put it in a file called `.env`, and put `.env` in
`.gitignore`.

```bash
echo "GITHUB_TOKEN=ghp_yourtokenhere" > .env
echo ".env" >> .gitignore
```

```python
import os

token = os.environ.get("GITHUB_TOKEN")
```

!!! warning "A committed key is a public key"

    This is the same lesson as the Git tutorial, where we did not put a token in
    a clone URL. Deleting a secret in a later commit does not remove it: it is
    still in the history, and bots scan public repositories for exactly this
    within minutes. If it happens, revoke the key immediately rather than trying
    to rewrite history.

## From API to DataFrame

Once you have the records, pandas takes over.

```python
import pandas as pd

ships = fetch_all("https://swapi-api.hbtn.io/api/starships/")
df = pd.DataFrame(ships)
print(df[["name", "passengers", "cost_in_credits"]].head())
```

When the JSON is nested, `json_normalize` flattens it into columns.

```python
df = pd.json_normalize(records, sep="_")
```

Save the raw response before you transform anything.

```python
import json

with open("starships_raw.json", "w") as f:
    json.dump(ships, f)
```

That file means the next twenty times you change your parsing logic you do not
touch the network. It also means you can prove later what the API actually said,
which matters when your results and somebody else's disagree.

## When it goes wrong

??? note "`JSONDecodeError`"

    The body was not JSON. Usually an error page or a redirect to a login.
    Print `response.status_code` and the first 200 characters of `response.text`
    and you will see what it really sent.

??? note "`KeyError: 'results'`"

    The response is not shaped the way you assumed, often because the request
    failed and you parsed the error body. Check the status code before reading
    fields.

??? note "`TypeError: '>=' not supported between instances of 'str' and 'int'`"

    You compared a field straight from the JSON to a number. Almost every
    numeric-looking field in SWAPI is a string. Parse it first.

??? note "403 Forbidden on GitHub"

    Either you are rate limited, in which case the reset header tells you when,
    or the endpoint needs authentication. `X-RateLimit-Remaining` at zero
    distinguishes the two.

??? note "My loop never ends"

    You are re-requesting the same page. Make sure you assign the next URL from
    the payload rather than incrementing something yourself, and print the URL
    each time round while you are debugging.

??? note "It works for me and fails for the checker"

    Often a trailing slash, an assumed page count, or a hard-coded number of
    results. Follow the links the API gives you and do not assume the data is
    the same size it was when you wrote the code.

## Check yourself

1. Request one page of starships and print the status code, the number of
   results, and the value of `next`.
2. Write `fetch_all` and confirm the total matches the `count` field on page one.
3. List every distinct value that appears in the `passengers` field across all
   ships. Decide what each one should become, and say why.
4. Implement `availableShips(passengerCount)` and check that an impossible
   number returns an empty list rather than raising.
5. Implement `sentientPlanets()` with a cache. Count how many requests you sent
   with it and without it.
6. Call the GitHub API for a user that does not exist, and for one that does.
   Print `X-RateLimit-Remaining` each time and watch it fall.
7. Save a raw response to disk and rebuild your DataFrame from the file with no
   network connection.

## Quick reference

| Task | Code |
|---|---|
| GET with parameters | `requests.get(url, params={"page": 2}, timeout=10)` |
| Status code | `response.status_code` |
| Raise on failure | `response.raise_for_status()` |
| Parse the body | `response.json()` |
| A header | `response.headers.get("X-RateLimit-Reset")` |
| Send a token | `headers={"Authorization": "Bearer " + token}` |
| Safe field access | `record.get("homeworld")` |
| Next page | `url = payload.get("next")` |
| Timestamp to minutes | `round((reset_at - time.time()) / 60)` |
| Read a secret | `os.environ.get("GITHUB_TOKEN")` |
| Records to a table | `pd.DataFrame(records)` |
| Flatten nested JSON | `pd.json_normalize(records, sep="_")` |

## Questions to sit with

1. An API publishes a rate limit and a documented format. A scraper has neither.
   What does that difference say about what the owner expects from you?
2. Fields like `"n/a"` and `"unknown"` and `"30-165"` all end up as `None` in
   your parser. What have you lost by treating them the same, and when would
   that matter?
3. Caching means your results reflect the data as it was when you first fetched
   it. When is that a feature, and when is it a bug?
4. Your script works because the API returns `next` links. What happens to
   everything you built if the provider changes that, and what would reduce the
   damage?

## Next

You can collect data from pages and from services. Next, what to do in the first
hour with a dataset you have never seen.

[Data Preparation and Visualization](data-preparation.md){ .h-button }

Worth your time: the
[requests documentation](https://requests.readthedocs.io/en/latest/user/quickstart/)
is one of the best written in Python, and
[GitHub's rate limit page](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api)
explains the headers used in the third task.

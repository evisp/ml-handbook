# Data Collection: Web Scraping

<span class="badge badge--time">75 min</span>
<span class="badge badge--level">Real data</span>
<span class="badge">Needs: Pandas</span>

Web scraping is pulling structured data out of pages that were built for people
to look at. This page covers how to do it, how to decide whether you should, and
how to tell which of the two tools you need.

## Why this matters

Most interesting data is not sitting in a CSV waiting for you. It is on a page
somewhere, formatted for a human reader, and getting it into a DataFrame is your
problem. Being able to solve that problem yourself is the difference between
working on datasets somebody handed you and working on questions you chose.

It is also the first thing in this programme that touches somebody else's
machine. Everything until now ran on your laptop and could only hurt you. A
scraper sends requests to a server you do not own, which is why half of this page
is about judgment rather than code.

## Check the cheaper routes first

Almost everybody's instinct is to open the page, look at the HTML and start
parsing. That is the slowest and most fragile of four options, and it should be
your third choice.

<svg viewBox="0 0 680 318" role="img" aria-labelledby="ld-title ld-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="ld-title">Four ways to get data from a website, best first</title>
<desc id="ld-desc">Try a documented API first. Then structured data embedded in the page. Then parsing the static HTML. Only use a real browser when nothing else works.</desc>
<defs><marker id="ld" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<line x1="34" y1="44" x2="34" y2="292" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#ld)"/>
<text x="22" y="36" font-size="11" font-weight="700" fill="var(--h-graphite)">try in</text>
<text x="22" y="50" font-size="11" font-weight="700" fill="var(--h-graphite)">this order</text>
<rect x="60" y="40" width="560" height="56" rx="10" fill="var(--h-cherry)" opacity="1.0"/>
<text x="80" y="64" font-size="13.5" font-weight="700" fill="#ffffff">A documented API</text>
<text x="80" y="83" font-size="11" fill="#ffffff" opacity="0.9">ask for the data the way the site intends</text>
<text x="636" y="70" text-anchor="end" font-size="0" fill="var(--h-graphite)"> </text>
<rect x="60" y="106" width="560" height="56" rx="10" fill="var(--h-cherry)" opacity="0.72"/>
<text x="80" y="130" font-size="13.5" font-weight="700" fill="#ffffff">Structured data inside the page</text>
<text x="80" y="149" font-size="11" fill="#ffffff" opacity="0.9">JSON-LD in a script tag, put there for search engines</text>
<text x="636" y="136" text-anchor="end" font-size="0" fill="var(--h-graphite)"> </text>
<rect x="60" y="172" width="560" height="56" rx="10" fill="var(--h-cherry)" opacity="0.46"/>
<text x="80" y="196" font-size="13.5" font-weight="700" fill="#ffffff">Static HTML</text>
<text x="80" y="215" font-size="11" fill="#ffffff" opacity="0.9">requests plus BeautifulSoup over the page source</text>
<text x="636" y="202" text-anchor="end" font-size="0" fill="var(--h-graphite)"> </text>
<rect x="60" y="238" width="560" height="56" rx="10" fill="var(--h-cherry)" opacity="0.22"/>
<text x="80" y="262" font-size="13.5" font-weight="700" fill="#ffffff">A real browser</text>
<text x="80" y="281" font-size="11" fill="#ffffff" opacity="0.9">Selenium, when the data only exists after JavaScript runs</text>
<text x="636" y="268" text-anchor="end" font-size="0" fill="var(--h-graphite)"> </text>
<text x="600" y="74" text-anchor="end" font-size="10.5" font-weight="700" fill="#ffffff" opacity="0.85">fastest, stable, usually allowed</text>
<text x="600" y="140" text-anchor="end" font-size="10.5" font-weight="700" fill="#ffffff" opacity="0.85">still just text, no parsing guesswork</text>
<text x="600" y="206" text-anchor="end" font-size="10.5" font-weight="700" fill="#ffffff" opacity="0.85">breaks when the layout changes</text>
<text x="600" y="272" text-anchor="end" font-size="10.5" font-weight="700" fill="#ffffff" opacity="0.85">slow, heavy, brittle, last resort</text>
<text x="340" y="308" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Most people start at the bottom. Almost every site you will meet can be scraped from higher up the list.</text>
</svg>

The reason is fragility. A parser built on the page layout breaks the next time
somebody changes a CSS class. An API returns the data as data, and keeps working.

You will do exactly this comparison in the project. One task scrapes the quotes
by parsing HTML. Another task gets the same quotes from
`/api/quotes?page=1`, which the site's own JavaScript uses. The second is shorter,
faster and more reliable, and the only way to find it was to look.

!!! tip "How to find the hidden API"

    Open the page, press F12, go to the **Network** tab, tick **Fetch/XHR** and
    reload. Every request the page makes to fetch its own data appears there.
    Click one and look at the response. If it is JSON, you have just skipped the
    hardest part of the project.

## Static and dynamic pages

This distinction decides which tool you need, and it is the single most useful
thing on this page.

<svg viewBox="0 0 680 296" role="img" aria-labelledby="sd-title sd-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="sd-title">A static page compared with a dynamic page</title>
<desc id="sd-desc">On a static page the server returns HTML that already contains the data. On a dynamic page the server returns an empty shell and JavaScript fetches the data afterwards, so a plain request sees nothing.</desc>
<defs><marker id="sd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<text x="20" y="28" font-size="13" font-weight="700" fill="var(--h-cherry)">static page</text>
<rect x="20" y="40" width="130" height="58" rx="9" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="85" y="66" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">requests.get</text>
<text x="85" y="84" text-anchor="middle" font-size="10" fill="var(--h-space)">your code</text>
<line x1="154" y1="69" x2="196" y2="69" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#sd)"/>
<rect x="200" y="40" width="130" height="58" rx="9" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="265" y="66" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">server</text>
<text x="265" y="84" text-anchor="middle" font-size="10" fill="var(--h-space)">builds the page</text>
<line x1="334" y1="69" x2="376" y2="69" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#sd)"/>
<rect x="380" y="40" width="280" height="58" rx="9" fill="var(--h-cherry)"/>
<text x="520" y="66" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">HTML with the quotes already in it</text>
<text x="520" y="84" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.88">BeautifulSoup can read this</text>
<text x="20" y="146" font-size="13" font-weight="700" fill="var(--h-graphite)">dynamic page</text>
<rect x="20" y="158" width="130" height="58" rx="9" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="85" y="184" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">requests.get</text>
<text x="85" y="202" text-anchor="middle" font-size="10" fill="var(--h-space)">your code</text>
<line x1="154" y1="187" x2="196" y2="187" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#sd)"/>
<rect x="200" y="158" width="130" height="58" rx="9" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="265" y="184" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">server</text>
<text x="265" y="202" text-anchor="middle" font-size="10" fill="var(--h-space)">sends a shell</text>
<line x1="334" y1="187" x2="376" y2="187" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#sd)"/>
<rect x="380" y="158" width="280" height="58" rx="9" fill="var(--h-surface)" stroke="var(--h-steel)" stroke-dasharray="5 4"/>
<text x="520" y="184" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-graphite)">HTML with no quotes in it</text>
<text x="520" y="202" text-anchor="middle" font-size="10" fill="var(--h-space)">JavaScript fetches them later, in a browser</text>
<text x="340" y="252" text-anchor="middle" font-size="12" fill="var(--h-graphite)">The test takes ten seconds: open the page source, or print the HTML, and search for a value you can see on screen.</text>
<text x="340" y="272" text-anchor="middle" font-size="12" fill="var(--h-graphite)">If it is there, requests is enough. If it is not, the data arrives by JavaScript and you need a browser.</text>
</svg>

A **static page** arrives complete. The server built the HTML, the data is in it,
and `requests` gets everything a browser would.

A **dynamic page** arrives as a shell. The HTML has the layout but not the
content, and JavaScript fetches the data after the page loads. `requests` does not
run JavaScript, so it sees an empty page and you get an empty list with no error
to explain it.

```python
import requests

html = requests.get("https://quotes.toscrape.com/").text
print("Einstein" in html)
html_js = requests.get("https://quotes.toscrape.com/js/").text
print("Einstein" in html_js)
```

```
True
False
```

Same site, same quotes on screen, two different answers. The `/js/` version
builds its page with JavaScript, which is why the project gives you both.

When the answer is `False`, you have two options and the order matters: look for
the underlying API first, and reach for a browser only if there is not one.

## Reading HTML

An HTML page is a tree of nested elements. Each element has a **tag** that says
what kind of thing it is, optional **attributes** including `class` and `id`, and
content that may be text or more elements.

```html
<div class="quote">
  <span class="text">The world as we have created it...</span>
  <small class="author">Albert Einstein</small>
  <div class="tags">
    <a class="tag" href="/tag/change/">change</a>
  </div>
</div>
```

`div` and `span` and `a` are tags. `class="quote"` is an attribute. Classes are
how a site labels elements that play the same role, which is exactly what makes
them useful to you: every quote on the page is wrapped in `div.quote`.

Inspect this yourself before writing any code. Right click on the thing you want,
choose **Inspect**, and the browser shows you the element and its classes.

## CSS selectors

A selector is a pattern that picks elements out of the tree. You need about six
of them.

<svg viewBox="0 0 680 312" role="img" aria-labelledby="ht-title ht-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="ht-title">The structure of one quote block and the selector for each part</title>
<desc id="ht-desc">A div with class quote contains a span with class text, a small with class author, and a div with class tags holding several anchors with class tag. Each part has a CSS selector that reaches it.</desc>
<text x="24" y="26" font-size="12" font-weight="700" fill="var(--h-graphite)">the page structure</text>
<text x="430" y="26" font-size="12" font-weight="700" fill="var(--h-cherry)">the selector that reaches it</text>
<rect x="20" y="36" width="380" height="244" rx="10" fill="var(--h-cherry-wash)" stroke="var(--h-cherry)" stroke-width="2"/>
<text x="36" y="58" font-size="12.5" font-weight="700" fill="var(--h-cherry)">div.quote</text>
<rect x="40" y="70" width="340" height="52" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="56" y="90" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">span.text</text>
<text x="56" y="108" font-size="10.5" fill="var(--h-space)">the quote itself</text>
<rect x="40" y="130" width="340" height="52" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="56" y="150" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">small.author</text>
<text x="56" y="168" font-size="10.5" fill="var(--h-space)">who said it</text>
<rect x="40" y="190" width="340" height="74" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="56" y="210" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">div.tags</text>
<rect x="58" y="222" width="94" height="30" rx="6" fill="var(--h-cherry)" opacity="0.8"/>
<text x="105" y="237" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="#ffffff">a.tag</text>
<rect x="162" y="222" width="94" height="30" rx="6" fill="var(--h-cherry)" opacity="0.8"/>
<text x="209" y="237" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="#ffffff">a.tag</text>
<rect x="266" y="222" width="94" height="30" rx="6" fill="var(--h-cherry)" opacity="0.8"/>
<text x="313" y="237" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="#ffffff">a.tag</text>
<line x1="404" y1="58" x2="426" y2="58" stroke="var(--h-steel)" stroke-width="1.5" stroke-dasharray="4 3"/>
<text x="432" y="62" font-size="12.5" font-weight="700" fill="var(--h-cherry)">div.quote</text>
<text x="432" y="78" font-size="10.5" fill="var(--h-space)">one whole block</text>
<line x1="404" y1="96" x2="426" y2="96" stroke="var(--h-steel)" stroke-width="1.5" stroke-dasharray="4 3"/>
<text x="432" y="100" font-size="12.5" font-weight="700" fill="var(--h-cherry)">.quote .text</text>
<text x="432" y="116" font-size="10.5" fill="var(--h-space)">the text inside it</text>
<line x1="404" y1="156" x2="426" y2="156" stroke="var(--h-steel)" stroke-width="1.5" stroke-dasharray="4 3"/>
<text x="432" y="160" font-size="12.5" font-weight="700" fill="var(--h-cherry)">.quote .author</text>
<text x="432" y="176" font-size="10.5" fill="var(--h-space)">the author inside it</text>
<line x1="404" y1="230" x2="426" y2="230" stroke="var(--h-steel)" stroke-width="1.5" stroke-dasharray="4 3"/>
<text x="432" y="234" font-size="12.5" font-weight="700" fill="var(--h-cherry)">.quote .tags a</text>
<text x="432" y="250" font-size="10.5" fill="var(--h-space)">every tag anchor</text>
<text x="340" y="302" text-anchor="middle" font-size="12" fill="var(--h-graphite)">A dot means a class. A space means "somewhere inside". That is almost every selector you will need.</text>
</svg>

| Selector | Picks |
|---|---|
| `div` | every div |
| `.quote` | every element with class quote |
| `#main` | the element with id main |
| `div.quote` | every div that has class quote |
| `.quote .text` | anything with class text inside something with class quote |
| `.quote > span` | a span that is a direct child, not a grandchild |
| `a[href]` | every anchor that has an href attribute |
| `li.next a` | the anchor inside the list item with class next |

The last one is the pagination link in the project. The space means "somewhere
inside", and the dot means "has this class". Those two ideas cover most real
selectors.

!!! warning "Do not parse HTML with regular expressions"

    The project forbids it, and the reason is not arbitrary. HTML nests, and
    regular expressions cannot count nesting. A pattern that works on the first
    page breaks on the one where somebody put a `<span>` inside a `<span>`, and
    it fails silently by matching the wrong thing rather than by raising.

## Scraping a static page

Two libraries. `requests` gets the page, BeautifulSoup parses it.

```bash
pip install requests beautifulsoup4
```

Start with fetching, and make it its own function.

```python
import requests

def fetch_html(url, headers=None, timeout=10):
    """Fetch a page and return its HTML as text."""
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text
```

Three things in four lines are worth naming. `timeout` stops your script hanging
forever on a server that never answers. `raise_for_status` turns a 404 or a 500
into an exception instead of letting you parse an error page as if it were data.
And `headers` is where you identify yourself, which comes up later.

Then parse.

```python
from bs4 import BeautifulSoup

def scrape_basic(url):
    """Return the quotes on one page as a list of dicts."""
    soup = BeautifulSoup(fetch_html(url), "html.parser")
    quotes = []
    for block in soup.select("div.quote"):
        quotes.append({
            "text": block.select_one("span.text").get_text(strip=True),
            "author": block.select_one("small.author").get_text(strip=True),
            "tags": [tag.get_text(strip=True) for tag in block.select("a.tag")],
        })
    return quotes
```

The pattern is always the same: **select the blocks, then select inside each
block.** Selecting all the texts and all the authors separately and zipping them
together works right up until one quote is missing an author, and then every row
after it is wrong.

`select` returns a list and `select_one` returns one element or `None`. Use
`get_text(strip=True)` rather than `.text` to lose the surrounding whitespace.

## Following pagination

Do not hard-code how many pages there are. Follow the link the site gives you and
stop when it is gone.

<svg viewBox="0 0 680 236" role="img" aria-labelledby="pg-title pg-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="pg-title">The pagination loop</title>
<desc id="pg-desc">Fetch a page, parse the items out of it, look for a next link. If there is one, wait politely and repeat. If there is not, stop.</desc>
<defs><marker id="pg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<rect x="5" y="46" width="152" height="70" rx="10" fill="var(--h-cherry)" stroke="var(--h-cherry)"/>
<text x="81" y="76" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">fetch the page</text>
<text x="81" y="96" text-anchor="middle" font-size="10.5" fill="#ffffff" opacity="0.85">one request</text>
<line x1="163" y1="81" x2="175" y2="81" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#pg)"/>
<rect x="175" y="46" width="152" height="70" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="251" y="76" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">parse the items</text>
<text x="251" y="96" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color)" opacity="0.85">collect the dicts</text>
<line x1="333" y1="81" x2="345" y2="81" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#pg)"/>
<rect x="345" y="46" width="152" height="70" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="421" y="76" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">look for next</text>
<text x="421" y="96" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color)" opacity="0.85">li.next a</text>
<line x1="503" y1="81" x2="515" y2="81" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#pg)"/>
<rect x="515" y="46" width="152" height="70" rx="10" fill="var(--h-cherry)" stroke="var(--h-cherry)"/>
<text x="591" y="76" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">wait, then repeat</text>
<text x="591" y="96" text-anchor="middle" font-size="10.5" fill="#ffffff" opacity="0.85">be polite</text>
<polyline points="601,120 601,158 81,158 81,122" fill="none" stroke="var(--h-steel)" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#pg)"/>
<text x="340" y="178" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--h-graphite)">while a next link exists</text>
<line x1="430" y1="120" x2="430" y2="196" stroke="var(--h-cherry)" stroke-width="2"/>
<text x="440" y="200" font-size="11.5" font-weight="700" fill="var(--h-cherry)">no next link, stop</text>
<text x="340" y="226" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Never hard-code the number of pages. Follow the link the site gives you, and stop when it disappears.</text>
</svg>

```python
import time
from urllib import parse

def scrape_paginated(base_url):
    """Follow the next links until there are none left."""
    all_quotes = []
    url = base_url
    while url:
        all_quotes.extend(scrape_basic(url))
        soup = BeautifulSoup(fetch_html(url), "html.parser")
        next_link = soup.select_one("li.next a")
        url = parse.urljoin(url, next_link["href"]) if next_link else None
        time.sleep(1)
    return all_quotes
```

`urljoin` is there because the href is `/page/2/`, a path rather than a full
address. Joining it to the page you are on gives you the address to request, and
it keeps working if the site moves to a different domain.

The `sleep` is not optional and it is not politeness theatre. Without it you send
requests as fast as your connection allows, which is what a denial of service
attack looks like from the other end.

!!! warning "Guard against the loop that never ends"

    If a site's next link points back at a page you have already seen, that
    `while` loop runs forever. Keep a set of visited URLs and stop if one repeats,
    and put a maximum page count on any scraper you leave running.

## Data that is already structured

Before parsing layout, check whether the page is carrying its data in a clean
form. Two places to look.

**A JSON API**, found in the Network tab as described above. On the quotes site:

```python
import json

def scrape_via_api(base_url):
    """Read the quotes from the site's own JSON endpoint."""
    quotes = []
    page = 1
    while True:
        payload = json.loads(fetch_html(f"{base_url}/api/quotes?page={page}"))
        for item in payload["quotes"]:
            quotes.append({
                "text": item["text"],
                "author": item["author"]["name"],
                "tags": item["tags"],
            })
        if not payload.get("has_next"):
            break
        page += 1
    return quotes
```

No selectors, no layout assumptions, nothing to break when the design changes.

**JSON-LD**, which many sites embed so that search engines can read them.

```python
def extract_json_ld(html):
    """Return the structured data blocks a page publishes for search engines."""
    soup = BeautifulSoup(html, "html.parser")
    blocks = []
    for script in soup.select('script[type="application/ld+json"]'):
        blocks.append(json.loads(script.string))
    return blocks
```

Product pages, recipes, articles and events very often carry this. It is the
site handing you clean data, and most people scraping the page never notice it
is there.

## When you need a real browser

Selenium drives an actual browser. It loads the page, runs the JavaScript, and
lets you read the result or interact with it. Use it when, and only when, the
data does not exist without JavaScript, or when you have to click, type, filter
or scroll to reach it.

```bash
pip install selenium
```

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
driver = webdriver.Chrome(options=options)
driver.get("https://quotes.toscrape.com/js/")
```

Headless means the browser runs without drawing a window. Keep it off while you
are developing, because watching what happens is how you debug, and turn it on
once the script works.

Always close it when you are finished, or the processes pile up until your
machine slows to a crawl.

```python
driver.quit()
```

### Waiting properly

This is the part everybody gets wrong first.

<svg viewBox="0 0 680 274" role="img" aria-labelledby="wt-title wt-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="wt-title">Fixed sleep compared with an explicit wait</title>
<desc id="wt-desc">After the page loads, JavaScript runs and the element appears some time later. A fixed sleep either wastes time or gives up too early. An explicit wait continues the moment the element exists.</desc>
<defs><marker id="wt" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<line x1="60" y1="70" x2="640" y2="70" stroke="var(--h-steel)" stroke-width="1.5"/>
<text x="60" y="34" font-size="11.5" font-weight="700" fill="var(--h-graphite)">page requested</text>
<line x1="60" y1="40" x2="60" y2="78" stroke="var(--h-graphite)" stroke-width="1.5"/>
<text x="220" y="34" font-size="11.5" font-weight="700" fill="var(--h-graphite)">HTML arrives</text>
<line x1="220" y1="40" x2="220" y2="78" stroke="var(--h-graphite)" stroke-width="1.5"/>
<text x="380" y="34" font-size="11.5" font-weight="700" fill="var(--h-cherry)">element appears</text>
<line x1="380" y1="40" x2="380" y2="78" stroke="var(--h-cherry)" stroke-width="2"/>
<text x="640" y="34" text-anchor="end" font-size="11" fill="var(--h-space)">time</text>
<rect x="60" y="100" width="160" height="26" rx="5" fill="var(--h-steel)" opacity="0.35"/>
<text x="140" y="116" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-graphite)">sleep(1) too short</text>
<text x="232" y="120" font-size="11" font-weight="700" fill="var(--h-cherry)">element not there yet, your code crashes</text>
<rect x="60" y="140" width="500" height="26" rx="5" fill="var(--h-steel)" opacity="0.35"/>
<text x="310" y="156" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--h-graphite)">sleep(5) works, and wastes 2 seconds on every page</text>
<rect x="60" y="180" width="320" height="26" rx="5" fill="var(--h-cherry)"/>
<text x="220" y="196" dy="0.36em" text-anchor="middle" font-size="10.5" font-weight="700" fill="#ffffff">WebDriverWait continues the instant it appears</text>
<text x="340" y="238" text-anchor="middle" font-size="12" fill="var(--h-graphite)">A fixed sleep is a guess about somebody else’s network. An explicit wait asks the page a question.</text>
<text x="340" y="258" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Over a thousand pages the difference is hours.</text>
</svg>

```python
quotes = WebDriverWait(driver, 10).until(
    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.quote"))
)
```

This means "wait up to ten seconds, and continue the moment these elements
exist". It is faster than a fixed sleep on a good connection and more reliable on
a bad one.

Two conditions cover most cases. `presence_of_element_located` means the element
is in the page. `element_to_be_clickable` means it is also visible and enabled,
which is what you need before a click.

!!! tip "Read the page with BeautifulSoup once the browser is done"

    ```python
    soup = BeautifulSoup(driver.page_source, "html.parser")
    ```

    You already know how to parse HTML. Let Selenium do the part only a browser
    can do, then hand the finished page to the parser you are comfortable with.

### Clicking, typing and filtering

```python
driver.get("https://quotes.toscrape.com/login")
driver.find_element(By.ID, "username").send_keys("student")
driver.find_element(By.ID, "password").send_keys("student")
driver.find_element(By.CSS_SELECTOR, "input[type='submit']").click()
```

Filters in a user interface work the same way: find the control, set it, click
the button, then wait for the results to appear. The wait matters, because the
old results are still on screen for a moment after you click and you will scrape
them by accident.

### Cookie banners and popups

A banner covering the page will make clicks fail with an error about the element
being intercepted. Dismiss it first, and do not assume it is always there.

```python
from selenium.common.exceptions import TimeoutException

try:
    button = WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, ".cookie-accept"))
    )
    button.click()
except TimeoutException:
    pass
```

The `try` is the point. A banner that appears only for new visitors will be
missing when you run the script a second time, and an unhandled exception there
kills an otherwise working scraper.

### Infinite scroll

Some pages load more content as you scroll rather than paginating. The pattern is
to scroll to the bottom, wait, and check whether the page got taller.

```python
import time

def scroll_to_bottom(driver, pause=1.5, max_rounds=50):
    """Scroll until the page stops growing."""
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(max_rounds):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            return
        last_height = new_height
```

`max_rounds` is a safety net. Some feeds are genuinely endless, and without a
limit your script runs until the machine runs out of memory.

## Not collecting the same thing twice

Pagination that overlaps, a scroll that reloads, a retry after a failure: all
three give you duplicates. Decide what makes a record unique and check it as you
go.

```python
def deduplicate(records, key_fields=("text", "author")):
    """Keep the first occurrence of each record."""
    seen = set()
    unique = []
    for record in records:
        key = tuple(record[field] for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique
```

A set membership test is instant no matter how many records you have already
seen, which a list is not. On ten thousand records that difference is the
difference between a second and a minute.

You can also do it at the end in pandas with `drop_duplicates`, but doing it
during the scrape means a duplicate can also tell you to stop early.

## Organising a scraper

The project's task numbering is the architecture, and it is worth noticing.

| Layer | Responsibility |
|---|---|
| `fetch_html` | Talk to the network. Headers, timeouts, retries, errors |
| `scrape_basic` | Turn one page of HTML into records |
| `scrape_paginated` | Decide which pages to visit and in what order |

Each layer knows nothing about the ones above it. That means you can test the
parser on a saved HTML file with no network at all, which is the single biggest
improvement you can make to how quickly you develop a scraper.

!!! tip "Save the page while you are developing"

    ```python
    html = fetch_html(url)
    open("page.html", "w", encoding="utf-8").write(html)
    ```

    Then work on your selectors against the saved copy. You will run the parser
    fifty times before it is right, and there is no reason for the server to see
    any of that.

## Scraping responsibly

<svg viewBox="0 0 680 286" role="img" aria-labelledby="rs-title rs-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="rs-title">The checks to make before you scrape</title>
<desc id="rs-desc">Check robots.txt, check the terms of service, and check that you are taking only what you need. If all three pass, scrape slowly and identify yourself. If any fails, do not scrape.</desc>
<defs><marker id="rs" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<rect x="10" y="30" width="204" height="76" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="112" y="58" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">robots.txt</text>
<text x="112" y="80" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">does it allow this path?</text>
<line x1="112" y1="110" x2="112" y2="146" stroke="var(--h-steel)" stroke-width="1.5" marker-end="url(#rs)"/>
<line x1="218" y1="68" x2="230" y2="68" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#rs)"/>
<rect x="234" y="30" width="204" height="76" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="336" y="58" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">terms of service</text>
<text x="336" y="80" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">do they forbid automated access?</text>
<line x1="336" y1="110" x2="336" y2="146" stroke="var(--h-steel)" stroke-width="1.5" marker-end="url(#rs)"/>
<line x1="442" y1="68" x2="454" y2="68" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#rs)"/>
<rect x="458" y="30" width="204" height="76" rx="10" fill="var(--h-surface)" stroke="var(--h-cherry-line)"/>
<text x="560" y="58" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">what you take</text>
<text x="560" y="80" text-anchor="middle" font-size="10.5" fill="var(--h-graphite)">only what you need, never personal data</text>
<line x1="560" y1="110" x2="560" y2="146" stroke="var(--h-steel)" stroke-width="1.5" marker-end="url(#rs)"/>
<rect x="10" y="150" width="652" height="54" rx="10" fill="var(--h-cherry)"/>
<text x="336" y="172" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">all three pass: scrape politely</text>
<text x="336" y="192" text-anchor="middle" font-size="11" fill="#ffffff" opacity="0.9">a delay between requests, a user agent that says who you are, and cache what you already have</text>
<text x="340" y="232" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-cherry)">any one of them fails: do not scrape it</text>
<text x="340" y="258" text-anchor="middle" font-size="12" fill="var(--h-graphite)">Fifteen students hitting one server from one building looks exactly like an attack, and gets one IP banned.</text>
<text x="340" y="276" text-anchor="middle" font-size="12" fill="var(--h-graphite)">That IP belongs to the school.</text>
</svg>

**Read robots.txt.** It is at the root of every site, and Python will read it for
you.

```python
from urllib import robotparser

rp = robotparser.RobotFileParser()
rp.set_url("https://quotes.toscrape.com/robots.txt")
rp.read()
print(rp.can_fetch("*", "https://quotes.toscrape.com/page/2/"))
```

It is a request rather than a lock, which is exactly why ignoring it says
something about you.

**Say who you are.** The default user agent announces you as a Python script,
and many sites block it. Do not disguise yourself as a browser you are not. Say
what you are and leave a way to contact you.

```python
headers = {"User-Agent": "HolbertonML-student-project (evis.plaku@example.com)"}
html = fetch_html(url, headers=headers)
```

**Go slowly.** One request per second is a reasonable default for a small site.
There is no prize for finishing first, and the cost of going too fast is paid by
somebody else.

**Take only what you need.** Scraping personal data brings you under data
protection law, which you read about in Fundamentals. Do not collect names,
emails or profiles because they happen to be there.

**Cache.** If you have already downloaded a page, use your copy. Re-downloading
the same thing because it was easier than saving it is the most common way
students generate thousands of pointless requests.

!!! warning "You are sharing an IP address"

    Every student in the building appears to a server as the same address. If
    fifteen scrapers run without delays at the same time, that server sees an
    attack and blocks the school, not you. This is the reason the project uses
    a practice site built for the purpose.

## When it goes wrong

??? note "My list is empty and there is no error"

    Either the selector matches nothing or the data is not in the HTML. Print
    `len(html)` and search it for a value you can see on screen. If it is
    missing, the page is dynamic and you need the API or a browser.

??? note "`AttributeError: 'NoneType' object has no attribute 'get_text'`"

    `select_one` found nothing and returned `None`. Some blocks are missing the
    element you assumed was always there. Check before using it, or use a
    default.

??? note "403 Forbidden"

    The site is refusing your request, often because of the default user agent.
    Set a real one that identifies you. If it still refuses, that is an answer
    and the answer is no.

??? note "429 Too Many Requests"

    You are going too fast. Increase the delay, and back off further each time
    you see this rather than retrying immediately.

??? note "`ElementClickInterceptedException`"

    Something is on top of the element, usually a cookie banner or a modal.
    Dismiss it first.

??? note "`NoSuchElementException` even though I can see the element"

    Selenium looked before JavaScript had finished. Use `WebDriverWait` rather
    than reading immediately after `get`.

??? note "Everything worked yesterday and today it returns nothing"

    The site changed its layout. This is normal, it is the cost of scraping, and
    it is why an API is worth looking for first.

## Check yourself

Use `quotes.toscrape.com` for all of these.

1. Write `fetch_html` and prove it raises on a URL that returns 404.
2. Scrape the first page into a list of dicts and load it into a DataFrame.
   Confirm you have ten rows and the tags are lists.
3. Follow the pagination to the end. How many quotes are there in total, and how
   did your code know when to stop?
4. Find the JSON endpoint in the Network tab and scrape the same data through
   it. Compare the two results and the time each took.
5. Open `/js/` with `requests` and show that the quotes are not in the HTML. Then
   get them with Selenium and an explicit wait.
6. Combine the results of two overlapping scrapes and deduplicate them. Say what
   you chose as the key and why.
7. Read the site's robots.txt and say, in one sentence, what it permits.

## Quick reference

| Task | Code |
|---|---|
| Fetch a page | `requests.get(url, headers=h, timeout=10)` |
| Fail on error status | `response.raise_for_status()` |
| Parse | `BeautifulSoup(html, "html.parser")` |
| All matches | `soup.select("div.quote")` |
| First match | `soup.select_one("span.text")` |
| Clean text | `el.get_text(strip=True)` |
| Attribute | `el["href"]` |
| Absolute URL | `parse.urljoin(base, href)` |
| Start a browser | `webdriver.Chrome(options=options)` |
| Wait for an element | `WebDriverWait(driver, 10).until(EC.presence_of_element_located(...))` |
| Click, type | `el.click()`, `el.send_keys("text")` |
| Scroll | `driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")` |
| Hand over to the parser | `BeautifulSoup(driver.page_source, "html.parser")` |
| Close the browser | `driver.quit()` |
| Check permission | `robotparser.RobotFileParser()` |

## Questions to sit with

1. The data was public on a page anybody could open. Does that make collecting
   all of it, automatically, the same thing? What changes between one visit and
   ten thousand?
2. Your scraper is polite, identifies itself and respects robots.txt, and the
   site still blocks it. What does that tell you, and what do you do next?
3. A site publishes JSON-LD so search engines can read it. Is using that data
   different from parsing the visible page, and should it be?
4. Scrapers break when layouts change, so the data you collected last month
   cannot be collected the same way today. What does that mean for anyone trying
   to reproduce your results?

## Next

Scraping is what you do when nobody gave you a proper way in. Next, the proper
way in.

[Data Collection: APIs](apis.md){ .h-button }

Worth your time: the
[BeautifulSoup documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
is short and genuinely readable, and the
[Selenium waits guide](https://www.selenium.dev/documentation/webdriver/waits/)
explains the difference between the wait types better than any tutorial.

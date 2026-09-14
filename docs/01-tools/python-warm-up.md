# Python warm up

<span class="badge badge--time">60 min</span>
<span class="badge badge--level">Foundations</span>
<span class="badge">Needs: Python running, from the last tutorial</span>

Seven small tasks on realistic data. Loops, functions, lists, and dictionaries.
Nothing here is new material, and that is the point. These are the moves you
will use every day for the rest of the programme.

## Why this matters

Most machine learning work is not modelling. It is getting data out of an
awkward shape and into a useful one. Before a model sees anything, somebody
loops over records, pulls out the fields that matter, groups them, counts them,
and checks what is missing.

That somebody is you. The exercises below are that job in miniature.

## The data

You will use the sample users from
[JSONPlaceholder](https://jsonplaceholder.typicode.com/users). It is fake data
shaped like a real web API, which means it is nested. Fields inside fields.

<svg viewBox="0 0 680 276" role="img" aria-labelledby="json-title json-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="json-title">The shape of one user record</title>
<desc id="json-desc">Data is a list of users. Each user has name, username and email at the top level, plus an address object containing street, suite, city and zipcode, and a company object containing name.</desc>
<rect x="3" y="12" width="200" height="50" rx="10" fill="var(--h-cherry)"/>
<text x="103" y="34" text-anchor="middle" font-size="13.5" font-weight="700" fill="#ffffff">data</text>
<text x="103" y="52" text-anchor="middle" font-size="11.5" fill="#ffffff" opacity="0.88">a list of 10 users</text>
<line x1="103" y1="62" x2="103" y2="86" stroke="var(--h-steel)" stroke-width="2"/>
<rect x="3" y="86" width="200" height="50" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="103" y="108" text-anchor="middle" font-size="13" font-weight="700" fill="var(--md-default-fg-color)">data[0]</text>
<text x="103" y="126" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">one user, a dictionary</text>
<line x1="103" y1="136" x2="103" y2="240" stroke="var(--h-steel)" stroke-width="2"/>
<line x1="103" y1="162" x2="243" y2="162" stroke="var(--h-steel)" stroke-width="2"/>
<rect x="243" y="140" width="434" height="44" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="263" y="160" font-size="12" font-weight="600" fill="var(--md-default-fg-color)">"name", "username", "email"</text>
<text x="263" y="176" font-size="11.5" fill="var(--h-graphite)">plain text, one level down</text>
<line x1="103" y1="206" x2="243" y2="206" stroke="var(--h-steel)" stroke-width="2"/>
<rect x="243" y="190" width="434" height="44" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-cherry)"/>
<text x="263" y="210" font-size="12" font-weight="600" fill="var(--md-default-fg-color)">"address" holds "street", "suite", "city", "zipcode"</text>
<text x="263" y="226" font-size="11.5" fill="var(--h-graphite)">a dictionary inside the dictionary</text>
<line x1="103" y1="256" x2="243" y2="256" stroke="var(--h-steel)" stroke-width="2"/>
<rect x="243" y="240" width="434" height="32" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-cherry)"/>
<text x="263" y="261" font-size="12" font-weight="600" fill="var(--md-default-fg-color)">"company" holds "name"</text>
</svg>

So a city is not `user["city"]`. It is `user["address"]["city"]`. Getting that
wrong is the single most common error in these tasks, and reading the diagram
above is faster than guessing.

## Getting the data

=== "Fetch it live"

    ```python
    import requests

    data = requests.get("https://jsonplaceholder.typicode.com/users").json()
    ```

    If Python complains that requests is missing, install it first:

    ```bash
    pip install requests
    ```

=== "Paste it in"

    Open [the URL](https://jsonplaceholder.typicode.com/users) in your browser,
    copy everything, and paste it into your file as a list:

    ```python
    data = [
        {"id": 1, "name": "Leanne Graham", ...},
    ]
    ```

    This works offline and is a fine choice.

Before you write anything else, look at one record:

```python
print(data[0])
print(data[0]["address"]["city"])
```

Always do this with unfamiliar data. Ten seconds of looking saves twenty minutes
of guessing at key names.

## The tasks

Create a file called `user_processing.py`. Each task is one function.

### Task 1: Print every name

Loop through `data` and print each user's `name`.

```
Leanne Graham
Ervin Howell
...
```

### Task 2: Collect the .biz emails

Return a list of every email address ending in `.biz`.

Return the list, do not print it inside the function. A function that returns a
value can be used by other code. A function that only prints is a dead end.

### Task 3: Find users by city

Take a city name as an argument and print the name and email of everyone living
there. Remember the nesting: `user["address"]["city"]`.

### Task 4: Count users per company

Build a dictionary where the key is a company name and the value is how many
users work there.

```
{'Romaguera-Crona': 1, 'Deckow-Crist': 1, ...}
```

??? tip "Stuck on the counting pattern?"

    The shape is always the same. Check whether the key exists, add it with a
    starting value if it does not, then increase it.

    ```python
    counts = {}
    for item in things:
        key = item["something"]
        counts[key] = counts.get(key, 0) + 1
    ```

    `dict.get(key, 0)` returns zero instead of crashing when the key is missing.
    You will use this pattern constantly.

### Task 5: List the unique zip codes

Pull every zip code out of the address section, remove duplicates, and print
them sorted.

A `set` removes duplicates for you. `sorted()` turns it back into an ordered
list.

### Task 6: Group users by the first letter of their username

Build a dictionary where each key is a first letter and each value is a list of
names.

```
{'B': ['Leanne Graham'], 'A': ['Ervin Howell', 'Chelsey Dietrich'], ...}
```

This is grouping, and it is the same operation as `groupby` in pandas, which
you will meet in the Data module. Doing it by hand once makes the pandas version
obvious later.

### Task 7: Print a full address

Take a username and print that person's address on one line:

```
Leanne Graham: Apt. 556, Kulas Light, Gwenborough, 92998-3874
```

Decide what your function does when the username does not exist. Printing a
clear message beats crashing, and thinking about the missing case is a habit
worth building now.

## How to hand it in

Put every task in its own function and call them from a main block:

```python
def print_all_names(data):
    """Print the name of every user."""
    for user in data:
        print(user["name"])


if __name__ == "__main__":
    print_all_names(data)
```

The `if __name__ == "__main__":` line means the code below it runs when you run
the file directly, but not when another file imports your functions. It is
standard in Python and you should use it from now on.

Then commit and push:

```bash
git add user_processing.py
git commit -m "Add user processing exercises"
git push
```

## Check yourself

Before you submit, confirm each of these:

1. Every task is a separate function with a name that says what it does.
2. Tasks 2 and 6 return their result rather than only printing it.
3. Running `python3 user_processing.py` produces output for all seven tasks
   with no errors.
4. Each function has a one line docstring.
5. Your work is pushed to GitHub and you can see it in the browser.

!!! tip "One test of whether you have really got this"

    Close the file, open a new one, and write Task 4 again from memory. If you
    can, the pattern is yours. If you cannot, that is useful information and it
    is better to find out now than in the Data module.

## You have finished Tools

Your accounts work, your code is backed up on GitHub, you can experiment in
notebooks, and you have a Linux environment for real programs.

The next module starts with data. Watch the cohort channel for when it opens.

# Jupyter notebooks

<span class="badge badge--time">25 min</span>
<span class="badge badge--level">Foundations</span>
<span class="badge">Needs: a computer you can install software on</span>

A notebook lets you run code one piece at a time and see the result right
underneath it. You will spend more hours in notebooks than in any other tool
this year.

## Why this matters

Machine learning is guesswork with evidence. You load some data, look at it,
try something, look again. Running a whole script every time you want to check
one number is slow and it hides what went wrong.

Notebooks fix that. Each piece of code stays on screen next to its output, so
your experiment and your record of the experiment are the same document.

## What a notebook is made of

<svg viewBox="0 0 680 286" role="img" aria-labelledby="nb-title nb-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="nb-title">The parts of a notebook</title>
<desc id="nb-desc">A notebook holds markdown cells for notes and code cells. Running a code cell sends it to the kernel, which is the Python process that remembers your variables, and the result appears as output below the cell.</desc>
<rect x="3" y="10" width="440" height="266" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="24" y="38" font-size="12" font-weight="700" fill="var(--h-space)">notebook.ipynb</text>
<rect x="24" y="52" width="398" height="46" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="42" y="72" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">Markdown cell</text>
<text x="42" y="90" font-size="11.5" fill="var(--h-graphite)">what you tried and why</text>
<rect x="24" y="110" width="398" height="46" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-cherry)"/>
<text x="42" y="130" font-size="12.5" font-weight="700" fill="var(--md-default-fg-color)">Code cell</text>
<text x="42" y="148" font-size="11.5" fill="var(--h-graphite)">df.head()</text>
<rect x="24" y="168" width="398" height="46" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)" stroke-dasharray="4 4"/>
<text x="42" y="188" font-size="12.5" font-weight="700" fill="var(--h-graphite)">Output</text>
<text x="42" y="206" font-size="11.5" fill="var(--h-graphite)">the table, the plot, the error</text>
<rect x="24" y="226" width="398" height="34" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="42" y="248" font-size="12.5" fill="var(--h-graphite)">the next cell, still empty</text>
<rect x="487" y="94" width="190" height="88" rx="12" fill="var(--h-cherry)"/>
<text x="582" y="130" text-anchor="middle" font-size="15" font-weight="700" fill="#ffffff">Kernel</text>
<text x="582" y="152" text-anchor="middle" font-size="11.5" fill="#ffffff" opacity="0.88">the Python that remembers</text>
<text x="582" y="168" text-anchor="middle" font-size="11.5" fill="#ffffff" opacity="0.88">your variables</text>
<line x1="447" y1="126" x2="481" y2="126" stroke="var(--h-steel)" stroke-width="2"/>
<line x1="481" y1="150" x2="447" y2="150" stroke="var(--h-steel)" stroke-width="2" stroke-dasharray="4 4"/>
<text x="464" y="114" text-anchor="middle" font-size="11" font-weight="600" fill="var(--h-cherry)">run</text>
<text x="464" y="172" text-anchor="middle" font-size="11" font-weight="600" fill="var(--h-graphite)">result</text>
</svg>

The kernel is the part people forget, and it causes most notebook confusion.
It is a Python process running in the background. It remembers every variable
you have created, in the order you ran the cells, not the order they appear on
screen.

## Step 1: Install Anaconda

Anaconda is Python plus about 250 packages, including everything you need for
the next few months. Installing it is easier than installing the pieces one by
one.

Download the installer for your system from the
[Anaconda download page](https://www.anaconda.com/download) and run it.

=== "Windows"

    Double click the `.exe` file and follow the installer.

    Choose **Just Me**, keep the default folder, and leave **Add Anaconda to
    PATH** unchecked. Do allow it to register as your default Python.

    Leaving it off PATH looks wrong but is correct. It stops Anaconda's Python
    from shadowing the system Python and breaking other software. You launch it
    from the Anaconda Prompt instead.

=== "macOS"

    Double click the `.pkg` file and follow the installer. The defaults are
    fine.

=== "Linux or WSL"

    ```bash
    bash ~/Downloads/Anaconda3-2025.06-Linux-x86_64.sh
    ```

    Adjust the filename to match what you downloaded. Accept the license, keep
    the default location, and answer yes when it offers to run `conda init`.
    Then close and reopen your terminal.

You are done when Anaconda Navigator opens from your applications menu.

## Step 2: Launch Jupyter

=== "Anaconda Navigator"

    Open Anaconda Navigator and click **Launch** under Jupyter Notebook. This
    is the simplest way and the one to use if anything else fails.

=== "Terminal"

    ```bash
    jupyter notebook
    ```

    On Windows, run this in the Anaconda Prompt rather than the normal Command
    Prompt.

Your browser opens at `http://localhost:8888` showing your files. That page is
the dashboard, not a notebook yet.

!!! tip "Start Jupyter from the folder you want to work in"

    Jupyter can only see the folder it was started in and anything below it. Use
    `cd` to get to your project folder first, then run `jupyter notebook`. It
    saves a lot of hunting.

## Step 3: Run your first cells

Click **New**, then **Python 3**. An empty cell appears.

Type this into it and press ++shift+enter++.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df.head()
```

You should see a small table of flower measurements. If you do, every library
you need for the next few months is installed and working.

Notice the last line has no `print()`. A notebook shows the value of the final
line in a cell automatically, and for a table it renders it properly instead of
dumping text.

Now add a second cell and run it:

```python
df["sepal length (cm)"].mean()
```

It works because the kernel still remembers `df` from the first cell. That is
the whole idea.

### The shortcuts worth learning

| Keys | What it does |
|---|---|
| ++shift+enter++ | Run the cell and move to the next one |
| ++ctrl+enter++ | Run the cell and stay where you are |
| ++b++ | New cell below (press ++esc++ first) |
| ++a++ | New cell above |
| ++m++ | Turn this cell into notes (markdown) |
| ++y++ | Turn it back into code |
| ++d++ ++d++ | Delete the cell |

Press ++esc++ before the single letter shortcuts. Otherwise you just type the
letter into your code.

## Step 4: Keep your notebooks trustworthy

Cells remember their results, so a notebook can look like it works when it does
not. You edited a cell, ran a later one, went back, and now the numbers on
screen came from code that no longer exists.

There is one habit that prevents this. Before you trust a result or show it to
anyone, use **Kernel**, then **Restart and Run All**. That wipes the kernel's
memory and runs every cell from the top in order. If it still works, it really
works.

!!! warning "Clear your outputs before committing to Git"

    A notebook stores its outputs inside the file, including every plot as
    encoded image data. Commit it as is and your repository fills with noise
    that nobody can read in a diff. Use **Kernel**, then **Restart and Clear
    Output**, before you `git add` the notebook.

A few more things that will save you later:

- Name notebooks so the order is obvious: `01-explore-data.ipynb`, then
  `02-train-model.ipynb`.
- Write a markdown cell above each section saying what you are about to try.
  Your future self reads those, and so does your instructor.
- When a notebook gets long and the code is settled, move it into a `.py` file.
  Notebooks are for figuring things out, not for keeping things.

## Check yourself

1. Create a notebook called `01-first-look.ipynb`.
2. Load the iris data and show the first five rows.
3. Add a markdown cell above it saying what the dataset contains.
4. Add a cell that prints the average of each column.
5. Run **Restart and Run All** and confirm everything still works from a clean
   start.

## Next

You can now experiment quickly. Next you set up the editor for writing real
programs, the ones with more than a few cells in them.

[VS Code setup](vscode-setup.md){ .h-button }

More depth when you want it: the
[Jupyter documentation](https://jupyter-notebook.readthedocs.io/) is the
official reference.

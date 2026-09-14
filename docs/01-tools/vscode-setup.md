# VS Code and WSL

<span class="badge badge--time">35 min</span>
<span class="badge badge--level">Foundations</span>
<span class="badge">Needs: Windows 10 or 11, admin rights</span>

Notebooks are for experiments. Once your code grows past a few cells you want a
real editor. This tutorial sets up VS Code on Windows with Linux running
underneath it.

## Why this matters

Almost every machine learning system in production runs on Linux. If you learn
on Windows paths and Windows commands, you relearn everything the first time
you deploy something.

WSL solves this. It gives you a real Ubuntu inside Windows. You keep the
Windows you are used to, your code runs on Linux, and the two share files.

!!! note "On a Mac or Linux already?"

    You do not need WSL. Install VS Code, skip to Step 4, and use your normal
    terminal wherever this page says WSL terminal.

## How the pieces fit

<svg viewBox="0 0 680 216" role="img" aria-labelledby="wsl-title wsl-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="wsl-title">VS Code on Windows connected to Ubuntu in WSL</title>
<desc id="wsl-desc">VS Code runs on Windows and shows the interface. The WSL extension connects it to Ubuntu, where your code, Python and packages actually run. Windows files are reachable from Ubuntu under slash mnt slash c.</desc>
<rect x="3" y="14" width="300" height="188" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="24" y="44" font-size="12" font-weight="700" fill="var(--h-space)">WINDOWS</text>
<rect x="24" y="60" width="258" height="58" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="44" y="84" font-size="14" font-weight="700" fill="var(--md-default-fg-color)">VS Code</text>
<text x="44" y="104" font-size="11.5" fill="var(--h-graphite)">what you look at and type in</text>
<rect x="24" y="130" width="258" height="52" rx="8" fill="var(--md-default-bg-color)" stroke="var(--h-surface-line)"/>
<text x="44" y="152" font-size="13" font-weight="600" fill="var(--md-default-fg-color)">Your Windows files</text>
<text x="44" y="170" font-size="11.5" fill="var(--h-graphite)">C:\Users\you\Documents</text>
<rect x="377" y="14" width="300" height="188" rx="12" fill="var(--h-cherry)"/>
<text x="398" y="44" font-size="12" font-weight="700" fill="#ffffff" opacity="0.8">UBUNTU, INSIDE WSL</text>
<rect x="398" y="60" width="258" height="58" rx="8" fill="rgba(255,255,255,0.14)"/>
<text x="418" y="84" font-size="14" font-weight="700" fill="#ffffff">Python and your packages</text>
<text x="418" y="104" font-size="11.5" fill="#ffffff" opacity="0.88">where the code actually runs</text>
<rect x="398" y="130" width="258" height="52" rx="8" fill="rgba(255,255,255,0.14)"/>
<text x="418" y="152" font-size="13" font-weight="600" fill="#ffffff">Your project files</text>
<text x="418" y="170" font-size="11.5" fill="#ffffff" opacity="0.88">~/ml-projects</text>
<line x1="307" y1="88" x2="371" y2="88" stroke="var(--h-steel)" stroke-width="2"/>
<text x="339" y="76" text-anchor="middle" font-size="11" font-weight="600" fill="var(--h-cherry)">WSL extension</text>
<line x1="371" y1="156" x2="307" y2="156" stroke="var(--h-steel)" stroke-width="2" stroke-dasharray="4 4"/>
<text x="339" y="176" text-anchor="middle" font-size="11" font-weight="600" fill="var(--h-graphite)">/mnt/c</text>
</svg>

One thing to hold on to: the window is Windows, the code is Linux.

## Step 1: Install VS Code

Download it from [code.visualstudio.com](https://code.visualstudio.com/) and
run the installer. Tick these three boxes when offered:

- Add "Open with Code" to the file context menu
- Add "Open with Code" to the directory context menu
- Add to PATH

The last one lets you type `code .` in any terminal to open the current folder.
You will use it constantly.

## Step 2: Install WSL

Right click the Start button and choose **Terminal (Admin)**, then run:

```powershell
wsl --install
```

This turns on the Windows features WSL needs, downloads the Linux kernel, and
installs Ubuntu. Restart when it asks.

After the restart, Ubuntu opens and asks for a username and password. They do
not have to match your Windows ones. Write the password down somewhere, because
you need it every time you run `sudo` and Linux does not show any characters
while you type it.

Check it worked:

```powershell
wsl --list --verbose
```

You want to see Ubuntu with VERSION 2:

```
  NAME      STATE           VERSION
* Ubuntu    Running         2
```

??? note "`wsl --install` fails or WSL version says 1"

    Virtualisation is probably turned off in your BIOS. Restart, enter BIOS
    setup, and enable Intel VT-x or AMD-V. If WSL installed but shows version
    1, run `wsl --set-version Ubuntu 2` and wait. It can take a few minutes.

## Step 3: Connect VS Code to Ubuntu

1. Open VS Code and press ++ctrl+shift+x++ for Extensions.
2. Search for **WSL** by Microsoft and install it.
3. Press ++ctrl+shift+p++ and run **WSL: Connect to WSL**.

A new window opens. Look at the bottom left corner. It should say **WSL:
Ubuntu**. That label is how you know which side you are on, and you should check
it whenever something behaves oddly.

Open the terminal inside VS Code with ++ctrl+j++, or with Ctrl and the backtick
key. The prompt now looks like this:

```bash
yourname@YOUR-PC:~$
```

That is Linux. Everything you type there runs on Ubuntu.

## Step 4: Set up Python on the Linux side

Update the system first:

```bash
sudo apt update && sudo apt upgrade -y
```

Then install the basics:

```bash
sudo apt install python3 python3-pip python3-venv git curl -y
python3 --version
```

Now make a virtual environment. A virtual environment is a private folder of
packages for one project. Without it, two projects that need different versions
of the same library fight each other, and the loser is whichever one you open
second.

```bash
python3 -m venv ~/ml-env
source ~/ml-env/bin/activate
pip install numpy pandas matplotlib scikit-learn jupyter
```

Your prompt now starts with `(ml-env)`. That tells you the environment is
active. It deactivates when you close the terminal, so run the `source` line
again each session.

!!! note "You installed Anaconda already. Is this a duplicate?"

    No. Anaconda is on the Windows side, WSL is a separate machine, and neither
    can see the other's packages. Use Anaconda and its notebooks for
    exploration on Windows, and this environment for the code you run in
    Linux. If you prefer one place for everything, install Anaconda inside WSL
    instead and use it for both.

## Step 5: Work in the right folder

This is the one WSL rule worth memorising.

**Keep your projects in the Linux home folder.** Files there live on the Linux
filesystem and are fast. Files under `/mnt/c` live on the Windows filesystem
and every read crosses a bridge, which makes Git and Python noticeably slow on
a large project.

```bash
mkdir -p ~/ml-projects
cd ~/ml-projects
code .
```

That last command opens the folder in VS Code, still connected to Ubuntu. When
you do need something from Windows, it is under `/mnt/c/Users/YourName/`.

Finally, install two extensions in this WSL window: **Python** and **Jupyter**,
both from Microsoft. Extensions install per side, so having them on Windows is
not enough.

### Shortcuts worth knowing

| Keys | What it does |
|---|---|
| ++ctrl+shift+p++ | Command palette. Everything VS Code can do is in here |
| ++ctrl+j++ | Show or hide the terminal |
| ++ctrl+shift+e++ | File explorer |
| ++ctrl+b++ | Show or hide the sidebar |
| ++ctrl+p++ | Jump to a file by typing part of its name |

## Check yourself

1. Open VS Code and confirm the bottom left corner says WSL: Ubuntu.
2. In the terminal, run `pwd`. You should see `/home/yourname/ml-projects`, not
   a `C:` path.
3. Create `hello.py` with `print("Linux is running my code")` and run it with
   `python3 hello.py`.
4. Activate `ml-env` and run `python3 -c "import pandas; print(pandas.__version__)"`.
5. Run `git --version` to confirm Git is available on this side too.

If step 2 shows a `/mnt/c` path, you opened a Windows folder. Close the window
and start again from `~/ml-projects`.

## Next

You have an editor, a Linux environment, and somewhere to run code. Time to
write some.

[Python warm up](python-warm-up.md){ .h-button }

Official guide if you want more:
[VS Code with WSL](https://code.visualstudio.com/docs/remote/wsl-tutorial).

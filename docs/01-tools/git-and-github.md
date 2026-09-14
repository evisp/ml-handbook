# Git and GitHub

<span class="badge badge--time">45 min</span>
<span class="badge badge--level">Foundations</span>
<span class="badge">Needs: a terminal, a GitHub account</span>

Git saves snapshots of your work. GitHub stores those snapshots online so you
do not lose them and other people can see them. In this tutorial you set both
up, then push your first project to GitHub.

## Why this matters

You will write a lot of code in the next nine months. Some of it will work and
then stop working, and you will want the version from yesterday back. Git gives
you that.

There is a second reason. When you apply for a job, your GitHub profile is the
first thing a hiring manager opens. Every project you finish here should end up
there. Start building that history on day one.

## How Git works

Four places hold your work. Three of them are on your computer, one is on
GitHub. Each command below moves your work from one place to the next.

<svg viewBox="0 0 680 172" role="img" aria-labelledby="git-flow-title git-flow-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="git-flow-title">How work moves through Git</title>
<desc id="git-flow-desc">Your folder, then the staging area using git add, then the local repository using git commit, then GitHub using git push. The git pull command brings work back from GitHub to your folder.</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker></defs>
<rect x="3" y="40" width="146" height="64" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="76" y="68" text-anchor="middle" font-size="13.5" font-weight="700" fill="var(--md-default-fg-color)">Your folder</text>
<text x="76" y="87" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">files you edit</text>
<rect x="179" y="40" width="146" height="64" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="252" y="68" text-anchor="middle" font-size="13.5" font-weight="700" fill="var(--md-default-fg-color)">Staging area</text>
<text x="252" y="87" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">picked to save</text>
<rect x="355" y="40" width="146" height="64" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="428" y="68" text-anchor="middle" font-size="13.5" font-weight="700" fill="var(--md-default-fg-color)">Local history</text>
<text x="428" y="87" text-anchor="middle" font-size="11.5" fill="var(--h-graphite)">saved on your machine</text>
<rect x="531" y="40" width="146" height="64" rx="10" fill="var(--h-cherry)"/>
<text x="604" y="68" text-anchor="middle" font-size="13.5" font-weight="700" fill="#ffffff">GitHub</text>
<text x="604" y="87" text-anchor="middle" font-size="11.5" fill="#ffffff" opacity="0.85">online, shareable</text>
<line x1="153" y1="72" x2="173" y2="72" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#arrow)"/>
<line x1="329" y1="72" x2="349" y2="72" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#arrow)"/>
<line x1="505" y1="72" x2="525" y2="72" stroke="var(--h-steel)" stroke-width="2" marker-end="url(#arrow)"/>
<text x="163" y="28" text-anchor="middle" font-size="12" font-weight="600" fill="var(--h-cherry)">git add</text>
<text x="339" y="28" text-anchor="middle" font-size="12" font-weight="600" fill="var(--h-cherry)">git commit</text>
<text x="515" y="28" text-anchor="middle" font-size="12" font-weight="600" fill="var(--h-cherry)">git push</text>
<polyline points="604,110 604,134 76,134 76,112" fill="none" stroke="var(--h-steel)" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#arrow)"/>
<text x="340" y="153" text-anchor="middle" font-size="12" font-weight="600" fill="var(--h-graphite)">git pull</text>
</svg>

Keep this picture in your head. Most Git confusion comes from not knowing which
of the four boxes your work is sitting in right now.

## Step 1: Tell Git who you are

Git writes your name on every snapshot you save. Set it once and you never
think about it again.

```bash
git config --global user.name "Your Full Name"
git config --global user.email "your-email@example.com"
```

Use the same email address as your GitHub account. If the two do not match,
your commits show up on GitHub without your name attached to them.

Nothing prints when this works. Check it like this:

```bash
git config --global --list
```

## Step 2: Let your computer talk to GitHub

GitHub stopped accepting passwords in the terminal. You need a token instead.
There are two ways to set this up. The first is easier, so try it first.

=== "GitHub CLI (easier)"

    The `gh` tool does the whole login for you and stores the token safely.

    ```bash
    sudo apt update && sudo apt install gh    # Ubuntu or WSL
    gh auth login
    ```

    Answer the questions like this: GitHub.com, HTTPS, yes authenticate Git
    with your GitHub credentials, login with a web browser. Copy the one time
    code it shows you, press ++enter++, and finish in the browser that opens.

    Check it worked:

    ```bash
    gh auth status
    ```

=== "Personal access token"

    Use this if `gh` will not install on your machine.

    1. On GitHub, open **Settings**, then **Developer settings**, then
       **Personal access tokens**, then **Fine-grained tokens**.
    2. Click **Generate new token**. Give it a name and an expiry of 90 days.
    3. Under **Repository access**, choose **All repositories**. Under
       **Permissions**, set **Contents** to **Read and write**.
    4. Click generate, then copy the token. GitHub shows it once and never
       again, so paste it somewhere safe right now.

    Then tell Git to remember it:

    ```bash
    git config --global credential.helper store
    ```

    The first time you push, Git asks for a username and password. Type your
    GitHub username, then paste the token as the password. Git saves it and
    stops asking.

!!! warning "Do not put your token in a URL"

    You will find advice online that says to clone like this:
    `git clone https://username:token@github.com/...`. It works, and it is a
    bad habit. Git writes that URL into a file inside your project, and your
    shell saves it in your command history. One screenshot in a chat channel
    and someone else has full access to your account. Use one of the two
    methods above instead.

## Step 3: Make the repository on GitHub first

A repository, or repo, is one project folder that Git is tracking. It is easier
to create it on GitHub and copy it down than to create it locally and connect
it afterwards.

1. On GitHub, click the green **New** button.
2. Name it `my-ml-project`.
3. Leave it public.
4. Leave **Add a README file** unchecked.
5. Click **Create repository**.

You now have an empty repo and GitHub shows you its address.

## Step 4: Copy it to your computer

Cloning downloads the repo and connects your local folder to the GitHub one.

```bash
git clone https://github.com/your-username/my-ml-project.git
cd my-ml-project
```

You should see something like this:

```
Cloning into 'my-ml-project'...
warning: You appear to have cloned an empty repository.
```

The warning is fine. The repo is empty because you have not put anything in it
yet.

## Step 5: The four commands you will use every day

Make a file, check what changed, save it, send it to GitHub. That is the whole
loop, and you will run it hundreds of times.

```bash
echo "# My ML Project" > README.md
```

**See what Git noticed.** Run this whenever you are unsure what state you are
in. It is the most useful command in Git.

```bash
git status
```

Git reports `README.md` as untracked. It can see the file but is not saving it
yet.

**Pick what to save.**

```bash
git add README.md
```

**Save it.** The message says what you changed. Write it for the version of you
who reads it in three months.

```bash
git commit -m "Add project README"
```

**Send it to GitHub.**

```bash
git push origin main
```

Refresh the repo page in your browser. Your file is there.

!!! tip "Write commit messages in the present tense"

    Say "Add data loader", not "Added data loader". Git's own messages are
    written that way, so your history reads consistently. Also, if your message
    needs the word "and", you are probably saving two things at once. Make two
    commits.

## When something goes wrong

??? note "`fatal: not a git repository`"

    You are in the wrong folder. Run `pwd` to see where you are, then `cd` into
    the project folder you cloned.

??? note "`Authentication failed` when you push"

    Your token is wrong, expired, or you typed your GitHub password instead of
    the token. Generate a new token and run `gh auth login` again, or delete
    the saved credentials with `rm ~/.git-credentials` and push again.

??? note "`Updates were rejected because the remote contains work`"

    Someone changed the repo on GitHub after you last downloaded it. Run
    `git pull`, deal with any conflicts, then push again.

??? note "`Please tell me who you are`"

    You skipped Step 1. Set your name and email, then commit again.

## Check yourself

Do this now, before moving on. It takes five minutes and proves the whole setup
works.

1. Add a line to your `README.md` describing what you plan to build this year.
2. Run `git status` and read what it tells you.
3. Stage, commit, and push the change.
4. Open the repo on GitHub and find your commit under the commits link.
5. Post the repository link in the cohort channel.

If step 3 fails, the problem is almost always authentication. Go back to
Step 2.

## Quick reference

| Command | What it does | When you run it |
|---|---|---|
| `git status` | Shows what changed and what is staged | Any time you are unsure |
| `git add file.py` | Stages one file | After editing |
| `git add .` | Stages everything you changed | When all of it belongs together |
| `git commit -m "message"` | Saves a snapshot on your machine | After staging |
| `git push` | Sends your snapshots to GitHub | After committing |
| `git pull` | Brings down other people's work | Before you start working |
| `git log --oneline` | Lists your past snapshots | When looking for an old version |

## Next

You have version control working. Next you need somewhere to run code and see
results straight away.

[Jupyter notebook setup](notebook-setup.md){ .h-button }

Want to go deeper on Git later: the [Pro Git book](https://git-scm.com/book) is
free and thorough, and [Learn Git Branching](https://learngitbranching.js.org/)
lets you practise branching visually.

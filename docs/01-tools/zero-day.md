# Zero day

<span class="badge badge--time">20 min</span>
<span class="badge badge--level">Foundations</span>
<span class="badge">Needs: the email Holberton sent you</span>

Four accounts run your training. This tutorial gets all four working. Do it
before your first day, not during it.

## Why this matters

Every hour you spend chasing a login is an hour you are not learning. These
four platforms are also how your work gets seen and graded, so a broken account
in week one turns into a missing submission in week two.

## The four platforms

<svg viewBox="0 0 680 236" role="img" aria-labelledby="platforms-title platforms-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="platforms-title">The four platforms you need</title>
<desc id="platforms-desc">Intranet for course materials, Slack for communication, GitHub for your code, and Containers on Demand for your development machine. The intranet, Slack and Containers on Demand share one login. GitHub is a separate account.</desc>
<rect x="3" y="14" width="330" height="96" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="26" y="48" font-size="15" font-weight="700" fill="var(--md-default-fg-color)">Intranet</text>
<text x="26" y="70" font-size="12.5" fill="var(--h-graphite)">Projects, deadlines, your grades</text>
<text x="26" y="90" font-size="11.5" font-weight="600" fill="var(--h-cherry)">intranet.hbtn.io</text>
<rect x="347" y="14" width="330" height="96" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="370" y="48" font-size="15" font-weight="700" fill="var(--md-default-fg-color)">Slack</text>
<text x="370" y="70" font-size="12.5" fill="var(--h-graphite)">Questions, announcements, your cohort</text>
<text x="370" y="90" font-size="11.5" font-weight="600" fill="var(--h-cherry)">same login as intranet</text>
<rect x="3" y="126" width="330" height="96" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="26" y="160" font-size="15" font-weight="700" fill="var(--md-default-fg-color)">GitHub</text>
<text x="26" y="182" font-size="12.5" fill="var(--h-graphite)">Your code, and your public portfolio</text>
<text x="26" y="202" font-size="11.5" font-weight="600" fill="var(--h-cherry)">your own separate account</text>
<rect x="347" y="126" width="330" height="96" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="370" y="160" font-size="15" font-weight="700" fill="var(--md-default-fg-color)">Containers on Demand</text>
<text x="370" y="182" font-size="12.5" fill="var(--h-graphite)">A Linux machine in the browser</text>
<text x="370" y="202" font-size="11.5" font-weight="600" fill="var(--h-cherry)">cod.hbtn.io</text>
</svg>

Three of them use one password. GitHub is the exception, and it is the one that
follows you after graduation.

## Step 1: Set up the intranet

Go to [intranet.hbtn.io](https://intranet.hbtn.io/) and log in with the
credentials from your welcome email. Bookmark the page. You will open it every
day.

Then finish your profile:

1. Click your profile icon in the bottom left corner.
2. Fill in every field marked with an asterisk.
3. Add a photo. A plain head and shoulders shot is fine. This is the photo your
   instructors and future employers will see.
4. Save.

!!! warning "Do not change your password yet"

    Slack and Containers on Demand read your account from the intranet. Change
    the password now and you may lock yourself out of the other two before they
    have synced. Wait until all four platforms work.

## Step 2: Join Slack

Open the intranet and find the Slack link in the left navigation panel. Log in
with the same credentials.

You are added to your cohort channel automatically. Download the desktop or
phone app as well, because the browser version is easy to forget and that is
where schedule changes get announced.

## Step 3: Create your GitHub account

Already have one you use professionally? Use it and skip to Step 4.

Otherwise go to [github.com/signup](https://github.com/signup). Two choices
matter here:

**The email.** Use one you will still read in five years. Not a school address
that gets shut off when you graduate.

**The username.** This becomes part of your professional identity. It appears
in every project link you ever send to an employer. Something close to your
real name works well. Something you thought was funny at nineteen does not.

When you have it, go back to your intranet profile and add the username there.
The automated grading system uses it to find your work, so a typo means your
projects do not get marked.

## Step 4: Start your development machine

Containers on Demand gives everyone the same Linux machine with Python and the
ML libraries already installed. Nobody has to debug somebody else's laptop.

1. Go to [cod.hbtn.io](https://cod.hbtn.io/sign_in) and log in with your
   intranet credentials.
2. Set the **Region** dropdown at the top of the page to **Europe**. This
   affects how fast the connection feels from Tirana.
3. Find **ml_ubuntu_2204** in the container list and click **Spin Up
   Container**. Give it up to a minute.
4. Click **Actions**, then **VS Code**. A full editor and a Linux terminal open
   in your browser.

!!! warning "Containers stop after four hours"

    Your container shuts down four hours after you start it, and anything you
    have not pushed to GitHub is gone. You can add more time while you work,
    from the same Actions menu. Two habits protect you: commit often, and never
    leave your only copy of something inside a container.

## Check yourself

Five minutes, and it proves all four accounts work.

1. Log in to the intranet and confirm your profile shows your photo and your
   GitHub username.
2. Post a short hello in your cohort channel on Slack.
3. Open your GitHub profile page and check the username reads the way you want
   an employer to read it.
4. Spin up a container, open the terminal, and run `python3 --version`. You
   should see Python 3.10 or higher.

If any of the four fails, ask in Slack now. This is exactly what the channel is
for.

## Quick reference

| Platform | Where | What it is for | Login |
|---|---|---|---|
| Intranet | intranet.hbtn.io | Projects, deadlines, grades | The one from your email |
| Slack | Link inside the intranet | Questions and announcements | Same as intranet |
| GitHub | github.com | Your code and portfolio | Your own account |
| Containers on Demand | cod.hbtn.io | Linux machine for coding | Same as intranet |

## Next

Your accounts work. Now set up the tool that saves your work.

[Git and GitHub](git-and-github.md){ .h-button }

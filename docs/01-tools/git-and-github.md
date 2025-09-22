# Git and GitHub Setup and Essentials

This tutorial shows you how to set up and use Git and GitHub for effective version control and collaboration in machine learning projects.

***

## What Git Does: Key Features at a Glance

Git helps you track and manage changes in your project files, making teamwork and project history easy to handle. Here are the main features:

- **Version Tracking:** Saves snapshots of your project so you can review or go back to any earlier version anytime.  
- **Branching and Merging:** Work on new features separately in branches and merge them when ready, allowing multiple people to collaborate smoothly.  
- **Distributed System:** You have a full copy of the project history locally, so you can work offline and sync changes later.  
- **Staging Area:** Choose which changes to include in the next save (commit), giving control over what to record.  
- **History Log:** Track who changed what and when, helping understand project evolution and for troubleshooting.

***

## Quick Reference: Common Git Commands

| Command                      | What It Does                                      |
|-----------------------------|-------------------------------------------------|
| `git init`                  | Create a new Git repository in a folder.        |
| `git clone <repo>`          | Download a remote repository to your computer.  |
| `git status`                | See what files have changed or need staging.    |
| `git add <file>`            | Mark files to include in the next commit.       |
| `git commit -m "message"`  | Save staged changes with a clear message.        |
| `git remote add origin <url>` | Link your local repo to a GitHub repository.  |
| `git push`                  | Send your commits to GitHub.                     |
| `git pull`                  | Get and combine changes from GitHub.             |

***

## 1. Set Your Identity

Git uses your name and email to link changes to you. Configure them once on your machine:

```bash
git config --global user.name "Your Full Name"
git config --global user.email "your-email@example.com"
```

Use the same details you use for your GitHub account for clarity.

***

## 2. Create a GitHub Personal Access Token (PAT)

GitHub now requires a personal access token (PAT) instead of a password for security.

- Log in to GitHub.  
- Go to **Settings → Developer settings → Personal access tokens → Tokens (classic)**.  
- Click **Generate new token**, select scopes including `repo`, and create it.  
- Copy the token immediately; you won’t see it again.  

Use this token instead of your password when pushing code.

***

## 3. Basic Git Commands and Setup

### Initialize a Repository

Create a new Git repository locally in your project folder:

```bash
git init
```

### Check What Changed

See what files are modified, staged, or new:

```bash
git status
```

### Stage Files

Add specific files or all changed files to be included in the next commit:

```bash
git add filename
# or add everything:
git add .
```

### Commit Changes

Save your staged changes with a descriptive message:

```bash
git commit -m "Describe the changes you made"
```

### Connect to GitHub Repository

Link your local Git repo to GitHub:

```bash
git remote add origin https://github.com/username/repository.git
```

Or authenticate directly using your personal token in the URL:

```bash
git remote remove origin  # if already exists
git remote add origin https://<TOKEN>@github.com/username/repository.git
```

*(Replace `<TOKEN>` with your actual GitHub token.)*

*Note:* Avoid exposing tokens in shared scripts or public places for security.

### Push Your Work

Send your commits to GitHub’s main branch:

```bash
git push -u origin main
```

### Pull Latest Changes

Update your local copy with changes from GitHub:

```bash
git pull
```

### Clone a Repository

Copy an existing GitHub repository to your computer:

```bash
git clone https://github.com/username/repository.git
```

***

## 4. Great Tutorials to Learn More

- [Git and GitHub Tutorial for Beginners - freeCodeCamp (YouTube)](https://www.youtube.com/watch?v=tRZGeaHPoaw)  
- [An Intro to Git and GitHub for Beginners (HubSpot Blog)](https://product.hubspot.com/blog/git-and-github-tutorial-for-beginners)  
- [Set up Git - Official GitHub Docs](https://docs.github.com/en/get-started/git-basics/set-up-git)  

***

Mastering Git and GitHub is key to managing your code and collaborating effectively throughout your machine learning journey.
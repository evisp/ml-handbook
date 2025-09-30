# Git and GitHub Setup and Essentials

This tutorial shows you how to set up and use Git and GitHub for effective version control and collaboration in machine learning projects. You'll learn essential commands, authentication setup, and workflows that form the foundation of professional ML development. 

**Estimated time:** 45 minutes

## Why This Matters

**Problem statement:** Managing code changes, collaborating with teammates, and maintaining project history becomes impossible without proper version control as ML projects grow in complexity.

**Practical benefits:** Git and GitHub enable you to track experiments, collaborate safely with teammates, maintain reproducible ML workflows, and recover from mistakes without losing work. These skills are essential for any ML role where you'll work on shared codebases and need to manage model iterations.

**Professional context:** Every ML team uses Git for code management, GitHub for collaboration, and version control for tracking model experiments and dataset changes. Mastering these tools early accelerates your integration into professional ML workflows.

![Git Workflow](https://i.imgur.com/1P7SAo0.png)


## Prerequisites & Learning Objectives

**Required knowledge:**
- Basic command line navigation
- Understanding of files and folders
- GitHub account (create at github.com)

**Required tools:**
- Git installed on your system
- Command line access (Terminal/Command Prompt)
- Text editor or IDE

**Learning outcomes:**
- Configure Git with your identity and GitHub authentication
- Create, stage, commit, and push changes to repositories
- Clone existing projects and collaborate through pull/push workflows
- Troubleshoot common Git authentication and workflow issues

**High-level approach:** You'll configure Git locally, set up secure GitHub authentication, practice core workflow commands, and connect local repositories to GitHub for collaboration.

## Step-by-Step Instructions

### Step 1: Configure Your Git Identity

Git needs to know who you are to properly track changes and contributions.

```bash
git config --global user.name "Your Full Name"
git config --global user.email "your-email@example.com"
```

**Why this matters:** These details appear in your commit history and help teammates identify who made changes. Use the same email associated with your GitHub account for consistency.

**Expected output:** No output means success. Verify with:
```bash
git config --global --list
```

### Step 2: Set Up GitHub Authentication

GitHub requires secure authentication through Personal Access Tokens (PATs) instead of passwords.

**Create your PAT:**
- Log in to GitHub
- Navigate to **Settings → Developer settings → Personal access tokens → Tokens (classic)**
- Click **Generate new token (classic)**
- Select scopes: `repo`, `workflow`, `write:packages`
- Set expiration (no expiration, or 90 days)
- Copy the token immediately
- See [short video tutorial](https://www.youtube.com/watch?v=0C-B6bFuQYU)

**Store your token securely:** Save it in a password manager or secure note - you won't see it again.

**Expected outcome:** You have a PAT that starts with `ghp_` and is 40+ characters long.

### Step 3: Create Your Repository on GitHub First

The most streamlined approach is to **create your repository directly on GitHub, then clone it to your local machine**. This eliminates potential conflicts and simplifies the authentication process.

**Create a new repository on GitHub:**
- Navigate to github.com and click the green "New" button
- Name your repository (e.g., `my-ml-project`)
- Keep it public for learning purposes
- **Important:** Leave "Initialize this repository with a README" **unchecked**
- Click "Create repository"

**Why this approach works better:** Creating an empty repository first gives you a clean foundation and GitHub generates the exact clone URL you need, eliminating configuration errors.

### Step 4: Clone Your Repository Locally

Since you already have your Personal Access Token, you can clone the repository using the embedded authentication method:

```bash
git clone https://username:token@github.com/username/repository-name.git
```

**Make sure to replace `username`, `token` and `repository-name` with your actual data. 

**Example:**
```bash
git clone https://your-username:ghp_your_token_here@github.com/your-username/my-ml-project.git
cd my-ml-project
```

**Why embed the token:** This method stores your credentials temporarily and eliminates repeated authentication prompts during your session.

**Expected output:**
```
Cloning into 'my-ml-project'...
remote: Enumerating objects: 3, done.
remote: Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
Receiving objects: 100% (3/3), done.
```

### Step 5: Master the Essential Git Workflow

Now that your repository is connected, follow the fundamental pattern: **create/modify → stage → commit → push**.

**Create your first file:**
```bash
echo "# My ML Project" > README.md # this creates a file with heading "My ML Project"
echo "This repository contains my machine learning experiments." >> README.md
```

**Check what's changed:**
```bash
git status
```

**Stage your changes:**
```bash
git add README.md
```

**Commit with a descriptive message:**
```bash
git commit -m "Add project README with initial description"
```

**Push to GitHub:**
```bash
git push origin main
```

**Why this workflow matters:** This four-step process (modify → add → commit → push) forms the essential of version control. Each commit creates a checkpoint you can return to, and pushing synchronizes your work with GitHub.

**Expected output after pushing:**
```
Enumerating objects: 3, done.
Counting objects: 100% (3/3), done.
Writing objects: 100% (3/3), 245 bytes | 245.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0)
To https://github.com/username/my-ml-project.git
 * [new branch]      main -> main
```

**Pro tip:** Always use present-tense commit messages ("Add feature" not "Added feature") to maintain consistency with Git's own messaging style.

### Quick reference for daily use

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `git status` | Check what changed | Before staging files |
| `git add .` | Stage all changes | When ready to commit everything |
| `git commit -m "message"` | Save changes locally | After staging files |
| `git push` | Upload to GitHub | After committing changes |
| `git pull` | Download latest changes | Before starting new work |

## Summary & Next Steps

**Key accomplishments:** You've configured Git with your identity, set up secure GitHub authentication, mastered the core workflow (stage → commit → push), and can now collaborate on repositories professionally.

**Best practices for ML development:**
- **Commit frequently** with descriptive messages to track experiment iterations
- **Pull before pushing** to avoid conflicts when collaborating  
- **Use `.gitignore`** to exclude large model files and sensitive data
- **Branch for experiments** to keep main branch stable during model development


**External resources for deeper learning:**

- [Pro Git Book](https://git-scm.com/book) - comprehensive Git reference
- [GitHub Flow](https://guides.github.com/introduction/flow/) - branching strategies for teams  
- [Git Branching Interactive Tutorial](https://learngitbranching.js.org/) - visual practice environment

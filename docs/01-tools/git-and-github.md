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
- Set expiration (90 days recommended for learning)
- Copy the token immediately
- See [short video tutorial](https://www.youtube.com/watch?v=0C-B6bFuQYU)

**Store your token securely:** Save it in a password manager or secure note - you won't see it again.

**Expected outcome:** You have a PAT that starts with `ghp_` and is 40+ characters long.

### Step 3: Initialize and Configure Your First Repository

Create a new project folder and initialize Git tracking:

```bash
mkdir my-ml-project
cd my-ml-project
git init
```

**Why this works:** `git init` creates a hidden `.git` folder that tracks all changes in your project.

**Expected output:**
```
Initialized empty Git repository in /path/to/my-ml-project/.git/
```

### Step 4: Master the Core Git Workflow

The fundamental Git workflow follows this pattern: **modify → stage → commit → push**.

![Git Workflow](https://miro.medium.com/v2/1*W1LPtxxrJ0J1cq_Pv_OWbQ.png)


**Check repository status:**
```bash
git status
```

**Create a test file and stage it:**
```bash
echo "# My ML Project" > README.md
git add README.md
```

**Commit your changes:**
```bash
git commit -m "Add initial README file"
```

**Why descriptive messages matter:** Clear commit messages help you and teammates understand project evolution. Use present tense ("Add feature" not "Added feature").

**Expected output after commit:**
```
[main (root-commit) abc1234] Add initial README file
 1 file changed, 1 insertion(+)
 create mode 100644 README.md
```

### Step 5: Connect to GitHub and Push Changes

**Create a repository on GitHub:**
- Go to github.com, click "New repository"
- Name it `my-ml-project`
- Leave "Initialize with README" unchecked
- Click "Create repository"

**Link your local repository to GitHub:**
```bash
git remote add origin https://github.com/YOUR_USERNAME/my-ml-project.git
git branch -M main
git push -u origin main
```

**Authenticate when prompted:** Use your GitHub username and your PAT as the password.

**Troubleshooting authentication:** If push fails with authentication errors:
```bash
# Remove existing remote
git remote remove origin
# Add remote with token in URL (for persistent authentication)
git remote add origin https://YOUR_PAT@github.com/YOUR_USERNAME/my-ml-project.git
git push -u origin main
```

**Expected outcome:** Your code appears on GitHub, and future `git push` commands work without additional setup.

### Step 6: Practice Collaboration Workflow

**Pull changes from GitHub:**
```bash
git pull
```

**Clone existing repositories:**
```bash
git clone https://github.com/username/repository.git
cd repository
```

**Quick reference for daily use:**

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

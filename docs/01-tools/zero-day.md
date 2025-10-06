# Zero Day: Essential Setup for Machine Learning Training

This tutorial guides you through setting up all accounts and tools required for your Machine Learning training at Holberton. You'll configure your learning platform access, communication channels, version control, and development environment to ensure you're ready for day one of training.

**Estimated time:** 20 minutes

## Why This Matters

Starting ML training without proper account setup and tool configuration leads to lost time troubleshooting access issues, missed communications, and inability to submit assignments when you should be focusing on learning.

![Zero day](https://i.imgur.com/4VGN8Y2.png)


## Step-by-Step Instructions

### Step 1: Access and Configure Your Holberton Intranet

The intranet serves as your central hub for all course materials, project specifications, and progress tracking.

**Access the platform:**
- Navigate to [https://intranet.hbtn.io/](https://intranet.hbtn.io/)
- Log in with the credentials provided to you by email
- Bookmark this page for daily access

**Why this matters:** The intranet contains all project requirements, learning resources, and submission deadlines. It's your single source of truth throughout the program.

**Complete your profile:**
- Click on your profile icon (bottom left corner)
- **Critical step:** Fill in all mandatory fields marked with asterisks (*)
- Add a (professional) profile photo
- Save your changes
- **At this phase, do not yet change the password**.




### Step 2: Connect to Slack for Team Communication

Slack serves as your real-time communication channel with instructors, mentors, and fellow students throughout the program.

**Access Slack:**
- Locate the Slack icon or link within the intranet (left side panel on the navigation menu)
- Click to launch Slack
- **Important:** Use the **same credentials** as your intranet login
- Alternatively, download the Slack desktop app for better notifications


**Join your cohort channel:**
- You'll automatically be added to your cohort's group channel

### Step 3: Create Your GitHub Account

GitHub hosts your code repositories and integrates with the platform's automated grading system.

> If you already have a GitHub account you may use it. 

Alternatively,

**Create your account:**
- Navigate to [https://github.com/signup](https://github.com/signup)
- Enter your email address (use a professional email you'll access long-term)
- Create a strong password
- Choose a professional username (avoid numbers or special characters if possible)

**Why username matters:** Your GitHub username becomes part of your professional identity. Choose something you'd be comfortable sharing with future employers, as your ML projects will remain visible in your portfolio.

> **Next action:** Remember to add this exact username to your intranet profile as described in Step 1.


### Step 4: Access Your Cloud Development Environment

Containers on Demand (COD) provides pre-configured Linux machines with all necessary ML libraries installed, eliminating local setup complexity.

**Access the platform:**
- Navigate to [https://cod.hbtn.io/sign_in](https://cod.hbtn.io/sign_in)
- Log in with your **same intranet credentials**
- Wait for the dashboard to load

**Why cloud environments matter:** COD ensures everyone works in identical environments with consistent library versions, eliminating "it works on my machine" problems common in ML development.

**Configure your container settings:**

**Step 4.1: Select your region**
- Locate the "Region" dropdown at the top of the page
- **Important:** Select **Europe** for optimal performance and compliance
- This choice affects connection speed and data residency

**Step 4.2: Choose your container**
- Scroll through the container list
- Find and select **ml_ubuntu_2204**
- Click `Spin Up Container`
- Wait 30-60 seconds for the container to initialize


**Step 4.3: Access your development environment**
- Click "`Actions` and select `VS Code` to launch the web-based VS Code interface
- The interface loads with a Linux terminal and file explorer

**Why this container:** `ml_ubuntu_2204` comes pre-installed with Python, NumPy, pandas, scikit-learn, TensorFlow, PyTorch, and other essential ML libraries on Ubuntu 22.04 LTS.

> Important info: the container on demand expands after 4 hours. You need to repeat this process any time you work witht the platform. You can add *more time* as you are working.

### Quick Reference for Daily Workflow

| Platform | URL | Purpose | Credentials |
|----------|-----|---------|-------------|
| Intranet | intranet.hbtn.io | Course materials, projects, progress | Primary account |
| Slack | Via intranet link | Communication, support | Same as intranet |
| GitHub | github.com | Code hosting, version control | Separate account |
| COD | cod.hbtn.io | Development environment | Same as intranet |

## Summary & Next Steps

**Key accomplishments:** You've configured your Holberton intranet profile with GitHub integration, connected to Slack for team communication, created a professional GitHub account, and launched your pre-configured ML development environment with VS Code customization.

**Next tutorial:** Complete the [Git and GitHub](./git-and-github.md) tutorial to finish setting up version control and learn the essential workflow for submitting projects.
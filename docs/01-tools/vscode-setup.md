# Visual Studio Code Setup with WSL for Machine Learning

This tutorial shows you how to install and configure Visual Studio Code with Windows Subsystem for Linux (WSL) to create a professional ML development environment. You'll learn to combine Windows usability with Linux power for seamless machine learning workflows.

**Estimated time:** 35 minutes

## Why This Matters

**Problem statement:** Machine learning development often requires Linux tools and libraries, but many developers use Windows machines. Traditional solutions like dual-boot or virtual machines create workflow friction and performance overhead.

**Practical benefits:** VS Code with WSL provides the best of both worlds - Windows productivity tools with native Linux performance for ML development. You get access to Linux package managers, Python environments, and ML frameworks while maintaining familiar Windows workflows and file system integration.

**Professional context:** Most ML production environments run on Linux, making WSL essential for developing code that matches deployment targets. VS Code with remote development is the industry standard for teams working across different operating systems and cloud environments.

![VS Code](https://i.imgur.com/p0HjIIW.png)

## Prerequisites & Learning Objectives

**Required knowledge:**
- Basic Windows navigation and file management
- Understanding of command line concepts
- Familiarity with text editors or IDEs

**Learning outcomes:**
- Install and configure VS Code with essential ML extensions
- Set up WSL with Ubuntu for Linux development environment
- Connect VS Code to WSL for seamless remote development
- Navigate between Windows and Linux file systems efficiently
- Configure Python environments and run ML code in integrated terminals

**High-level approach:** You'll install VS Code first, then enable WSL with Ubuntu, install the Remote-WSL extension, and establish an integrated development workflow that combines Windows interface with Linux backend.

## Step-by-Step Instructions

### Step 1: Install Visual Studio Code

**Download and install VS Code:**
- Visit [https://code.visualstudio.com/](https://code.visualstudio.com/)
- Download the Windows installer (64-bit recommended)
- Run the installer with these important settings:
  - **Add "Open with Code" action to Windows Explorer file context menu**
  - **Add "Open with Code" action to Windows Explorer directory context menu**
  - **Add to PATH** (enables command line access)

**Why these settings matter:** Adding to PATH allows you to open VS Code from any terminal, while context menu integration provides quick access to editing files and folders directly from Windows Explorer.

**Expected outcome:** VS Code launches successfully and you can access it from Start menu, desktop shortcut, or by typing `code` in any command prompt.

### Step 2: Enable and Install WSL with Ubuntu

**Install WSL using PowerShell:**
```powershell
# Open PowerShell as Administrator
# Right-click Start button → "Terminal (Admin)" or "PowerShell (Admin)"
wsl --install
```

**What this command accomplishes:**
- Enables WSL and Virtual Machine Platform features
- Downloads and installs the Linux kernel
- Sets WSL 2 as default (better performance)
- Installs Ubuntu Linux distribution automatically[2]

**Complete the installation:**
- Restart your computer when prompted
- After restart, Ubuntu setup will launch automatically
- Create a Linux username and password (can be different from Windows)
- **Important:** Remember this password - you'll need it for `sudo` commands

**Verify WSL installation:**
```bash
# In any command prompt or PowerShell
wsl --list --verbose
```

**Expected output:**
```
  NAME      STATE           VERSION
* Ubuntu    Running         2
```

### Step 3: Install WSL Extension and Connect VS Code

**Install the Remote-WSL extension:**
- Open VS Code
- Press `Ctrl + Shift + X` to open Extensions
- Search for "Remote - WSL" by Microsoft
- Click **Install**

**Connect to WSL:**
- Press `Ctrl + Shift + P` to open Command Palette
- Type and select **"Remote-WSL: New WSL Window"**
- Choose **Ubuntu** from the distribution list
- A new VS Code window opens connected to your Linux environment

**Verify the connection:**
- Look for **"WSL: Ubuntu"** in the bottom-left corner of VS Code
- Open integrated terminal with `Ctrl + ` (backtick)
- The terminal should show your Linux username and prompt

**Expected terminal prompt:**
```bash
username@DESKTOP-NAME:~$
```

### Step 4: Configure Your Linux Development Environment

**Update Ubuntu packages:**
```bash
# Run in the WSL-connected VS Code terminal
sudo apt update && sudo apt upgrade -y
```

**Install essential development tools:**
```bash
# Python and development essentials (if not already installed)
sudo apt install python3 python3-pip python3-venv git curl wget -y

# Verify installations
python3 --version
pip3 --version
git --version
```

**Install Python ML libraries:**
```bash
# Create a virtual environment for ML projects
python3 -m venv ~/ml-env
source ~/ml-env/bin/activate

# Install essential ML packages
pip install numpy pandas matplotlib seaborn scikit-learn jupyter notebook
```

**Why virtual environments are essential:** Virtual environments isolate project dependencies, preventing conflicts between different ML projects and ensuring reproducible development across team members.

**Expected outcome:** You can run Python ML code natively in Linux while editing comfortably in VS Code.

### Step 5: Master the Integrated Workflow

**File system navigation:**
```bash
# Access Windows files from WSL
cd /mnt/c/Users/YourUsername/Documents

# Create ML project in Linux home (recommended for performance)
cd ~
mkdir ml-projects
cd ml-projects
```

**Open projects efficiently:**
```bash
# From WSL terminal, open current directory in VS Code
code .

# Create and edit files directly
code my_ml_script.py
```

**Essential VS Code shortcuts for remote development:**
- `Ctrl + Shift + P`: Command Palette (most important!)
- `Ctrl + `: Toggle integrated terminal
- `Ctrl + Shift + E`: File Explorer
- `Ctrl + B`: Toggle sidebar
- `F1`: Alternative Command Palette access

**Python development workflow:**
```bash
# Activate your ML environment
source ~/ml-env/bin/activate

# Run Python scripts
python my_ml_script.py
```

### Step 6: Install Essential VS Code Extensions for ML

**Install Python development extensions:**
- Press `Ctrl + Shift + X` in your WSL-connected window
- Install these essential extensions:
  - **Python** (Microsoft) - Python language support
  - **Jupyter** (Microsoft) - Notebook support in VS Code

**File system best practices:**
- Store Linux-specific projects in `~/` (Linux home) for best performance
- Access Windows files when needed via `/mnt/c/Users/YourUsername/`
- Use WSL for running code, Windows for file management when convenient

## Summary & Next Steps

**Key accomplishments:** You've installed VS Code with WSL integration, created a Linux development environment with Python ML libraries, mastered file system navigation between Windows and Linux, and established a professional workflow for ML development.

**Best practices for ML development:**
- **Store ML projects in Linux home** (`~/ml-projects/`) for optimal performance and native tool access
- **Use virtual environments** for each project to maintain clean dependency management
- **Leverage integrated terminal** to avoid switching between VS Code and separate command windows
- **Keep sensitive data in WSL** to benefit from Linux security and permissions model

**For non-Windows users:**
- **macOS/Linux users**: Install VS Code directly and use local Python development - no WSL needed
- **All platforms**: Consider Remote-SSH extension for connecting to remote servers or cloud instances


**External resources for deeper learning:**
- [VS Code WSL Tutorial](https://code.visualstudio.com/docs/remote/wsl-tutorial) - official comprehensive guide[4]
- [WSL Best Practices](https://docs.microsoft.com/windows/wsl/setup/environment) - Microsoft's development environment guide[5]
- [Remote Development Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) - additional remote development capabilities[1]

**Practice exercises:**
- Create a sample ML project using scikit-learn in your WSL environment
- Practice navigating between Windows and Linux file systems
- Set up a Git repository and commit code from VS Code with WSL


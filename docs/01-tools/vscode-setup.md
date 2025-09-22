# Visual Studio Code (VS Code) Setup with WSL (Windows Subsystem for Linux)

This guide helps Windows users install VS Code and WSL to write and run machine learning code using Linux tools while editing in VS Code on Windows.

***

## Why Use VS Code with WSL?

WSL lets you run Linux commands and software on Windows easily. VS Code can connect to WSL so you can edit and run Linux files inside a powerful code editor. This is helpful because many data science and machine learning tools are designed for Linux.

***

## 1. Install VS Code

- Go to https://code.visualstudio.com/ and download the installer for Windows.  
- Run the installer and follow the setup steps. The default options are fine.  
- During installation, check “Add to PATH” to let you open VS Code from the terminal.  

***

## 2. Install WSL on Windows

- Open PowerShell as Administrator (right-click Start → Windows Terminal (Admin) or PowerShell (Admin)).  
- Enter this command to install WSL and the Ubuntu Linux distribution:

```powershell
wsl --install
```

- Restart your computer if asked.  
- After restarting, open PowerShell or Command Prompt and type:

```bash
wsl
```

- You will enter Linux and be asked to create a Linux username and password.  

***

## 3. Use VS Code with WSL

- Launch VS Code.  
- Press **Ctrl + Shift + P** to open the Command Palette.  
- Type **Remote-WSL: New Window** and select it.  
- A new VS Code window opens connected to Linux inside WSL.  
- Open folders and files stored in your Linux file system (e.g., /home/yourname).  
- When you open a terminal in VS Code (Ctrl + `), it runs Linux shell inside WSL.  

***

## 4. Run Python and ML Code in VS Code with WSL

- In the WSL-connected VS Code window, open Python scripts or Jupyter notebooks.  
- Use the integrated terminal to run commands, install packages, and test code just like in Linux.  
- Use VS Code’s editor, debugging, and file management features while working inside Linux.

***

## 5. Opening VS Code from WSL Terminal

- You can also start VS Code connected to WSL by running this in the WSL terminal:

```bash
code .
```

- This command opens the current folder in VS Code with WSL integration.

- If this doesn’t work, restart your terminal or ensure VS Code is added to your PATH.

***


## 6. Using VS Code without WSL

- On macOS, Linux, or Windows without WSL, just open your projects in VS Code and run Python directly using local Python interpreters.

***

## 7. Tips for Working Smoothly

- Save files often (Ctrl+S).  
- Use the terminal inside VS Code to avoid switching windows.  
- Become familiar with keyboard shortcuts like Command Palette (Ctrl+Shift+P).  
- Keep Linux files inside WSL folders (/home directory) for best performance.

***

## 8. Further Learning

- [Install and Use WSL - Microsoft Docs](https://learn.microsoft.com/en-us/windows/wsl/install)  
- [VS Code Remote Development with WSL](https://code.visualstudio.com/docs/remote/wsl)  
- [VS Code Terminal User Guide](https://code.visualstudio.com/docs/terminal/shell-integration)  

***

This setup gives you the ease of Windows with the power of Linux tools, all managed from VS Code for a strong machine learning development workflow.

[1](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-vscode)
[2](https://code.visualstudio.com/docs/remote/wsl)
[3](https://learn.microsoft.com/en-us/windows/wsl/setup/environment)
[4](https://dev.to/jennieji/a-quick-setup-for-front-end-development-with-vscode-in-windows-10-3pok)
[5](https://ajeet.dev/developing-in-wsl-using-visual-studio-code/)
[6](https://www.youtube.com/watch?v=CokQE8kxxHQ)
[7](https://code.visualstudio.com/docs/setup/setup-overview)
[8](https://code.visualstudio.com/docs/getstarted/getting-started)
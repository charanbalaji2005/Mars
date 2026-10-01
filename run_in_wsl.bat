@echo off
REM NeuroTune WSL2 Helper Script
REM Allows running neurotune commands directly from Windows cmd.exe or PowerShell.

if "%~1"=="" (
    echo Opening interactive NeuroTune shell in WSL2...
    wsl -d Ubuntu -e bash -c "cd '/mnt/c/Users/Charan Balaji/Downloads/neurotune/neurotune' && source /home/charan_balaji/neurotune-venv/bin/activate && exec bash"
) else (
    wsl -d Ubuntu -e bash -c "cd '/mnt/c/Users/Charan Balaji/Downloads/neurotune/neurotune' && source /home/charan_balaji/neurotune-venv/bin/activate && %*"
)

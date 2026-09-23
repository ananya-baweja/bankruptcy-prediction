@echo off
REM ===================================================================
REM  Bankruptcy Prediction Project - first commit helper
REM  Double-click this file, or run it from a Command Prompt.
REM  It initialises git, stages everything, checks that no data or PDFs
REM  are about to be committed, and only then makes the first commit.
REM  It does NOT push - you do that after creating the GitHub repo.
REM ===================================================================
setlocal
cd /d "%~dp0"
echo.
echo ==== Bankruptcy Prediction Project : first commit ====
echo Folder: %CD%
echo.

REM ---- 1. is git installed? --------------------------------------
where git >nul 2>nul
if errorlevel 1 (
  echo ERROR: git is not installed, or not on PATH.
  echo Install Git for Windows from https://git-scm.com/download/win and run this again.
  echo.
  pause
  exit /b 1
)

REM ---- 2. is the git identity set? -------------------------------
git config user.name >nul 2>nul
if errorlevel 1 (
  echo ERROR: git does not know who you are. Run these two commands, then try again:
  echo.
  echo     git config --global user.name "Your Name"
  echo     git config --global user.email "your-github-email@example.com"
  echo.
  pause
  exit /b 1
)

REM ---- 3. initialise the repository ------------------------------
if exist ".git" (
  echo This folder is already a git repository - skipping init.
) else (
  git init -b main
  if errorlevel 1 (
    echo ERROR: git init failed.
    pause
    exit /b 1
  )
)
echo.

REM ---- 4. stage everything ---------------------------------------
git add -A
echo.

REM ---- 5. safety check: nothing that looks like data or a PDF -----
echo ---- Checking for data files and PDFs in the staged list ----
git ls-files --cached | findstr /i /c:".pdf" /c:"data/raw/" /c:"data/interim/" /c:"data/processed/" /c:".env" /c:"__pycache__"
if errorlevel 1 (
  echo OK: no PDFs, data files or caches are staged.
) else (
  echo.
  echo STOP. The files listed above look like data, PDFs or caches.
  echo They must not go into git history. Check .gitignore, run:
  echo     git rm -r --cached ^<the offending path^>
  echo and run this script again.
  echo.
  pause
  exit /b 1
)
echo.

REM ---- 6. show what will be committed ----------------------------
echo ---- Files that will be committed ----
git ls-files --cached
echo.
git ls-files --cached > staged_files_list.txt
for /f %%A in ('git ls-files --cached ^| find /c /v ""') do echo Total files staged: %%A
echo (The same list has been saved to staged_files_list.txt)
echo.
echo Check the list above. You should see: src, tests, docs, configs,
echo notebooks, README.md, requirements.txt, environment.yml, pyproject.toml,
echo and only the *_TEMPLATE.csv files under data\manual.
echo.
echo Press any key to make the commit, or close this window to stop.
pause >nul

REM ---- 7. commit --------------------------------------------------
git commit -m "Phases 0-2: cohort building and document processing"
if errorlevel 1 (
  echo.
  echo The commit did not complete. Read the message above.
  pause
  exit /b 1
)
echo.
echo ==== Commit done. ====
echo.
echo Next: create the GitHub repository and push. See docs\05_github_and_collaboration.md
echo.
pause

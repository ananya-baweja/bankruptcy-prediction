@echo off
REM ===================================================================
REM  Bankruptcy Prediction Project - connect to GitHub and push
REM  Run this AFTER git_setup.bat has made the first commit, and AFTER
REM  you have created an EMPTY private repository on github.com.
REM
REM  This script asks you to type your GitHub username, so you never
REM  have to paste a URL with angle brackets into Command Prompt.
REM  (In cmd, "<" means "read from a file", which is why pasting
REM   https://github.com/<your-username>/... fails.)
REM ===================================================================
setlocal
cd /d "%~dp0"
echo.
echo ==== Bankruptcy Prediction Project : push to GitHub ====
echo Folder: %CD%
echo.

REM ---- 1. is this a git repository? -------------------------------
if not exist ".git" (
  echo ERROR: this folder is not a git repository yet.
  echo Run git_setup.bat first to make the first commit.
  echo.
  pause
  exit /b 1
)

REM ---- 2. is there at least one commit? ---------------------------
git rev-parse --verify HEAD >nul 2>nul
if errorlevel 1 (
  echo ERROR: there are no commits yet.
  echo Run git_setup.bat first to make the first commit.
  echo.
  pause
  exit /b 1
)

echo Last commit:
git --no-pager log -1 --oneline
echo.

REM ---- 3. ask for the GitHub username -----------------------------
echo Before continuing, make sure you have created an EMPTY repository
echo on github.com named: bankruptcy-prediction
echo (New repository, Private, and do NOT tick README / .gitignore / license)
echo.
set "GHUSER="
set /p GHUSER=Type your GitHub username exactly as it appears on github.com, then press Enter:
if "%GHUSER%"=="" (
  echo.
  echo ERROR: no username entered. Nothing was changed.
  pause
  exit /b 1
)

set "GHURL=https://github.com/%GHUSER%/bankruptcy-prediction.git"
echo.
echo The repository URL will be:
echo     %GHURL%
echo.
echo If that is wrong, close this window now. Otherwise press any key.
pause >nul
echo.

REM ---- 4. replace any existing / broken remote --------------------
git remote get-url origin >nul 2>nul
if not errorlevel 1 (
  echo An "origin" remote already exists - replacing it.
  git remote remove origin
)
git remote add origin "%GHURL%"
if errorlevel 1 (
  echo ERROR: could not add the remote. Read the message above.
  pause
  exit /b 1
)

echo.
echo ---- Remotes now configured ----
git remote -v
echo.

REM ---- 5. push ----------------------------------------------------
echo Pushing "main" to GitHub. A browser window may open so you can sign in.
echo Sign in there. Do NOT type a personal access token into a chat or a document.
echo.
git push -u origin main
if errorlevel 1 (
  echo.
  echo ==== The push did not complete. ====
  echo.
  echo Common causes:
  echo   * "Repository not found"  - the repo does not exist on github.com yet,
  echo                               or the username above is spelt differently.
  echo                               Check the spelling on your GitHub profile page.
  echo   * "rejected / fetch first" - the repo was created WITH a README.
  echo                               Delete it and create an empty one, or run:
  echo                                   git pull --rebase origin main
  echo                               and then push again.
  echo   * Sign-in window closed    - just run this script again.
  echo.
  pause
  exit /b 1
)

echo.
echo ==== Pushed successfully. ====
echo.
echo Next steps on the repository page at %GHURL:~0,-4%
echo   1. Settings -^> Collaborators -^> Add people -^> add your teammate's
echo      GitHub username and send the invitation.
echo   2. Send him this clone URL:
echo          %GHURL%
echo      He must accept the emailed invitation before he can clone it.
echo.
pause

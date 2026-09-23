# Putting the project on GitHub and working as a team

One person (Ananya) does Part A and B once. Everyone else does Part C.

> **Placeholders in this document are written as `YOUR-USERNAME`, with no angle brackets.**
> Never type `<` or `>` into a Command Prompt. In Windows, `<` means "read input from a file",
> so a command containing `<your-username>` fails with *"The system cannot find the file
> specified"* and git never runs at all.

---

## Part A — First commit (on the laptop that holds the project)

Either double-click `git_setup.bat` in the project folder, or run the commands below from a
Command Prompt. The script does the same thing and refuses to continue if a PDF or data file
has been staged by mistake.

```bat
cd /d "V:\Projects\Bankruptcy Prediction Project"

git config --global user.name "Your Name"
git config --global user.email "your-github-email@example.com"

git init -b main
git add -A
git status
```

**Read the `git status` output before committing.** You should see:

- `src/`, `tests/`, `docs/`, `configs/`, `notebooks/`
- `README.md`, `requirements.txt`, `environment.yml`, `pyproject.toml`, `.gitignore`, `.gitattributes`
- under `data/`: only the four `*_TEMPLATE.csv` files, `data/manual/README.md` and the `.gitkeep` markers

You should **not** see any `.pdf`, anything under `data/raw`, `data/interim`, `data/processed`,
or any `__pycache__` folder. If you do, stop and fix `.gitignore` first: committing an annual
report or a scraped dataset is hard to undo, especially in a public repository.

Then:

```bat
git commit -m "Phases 0-2: cohort building and document processing"
```

---

## Part B — Create the repository and push

### Step 1: create an empty repository

1. Go to github.com → **New repository**.
2. Name: `bankruptcy-prediction`. Visibility: **Private**.
3. Do **not** tick "Add a README", "Add .gitignore" or "Choose a license" — the repository must be
   empty, otherwise the first push is rejected.
4. Create it.

### Step 2: connect and push

**Easiest — double-click `git_push.bat`.** It asks you to type your GitHub username, builds the
URL for you, replaces any broken remote, pushes, and explains any error it hits.

**By hand**, from a Command Prompt in the project folder. Replace `YOUR-USERNAME` with your real
GitHub username — type it in, do not keep the capital letters or add angle brackets:

```bat
git remote add origin https://github.com/YOUR-USERNAME/bankruptcy-prediction.git
git remote -v
git push -u origin main
```

`git remote -v` must print two lines showing your URL. If it prints nothing, the remote was not
added — check the command for stray `<` or `>` characters and run it again. If it shows the wrong
URL, run `git remote remove origin` first, then add it again.

When the browser window appears, sign in to GitHub. Git for Windows stores the credentials, so
later pushes will not ask again. Never type a personal access token into a chat or a document.

### If the push fails

| Message | Cause | Fix |
| --- | --- | --- |
| `The system cannot find the file specified` | The command contained `<` or `>` | Retype the URL with your real username and no angle brackets |
| `'origin' does not appear to be a git repository` | The remote was never added (usually the row above) | Run `git remote -v`; if empty, add the remote again |
| `Repository not found` | The repo does not exist on GitHub yet, or the username is spelt differently | Create it, or check the spelling on your GitHub profile page |
| `Updates were rejected / fetch first` | The repo was created **with** a README | `git pull --rebase origin main`, then push again |

### Add your teammates as collaborators

On the repository page: **Settings → Collaborators → Add people**, type each teammate's GitHub
username, and send the invitation. They receive an email and must accept it before they can clone
a private repository.

Then send them the clone URL (with your real username in place of `YOUR-USERNAME`):

```
https://github.com/YOUR-USERNAME/bankruptcy-prediction.git
```

**If a teammate cannot reach the private repository** (for example, an automated environment
without a GitHub login), the fallback is a zip of the code and docs. There is no data in the
repository, so a zip is a complete copy. The other option is to make the repository public, which
is acceptable here because it holds only code and documentation — but only after you have confirmed
the `git status` list is clean.

---

## Part C — Everyone else: clone and set up

Replace `OWNER` with the GitHub username that owns the repository.

```bat
cd /d "C:\wherever\you\keep\projects"
git clone https://github.com/OWNER/bankruptcy-prediction.git
cd bankruptcy-prediction

conda env create -f environment.yml
conda activate bpp
pytest
bpp demo
```

`data/` is intentionally almost empty in the repository. The real data lives in the shared Google
Drive folder. Point the tools at it with:

```bat
bpp --data-dir "G:\My Drive\BPP-data" status
```

---

## How we work day to day

| Rule | Reason |
| --- | --- |
| `git pull` before you start, `git push` when something works | Avoids painful merges |
| One branch per person or feature: `git checkout -b phase3-ratios` | Keeps `main` working |
| Merge into `main` through a pull request, with one teammate reviewing | Someone else sees every change |
| Never commit anything under `data/` except templates | Data belongs in Drive; the repo stays small and shareable |
| Never commit tokens, passwords or `.env` files | These cannot be fully removed from history later |
| Write what you changed in `docs/progress_log.md` | The log is what the report and the viva are built from |

### If something does get committed by mistake

Stop and tell the team before pushing. If it has not been pushed yet — replace `PATH` with the
offending file or folder:

```bat
git rm -r --cached PATH
git commit --amend -C HEAD
```

If it has already been pushed, the history has to be rewritten (`git filter-repo`) and everyone
must re-clone. This is why the check in Part A matters.

# Predicting Electric Vehicle Purchases 

---

## Repository Structure

```text
├── .github/workflows/     # CI/CD: Automated linting & notebook check
├── data/                  # Dataset files (train, test, sample submission)
├── model/                 # Model definitions, scripts, and checkpoints
├── notebook/              # Jupyter Notebooks for EDA and experiments
├── src/                   # Reusable data processing and utility modules
├── .gitattributes         # Line-ending normalization & notebook filters
├── .gitignore             # Ignores virtualenvs, caches, artifacts, checkpoints
├── .pre-commit-config.yaml# Git hooks (Ruff formatter, nbstripout, hygiene)
├── pyproject.toml         # Central tool configuration (Ruff) & project metadata
├── requirements.txt       # Core project dependencies
└── requirements-dev.txt   # Development, formatting, and linting tools
```

---

## Team Set-Up

### 1. Clone the repository
```bash
git clone https://github.com/3oFiz4/Predicting-Electric-Vehicle-Purchases.git
cd Predicting-Electric-Vehicle-Purchases
```

### 2. Create and activate a virtual environment
```bash
# Using python venv:
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements-dev.txt
```

### 4. Enable pre-commit hooks (Crucial!)
```bash
pre-commit install
```
> **What this does:** Whenever `git commit` executed, `pre-commit` will automatically:
> 1. Strip all execution outputs and cell execution counts from `.ipynb` files (`nbstripout`).
> 2. Format your Python scripts and notebook cells using **Ruff**.
> 3. Fix trailing whitespaces and normalize line endings.

---

## File info
### Responsibility
- **`nbstripout`**: Automatically strips cell outputs before committing, so only your actual code and markdown are tracked in Git.
- **`ruff`**: An ultra-fast Python linter and formatter that unifies style across `.py` and `.ipynb` files.
- **`src/` vs `notebook/` pattern**:
  - Keep notebooks clean and focused on **EDA, visualization, and storytelling**.
  - Extract reusable functions, data pipelines, and feature engineering logic into `src/` as pure Python modules (`.py`).
  - Import functions from `src/` into your notebooks instead of writing huge monolithic code blocks inside notebooks.

### Manual Commands
If you ever want to run checks manually:
```bash
# Check code style & linting:
ruff check .

# Automatically fix lint issues:
ruff check --fix .

# Auto-format all .py and .ipynb files:
ruff format .

# Manually strip outputs from all notebooks:
nbstripout notebook/*.ipynb
```

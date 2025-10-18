#!/usr/bin/env bash
# setup_project.sh
# Creates the Finance_Case_Studies project skeleton

set -e

ROOT="$(pwd)"

echo "Creating project structure under: $ROOT"

# Governance
mkdir -p governance

# Case 1A - Monthly
mkdir -p Case1A_MonthlyForecast/{data/{raw,processed},eda,features,models,deployment,tests,docs,notebooks}

# Case 1B - Quarterly
mkdir -p Case1B_QuarterlyForecast/{data/{raw,processed},eda,features,models,deployment,tests,docs,notebooks}

# Add placeholder files
for d in governance Case1A_MonthlyForecast Case1B_QuarterlyForecast; do
  [ -f "$d/README.md" ] || echo "# $d" > "$d/README.md"
done

# Root files
[ -f README.md ] || echo "# Finance_Case_Studies" > README.md
[ -f .gitignore ] || cat > .gitignore <<'GITIGNORE'
__pycache__/
.venv/
venv/
.env
data/
*.csv
*.parquet
.ipynb_checkpoints
GITIGNORE

chmod +x setup_project.sh

echo "Scaffold complete. Add your data and start working."

.PHONY: help install lint format test run clean

# Default target executed when no arguments are given to make.
help:
	@echo "Commands:"
	@echo "  install    : Install dependencies using requirements.txt"
	@echo "  format     : Format code using black"
	@echo "  lint       : Lint code using flake8 and check formatting with black"
	@echo "  test       : Run tests using pytest"
	@echo "  run        : Execute the demo notebook non-interactively"
	@echo "  clean      : Remove temporary files"
	@echo "  setup-hooks: Install pre-commit hooks"

# Variables
PYTHON = python
VENV_DIR = venv
PIP = $(VENV_DIR)/bin/pip
PYTEST = $(VENV_DIR)/bin/pytest
BLACK = $(VENV_DIR)/bin/black
FLAKE8 = $(VENV_DIR)/bin/flake8
PRECOMMIT = $(VENV_DIR)/bin/pre-commit
JUPYTER = $(VENV_DIR)/bin/jupyter
NOTEBOOK_FILE = notebooks/noise_analysis_demo.ipynb

# Check if virtual environment exists
VENV_CHECK = $(wildcard $(VENV_DIR)/bin/activate)

# Environment setup
$(VENV_DIR)/bin/activate: requirements.txt
	test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Virtual environment created and dependencies installed."
	@touch $(VENV_DIR)/bin/activate

install: $(VENV_DIR)/bin/activate
	@echo "Dependencies are up to date."

# Code Quality & Formatting
format: install
	$(BLACK) *.py src/ tests/ notebooks/
	@echo "Code formatting complete."

lint: install
	@echo "Running flake8 linter..."
	$(FLAKE8) src/ tests/ notebooks/ *.py
	@echo "Checking code formatting with black..."
	$(BLACK) --check src/ tests/ notebooks/ *.py
	@echo "Linting and format check complete."

# Testing
test: install
	@echo "Running tests with coverage..."
	$(PYTEST) tests/ --cov=notebooks --cov-report=term-missing
	@echo "Tests complete."

# Running the demo
run: install
	@echo "Executing notebook: $(NOTEBOOK_FILE)..."
	$(JUPYTER) nbconvert --to notebook --execute $(NOTEBOOK_FILE) --output $(NOTEBOOK_FILE) --ExecutePreprocessor.timeout=-1
	@echo "Notebook execution finished. Check $(NOTEBOOK_FILE) and the /figures directory."

# Pre-commit hooks
setup-hooks: install
	$(PRECOMMIT) install
	@echo "Pre-commit hooks installed."

# Cleaning
clean:
	rm -rf .pytest_cache/ .mypy_cache/ .coverage htmlcov/ build/ dist/ *.egg-info/ __pycache__/ */__pycache__/
	rm -rf $(VENV_DIR)
	rm -f figures/*.png
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	@echo "Cleaned up temporary files."

# Prevent make from deleting intermediate files
.PRECIOUS: $(VENV_DIR)/bin/activate
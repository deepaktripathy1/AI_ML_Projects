# Makefile for Model Training Project
# === Environment Setup ===
VENV_NAME := env
VENV_PATH := ./$(VENV_NAME)
PYTHON_VENV := $(VENV_PATH)/Scripts/python.exe
PIP_VENV := $(VENV_PATH)/Scripts/pip.exe
MLFLOW_PORT := 5000
MLFLOW_HOST := 127.0.0.1

# === Colors for output ===
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# === Default Target ===
.DEFAULT_GOAL := help

# === Check if virtual environment exists ===
.PHONY: check-venv
check-venv:
	@if [ -d "$(VENV_PATH)" ]; then \
		printf "$(GREEN)Virtual environment '$(VENV_NAME)' exists$(NC)"; \
	else \
		printf "$(RED)Virtual environment '$(VENV_NAME)' not found$(NC)"; \
		printf "$(YELLOW)Creating virtual environment...$(NC)"; \
		$(MAKE) create-venv; \
	fi

# === Create virtual environment ===
.PHONY: create-venv
create-venv:
	@printf "$(BLUE)Creating virtual environment: $(VENV_NAME)$(NC)"
	$(PYTHON) -m venv $(VENV_PATH)
	@printf "$(BLUE)Installing requirements...$(NC)"
	$(PIP_VENV) install --upgrade pip
	$(PIP_VENV) install -r requirements.txt
	@printf "$(GREEN)Virtual environment created and dependencies installed$(NC)"


# === Activate virtual environment (info) ===
.PHONY: activate-venv
activate-venv:
	@printf "$(YELLOW)To activate the virtual environment, run:$(NC)"
	@echo "source $(VENV_PATH)/Scripts/activate"

# === Install/Update requirements ===
.PHONY: install-requirements
install-requirements: check_venv
	@printf "$(BLUE)Installing/Updating requirements...$(NC)"
	$(PIP_VENV) install --upgrade pip
	$(PIP_VENV) install -r requirements.txt
	@printf "$(GREEN)Requirements installed$(NC)"

# === Check if MLflow server is running ===
.PHONY: check-mlflow
check-mlflow:
	@tasklist | grep -i "mlflow" > /dev/null && \
		{ \
			printf "$(GREEN)MLflow server is already running$(NC)"; \
			printf "$(BLUE)MLflow UI: http://$(MLFLOW_HOST):$(MLFLOW_PORT)$(NC)"; \
		} || \
		{ \
			printf "$(YELLOW)MLflow server is not running$(NC)"; \
		}

# === Start MLflow Tracking UI ===
.PHONY: start-mlflow
start-mlflow: check-venv
	@printf "\n$(BLUE)Starting MLflow server on port $(MLFLOW_PORT)...$(NC)"; \
	if tasklist | grep -i "mlflow" > /dev/null 2>&1; then \
		printf "$(YELLOW)MLflow server is already running$(NC)"; \
		printf "$(BLUE)MLflow UI: http://$(MLFLOW_HOST):$(MLFLOW_PORT)$(NC)"; \
	else \
		$(PYTHON_VENV) -m mlflow ui --host $(MLFLOW_HOST) --port $(MLFLOW_PORT) > mlflow.log 2>&1 & \
		printf "$(BLUE)MLflow server started in the background$(NC)"; \
		printf "$(BLUE)Logs: tail -f mlflow.log$(NC)"; \
		printf "$(BLUE)MLflow UI: http://$(MLFLOW_HOST):$(MLFLOW_PORT)$(NC)"; \
	fi

# === Stop MLflow Tracking UI ===
.PHONY: stop-mlflow
stop-mlflow:
	@printf "$(BLUE)Stopping MLflow server...$(NC)\n"
	@# Find process using port 5000
	@pid=$$(netstat -ano 2>/dev/null | grep "127.0.0.1:5000.*LISTENING" | awk '{print $$5}' | head -1); \
	if [ -n "$$pid" ] && [ "$$pid" != "0" ]; then \
		printf "$(BLUE)Found process using port 5000 (PID $$pid)$(NC)\n"; \
		if taskkill.exe //PID $$pid //F >/dev/null 2>&1; then \
			printf "$(GREEN)MLflow server (PID $$pid) stopped successfully$(NC)\n"; \
		else \
			printf "$(RED)Failed to stop MLflow server (PID $$pid)$(NC)\n"; \
		fi; \
	else \
		printf "$(YELLOW)No process found listening on port 5000$(NC)\n"; \
	fi
	@# Verify MLflow is stopped
	@sleep 1; \
	if netstat -an 2>/dev/null | grep "127.0.0.1:5000.*LISTENING" >/dev/null; then \
		printf "$(RED)MLflow server may still be running on port 5000$(NC)\n"; \
	else \
		printf "$(GREEN)MLflow server successfully stopped$(NC)\n"; \
	fi

# === Reproduce DVC pipeline ===
.PHONY: dvc-repro
dvc-repro: check-venv
	@printf "\n$(BLUE)Running DVC pipeline...$(NC)"
	$(PYTHON_VENV) -m dvc repro
	@printf "$(GREEN)DVC pipeline completed$(NC)"

# === Check DVC status ===
.PHONY: dvc-status
dvc-status: check-venv
	@printf "\n$(BLUE)Checking DVC pipeline status...$(NC)"
	$(PYTHON_VENV) -m dvc status

# === Show DVC metrics ===
.PHONY: dvc-metrics
dvc-metrics: check-venv
	@printf "\n$(BLUE)Showing DVC metrics...$(NC)"
	$(PYTHON_VENV) -m dvc metrics show

# === Clean DVC cache ===
.PHONY: dvc-clean
dvc-clean: check-venv
	@printf "\n$(BLUE)Cleaning DVC cache...$(NC)\n"
	$(PYTHON_VENV) -m dvc gc -w -f
	@printf "$(GREEN)DVC cache cleaned$(NC)\n"

# === Git Status ===
.PHONY: git-status
git-status:
	@printf "$(BLUE)Git status:$(NC)"
	cd../.. && git status

# === git-add ===
.PHONY: git-add
git-add:
	@printf "$(BLUE)Adding all changes to git...$(NC)"
	cd ../.. && git add Used_Car_Price_Prediction/ .gitignore README.md
	@printf "$(GREEN)Changes added to git$(NC)"

# === git-commit ===
.PHONY: git-commit
git-commit: git-add
	@if [ -z "$(MSG)" ]; then \
		printf "$(RED)Error: Please provide a commit message using MSG='your message'$(NC)"; \
		printf "$(YELLOW)Example: make git-commit MSG='Updated model training pipeline'$(NC)"; \
		exit 1; \
	fi
	@printf "$(BLUE)Committing changes...$(NC)"
	cd ../.. && git commit -m "$(MSG)"
	@printf "$(GREEN)Changes committed$(NC)"

# === git-push ===
.PHONY: git-push
git-push:
	@printf "$(BLUE)Pushing to remote repository...$(NC)"
	cd ../.. && git push
	@printf "$(GREEN)Changes pushed to remote$(NC)"

# === git auto-commit and push ===
.PHONY: git-auto
git-auto:
	@if [ -z "$(MSG)" ]; then \
		MSG="Auto commit: Pipeline execution $$(date '+%Y-%m-%d %H:%M:%S')"; \
	fi
	@printf "$(BLUE)Auto git commit and push...$(NC)"
	$(MAKE) git-commit MSG="$$MSG"
	$(MAKE) git-push

# Clean generated files
.PHONY: clean
clean:
	@printf "$(BLUE)Cleaning generated files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf *.egg-info
	@printf "$(GREEN)Cleanup completed$(NC)"

# Deep clean (includes models and cache)
.PHONY: clean-all
clean-all: clean dvc-clean stop-mlflow
	@printf "$(BLUE)Deep cleaning...$(NC)"
	rm -rf models/*
	rm -rf mlruns/*
	rm -rf plots/*
	rm -rf data/processed/*
	rm -rf outputs/*
	@printf "$(GREEN)Deep cleanup completed$(NC)"

# === Run entire pipeline ===
.PHONY: pipeline
pipeline: check-venv start-mlflow dvc-repro dvc-metrics
	@printf "$(GREEN)Full pipeline completed successfully$(NC)"
	@printf "$(BLUE)Check MLflow UI at: http://$(MLFLOW_HOST):$(MLFLOW_PORT)$(NC)"

# === Run everything (Target) ===
.PHONY: all
all: check-venv start-mlflow pipeline
	@printf "$(GREEN)All tasks completed successfully$(NC)"
	@printf "$(BLUE)MLflow UI: http://$(MLFLOW_HOST):$(MLFLOW_PORT)$(NC)"
	@printf "$(YELLOW)Note: Use 'make git-auto MSG=\"your message\"' for git operations$(NC)"

# === Development setup ===
.PHONY: setup
setup: create-venv install-requirements start-mlflow
	@printf "$(GREEN)Development environment setup completed$(NC)"
	@printf "$(BLUE)MLflow UI: http://$(MLFLOW_HOST):$(MLFLOW_PORT)$(NC)"
	@printf "$(YELLOW)Run 'make all' to execute the full pipeline$(NC)"

# === Show help message ===
.PHONY: help
help:
	@printf "$(BLUE)Used Car Price Prediction - ML Pipeline Makefile$(NC)"
	@echo ""
	@printf "$(YELLOW)Main Commands:$(NC)\n"
	@printf "  $(GREEN)make all$(NC)                      - Run full pipeline (env check + MLflow + DVC)\n"
	@printf "  $(GREEN)make setup$(NC)                    - Initial development environment setup\n"
	@printf "  $(GREEN)make pipeline$(NC)                 - Run DVC pipeline with MLflow tracking\n"
	@echo ""
	@printf "$(YELLOW)Environment:$(NC)\n"
	@printf "  $(GREEN)make check-venv$(NC)               - Check if virtual environment exists\n"
	@printf "  $(GREEN)make create-venv$(NC)              - Create virtual environment and install deps\n"
	@printf "  $(GREEN)make install-requirements$(NC)     - Install/update requirements\n"
	@printf "  $(GREEN)make activate-venv$(NC)            - Show activation command\n"
	@echo ""
	@printf "$(YELLOW)MLflow:$(NC)\n"
	@printf "  $(GREEN)make start-mlflow$(NC)             - Start MLflow server (port $(MLFLOW_PORT))\n"
	@printf "  $(GREEN)make stop-mlflow$(NC)              - Stop MLflow server\n"
	@printf "  $(GREEN)make check-mlflow$(NC)             - Check MLflow server status\n"
	@echo ""
	@printf "$(YELLOW)DVC Pipeline:$(NC)\n"
	@printf "  $(GREEN)make dvc-repro$(NC)                - Run DVC pipeline reproduction\n"
	@printf "  $(GREEN)make dvc-status$(NC)               - Check DVC pipeline status\n"
	@printf "  $(GREEN)make dvc-metrics$(NC)              - Show DVC metrics\n"
	@printf "  $(GREEN)make dvc-clean$(NC)                - Clean DVC cache\n"
	@echo ""
	@printf "$(YELLOW)Git Operations:$(NC)\n"
	@printf "  $(GREEN)make git-status$(NC)               - Show git status\n"
	@printf "  $(GREEN)make git-commit MSG='message'$(NC) - Commit with message\n"
	@printf "  $(GREEN)make git-push$(NC)                 - Push to remote\n"
	@printf "  $(GREEN)make git-auto$(NC)                 - Auto commit and push\n"
	@echo ""
	@printf "$(YELLOW)Cleanup:$(NC)\n"
	@printf "  $(GREEN)make clean$(NC)                    - Clean Python cache and logs\n"
	@printf "  $(GREEN)make clean-all$(NC)                - Deep clean (models, cache, MLflow)\n"
	@echo ""
	@printf "$(BLUE)Examples:$(NC)\n"
	@echo "  make git-commit MSG='Updated feature engineering'"
	@echo "  make git-auto MSG='Pipeline improvements'"

# Makefile for Four Rooms Reinforcement Learning Assignment

# Python interpreter
PYTHON = python3

# Default target
all: run_all

# Run all scenarios
run_all: run_scenario1 run_scenario2 run_scenario3

# Run individual scenarios
run_scenario1:
	$(PYTHON) Scenario1.py

run_scenario1_stochastic:
	$(PYTHON) Scenario1.py -stochastic

run_scenario2:
	$(PYTHON) Scenario2.py

run_scenario2_stochastic:
	$(PYTHON) Scenario2.py -stochastic

run_scenario3:
	$(PYTHON) Scenario3.py

run_scenario3_stochastic:
	$(PYTHON) Scenario3.py -stochastic

# Clean up generated files and directories
clean:
	rm -rf epsilon_paths/
	rm -rf softmax_paths/
	rm -f *.png
	rm -f *.pyc
	rm -rf __pycache__/

# Help command
help:
	@echo "Available commands:"
	@echo "  make run_all              - Run all scenarios"
	@echo "  make run_scenario1        - Run Scenario 1 (deterministic)"
	@echo "  make run_scenario1_stochastic - Run Scenario 1 (stochastic)"
	@echo "  make run_scenario2        - Run Scenario 2 (deterministic)"
	@echo "  make run_scenario2_stochastic - Run Scenario 2 (stochastic)"
	@echo "  make run_scenario3        - Run Scenario 3 (deterministic)"
	@echo "  make run_scenario3_stochastic - Run Scenario 3 (stochastic)"
	@echo "  make clean                - Remove generated files and directories"
	@echo "  make help                 - Show this help message"

.PHONY: all run_all run_scenario1 run_scenario1_stochastic run_scenario2 run_scenario2_stochastic run_scenario3 run_scenario3_stochastic clean help 
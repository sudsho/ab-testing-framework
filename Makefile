.PHONY: smoke test install clean

# Offline smoke: synthetic experiment data + core stats, no keys/downloads.
smoke:
	python scripts/smoke.py

# Full unit + integration suite (pymc3 test stays opt-in via RUN_PYMC_TESTS=1).
test:
	pytest -q

install:
	pip install -r requirements.txt

clean:
	rm -rf .pytest_cache __pycache__ src/__pycache__ tests/__pycache__

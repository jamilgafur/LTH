"""Thin entrypoint for adversarial analysis.

The implementation lives in modular files:
- adversarial_core.py
- adversarial_plotting.py
- adversarial_reporting.py
- adversarial_experiments.py
- adversarial_compute_tradeoff.py
- adversarial_correlations.py
- adversarial_pipeline.py
"""

from adversarial_pipeline import main


if __name__ == "__main__":
    main()

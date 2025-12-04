"""
Self‑Improving Prompt Optimisation Package
=========================================

This package exposes utility functions and classes for building and testing
self‑improving prompts. It wraps the functionality originally implemented in
the Jupyter notebook into a more modular and reusable form. See
``main.py`` for a command line interface. The primary public components are:

* :class:`prompt_optimization.config.PromptOptimisationConfig` – holds
  configuration parameters for the optimisation loop.
* :mod:`prompt_optimization.utils` – helper functions for loading data,
  building prompts and computing distances.
* :func:`prompt_optimization.main.run_optimisation` – runs the end‑to‑end
  optimisation loop.

You can import these modules directly in your own code to integrate custom
datasets or alternative evaluation metrics.
"""

from .config import PromptOptimisationConfig  # noqa: F401
from .utils import load_training_data, sample_exemplars, build_prompt, apply_unified_diff  # noqa: F401

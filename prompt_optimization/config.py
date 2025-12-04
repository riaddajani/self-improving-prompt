"""
Configuration classes and utilities for the prompt optimization project.

This module defines dataclasses encapsulating the hyper‑parameters and paths used
throughout the repository. By grouping related settings into dataclasses we
enable type checking, auto‑completion and an easy way to override individual
parameters from the command line. The default values are chosen to work with
the provided notebook and sample data, but can be customised when running the
CLI.

In general, keeping configuration in a dedicated module rather than hard
coding constants in the algorithm implementation improves readability and
testability of the code base.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PromptOptimisationConfig:
    """Configuration for the prompt optimisation loop.

    Attributes
    ----------
    csv_path:
        Path to the training data CSV. This file must contain at least two
        columns: one for the social media comment and another for the ground
        truth response. The default path points at the dataset used in the
        original notebook.
    profile_path:
        Path to a text file containing persona information. The contents of
        this file will be inserted into the prompt to steer the generated
        replies towards a consistent tone and style.
    n_exemplars:
        Number of exemplar comment–reply pairs to include in the few‑shot
        prompt. See Lilian Weng’s article on prompt engineering for an
        introduction to zero‑ and few‑shot prompting【179192835280262†L36-L77】.
    train_size:
        Number of comments sampled for each training batch during optimisation.
    val_size:
        Number of comments sampled for validation at each iteration.
    n_iterations:
        Number of optimisation rounds to perform. Each iteration attempts to
        improve the prompt and then measures the average cosine distance on
        validation data.
    dist_threshold:
        Cosine distance threshold above which generated replies are considered
        misses and included in the critique step. The value of 0.20 follows the
        heuristic from the notebook.
    good_threshold:
        Target distance below which the optimisation loop will stop early.
    patience:
        Number of successive iterations with no significant improvement before
        early stopping. This prevents unnecessary calls when progress has
        plateaued.
    min_improvement:
        Minimum reduction in average distance required to accept a new prompt.
    model_name:
        Name of the OpenAI model to use. For example ``gpt-4-turbo`` or
        ``gpt-4.1`` depending on availability. Keep in mind that larger models
        incur higher latency and cost.
    evolution_modes:
        Optional list of experiment names to run sequentially. When provided
        the optimisation script will train a separate prompt for each mode
        (e.g. different critique styles). This is left unused by default.
    """

    csv_path: str = "training_data.csv"
    profile_path: str = "profile.txt"
    n_exemplars: int = 10
    train_size: int = 20
    val_size: int = 20
    n_iterations: int = 10
    dist_threshold: float = 0.20
    good_threshold: float = 0.10
    patience: int = 5
    min_improvement: float = 1e-3
    model_name: str = "gpt-4-turbo"
    evolution_modes: Optional[List[str]] = field(default=None)

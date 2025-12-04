"""
Utility functions for self‑improving prompt optimisation.

This module exposes helper functions used by the optimisation loop. Where
possible we try to keep heavy dependencies optional and document external
requirements in the README and requirements file. Many functions are pure and
therefore easily testable. Functions that interact with the OpenAI API are
isolated so that they can be mocked during unit tests.

Note that the original notebook includes a mix of synchronous and asynchronous
API calls. In this refactored code we choose synchronous calls for
simplicity. If you wish to parallelise requests consider switching to
``openai.AsyncOpenAI`` and using ``asyncio.gather``.
"""

from __future__ import annotations

import os
import difflib
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd

try:
    # ``openai`` is an optional dependency. If it's not installed the functions
    # depending on it will raise informative errors.
    import openai
except ImportError:  # pragma: no cover - openai is optional for tests
    openai = None  # type: ignore


def load_training_data(csv_path: str) -> pd.DataFrame:
    """Load the training dataset from a CSV file.

    The CSV is expected to have at least two columns named ``comment`` and
    ``response``. If additional columns are present they are ignored. Rows
    containing NaN values in these columns are dropped.

    Parameters
    ----------
    csv_path:
        Path to the CSV file on disk.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the cleaned comment and response pairs.
    """
    df = pd.read_csv(csv_path)
    if not {"comment", "response"}.issubset(df.columns):
        raise ValueError(
            f"CSV file must contain 'comment' and 'response' columns, got {df.columns.tolist()}"
        )
    return df[["comment", "response"]].dropna().reset_index(drop=True)


def sample_exemplars(df: pd.DataFrame, n: int, seed: Optional[int] = None) -> pd.DataFrame:
    """Randomly sample exemplar pairs from the dataset.

    Few‑shot prompting benefits from including well‑chosen examples of input
    comments and their desired replies【179192835280262†L60-L77】. This function samples without
    replacement using an optional random seed for reproducibility.

    Parameters
    ----------
    df:
        DataFrame with at least columns ``comment`` and ``response``.
    n:
        Number of exemplars to sample.
    seed:
        Optional seed for the random number generator.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of length ``n`` containing exemplar pairs. If ``n`` is
        larger than the number of rows in ``df`` the entire dataset is
        returned.
    """
    if n <= 0:
        raise ValueError("Number of exemplars must be positive")
    n = min(n, len(df))
    return df.sample(n, random_state=seed).reset_index(drop=True)


def build_prompt(profile: str, exemplars: pd.DataFrame) -> str:
    """Construct an initial prompt from persona information and exemplar pairs.

    The persona or profile describes the desired tone, domain knowledge and
    behavioural constraints for the AI assistant. Exemplars provide
    demonstrations of correct behaviour in response to real user comments. Both
    pieces are concatenated into a single string used as input to the model.

    Parameters
    ----------
    profile:
        A string containing persona information loaded from a text file.
    exemplars:
        DataFrame with columns ``comment`` and ``response`` containing the
        demonstration pairs.

    Returns
    -------
    str
        A formatted prompt string.
    """
    lines: List[str] = []
    lines.append(profile.strip())
    lines.append("")
    lines.append("Below are examples of how to reply to comments:")
    for _, row in exemplars.iterrows():
        lines.append(f"Comment: {row['comment']}")
        lines.append(f"Reply: {row['response']}")
        lines.append("")
    lines.append("When generating a reply to a new comment, adhere to the style shown above.")
    return "\n".join(lines)


def apply_unified_diff(original: str, diff_text: str) -> str:
    """Apply a minimal unified‑diff patch to a string.

    This helper implements a simple unified diff parser similar to that used in
    the original notebook. It ignores fencing markers (``````, ``---``, ``+++``)
    and applies line level additions and deletions. Unsupported diff operations
    such as context or hunk headers are skipped. If the diff removes more
    characters than exist in the original prompt an IndexError will be raised.

    Parameters
    ----------
    original:
        The original string to be patched.
    diff_text:
        The unified diff text (e.g. produced by GPT). Lines beginning with
        ``+`` are inserted, lines beginning with ``-`` are deleted and lines
        beginning with a space are kept unchanged.

    Returns
    -------
    str
        The patched string.
    """
    orig_lines = original.splitlines()
    new_lines: List[str] = []
    idx = 0
    for raw in diff_text.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("```") or line.startswith(("---", "+++", "@@")) or line == "":
            # Skip diff metadata and code fences
            continue
        if line.startswith(" "):
            new_lines.append(orig_lines[idx])
            idx += 1
        elif line.startswith("-"):
            # Delete the corresponding original line
            idx += 1
        elif line.startswith("+"):
            # Insert new line (remove leading '+')
            new_lines.append(line[1:])
        else:
            # Unknown prefix, treat as unchanged
            new_lines.append(line)
            idx += 1
    # Append remaining unchanged lines
    if idx < len(orig_lines):
        new_lines.extend(orig_lines[idx:])
    return "\n".join(new_lines)


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute the cosine distance between two 1‑D vectors.

    Cosine distance is defined as ``1 - cos(theta)`` where ``cos(theta)`` is
    the cosine similarity. It ranges from 0 (identical) to 2 (opposite). The
    measure is commonly used to evaluate similarity between embedding vectors
    representing natural language sentences.

    Parameters
    ----------
    vec_a, vec_b:
        One‑dimensional numpy arrays of the same length.

    Returns
    -------
    float
        The cosine distance between the vectors.
    """
    numerator = float(np.dot(vec_a, vec_b))
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 1.0
    return 1.0 - numerator / denom


def get_embedding(text: str, model: str = "text-embedding-ada-002") -> np.ndarray:
    """Retrieve a text embedding from the OpenAI API.

    This function calls the OpenAI embeddings endpoint to convert input text
    into a fixed length vector. It requires the ``OPENAI_API_KEY`` environment
    variable to be set and the ``openai`` Python package to be installed. If
    those conditions are not met an informative RuntimeError is raised.

    Parameters
    ----------
    text:
        The input string to embed. Avoid sending extremely long texts because
        they will be truncated by the API.
    model:
        Identifier of the embedding model to use. See the OpenAI documentation
        for available models and their trade‑offs. The default ``text-embedding-ada-002``
        is inexpensive and performs well on semantic similarity tasks.

    Returns
    -------
    numpy.ndarray
        A one‑dimensional vector representing the input text.
    """
    if openai is None:
        raise RuntimeError(
            "openai Python package is not installed. Install it with 'pip install openai'."
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Environment variable OPENAI_API_KEY must be set to call the OpenAI API."
        )
    response = openai.Embedding.create(model=model, input=[text])  # type: ignore
    data = response["data"][0]["embedding"]
    return np.array(data, dtype=float)


def distance_between_texts(a: str, b: str, model: str = "text-embedding-ada-002") -> float:
    """Compute the cosine distance between two strings via embeddings.

    Under the hood this function fetches embeddings for both inputs and then
    computes the cosine distance. It is separated here for convenience so that
    embedding retrieval can be mocked during unit tests.

    Parameters
    ----------
    a, b:
        The two strings to compare.
    model:
        The OpenAI embedding model identifier.

    Returns
    -------
    float
        The cosine distance between the embeddings of the two strings.
    """
    vec_a = get_embedding(a, model=model)
    vec_b = get_embedding(b, model=model)
    return cosine_distance(vec_a, vec_b)

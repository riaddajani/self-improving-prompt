"""
Command line entry point for running self‑improving prompt optimisation.

This script orchestrates the end‑to‑end loop described in the original
notebook. It loads the training data, constructs an initial prompt using a
persona profile and exemplar examples, then iteratively improves the prompt
based on feedback from the model. At each iteration the current prompt is
evaluated on a held‑out validation set using cosine distance between the
generated replies and the ground truth responses. If a candidate prompt
achieves a lower average distance by more than a configurable threshold it is
accepted. Otherwise the previous prompt is retained. The loop stops after a
fixed number of iterations, when a satisfactory distance has been reached,
or when no improvements are seen for a number of rounds (patience).

Running this script requires a valid OpenAI API key. Set the
``OPENAI_API_KEY`` environment variable before executing. See the README for
installation instructions and safety considerations.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import PromptOptimisationConfig
from . import utils

try:
    import openai
except ImportError:
    openai = None  # type: ignore


def parse_arguments(argv: List[str]) -> PromptOptimisationConfig:
    """Parse command line arguments into a configuration dataclass.

    Each attribute of :class:`PromptOptimisationConfig` is exposed as an
    optional argument. For example, to override the number of iterations run:

    ``python -m prompt_optimization.main --n-iterations 20``

    Parameters
    ----------
    argv:
        Raw command line arguments (excluding the executable name).

    Returns
    -------
    PromptOptimisationConfig
        An instance populated with user provided values or defaults.
    """
    parser = argparse.ArgumentParser(description="Run self‑improving prompt optimisation.")
    parser.add_argument("--csv-path", type=str, default=PromptOptimisationConfig.csv_path,
                        help="Path to the training CSV containing comment and response columns.")
    parser.add_argument("--profile-path", type=str, default=PromptOptimisationConfig.profile_path,
                        help="Path to the persona profile text file.")
    parser.add_argument("--n-exemplars", type=int, default=PromptOptimisationConfig.n_exemplars,
                        help="Number of exemplar pairs to include in the initial prompt.")
    parser.add_argument("--train-size", type=int, default=PromptOptimisationConfig.train_size,
                        help="Number of comments sampled for training in each iteration.")
    parser.add_argument("--val-size", type=int, default=PromptOptimisationConfig.val_size,
                        help="Number of comments sampled for validation in each iteration.")
    parser.add_argument("--n-iterations", type=int, default=PromptOptimisationConfig.n_iterations,
                        help="Maximum number of optimisation rounds.")
    parser.add_argument("--dist-threshold", type=float, default=PromptOptimisationConfig.dist_threshold,
                        help="Cosine distance above which outputs are considered misses.")
    parser.add_argument("--good-threshold", type=float, default=PromptOptimisationConfig.good_threshold,
                        help="Target distance below which optimisation stops early.")
    parser.add_argument("--patience", type=int, default=PromptOptimisationConfig.patience,
                        help="Number of iterations with no improvement before stopping.")
    parser.add_argument("--min-improvement", type=float, default=PromptOptimisationConfig.min_improvement,
                        help="Minimum reduction in average distance to accept a new prompt.")
    parser.add_argument("--model-name", type=str, default=PromptOptimisationConfig.model_name,
                        help="Name of the OpenAI model to use for generation and critique.")
    args = parser.parse_args(argv)
    return PromptOptimisationConfig(
        csv_path=args.csv_path,
        profile_path=args.profile_path,
        n_exemplars=args.n_exemplars,
        train_size=args.train_size,
        val_size=args.val_size,
        n_iterations=args.n_iterations,
        dist_threshold=args.dist_threshold,
        good_threshold=args.good_threshold,
        patience=args.patience,
        min_improvement=args.min_improvement,
        model_name=args.model_name
    )


def load_profile(profile_path: str) -> str:
    """Load persona text from a file.

    Parameters
    ----------
    profile_path:
        Path to a plain text file containing the persona description.

    Returns
    -------
    str
        Contents of the file as a single string.
    """
    with open(profile_path, "r", encoding="utf8") as f:
        return f.read()


def generate_reply(model_name: str, system_prompt: str, user_message: str) -> str:
    """Call the OpenAI chat completion API to generate a reply.

    This helper builds a chat request using a system prompt and a user message
    describing the comment to reply to. It returns the content of the assistant
    message. If the OpenAI package is not installed or the API key is missing
    an informative error is raised.

    Parameters
    ----------
    model_name:
        The model identifier to call, e.g. ``gpt-4-turbo``.
    system_prompt:
        The prompt containing persona and exemplars.
    user_message:
        A string describing the current comment. Should end with ``Reply:`` to
        indicate where the model should insert its response.

    Returns
    -------
    str
        The generated reply.
    """
    if openai is None:
        raise RuntimeError("openai package is not installed. Install it with 'pip install openai'.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable must be set to call the API.")
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=200,
    )  # type: ignore
    return completion["choices"][0]["message"]["content"].strip()


def evaluate_prompt_on_batch(
    prompt: str,
    model_name: str,
    comments: List[str],
    ground_truth: List[str],
    embedding_model: str = "text-embedding-ada-002",
) -> Tuple[float, List[Tuple[str, str, str, float]]]:
    """Evaluate a prompt by generating replies and computing cosine distances.

    For each comment in the batch this function calls the generation API to
    produce a reply, then computes the cosine distance between the generated
    reply and the corresponding ground truth response. The distances are
    averaged and returned along with a list of tuples containing the inputs
    and distances.

    Parameters
    ----------
    prompt:
        The current prompt (system message) used for generation.
    model_name:
        The OpenAI model name to call.
    comments:
        A list of comments to reply to.
    ground_truth:
        The expected responses for the comments.
    embedding_model:
        Embedding model name used for distance calculations.

    Returns
    -------
    Tuple[float, List[Tuple[str, str, str, float]]]
        The average cosine distance and a list of results containing comment,
        ground truth, generated reply and distance.
    """
    results: List[Tuple[str, str, str, float]] = []
    distances = []
    for comment, gt in zip(comments, ground_truth):
        # Compose user message: include the comment and a cue for the model to reply
        user_message = f"Comment: {comment}\nReply:"
        generated = generate_reply(model_name=model_name, system_prompt=prompt, user_message=user_message)
        dist = utils.distance_between_texts(generated, gt, model=embedding_model)
        distances.append(dist)
        results.append((comment, gt, generated, dist))
    avg_dist = float(np.mean(distances)) if distances else float('inf')
    return avg_dist, results


def build_critique_message(misses: List[Tuple[str, str, str, float]]) -> str:
    """Construct a critique message summarising the worst performing examples.

    The model is asked to identify why the generated replies differ from the
    ground truth and to suggest improvements to the prompt. We follow the
    structure used in the notebook: each miss is presented with the comment,
    ground truth, generated reply and the cosine distance. The message ends
    with a request for specific feedback.

    Parameters
    ----------
    misses:
        A list of tuples ``(comment, ground_truth, generated, distance)``
        representing misaligned examples where the distance exceeds the
        threshold.

    Returns
    -------
    str
        The user message sent to the critique model.
    """
    lines = []
    lines.append("The following replies deviate from the desired behaviour. Please critique and suggest improvements to the prompt to reduce these mismatches:")
    for comment, gt, gen, dist in misses:
        lines.append("")
        lines.append(f"Comment: {comment}")
        lines.append(f"Expected reply: {gt}")
        lines.append(f"Generated reply: {gen}")
        lines.append(f"Cosine distance: {dist:.3f}")
    lines.append("")
    lines.append("Please provide actionable suggestions. Do not rewrite the entire prompt, only point out what to adjust.")
    return "\n".join(lines)


def critique_and_patch_prompt(
    current_prompt: str,
    model_name: str,
    misses: List[Tuple[str, str, str, float]],
) -> str:
    """Ask the model to critique the current prompt and return a patched prompt.

    This function sends the critique request to the language model and then a
    follow‑up request asking for a unified diff patch. It applies the diff to
    the current prompt and returns the improved prompt. If the API key is
    unavailable, the original prompt is returned unchanged.

    Parameters
    ----------
    current_prompt:
        The current system prompt to be improved.
    model_name:
        The OpenAI model name to use for both critique and diff generation.
    misses:
        Misaligned examples used as evidence for critique.

    Returns
    -------
    str
        The improved prompt if available; otherwise the current prompt.
    """
    if not misses:
        return current_prompt
    # Build critique prompt
    critique_system = (
        "You are a prompt engineer and strict evaluator. Analyse the following examples and recommend concrete improvements to the prompt to minimise the cosine distance between generated and expected replies."
    )
    critique_user = build_critique_message(misses)
    # Query model for critique suggestions
    if openai is None:
        raise RuntimeError("openai package is not installed. Install it with 'pip install openai'.")
    critique_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": critique_system},
            {"role": "user", "content": critique_user},
        ],
        temperature=0.0,
        max_tokens=500,
    )  # type: ignore
    critique_text = critique_completion["choices"][0]["message"]["content"].strip()
    # Ask for unified diff based on critique
    diff_system = (
        "You are a prompt‑engineer bot. Your job is to output a unified diff that transforms the ORIGINAL prompt into an IMPROVED prompt incorporating the critique suggestions. Only output the diff."
    )
    diff_user = f"ORIGINAL_PROMPT\n<<<\n{current_prompt}\n>>>\n\nCRITIQUE_SUGGESTIONS\n<<<\n{critique_text}\n>>>\n\nPlease produce a git‑style unified diff showing only the lines that change. Keep placeholders intact."
    diff_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": diff_system},
            {"role": "user", "content": diff_user},
        ],
        temperature=0.0,
        max_tokens=500,
    )  # type: ignore
    diff_text = diff_completion["choices"][0]["message"]["content"].strip()
    # Apply the diff to the current prompt
    return utils.apply_unified_diff(current_prompt, diff_text)


def run_optimisation(config: PromptOptimisationConfig) -> None:
    """Run the full optimisation loop as defined by the configuration.

    This function ties together loading data, building the initial prompt,
    iterative training, validation and prompt updates. It prints intermediate
    results to stdout so that progress can be monitored. A simple matplotlib
    chart is displayed at the end if running in an interactive environment.

    Parameters
    ----------
    config:
        A :class:`PromptOptimisationConfig` instance describing hyper‑parameters
        and paths.
    """
    # Load dataset and persona
    df = utils.load_training_data(config.csv_path)
    profile = load_profile(config.profile_path)
    # Build initial prompt
    exemplars = utils.sample_exemplars(df, config.n_exemplars, seed=0)
    current_prompt = utils.build_prompt(profile, exemplars)
    best_distance = float('inf')
    best_prompt = current_prompt
    stagnation = 0
    history: List[float] = []
    # Loop over iterations
    for iteration in range(1, config.n_iterations + 1):
        # Sample train and validation batches
        train_df = df.sample(config.train_size, random_state=iteration)
        val_df = df.sample(config.val_size, random_state=iteration * 37)
        # Evaluate current prompt on training set
        avg_train_dist, train_results = evaluate_prompt_on_batch(
            prompt=current_prompt,
            model_name=config.model_name,
            comments=train_df.comment.tolist(),
            ground_truth=train_df.response.tolist(),
        )
        # Identify misses above threshold
        misses = [(comment, gt, gen, dist) for comment, gt, gen, dist in train_results if dist > config.dist_threshold]
        # Generate improved prompt via critique
        improved_prompt = critique_and_patch_prompt(current_prompt, config.model_name, misses)
        # Evaluate improved prompt on validation set
        avg_val_dist, _ = evaluate_prompt_on_batch(
            prompt=improved_prompt,
            model_name=config.model_name,
            comments=val_df.comment.tolist(),
            ground_truth=val_df.response.tolist(),
        )
        print(f"Iteration {iteration}: train distance = {avg_train_dist:.4f}, val distance = {avg_val_dist:.4f}")
        history.append(avg_val_dist)
        # Decide whether to accept improved prompt
        if avg_val_dist + config.min_improvement < best_distance:
            print(f"Improvement detected: {best_distance:.4f} -> {avg_val_dist:.4f}")
            best_distance = avg_val_dist
            best_prompt = improved_prompt
            current_prompt = improved_prompt
            stagnation = 0
        else:
            print("No significant improvement; retaining previous prompt.")
            stagnation += 1
        # Early stopping conditions
        if best_distance <= config.good_threshold:
            print("Good threshold reached; stopping early.")
            break
        if stagnation >= config.patience:
            print("No improvement for several iterations; stopping.")
            break
    # Summary
    print("\nOptimisation finished.")
    print(f"Best average distance: {best_distance:.4f}")
    print("Best prompt:\n")
    print(best_prompt)
    # Plot history if possible
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(range(1, len(history) + 1), history, marker="o")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Average cosine distance")
        ax.set_title("Prompt optimisation progress")
        plt.show()
    except Exception:
        pass


def main(argv: List[str] | None = None) -> None:
    """Parse arguments and run the optimisation loop.

    This function serves as the entry point when the module is executed as a
    script. It is separated from :func:`run_optimisation` to improve
    testability.
    """
    config = parse_arguments(argv or sys.argv[1:])
    run_optimisation(config)


if __name__ == "__main__":  # pragma: no cover
    main()

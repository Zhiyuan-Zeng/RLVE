"""Run the environment checks on a few generated problems per environment and on the stored held-out problems.

RLVE_TEST_DIFFICULTIES and RLVE_TEST_SEEDS (comma-separated integers) choose the generated problems.
"""

import collections
import functools
import json
import os
import pathlib
import random

import pytest

from Gym.environments import identifier2environment
from Gym.tests import environment_checks as checks

DIFFICULTIES = [int(value) for value in os.environ.get("RLVE_TEST_DIFFICULTIES", "0,1,2").split(",")]
SEEDS = [int(value) for value in os.environ.get("RLVE_TEST_SEEDS", "0,1").split(",")]
IDENTIFIERS = sorted(identifier2environment)
HELD_OUT_FILE = pathlib.Path(__file__).resolve().parents[2] / "data" / "HELD-OUT_ENVIRONMENTS" / "test.json"

# Found by search: with these values the second Numbrix generation starts in the middle of the first one's search.
NUMBRIX_SEEDS = (1, 0)
NUMBRIX_SHUFFLES_BEFORE_SECOND_GENERATION = 4


@functools.lru_cache(maxsize=None)
def generated_problems(identifier):
    """One entry per (difficulty, seed): the problem, or None when generation failed."""
    return [
        (difficulty, seed, checks.generate_problem(identifier, difficulty, seed))
        for difficulty in DIFFICULTIES
        for seed in SEEDS
    ]


def problems(identifier):
    return [problem for _, _, problem in generated_problems(identifier) if problem is not None]


def held_out_problems():
    """The stored held-out test problems that have a reference answer, grouped by environment."""
    grouped = collections.defaultdict(list)
    if not HELD_OUT_FILE.exists():
        return grouped
    for index, item in enumerate(json.loads(HELD_OUT_FILE.read_text(encoding="utf-8"))):
        metadata = json.loads(item["metadata"])
        problem = checks.Problem(
            metadata["environment"], None, None, metadata["config"], source=f"held-out problem {index}"
        )
        if problem.reference_answer() is not None:
            grouped[problem.identifier].append(problem)
    return grouped


HELD_OUT_PROBLEMS = held_out_problems()


def assert_no_failures(failures):
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("identifier", IDENTIFIERS)
def test_problem_generation(identifier):
    failures = []
    for difficulty, seed, problem in generated_problems(identifier):
        if problem is None:
            failures.append(f"{identifier} (difficulty {difficulty}, seed {seed}): generator() returned False")
        else:
            failures.extend(checks.check_generation(problem, checks.generate_problem(identifier, difficulty, seed)))
    assert_no_failures(failures)


@pytest.mark.parametrize("identifier", IDENTIFIERS)
def test_strategy_names(identifier):
    assert_no_failures(checks.check_strategy_names(identifier))


@pytest.mark.parametrize("identifier", IDENTIFIERS)
def test_reference_answer_passes(identifier):
    assert_no_failures([failure for problem in problems(identifier) for failure in checks.check_reference_answer(problem)])


@pytest.mark.parametrize("identifier", IDENTIFIERS)
def test_probe_answers(identifier):
    assert_no_failures([failure for problem in problems(identifier) for failure in checks.check_probe_answers(problem)])


@pytest.mark.parametrize("identifier", sorted(HELD_OUT_PROBLEMS))
def test_held_out_reference_answers_pass(identifier):
    assert_no_failures(
        [failure for problem in HELD_OUT_PROBLEMS[identifier] for failure in checks.check_reference_answer(problem)]
    )


def test_numbrix_generations_do_not_share_search_state(monkeypatch):
    """A generator thread that timed out keeps running, so a second Numbrix generation can start before the first ends."""
    numbrix = identifier2environment["Numbrix"]
    parameter = {"MAX_N_M": 5, "sparsity": 0.5}
    first, second = numbrix(), numbrix()
    shuffle = random.shuffle
    calls = []

    def shuffle_then_start_the_second_generation(sequence):
        shuffle(sequence)
        calls.append(None)
        if len(calls) == NUMBRIX_SHUFFLES_BEFORE_SECOND_GENERATION:
            state = random.getstate()
            assert second.generator(seed=NUMBRIX_SEEDS[1], parameter=dict(parameter))
            random.setstate(state)

    monkeypatch.setattr(random, "shuffle", shuffle_then_start_the_second_generation)
    assert first.generator(seed=NUMBRIX_SEEDS[0], parameter=dict(parameter))
    monkeypatch.undo()

    assert second.parameter is not None, "the first generation ended before the second one started"
    for environment in (first, second):
        assert checks.score(environment, environment.parameter["reference_answer"])["accuracy"] == 1

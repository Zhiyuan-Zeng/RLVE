"""Checks that every environment in Gym/ is expected to pass.

A problem is generated the way training generates it: a parameter from the environment's controller at
some difficulty, then `generator(seed, parameter)`. An answer is scored the way the reward function scores
it: a new environment object, `set_config(config)`, `verifier(output)`.

Each check returns a list of failure messages. An empty list means the check passed.
"""

import ast
import copy
import inspect
import json
import re
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from Gym.environments import identifier2environment
from Gym.parameter_controllers import identifier2controller

ANSWER_MARKERS = {"<answer></answer>": ("<answer>", "</answer>"), "\\boxed{}": ("\\boxed{", "}")}
STRATEGY_ARGUMENT = re.compile(r"""(self\.rewards\[["']rewarding_strategy\w*["']\]|rewarding_strategy\w*)""")
INTEGER = re.compile(r"-?\d+")


@dataclass
class Problem:
    """One problem: generated here from a difficulty and a seed, or loaded from a stored config."""

    identifier: str
    difficulty: Optional[int]
    seed: Optional[int]
    config: dict  # get_config() of the environment that generated the problem
    source: str = ""  # where a stored problem comes from

    def __str__(self) -> str:
        return f"{self.identifier} ({self.source or f'difficulty {self.difficulty}, seed {self.seed}'})"

    def environment(self, markers=None, through_json: bool = False, **arguments):
        """A new environment that holds this problem, built the way the reward function builds it."""
        config = json.loads(json.dumps(self.config)) if through_json else copy.deepcopy(self.config)
        environment = identifier2environment[self.identifier](answer_markers=markers, **arguments)
        environment.set_config(config)
        return environment

    def reference_answer(self) -> Optional[str]:
        reference = self.config["parameter"].get("reference_answer")
        return None if reference is None else str(reference)


def controller_parameter(identifier: str, difficulty: int, seed: int) -> dict:
    """The parameter the controller offers at this difficulty; the seed picks one when it offers several."""
    controller = identifier2controller[identifier]()
    for _ in range(difficulty):
        controller.update()
    parameters = controller.get_parameter_list()
    return copy.deepcopy(parameters[seed % len(parameters)])


def generate_problem(identifier: str, difficulty: int, seed: int, timeout_second: int = 10) -> Optional[Problem]:
    """Generate one problem, or return None when the generator fails or times out."""
    environment = identifier2environment[identifier]()
    parameter = controller_parameter(identifier, difficulty, seed)
    if not environment.generator(seed=seed, parameter=parameter, timeout_second=timeout_second):
        return None
    return Problem(identifier, difficulty, seed, environment.get_config())


def strategy_comparisons(environment_class) -> Dict[str, List[str]]:
    """Map each strategy name the class compares against to the expressions it is compared with.

    Any `<expression> == "<name>"` counts when the expression mentions rewards or strategies, so a branch
    that reads the wrong key or attribute is found as well.
    """
    comparisons: Dict[str, List[str]] = {}
    tree = ast.parse(textwrap.dedent(inspect.getsource(environment_class)))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)):
            continue
        right = node.comparators[0]
        if not (isinstance(right, ast.Constant) and isinstance(right.value, str)):
            continue
        left = ast.unparse(node.left)
        if "reward" in left.lower() or "strateg" in left.lower():
            comparisons.setdefault(right.value, []).append(left)
    return comparisons


def strategy_settings(environment_class) -> List[dict]:
    """Constructor arguments to score with: the defaults, then each strategy argument set to each other name."""
    parameters = inspect.signature(environment_class.__init__).parameters
    names = list(strategy_comparisons(environment_class))
    settings: List[dict] = [{}]
    for argument, parameter in parameters.items():
        if argument.startswith("rewarding_strategy"):
            settings.extend({argument: name} for name in names if name != parameter.default)
    return settings


def score(environment, answer: Optional[str]) -> dict:
    """verifier() on an output that holds `answer` between the answer markers (no markers when it is None)."""
    if answer is None:
        return environment.verifier("There is no answer here.")
    start, end = environment.answer_markers
    return environment.verifier(f"The answer is {start}{answer}{end}")


def is_unknown_strategy(environment, answer: str) -> bool:
    """True when the scorer does not know the strategy name it was built with (the name belongs to another argument)."""
    start, end = environment.answer_markers
    try:
        environment.scorer(f"{start}{answer}{end}")
    except NotImplementedError:
        return True
    except Exception:
        pass
    return False


def probe_answers(reference: Optional[str]) -> Dict[str, str]:
    """Answers that are malformed, or that differ from the reference answer in one systematic way."""
    probes = {
        "empty": "",
        "a word": "abc",
        "a decimal": "0.5",
        "zero": "0",
        "one": "1",
        "minus one": "-1",
        "a large number": "123456789",
    }
    if reference is None or not reference.strip():
        return probes
    text = reference.strip()
    tokens, lines = text.split(), text.splitlines()
    if len(tokens) > 1:
        probes["last item dropped"] = " ".join(tokens[:-1])
        probes["first item dropped"] = " ".join(tokens[1:])
        probes["items reversed"] = " ".join(reversed(tokens))
    if len(lines) > 1:
        probes["last line dropped"] = "\n".join(lines[:-1])
        probes["lines reversed"] = "\n".join(reversed(lines))
    if len(text) > 1:
        probes["last character dropped"] = text[:-1]
    probes["answer repeated"] = text + ("\n" if len(lines) > 1 else " ") + text
    integers = list(INTEGER.finditer(text))
    if integers:
        first, last = integers[0], integers[-1]

        def replace(match, value):
            return text[: match.start()] + str(value) + text[match.end() :]

        probes["last integer plus one"] = replace(last, int(last.group()) + 1)
        probes["last integer minus one"] = replace(last, int(last.group()) - 1)
        probes["first integer plus one"] = replace(first, int(first.group()) + 1)
        probes["last integer is -1"] = replace(last, -1)
        probes["last integer is 1000000"] = replace(last, 1000000)
        probes["last integer repeats the first"] = replace(last, first.group())
        probes["every integer plus one"] = INTEGER.sub(lambda m: str(int(m.group()) + 1), text)
        probes["every integer minus one"] = INTEGER.sub(lambda m: str(int(m.group()) - 1), text)
        probes["every integer negated"] = INTEGER.sub(lambda m: str(-int(m.group())), text)
        probes["every integer is 0"] = INTEGER.sub("0", text)
    return {name: answer for name, answer in probes.items() if answer != text}


def check_generation(problem: Problem, regenerated: Optional[Problem]) -> List[str]:
    """A second generation from the same seed and parameter gives the same problem; config and prompt survive JSON."""
    failures = []
    if regenerated is None:
        failures.append(f"{problem}: generator() returned False the second time")
    elif regenerated.config != problem.config:
        failures.append(f"{problem}: the same seed and parameter gave a different problem the second time")
    try:
        json.dumps(problem.config)
    except (TypeError, ValueError) as error:
        return failures + [f"{problem}: get_config() is not JSON serializable ({error})"]
    prompt = problem.environment().prompt_generator()
    if not isinstance(prompt, str) or not prompt.strip():
        failures.append(f"{problem}: prompt_generator() returned no text")
    if problem.environment(through_json=True).prompt_generator() != prompt:
        failures.append(f"{problem}: the prompt changes when the config goes through JSON")
    return failures


def check_strategy_names(identifier: str) -> List[str]:
    """Every comparison with a strategy name reads a rewarding-strategy argument."""
    failures = []
    for name, expressions in strategy_comparisons(identifier2environment[identifier]).items():
        for expression in expressions:
            if not STRATEGY_ARGUMENT.fullmatch(expression):
                failures.append(f'{identifier}: {expression} == "{name}" does not read a rewarding-strategy argument')
    return failures


def check_reference_answer(problem: Problem) -> List[str]:
    """The stored reference answer passes under every reward strategy, marker style and config form."""
    reference = problem.reference_answer()
    if reference is None:
        return []
    failures = []
    for arguments in strategy_settings(identifier2environment[problem.identifier]):
        if arguments and is_unknown_strategy(problem.environment(**arguments), reference):
            continue
        for marker_name, markers in ANSWER_MARKERS.items():
            for through_json in (False, True):
                environment = problem.environment(markers=markers, through_json=through_json, **arguments)
                for padding in ("", " ", "\n"):
                    result = score(environment, padding + reference + padding)
                    if result["accuracy"] != 1:
                        failures.append(
                            f"{problem}: the reference answer {reference!r} gets reward {result['reward']} "
                            f"with {arguments or 'default arguments'}, {marker_name} markers"
                            f"{', config through JSON' if through_json else ''}"
                            f"{', padded with whitespace' if padding else ''}"
                        )
                        break
    return failures


def score_probe_answers(problem: Problem) -> Tuple[List[str], Dict[str, str]]:
    """Score every probe answer on the generated config and on the config after JSON.

    Returns the failures and the probe answers that were scored as correct. A probe that is scored as correct
    is not a failure by itself, because many problems have several correct answers; the failures are a
    verifier() that raises, a score that changes after JSON, and a non-answer that is scored as correct.
    """
    failures: List[str] = []
    passing: Dict[str, str] = {}
    probes: Dict[str, Optional[str]] = dict(probe_answers(problem.reference_answer()))
    probes["no answer markers"] = None
    for name, answer in probes.items():
        results = []
        for through_json in (False, True):
            try:
                results.append(score(problem.environment(through_json=through_json), answer))
            except AssertionError as error:
                message = str(error).split("\n")[0]
                failures.append(f"{problem}: verifier() raises on the answer {answer!r} ({name}): {message}")
                break
        if len(results) < 2:
            continue
        if results[0] != results[1]:
            failures.append(
                f"{problem}: the answer {answer!r} ({name}) gets reward {results[0]['reward']} on the generated "
                f"config and {results[1]['reward']} after the config goes through JSON"
            )
        if results[0]["accuracy"] == 1:
            if name in ("no answer markers", "a word"):
                failures.append(f"{problem}: the answer {answer!r} ({name}) is scored as correct")
            else:
                passing[name] = answer
    return failures, passing


def check_probe_answers(problem: Problem) -> List[str]:
    """No answer makes verifier() raise or score differently after JSON, and non-answers never pass."""
    return score_probe_answers(problem)[0]

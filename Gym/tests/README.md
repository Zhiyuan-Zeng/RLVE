# Tests for the environments

`environment_checks.py` holds checks that every environment is expected to pass. A problem is generated the way training generates it (a parameter from the controller, then `generator(seed, parameter)`) and scored the way the reward function scores it (a new environment, `set_config(config)`, `verifier(output)`).

| Check | What it requires |
| --- | --- |
| generation | The problem regenerates identically from its seed and parameter. The config is JSON serializable, and the prompt is the same after the config goes through JSON. |
| strategy names | Every comparison with a rewarding-strategy name reads a rewarding-strategy argument. |
| reference answer | The stored `reference_answer` passes under every rewarding strategy, with `<answer></answer>` and `\boxed{}` markers, on the generated config and after JSON, with and without whitespace around it. |
| probe answers | For up to 25 malformed or perturbed answers: `verifier` does not raise, the score is the same after JSON, and an output with no answer markers or with `abc` as its answer is not scored as correct. |

Environments that store no `reference_answer` are covered by the generation and probe checks only.

The reference answer check also runs on the problems stored in `data/HELD-OUT_ENVIRONMENTS/test.json`.

Run the tests from the repository root. They need `pytest` and take about half a minute:

```bash
python -m pytest Gym/tests
```

`RLVE_TEST_DIFFICULTIES` and `RLVE_TEST_SEEDS` choose the problems (defaults `0,1,2` and `0,1`).

The sweep runs the same checks on more problems, one process per environment, and prints a report:

```bash
python -m Gym.tests.sweep --max-difficulty 10 --seeds 10 --jobs 8
```

With `--passing-probes` the report also lists the perturbed answers that were scored as correct. Most of them are correct, because many problems have more than one valid answer. The list is there to be read; it is not a failure.

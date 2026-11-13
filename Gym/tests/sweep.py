"""Run the environment checks on many problems and print a report.

    python -m Gym.tests.sweep --max-difficulty 10 --seeds 10 --jobs 8

Each environment runs in its own process, which is ended once it reports its last problem, because a
generator that times out leaves its thread running. An environment stops at the first problem whose
generation fails, or when its time budget is used up; the problems checked until then still count.
"""

import argparse
import collections
import multiprocessing
import time

from Gym.environments import identifier2environment
from Gym.tests import environment_checks as checks


def new_result(identifier):
    return {
        "identifier": identifier,
        "problems": 0,
        "with_reference": 0,
        "stopped": None,
        "failures": [],
        "passing_probes": collections.Counter(),
        "finished": False,
    }


def sweep_environment(identifier, max_difficulty, seeds, connection):
    """Check problem after problem and send the totals so far after each one."""
    result = new_result(identifier)
    result["failures"] += checks.check_strategy_names(identifier)
    for difficulty in range(max_difficulty + 1):
        for seed in range(seeds):
            problem = checks.generate_problem(identifier, difficulty, seed)
            regenerated = None if problem is None else checks.generate_problem(identifier, difficulty, seed)
            if regenerated is None:
                result["stopped"] = f"generation failed at difficulty {difficulty}, seed {seed}"
                break
            probe_failures, passing = checks.score_probe_answers(problem)
            result["problems"] += 1
            result["with_reference"] += problem.reference_answer() is not None
            result["failures"] += checks.check_generation(problem, regenerated)
            result["failures"] += checks.check_reference_answer(problem) + probe_failures
            result["passing_probes"].update(passing.keys())
            connection.send(result)
        if result["stopped"]:
            break
    result["finished"] = True
    connection.send(result)


def sweep(identifiers, max_difficulty, seeds, jobs, seconds):
    waiting = list(identifiers)
    running = {}
    results = []
    while waiting or running:
        while waiting and len(running) < jobs:
            identifier = waiting.pop(0)
            receiver, sender = multiprocessing.Pipe(duplex=False)
            process = multiprocessing.Process(target=sweep_environment, args=(identifier, max_difficulty, seeds, sender))
            process.start()
            sender.close()
            running[identifier] = [process, receiver, time.monotonic(), new_result(identifier)]
        for identifier, (process, receiver, started, latest) in list(running.items()):
            try:
                while receiver.poll():
                    latest = running[identifier][3] = receiver.recv()
            except EOFError:
                if not latest["finished"]:
                    latest["stopped"] = latest["stopped"] or "the process ended early"
                    latest["finished"] = True
            if not latest["finished"] and time.monotonic() - started > seconds:
                latest["stopped"] = f"time budget used up after {latest['problems']} problems"
                latest["finished"] = True
            if latest["finished"]:
                process.terminate()
                process.join()
                results.append(latest)
                del running[identifier]
        time.sleep(0.05)
    return sorted(results, key=lambda result: result["identifier"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--max-difficulty", type=int, default=5, help="difficulties 0 to this value")
    parser.add_argument("--seeds", type=int, default=5, help="seeds 0 to this value minus one, at every difficulty")
    parser.add_argument("--jobs", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--seconds", type=float, default=600, help="time budget for one environment")
    parser.add_argument("--environments", nargs="*", default=sorted(identifier2environment), metavar="IDENTIFIER")
    parser.add_argument("--passing-probes", action="store_true", help="also list probe answers that are scored as correct")
    arguments = parser.parse_args()

    results = sweep(arguments.environments, arguments.max_difficulty, arguments.seeds, arguments.jobs, arguments.seconds)

    print(f"environments: {len(results)}")
    print(f"problems generated: {sum(result['problems'] for result in results)}")
    print(f"problems with a reference answer: {sum(result['with_reference'] for result in results)}")
    print(f"environments with a reference answer: {sum(result['with_reference'] > 0 for result in results)}")
    stopped = [result for result in results if result["stopped"] is not None]
    print(f"environments that stopped before the last problem: {len(stopped)}")
    for result in stopped:
        print(f"  {result['identifier']}: {result['stopped']}")
    failing = [result for result in results if result["failures"]]
    print(f"environments with failures: {len(failing)}")
    for result in failing:
        print(f"  {result['identifier']}: {len(result['failures'])} failures, the first three:")
        for failure in result["failures"][:3]:
            print(f"    {failure}")
    if arguments.passing_probes:
        passing = [result for result in results if result["passing_probes"]]
        print(f"environments where a probe answer is scored as correct: {len(passing)}")
        for result in passing:
            kinds = ", ".join(f"{name} ({count})" for name, count in result["passing_probes"].most_common())
            print(f"  {result['identifier']} ({result['problems']} problems): {kinds}")
    raise SystemExit(1 if failing else 0)


if __name__ == "__main__":
    main()

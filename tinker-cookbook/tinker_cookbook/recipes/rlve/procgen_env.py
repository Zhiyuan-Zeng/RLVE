from functools import partial
from typing import Sequence, Any, Tuple, Optional, List, Dict
import random
import copy
import json
import os
import sys
from dataclasses import dataclass

# Go up 4 levels from this file to reach RLVE directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import chz
import numpy as np
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    StepResult,
    Trajectory,
    RLDataset,
    RLDatasetBuilder,
    StopCondition
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from Gym.environment import VerifiableEnvironment
from Gym.environments import identifier2environment
from Gym.parameter_controller import ParameterController
from Gym.parameter_controllers import identifier2controller


class RLVEManager :
    def __init__(self, args) :
        self.args = args
        
        assert args["environment_list"], "Environment list is not set."

        self.environment2difficulty = {environment : self.args["init_difficulty"] if args["ablation_fixed_difficulty"] is None else args["ablation_fixed_difficulty"] for environment in args["environment_list"]}
        if args["ablation_fixed_difficulty"] is not None :
            if "ZebraLogics" in args["environment_list"] :
                self.environment2difficulty["ZebraLogics"] = min(args["ablation_fixed_difficulty"], 2)
            if "CardColoringCounting" in args["environment_list"] :
                self.environment2difficulty["CardColoringCounting"] = min(args["ablation_fixed_difficulty"], 5)
        self.environment2accuracy = {environment : dict(accuracy = 0, num_samples = 0) for environment in args["environment_list"]}

        self.problem_generation_seed = 0

    def generate_problem(self) -> Tuple[str, int, Optional[VerifiableEnvironment]] :

        environment : str = random.choice(self.args["environment_list"])

        parameter_controller : ParameterController = identifier2controller[environment]()
        maximum_difficulty : int = self.environment2difficulty[environment]
        parameter_lists : List[List[Dict]] = []
        for problem_difficulty in range(maximum_difficulty + 1) :
            if problem_difficulty > maximum_difficulty - self.args["difficulty_sliding_window_size"] :
                parameter_lists.append((problem_difficulty, copy.deepcopy(parameter_controller.get_parameter_list())))
            parameter_controller.update()

        problem_difficulty, parameter_list = random.choice(parameter_lists)
        parameter : Dict = random.choice(parameter_list)
        problem : VerifiableEnvironment = identifier2environment[environment]()
        if problem.generator(seed = self.problem_generation_seed, parameter = parameter) :
            generated_problem = problem
        else :
            generated_problem = None
            print("Generating problem for environment {} failed\nparameter: {}\n\n\n".format(environment, parameter), flush=True)
        
        self.problem_generation_seed += 1

        return environment, problem_difficulty, generated_problem

    def get_state(self) -> Dict[str, Any] :
        return dict(
            environment2difficulty = self.environment2difficulty,
            environment2accuracy = self.environment2accuracy,
            problem_generation_seed = self.problem_generation_seed,
        )

    def set_state(self, state : Dict[str, Any]) -> None :
        self.environment2difficulty = state["environment2difficulty"]
        self.environment2accuracy = state["environment2accuracy"]
        self.problem_generation_seed = state["problem_generation_seed"]

    def update(self, samples : List[StepResult]) -> Dict[str, Any] :
        """
        Update accuracy statistics based on completed samples.
        Also update the difficulty when necessary.
        This should be called after rewards have been computed.
        """
        log_dict = {}

        for sample in samples :

            environment = sample.metadata["environment"]

            problem_difficulty, maximum_difficulty = sample.metadata["problem_difficulty"], self.environment2difficulty[environment]
            assert problem_difficulty <= maximum_difficulty, "The difficulty of the sample is higher than the current difficulty of the environment, which should not happen."
            if problem_difficulty < maximum_difficulty :
                continue
            self.environment2accuracy[environment]["num_samples"] += 1
            self.environment2accuracy[environment]["accuracy"] += sample.metrics["correct"]

        log_dict["rollout/problem_generation_seed"] = self.problem_generation_seed

        for environment in self.args["environment_list"] :
            num_samples, accuracy = self.environment2accuracy[environment]["num_samples"], self.environment2accuracy[environment]["accuracy"]
            if num_samples >= self.args["min_prompts_before_difficulty_check"] * self.args["n_samples_per_prompt"] :
                accuracy = accuracy / num_samples
                log_dict["RLVE/{}/accuracy".format(environment)] = accuracy

                if accuracy >= self.args["min_metric_to_increase_difficulty"] :
                    if self.args["ablation_fixed_difficulty"] is None :
                        self.environment2difficulty[environment] += 1
                    log_dict["RLVE/{}/difficulty".format(environment)] = self.environment2difficulty[environment]

                self.environment2accuracy[environment] = dict(accuracy = 0, num_samples = 0)

        return log_dict




class ProcgenEnv(ProblemEnv):

    def __init__(
        self,
        environment: str,
        problem_difficulty: int,
        problem: VerifiableEnvironment,
        reward_key: str,
        apply_chat_template: bool,
        answer_marker_type: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0,
    ):
        super().__init__(renderer, convo_prefix)
        self.environment = environment
        self.problem_difficulty = problem_difficulty
        self.problem = problem
        self.reward_key = reward_key
        self.apply_chat_template = apply_chat_template
        self.answer_marker_type = answer_marker_type
        self.renderer
        self.format_coef = format_coef

    def get_question(self):
        return self.problem.prompt_generator()

    def get_reference_answer(self) -> str:
        return self.problem.parameter.get("reference_answer", "")

    def check_answer(self, sample_str: str) -> bool:
        return self.problem.verifier(sample_str)["accuracy"]

    def check_format(self, sample_str: str) -> bool:
        return self.problem.verifier(sample_str)["format_score"]

    def verify(self, response : str) -> Dict[str, Any] :
        # NOTE: This is a hard-coded extraction logics based on template
        if self.answer_marker_type == r"\boxed{}" :
            answer_markers = (r"\boxed{", r"}")
            assert self.apply_chat_template
        elif self.answer_marker_type == r"<answer></answer>" :
            answer_markers = (r"<answer>", r"</answer>")
            assert not self.apply_chat_template
        else :
            raise NotImplementedError(f"Answer marker type {self.answer_marker_type} not implemented.")

        problem : VerifiableEnvironment = identifier2environment[self.environment](answer_markers = answer_markers)
        config = {"seed": self.problem.seed, "parameter": self.problem.parameter}
        if hasattr(self.problem, "passing_reward_threshold"):
            config["passing_reward_threshold"] = self.problem.passing_reward_threshold
        problem.set_config(config)
        return self.problem.verifier(response)

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        TinyZero_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. Show your work in <think> </think> tags, and return the final answer in <answer> </answer> tags."""
        user_prompt = self.get_question()
        if self.apply_chat_template:
            user_prompt = TinyZero_TEMPLATE + "\n\n" + user_prompt
        convo = self.convo_prefix + [
            {"role": "user", "content": user_prompt},
        ]
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action) # action is a list of int, i.e., token ids
        results = self.verify(message["content"])
        format_score = results["format_score"]
        accuracy = results["accuracy"]
        total_reward = results[self.reward_key] # choose from ["reward", "accuracy"]
        total_reward = total_reward + self.format_coef * format_score
        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": format_score,
                "correct": accuracy,
                "problem_difficulty": self.problem_difficulty
            },
            metadata=dict(
                            environment=self.environment,
                            problem_difficulty=self.problem_difficulty,
                            **self.problem.get_config()
                        )
        )


@dataclass(frozen=True)
class ProcgenEnvGroupBuilder(EnvGroupBuilder):
    renderer: renderers.Renderer
    num_envs: int

    rlve_manager: RLVEManager

    async def make_envs(self) -> Sequence[Env]:
        environment, problem_difficulty, problem = self.rlve_manager.generate_problem()
        return [
            ProcgenEnv(environment, 
                       problem_difficulty, 
                       problem, 
                       self.rlve_manager.args["reward_key"], 
                       self.rlve_manager.args["apply_chat_template"], 
                       self.rlve_manager.args["answer_marker_type"],
                       self.renderer)
            for _ in range(self.num_envs)
        ]



@chz.chz
class ProcgenDataset(RLDataset):
    renderer: renderers.Renderer
    n_batches: int
    batch_size: int
    group_size: int

    rlve_manager: RLVEManager

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            ProcgenEnvGroupBuilder(
                rlve_manager=self.rlve_manager,
                renderer=self.renderer,
                num_envs=self.group_size,
            )
            for i in range(self.batch_size)
        ]

    def __len__(self) -> int:
        return self.n_batches

    def save_state(self, log_path: str, name: str) -> str:
        """
        Save the RLVEManager state to a JSON file.

        Args:
            log_path: Directory where the state file will be saved
            name: Name prefix for the checkpoint (e.g., "000100")

        Returns:
            Path to the saved state file
        """
        state = self.rlve_manager.get_state()
        state_path = os.path.join(log_path, f"rlve_manager_{name}.json")

        os.makedirs(log_path, exist_ok=True)

        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        return state_path

    def load_state(self, state_path: str) -> None:
        """
        Load the RLVEManager state from a JSON file.

        Args:
            state_path: Path to the state file
        """
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"State file not found: {state_path}")

        with open(state_path, 'r') as f:
            state = json.load(f)

        self.rlve_manager.set_state(state)




@chz.chz
class ProcgenDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    n_batches: int
    group_size: int
    args: Dict

    async def __call__(self) -> tuple[ProcgenDataset, None]:

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        rlve_manager = RLVEManager(self.args)

        return ProcgenDataset(
            batch_size=self.batch_size,
            n_batches=self.n_batches,
            renderer=renderer,
            group_size=self.group_size,
            rlve_manager=rlve_manager,
        ), None

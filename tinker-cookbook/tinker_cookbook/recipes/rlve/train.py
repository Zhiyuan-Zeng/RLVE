import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Literal

# Go up 4 levels from this file to reach RLVE directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.rlve.procgen_env import ProcgenDatasetBuilder
from tinker_cookbook.rl.train import AsyncConfig, Config, main


from Gym.parameter_controllers import identifier2controller

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Simple command-line configuration for RL training."""

    # Model configuration
    model_name: str = "deepseek-ai/DeepSeek-V3.1"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Environment configuration
    env: str = "rlve"
    seed: int = 0  # Random seed for data shuffling

    # Training hyperparameters
    group_size: int = 16 # n_rollout
    groups_per_batch: int = 128 # train_bsz
    n_batches: int = 1000000 # #total_steps
    learning_rate: float = 1e-5
    max_tokens: int = 16384
    kl_penalty_coef: float = 0.0

    # RLVE configuration
    reward_key: str = "accuracy"
    init_difficulty: int = 0
    difficulty_sliding_window_size: int = 4
    ablation_fixed_difficulty = None
    min_prompts_before_difficulty_check: int = 8
    n_samples_per_prompt: int = group_size
    min_metric_to_increase_difficulty: float = 0.9
    apply_chat_template: bool = True
    answer_marker_type: str = r"\boxed{}" # r"\boxed{}" or r"<answer></answer>"

    def get_procgen_config(self):
        """Generate procgen config with actual CLI values."""
        return {
            "reward_key": self.reward_key,
            "environment_list": list(identifier2controller.keys()),
            "init_difficulty": self.init_difficulty,
            "difficulty_sliding_window_size": self.difficulty_sliding_window_size,
            "ablation_fixed_difficulty": self.ablation_fixed_difficulty,
            "min_prompts_before_difficulty_check": self.min_prompts_before_difficulty_check,
            "n_samples_per_prompt": self.n_samples_per_prompt,
            "min_metric_to_increase_difficulty": self.min_metric_to_increase_difficulty,
            "apply_chat_template": self.apply_chat_template,
            "answer_marker_type": self.answer_marker_type
        }

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = "tinker"
    wandb_name: str | None = None
    compute_post_kl: bool = True

    # Evals
    eval_every: int = 10

    # Checkpointing
    save_every: int = 10

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps_off_policy: int | None = None
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"



async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""

    # Get tokenizer for stop sequences
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    model_name = cli_config.model_name.replace("/", "-")
    model_name_for_log = cli_config.model_name.split("/")[-1]
    run_name = f"{cli_config.env}-{model_name}-{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-{cli_config.group_size}group-{cli_config.groups_per_batch}batch-{cli_config.loss_fn}-seed{cli_config.seed}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    # create log path if it doesn't exist
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"./logs/{model_name_for_log}-lr{cli_config.learning_rate}-bsz{cli_config.groups_per_batch}-offpolicy{cli_config.num_substeps}-init{cli_config.init_difficulty}-swsz{cli_config.difficulty_sliding_window_size}-minprompt{cli_config.min_prompts_before_difficulty_check}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = f"tinker-{model_name_for_log}-lr{cli_config.learning_rate}-bsz{cli_config.groups_per_batch}-offpolicy{cli_config.num_substeps}-init{cli_config.init_difficulty}-swsz{cli_config.difficulty_sliding_window_size}-minprompt{cli_config.min_prompts_before_difficulty_check}"

    # Create full config
    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=ProcgenDatasetBuilder(
        args=cli_config.get_procgen_config(),
        batch_size=cli_config.groups_per_batch,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        n_batches=cli_config.n_batches,
        group_size=cli_config.group_size,
        ),
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
        loss_fn=cli_config.loss_fn,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Run training
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))

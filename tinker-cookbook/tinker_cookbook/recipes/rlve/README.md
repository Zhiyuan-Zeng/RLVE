# Tinker for RLVE

## Installation

Install tinker-cookbook following the official guidance [here](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main?tab=readme-ov-file#installation):

```
1. Sign up for Tinker through the [waitlist](https://thinkingmachines.ai/tinker).
2. Once you have access, create an API key from the [console](https://tinker-console.thinkingmachines.ai) and export it as environment variable `TINKER_API_KEY`.
3. Install tinker python client via `pip install tinker`
4. We recommend installing `tinker-cookbook` in a virtual env either with `conda` or `uv`. For running most examples, you can install via `pip install -e .`.
```

Current implementation is compatible with `tinker==0.3.0`.


## Usage

### Default Setup in Paper
```bash
python -m tinker_cookbook.recipes.rlve.train model_name="Qwen/Qwen3-4B-Instruct-2507"
```

### Advanced Usage
```bash
# This should encourage faster difficulty update
python -m tinker_cookbook.recipes.rlve.train model_name="Qwen/Qwen3-4B-Instruct-2507" \
    init_difficulty=3 \
    difficulty_sliding_window_size=2 \
    min_prompts_before_difficulty_check=1
```

### Key Parameters

**RLVE Configuration:**
- `reward_key`: Reward metric to use (default: "accuracy")
- `init_difficulty`: Initial difficulty level (default: 0)
- `difficulty_sliding_window_size`: Sliding window size for difficulty (default: 4)
- `ablation_fixed_difficulty`: Fixed difficulty for ablation (default: None)
- `min_prompts_before_difficulty_check`: Minimum prompts before checking difficulty (default: 8)
- `n_samples_per_prompt`: Number of samples per prompt (default: group_size)
- `min_metric_to_increase_difficulty`: Minimum metric to increase difficulty (default: 0.9)

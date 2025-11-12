<div align="center">

# RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments
  
Zhiyuan Zeng*, Hamish Ivison*, Yiping Wang*, Lifan Yuan*, Shuyue Stella Li, Zhuorui Ye, Siting Li, Jacqueline He, Runlong Zhou, Tong Chen, Chenyang Zhao, Yulia Tsvetkov, Simon Shaolei Du, Natasha Jaques, Hao Peng, Pang Wei Koh, Hannaneh Hajishirzi
</div>

![Figure1](assets/Figure1.png)

## 🔗 Resources
- 📄 **[Paper](https://arxiv.org/abs/2511.07317)**
- 💾 **[Code & Data](https://github.com/Zhiyuan-Zeng/RLVE)**
- 🤗 **[Models](https://huggingface.co/collections/hamishivi/rlve)**

If you find our work useful, please consider citing:

```bibtex
@article{zeng2025rlve,
  title={RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments},
  author={Zeng, Zhiyuan and Ivison, Hamish and Wang, Yiping and Yuan, Lifan and Li, Shuyue Stella and Ye, Zhuorui and Li, Siting and He, Jacqueline and Zhou, Runlong and Chen, Tong and Zhao, Chenyang and Tsvetkov, Yulia and Du, Simon Shaolei and Jaques, Natasha and Peng, Hao and Koh, Pang Wei and Hajishirzi, Hannaneh},
  journal={arXiv preprint 2511.07317},
  year={2025}
}
```

## Bug Reports & Questions

If you have any questions about the code or the paper, feel free to contact [Zhiyuan Zeng](https://zhiyuan-zeng.github.io/) (`zhiyuan1zeng@gmail.com` or `zyzeng@cs.washington.edu`).

If you encounter any issues while using the code or want to report a bug, please open an issue. When reporting a problem, provide detailed information so we can assist you more effectively.

## Setup

Our environment setup is the same as the setup of the official [slime](https://github.com/THUDM/slime) framework. We use Docker:

```bash
# Pull the image
docker pull slimerl/slime:v0.5.0rc0-cu126

# Start the container
docker run -d --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --name RLVE \
  slimerl/slime:v0.5.0rc0-cu126 \
  tail -f /dev/null
```

After entering the Docker container, please follow these steps to clone the repository and install it:

```bash
cd /root/
git clone https://github.com/Zhiyuan-Zeng/RLVE.git
cd RLVE
pip install -e .
```

## Usage of Verifiable Environments in RLVE-Gym

Each environment has an identifier, which you can find [here](Gym/environments/__init__.py). We take the following example:

```python
identifier = "Sorting"
```

Each environment has two classes - one for the environment itself and another for controlling the problem difficulty:

```python
from Gym.environments import identifier2environment
from Gym.parameter_controllers import identifier2controller

environment = identifier2environment[identifier]
controller_class = identifier2controller[identifier]
```

Note that when creating the environment, you can specify the argument `answer_markers` to control how the answer is extracted from a model’s output during verification.
By default, it extracts the answer enclosed between `<answer>` and `</answer>`.

When generating a problem, we specify a random seed `seed` and a problem difficulty level `d` ($d \geq 0$):

```python
import random

seed = 42

controller = controller_class()
d = 3
for _ in range(d) :
    controller.update()

problem = environment()
parameter = random.choice(controller.get_parameter_list())
if problem.generator(seed = seed, parameter = parameter) :
    pass
else :
    problem = None
```

Problem generation is successful when `problem` is not `None`.

The input prompt is obtained via `problem.prompt_generator()`.  
The reward for a model output `output` is given by `problem.scorer(output)`, or you can access more detailed verification information using `problem.verifier(output)`.

You can check [here](Gym/environment.py) for more detailed usage.

## Model Download & Weight Conversion

In the following, we assume you are on a node with 8 GPUs, each equipped with at least 80GB of memory.
We recommend using H100 or H200 GPUs over A100s, due to recent findings regarding floating-point precision issues observed on A100 hardware.

We download models from Hugging Face and convert their weights from the Hugging Face format to the Megatron format:

```bash
# Qwen2.5-7B-Base
hf download Qwen/Qwen2.5-7B --local-dir ../Qwen2.5-7B
source scripts/models/qwen2.5-7B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ../Qwen2.5-7B \
    --save ../Qwen2.5-7B_torch_dist

# R1-Distill-Qwen-1.5B
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ../DeepSeek-R1-Distill-Qwen-1.5B
source scripts/models/deepseek-r1-distill-qwen-1.5B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ../DeepSeek-R1-Distill-Qwen-1.5B \
    --save ../DeepSeek-R1-Distill-Qwen-1.5B_torch_dist

# DeepScaleR-1.5B
hf download agentica-org/DeepScaleR-1.5B-Preview --local-dir ../DeepScaleR-1.5B-Preview
source scripts/models/deepseek-r1-distill-qwen-1.5B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ../DeepScaleR-1.5B-Preview \
    --save ../DeepScaleR-1.5B-Preview_torch_dist

# ProRL-1.5B-v2
hf download nvidia/Nemotron-Research-Reasoning-Qwen-1.5B --revision v2 --local-dir ../Nemotron-Research-Reasoning-Qwen-1.5B-v2
source scripts/models/deepseek-r1-distill-qwen-1.5B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ../Nemotron-Research-Reasoning-Qwen-1.5B-v2 \
    --save ../Nemotron-Research-Reasoning-Qwen-1.5B-v2_torch_dist

# OpenThinker3-1.5B
hf download open-thoughts/OpenThinker3-1.5B --local-dir ../OpenThinker3-1.5B
source scripts/models/qwen2.5-1.5B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ../OpenThinker3-1.5B \
    --save ../OpenThinker3-1.5B_torch_dist
```

**Warning:** Please make sure the weight conversion is successful before starting any training.
Even if the conversion fails, training can still launch without a runtime error, but the model will be trained from randomly initialized weights.


## Reward Shaping

By default, we use the original reward provided by each verifiable environment as the reward used in the RL algorithm.  
Many environments employ continuous reward shaping, meaning that a model output can still receive a positive reward even if its answer is not fully correct.  
We found that when the training scale is small, using a binary reward can sometimes yield slightly better results (i.e., the model output receives 1 if and only if its answer is fully correct, and 0 otherwise).  
You can switch to binary rewards by changing `--reward-key reward` to `--reward-key accuracy` in `scripts/training/[MODEL]/rlve.sh`.

## Wandb Setup

To use Weights & Biases (wandb), please set your wandb API key in the environment variable `WANDB_API_KEY`.  
Alternatively, you can comment out the wandb-related arguments in `WANDB_ARGS` within the training script to disable wandb.

## Environment Scaling

You can run our environment scaling experiments as follows:

```bash
# Qwen2.5-7B-Base
bash scripts/training/Qwen2.5-7B/rlve/num-environment=1.sh RLVE
bash scripts/training/Qwen2.5-7B/rlve/num-environment=4.sh RLVE
bash scripts/training/Qwen2.5-7B/rlve/num-environment=16.sh RLVE
bash scripts/training/Qwen2.5-7B/rlve/num-environment=256.sh RLVE

# R1-Distill-Qwen-1.5B
bash scripts/training/DeepSeek-R1-Distill-Qwen-1.5B/rlve/num-environment=1.sh RLVE
bash scripts/training/DeepSeek-R1-Distill-Qwen-1.5B/rlve/num-environment=4.sh RLVE
bash scripts/training/DeepSeek-R1-Distill-Qwen-1.5B/rlve/num-environment=16.sh RLVE
bash scripts/training/DeepSeek-R1-Distill-Qwen-1.5B/rlve/num-environment=256.sh RLVE

# DeepScaleR-1.5B
bash scripts/training/DeepScaleR-1.5B-Preview/rlve/num-environment=1.sh RLVE
bash scripts/training/DeepScaleR-1.5B-Preview/rlve/num-environment=4.sh RLVE
bash scripts/training/DeepScaleR-1.5B-Preview/rlve/num-environment=16.sh RLVE
bash scripts/training/DeepScaleR-1.5B-Preview/rlve/num-environment=256.sh RLVE

# ProRL-1.5B-v2
bash scripts/training/Nemotron-Research-Reasoning-Qwen-1.5B-v2/rlve/num-environment=1.sh RLVE
bash scripts/training/Nemotron-Research-Reasoning-Qwen-1.5B-v2/rlve/num-environment=4.sh RLVE
bash scripts/training/Nemotron-Research-Reasoning-Qwen-1.5B-v2/rlve/num-environment=16.sh RLVE
bash scripts/training/Nemotron-Research-Reasoning-Qwen-1.5B-v2/rlve/num-environment=256.sh RLVE
```

You can use `scripts/evaluation/[MODEL]/eval_HELD-OUT_ENVIRONMENTS.sh` to evaluate a checkpoint on the test set built from the 50 held-out environments. For example:

```bash
# Qwen2.5-7B-Base
bash scripts/evaluation/Qwen2.5-7B/eval_HELD-OUT_ENVIRONMENTS.sh "../Qwen2.5-7B"
bash scripts/evaluation/Qwen2.5-7B/eval_HELD-OUT_ENVIRONMENTS.sh "../[Qwen2.5-7B]_[num-environment=256]/iter_0000016"

# R1-Distill-Qwen-1.5B
bash scripts/evaluation/DeepSeek-R1-Distill-Qwen-1.5B/eval_HELD-OUT_ENVIRONMENTS.sh "../DeepSeek-R1-Distill-Qwen-1.5B"
bash scripts/evaluation/DeepSeek-R1-Distill-Qwen-1.5B/eval_HELD-OUT_ENVIRONMENTS.sh "../[DeepSeek-R1-Distill-Qwen-1.5B]_[num-environment=256]/iter_0000016"

# DeepScaleR-1.5B
bash scripts/evaluation/DeepScaleR-1.5B-Preview/eval_HELD-OUT_ENVIRONMENTS.sh "../DeepScaleR-1.5B-Preview"
bash scripts/evaluation/DeepScaleR-1.5B-Preview/eval_HELD-OUT_ENVIRONMENTS.sh "../[DeepScaleR-1.5B-Preview]_[num-environment=256]/iter_0000016"

# ProRL-1.5B-v2
bash scripts/evaluation/Nemotron-Research-Reasoning-Qwen-1.5B-v2/eval_HELD-OUT_ENVIRONMENTS.sh "../Nemotron-Research-Reasoning-Qwen-1.5B-v2"
bash scripts/evaluation/Nemotron-Research-Reasoning-Qwen-1.5B-v2/eval_HELD-OUT_ENVIRONMENTS.sh "../[Nemotron-Research-Reasoning-Qwen-1.5B-v2]_[num-environment=256]/iter_0000016"
```

## RL Training from ProRL-1.5B-v2

You can train ProRL-1.5B-v2 using RLVE with joint training across all 400 verifiable environments:

```bash
bash scripts/training/Nemotron-Research-Reasoning-Qwen-1.5B-v2/rlve/num-environment=400.sh RLVE
```

| Benchmark | AIME 2024 (Avg@64) | AIME 2025 (Avg@64) | OMEGA-500 (Avg@4) | OlympiadBench (Avg@4) | BBEH (Avg@4) | LiveCodeBench-v6 (Pass@8) |
|:-----------|:------------------:|:------------------:|:-----------------:|:----------------------:|:-------------:|:--------------------------:|
| [Nemotron-Research-Reasoning-Qwen-1.5B-v2](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B) | 51.93 | 33.96 | 24.15 | 57.85 | 7.27 | 25.90 |
| [Nemotron-Research-Reasoning-Qwen-1.5B-v2-RLVE](https://huggingface.co/hamishivi/Nemotron-Research-Reasoning-Qwen-1.5B-v2-RLVE) | **56.51** | **39.84** | **27.75** | **60.56** | **9.10** | **27.49** |

The training checkpoint at step 740 is available [here](https://huggingface.co/hamishivi/Nemotron-Research-Reasoning-Qwen-1.5B-v2-RLVE), corresponding to the final checkpoint reported in our paper (costing approximately 1.1K H100 GPU hours).

You can use `bash scripts/evaluation/Nemotron-Research-Reasoning-Qwen-1.5B-v2/eval_BENCHMARKS.sh` to evaluate a checkpoint on the five benchmarks AIME24/25, OMEGA-500, OlympiadBench, LiveCodeBench, and BBEH, and you can use `bash scripts/evaluation/Nemotron-Research-Reasoning-Qwen-1.5B-v2/eval_LiveCodeBench.sh` to evaluate a checkpoint specifically on LiveCodeBench (v6):

```bash
bash scripts/evaluation/Nemotron-Research-Reasoning-Qwen-1.5B-v2/eval_BENCHMARKS.sh "../[Nemotron-Research-Reasoning-Qwen-1.5B-v2]_[num-environment=400]/iter_0000016"
bash scripts/evaluation/Nemotron-Research-Reasoning-Qwen-1.5B-v2/eval_LiveCodeBench.sh "../[Nemotron-Research-Reasoning-Qwen-1.5B-v2]_[num-environment=400]/iter_0000016"
```

## RL Training from OpenThinker3-1.5B

You can train OpenThinker3-1.5B using RLVE with joint training across all 400 verifiable environments:

```bash
bash scripts/training/OpenThinker3-1.5B/rlve/num-environment=400.sh RLVE
```

The training checkpoint at step 400 is available [here](https://huggingface.co/hamishivi/OpenThinker3-1.5B-RLVE), corresponding to the final checkpoint reported in our paper.

| Benchmark | AIME 2024 (Avg@64) | AIME 2025 (Avg@64) | OMEGA-500 (Avg@4) | OlympiadBench (Avg@4) | BBEH (Avg@4) | LiveCodeBench-v6 (Pass@8) |
|:-----------|:------------------:|:------------------:|:-----------------:|:----------------------:|:-------------:|:--------------------------:|
| [OpenThinker3-1.5B](https://huggingface.co/open-thoughts/OpenThinker3-1.5B) | 54.32 | 42.03 | 25.15 | 56.85 | 4.00 | 28.17 |
| [OpenThinker3-1.5B-RLVE](https://huggingface.co/hamishivi/OpenThinker3-1.5B-RLVE) | **58.18** | **49.90** | **29.45** | **62.67** | **7.13** | **34.07** |

You can also train OpenThinker3-1.5B using RLVR on the DeepMath-103K dataset:

```bash
bash scripts/training/OpenThinker3-1.5B/rlvr-math/DeepMath-103K.sh RLVE
```

You can use `bash scripts/evaluation/OpenThinker3-1.5B/eval_BENCHMARKS.sh` to evaluate a checkpoint on the five benchmarks AIME24/25, OMEGA-500, OlympiadBench, LiveCodeBench, and BBEH, and you can use `bash scripts/evaluation/OpenThinker3-1.5B/eval_LiveCodeBench.sh` to evaluate a checkpoint specifically on LiveCodeBench (v6):

```bash
bash scripts/evaluation/OpenThinker3-1.5B/eval_BENCHMARKS.sh "../[OpenThinker3-1.5B]_[num-environment=400]/iter_0000016"
bash scripts/evaluation/OpenThinker3-1.5B/eval_LiveCodeBench.sh "../[OpenThinker3-1.5B]_[num-environment=400]/iter_0000016"
```

## RLVE on Tinker
First run `cd tinker-cookbook` to enter the directory, and then check out the [README](https://github.com/Zhiyuan-Zeng/RLVE/blob/main/tinker-cookbook/tinker_cookbook/recipes/rlve/README.md) for details on installation and usage of RLVE recipe.
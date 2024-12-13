# Cutting Stock Problem with Reinforcement Learning

A project implementing reinforcement learning solutions for the cutting stock optimization problem using A2C and PPO algorithms.

## Installation

1. Requires Python 3.11.5 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt

**Training**
The project supports training two reinforcement learning algorithms:

1. Advantage Actor-Critic (A2C)
2. Proximal Policy Optimization (PPO)
```bash
python main.py

To train a model:
```python
POLICY_TYPE = "ppo"  # Options: "a2c", "ppo"
RUNNING_TYPE = "train"
NUM_EPISODES = 10  # Adjust number of training episodes

Models are automatically saved:
Best performing model: model_ppo_best.pt or model_a2c_best.pt
Periodic saves every 10 episodes: model_ppo_episode_X.pt

**Benchmarking**
To benchmark trained models:

1. Set parameters in [main.py](main.py):

```python
RUNNING_TYPE = "benchmark"
NUM_EPISODES = 10  # Number of evaluation episodes

2. Run
```bash
python main.py
Benchmark results are saved to:

Individual logs: [benchmark_A2C.txt](benchmark_A2C.txt), [benchmark_PPO.txt](benchmark_PPO.txt)
Comparative visualization: [policy_comparison.png](policy_comparison.png)

**Model Loading**

To load and test a saved model:

```python
policy = ProximalPolicyOptimization()  # or ActorCriticPolicy2()
policy.load_model("saved_models/model_ppo_best.pt")

**Visualization**
The environment can be rendered by enabling:
```python

env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human"
)


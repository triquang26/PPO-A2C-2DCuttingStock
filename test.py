import gym_cutting_stock
import gymnasium as gym
from student_submissions.s2210xxx.ActorCriticPolicy import ActorCriticPolicy
from student_submissions.s2210xxx.ProximalPolicyOptimization import ProximalPolicyOptimization
# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",
)

# Initialize policy and load saved model
# policy = ActorCriticPolicy()
# model_loaded = policy.load_model("model_actor_critics_interrupted_step_26.pt")

# Initialize policy and load saved model
policy = ProximalPolicyOptimization()
model_loaded = policy.load_model("model_ppo_best.pt")  # hoặc model khác như model_ppo_step_100.pt


if not model_loaded:
    print("Failed to load model")
    exit()

print("Model loaded successfully, starting evaluation...")
policy.training = False  # Set to evaluation mode

# Test the loaded model
observation, info = env.reset(seed=1000)  # Use a fixed seed for testing
total_reward = 0
num_steps = 0

try:
    while True:
        action = policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        num_steps += 1
        
        print(f"Step {num_steps}")
        print(f"Action taken: {action}")
        print(f"Reward: {reward:.2f}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Filled ratio: {info['filled_ratio']:.3f}")
        print("-" * 50)
        
        if terminated or truncated:
            print("\nEpisode finished!")
            print(f"Final filled ratio: {info['filled_ratio']:.3f}")
            print(f"Total steps: {num_steps}")
            print(f"Total reward: {total_reward:.2f}")
            break

except KeyboardInterrupt:
    print("\nTest interrupted by user")
    print(f"Steps completed: {num_steps}")
    print(f"Total reward: {total_reward:.2f}")

finally:
    env.close()

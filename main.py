import gym_cutting_stock
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.ProximalPolicyOptimization import ProximalPolicyOptimization
from student_submissions.s2210xxx.A2C import ActorCriticPolicy2
from student_submissions.s2210xxx.EpisodeEvaluator import EpisodeEvaluator
import signal
import sys
# Add global variable for graceful shutdown
should_exit = False

def signal_handler(signum, frame):
    """Handle Ctrl+C signal"""
    global should_exit
    print('\nSignal received. Preparing to exit gracefully...')
    should_exit = True

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def calculate_metrics(observation):
    """Calculate trimloss and fill ratio from observation"""
    total_stocks = len(observation['stocks'])
    used_stocks = 0
    total_used_area = 0
    total_stock_area = 0
    
    for stock in observation['stocks']:
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        stock_area = stock_w * stock_h
        used_area = np.sum(stock >= 0)  # Count non-empty cells
        

        # Check if stock is used
        if used_area > 0:
            used_stocks += 1
            total_used_area += used_area
            total_stock_area += stock_area
    
    # Calculate trimloss (ratio of unused area in used stocks)
    trimloss = (total_stock_area - total_used_area) / total_stock_area if total_stock_area > 0 else 0.0
    
    # Calculate fill ratio (ratio of used stocks to total stocks)
    fill_ratio = used_stocks / total_stocks if total_stocks > 0 else 0.0
    
    return trimloss, fill_ratio

def ppo_policy(env, policy, num_episodes=100):
    evaluator = EpisodeEvaluator()
    log_file = open('training_log.txt', 'w')
    
    def log_info(message, console=True):
        log_file.write(message + '\n')
        log_file.flush()
        if console:
            print(message)

    observation, info = env.reset(seed=42)
    log_info("Initial info: " + str(info))

    # Training variables
    ep = 0
    best_reward = float('-inf')
    total_steps = 0

    try:
        while ep < num_episodes and not should_exit:
            episode_reward = 0
            step = 0
            final_observation = None
            
            # Training phase
            while True and not should_exit:
                action = policy.get_action(observation, info)
                next_observation, env_reward, terminated, truncated, info = env.step(action)
                calculated_reward = policy.calculate_reward(action, observation, info)
                episode_reward += calculated_reward
                policy.update_policy(calculated_reward, terminated or truncated)
                
                final_observation = next_observation
                observation = next_observation
                step += 1
                total_steps += 1

                if terminated or truncated:
                    # Only log if all products have been processed
                    remaining_products = sum(prod["quantity"] for prod in observation["products"])
                    if remaining_products == 0:
                        policy.log_episode_summary(
                            steps=step,
                            filled_ratio=info['filled_ratio'],
                            episode_reward=episode_reward,
                            observation=observation
                        )
                    
                    # Save model if it's the best so far
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        log_info("\nNew best reward achieved! Saving model...")
                        if isinstance(policy, ProximalPolicyOptimization):
                            policy.save_model("model_ppo_best")
                        elif isinstance(policy, ActorCriticPolicy2):
                            policy.save_model("model_a2c_best")
                    
                    # Save model every 10 episodes
                    if ep > 0 and ep % 10 == 0:
                        log_info(f"\nSaving model at episode {ep}...")
                        if isinstance(policy, ProximalPolicyOptimization):
                            policy.save_model("model_ppo_best")
                        elif isinstance(policy, ActorCriticPolicy2):
                            policy.save_model("model_a2c_best")
                    
                    observation, info = env.reset(seed=ep)
                    ep += 1
                    break
            
            
            # Add after printing final_observation['stocks']:
            # trimloss, fill_ratio = calculate_metrics(final_observation)
            # print(f"Episode TrimLoss: {trimloss:.3f}")
            # print(f"Episode Fill Ratio: {fill_ratio:.3f}")
            
            eval_data = {
                'episode_number': ep,
            }

            ep_summary = evaluator.evaluate_episode(final_observation, eval_data)
            print(ep_summary)

            # Evaluation phase every 10 episodes
            if ep > 0 and ep % 10 == 0:
                log_info("\nRunning evaluation phase...")
                policy.training = False  # Switch to evaluation mode
                
                # Run single evaluation episode
                eval_observation, eval_info = env.reset(seed=1000)
                eval_data = {
                    'episode_number': ep,
                    'steps': 0,
                    'total_reward': 0
                }
                
                while True:
                    eval_action = policy.get_action(eval_observation, eval_info)
                    eval_next_obs, _, eval_terminated, eval_truncated, eval_info = env.step(eval_action)
                    
                    # Calculate reward using policy's method
                    eval_reward = policy.calculate_reward(eval_action, eval_observation, eval_info)
                    eval_data['total_reward'] += eval_reward
                    eval_data['steps'] += 1
                    
                    eval_observation = eval_next_obs
                    
                    if eval_terminated or eval_truncated:
                        # Get evaluation summary using evaluator
                        summary = evaluator.evaluate_episode(eval_observation, eval_data)
                        print(summary)
                        log_info(summary)
                        break
                
                policy.training = True  # Switch back to training mode
        evaluator.plot_metrics()
    except Exception as e:
        log_info(f"Error occurred: {str(e)}")
        policy.save_model(f"model_ppo_error_step_{total_steps}")
        raise e
    finally:
        log_file.close()
        
    # Run final test
    # if not should_exit:
    #     test_policy(env, policy)


def simple_policy_run(env, policy, num_steps=1000):
    """
    Run a simple policy (random/greedy) without training
    
    Args:
        env: Gymnasium environment
        policy: Policy object implementing get_action()
        num_steps: Number of steps to run
    """
    observation, info = env.reset(seed=42)
    total_reward = 0
    
    for step in range(num_steps):
        if should_exit:
            break
            
        action = policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 100 == 0:
            print(f"Step {step}, Current Reward: {reward:.2f}, Total: {total_reward:.2f}")
            print(f"Current filled ratio: {info['filled_ratio']:.3f}")
        
        if terminated or truncated:
            print("\n" + "="*50)
            print(f"Episode finished after {step} steps")
            print(f"Final filled ratio: {info['filled_ratio']:.3f}")
            print(f"Total reward: {total_reward:.2f}")
            print("="*50 + "\n")
            observation, info = env.reset()
            total_reward = 0

# def test_policy(env, policy, num_episodes=5):
#     """
#     Test the trained policy.
    
#     Args:
#         env: Gymnasium environment
#         policy: Trained Policy object
#         num_episodes: Number of test episodes
#     """
#     print("\nTesting trained policy...")
#     policy.training = False
#     log_file = open('test_episode_log.txt', 'w')
#     evaluator = EpisodeEvaluator()
    
#     for episode in range(num_episodes):
#         observation, info = env.reset(seed=1000 + episode)
#         episode_data = {
#             'episode_number': episode + 1,
#             'trimloss': 0.0,       # Initialize trimloss
#             'fill_ratio': 0.0      # Initialize fill_ratio
#         }
#         done = False
        
#         while not done and not should_exit:
#             action = policy.get_action(observation, info)
#             next_obs, reward, terminated, truncated, info = env.step(action)
            
#             # Use policy's reward calculation instead of env reward
#             calculated_reward = policy.calculate_reward(action, observation, info)
#             episode_data['total_reward'] += calculated_reward
            
#             # Calculate waste for this step
#             waste_metrics = evaluator.calculate_waste({'stocks': [observation['stocks'][action['stock_idx']]]})
            
#             # Update trimloss and fill_ratio from info
#             ep_trimloss = info.get('trimloss', 0.0)
#             ep_fill_ratio = info.get('fill_ratio', 0.0)
#             print("Episode TrimLoss: ", ep_trimloss)
#             print("Episode Fill Ratio: ", ep_fill_ratio)

            
    #         episode_data['trimloss'] += info.get('trimloss', 0.0)
    #         episode_data['fill_ratio'] = info.get('fill_ratio', 0.0)  # Assuming fill_ratio is updated each step
            
    #         # Log step details
    #         step_log = f"\nEpisode {episode + 1}, Step {episode_data['steps'] + 1}:\n"
    #         step_log += f"Action: {action}\n"
    #         step_log += f"Step Reward: {calculated_reward:.3f}\n"
    #         step_log += f"Current Filled Ratio: {episode_data['fill_ratio']:.3f}\n"
    #         step_log += f"Step Waste: {waste_metrics.get('total_waste', 0)}\n"
    #         step_log += f"Episode Total Reward: {episode_data['total_reward']:.3f}\n"
    #         step_log += f"Episode TrimLoss: {episode_data['trimloss']:.3f}\n"
    #         step_log += "-" * 50 + "\n"
            
    #         log_file.write(step_log)
            
    #         episode_data['steps'] += 1
    #         observation = next_obs
    #         done = terminated or truncated
        
    #     # Calculate average trimloss per episode
    #     if episode_data['steps'] > 0:
    #         episode_data['trimloss'] /= episode_data['steps']
        
    #     # Evaluate and log episode summary
    #     summary = evaluator.evaluate_episode(observation, info, episode_data)     
    #     log_file.write(summary + "\n")
    #     print(summary)
        
    #     if done:
    #         ep += 1

    # log_file.close()

def benchmark_policies(env, policies, num_episodes=5, seed=1000):
    evaluators = {name: EpisodeEvaluator() for name in policies.keys()}
    
    for policy_name, policy in policies.items():
        print(f"\nTesting {policy_name}...")
        env.reset(seed=seed)
        policy.training = False
        log_file = open(f'benchmark_{policy_name}.txt', 'w')
        
        for episode in range(num_episodes):
            observation, info = env.reset(seed=seed + episode)
            episode_data = {
                'episode_number': episode + 1,
                'steps': 0,
                'trimloss': 0
            }
            done = False
            
            while not done:
                action = policy.get_action(observation, info)
                next_obs, reward, terminated, truncated, info = env.step(action)
                observation = next_obs
                episode_data['steps'] += 1
                done = terminated or truncated
            
            # Evaluate episode
            eval_data = {
                'episode_number': episode,
            }
            summary = evaluators[policy_name].evaluate_episode(observation, eval_data)
            log_file.write(summary + "\n")
            print(summary)
        
        log_file.close()
    
    # Plot comparative results
    plt.figure(figsize=(15, 5))
    
    # Plot filled ratios
    plt.subplot(1, 2, 1)
    for name, evaluator in evaluators.items():
        plt.plot(evaluator.history['episode_numbers'], 
                evaluator.history['filled_ratios'], 
                label=name)
    plt.title('Filled Ratio Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Filled Ratio')
    plt.grid(True)
    plt.legend()
    
    # Plot trim losses
    plt.subplot(1, 2, 2)
    for name, evaluator in evaluators.items():
        plt.plot(evaluator.history['episode_numbers'], 
                evaluator.history['trim_losses'], 
                label=name)
    plt.title('Trim Loss Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Trim Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('policy_comparison.png')
    plt.close()

if __name__ == "__main__":
    # Constants
    NUM_EPISODES = 10

    # Create environment
    env = gym.make(
        "gym_cutting_stock/CuttingStock-v0",
        render_mode="human",  # Comment this line to disable rendering
    )
    
    # Choose which policy to run
    POLICY_TYPE = "ppo"  # Options: "actor_critic", "ppo", "dqn", "random", "greedy"
    RUNNING_TYPE = "benchmark"  # Options: "train", "test", "benchmark"

    if RUNNING_TYPE == "train":
        if POLICY_TYPE == "a2c":
            policy = ActorCriticPolicy2()
            train_func = ppo_policy
        elif POLICY_TYPE == "ppo":
            policy = ProximalPolicyOptimization()
            train_func = ppo_policy
            # Run training and testing
        train_func(
                env=env,
                policy=policy,
                num_episodes=NUM_EPISODES,
        )
        # elif POLICY_TYPE == "random":
        #     policy = RandomPolicy()
        #     simple_policy_run(env=env, policy=policy)
        # elif POLICY_TYPE == "greedy":
        #     policy = GreedyPolicy()
        #     simple_policy_run(env=env, policy=policy)

    elif RUNNING_TYPE == "benchmark":
        # Run benchmarking
            # Load trained policies
        a2c_policy = ActorCriticPolicy2()
        a2c_policy.load_model("model_a2c_best")
        
        ppo_policy = ProximalPolicyOptimization()
        ppo_policy.load_model("model_ppo_best")

    # Create policies dictionary for benchmarking
        policies = {
            'A2C': a2c_policy,
            'PPO': ppo_policy
        }
        benchmark_policies(env, policies, NUM_EPISODES)
    # else:
    #     # Load trained policy
    #     if POLICY_TYPE == "a2c":
    #         policy = ActorCriticPolicy2()
    #         policy.load_model("model_a2c_best")
    #     elif POLICY_TYPE == "ppo":
    #         policy = ProximalPolicyOptimization()
    #         policy.load_model("model_ppo_best")

        
    #     test_policy(env, policy, num_episodes=5)
    
    env.close()


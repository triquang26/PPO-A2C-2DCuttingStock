import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.ActorCriticPolicy import ActorCriticPolicy
from student_submissions.s2210xxx.ProximalPolicyOptimization import ProximalPolicyOptimization
from student_submissions.s2210xxx.DeepQNetworkPolicy import DeepQNetworkPolicy
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

def actor_critics_policy(env, policy, num_episodes=100, logging_interval=300, save_interval=100):
    """
    Train and test an actor-critic policy
    
    Args:
        env: Gymnasium environment
        policy: Policy object implementing get_action() and update_policy()
        num_episodes: Number of training episodes
        logging_interval: Steps between detailed logging
        save_interval: Steps between model saves
    """
    # Reset environment
    observation, info = env.reset(seed=42)
    print("Initial info:", info)


    # Training variables
    ep = 0
    best_reward = float('-inf')
    episode_rewards = []
    total_steps = 0

    try:
        while ep < num_episodes and not should_exit:
            episode_reward = 0
            step = 0
            
            while True and not should_exit:
                # Get action from policy
                action = policy.get_action(observation, info)
                
                # Take action in environment
                next_observation, env_reward, terminated, truncated, info = env.step(action)
                
                # Calculate reward using policy's method (giống PPO)
                calculated_reward = policy.calculate_reward(action, observation, info)
                episode_reward += calculated_reward
                
                # Update policy với calculated reward
                policy.update_policy(calculated_reward, terminated or truncated)
                
                # Move to next state
                observation = next_observation
                step += 1
                total_steps += 1
                
                # Save model every SAVE_INTERVAL steps
                if total_steps % save_interval == 0:
                    print(f"\nSaving model at step {total_steps}...")
                    policy.save_model(f"model_actor_critics_step_{total_steps}")
                
                # Detailed logging every LOGGING_INTERVAL steps
                if step % logging_interval == 0:
                    print("\n" + "="*50)
                    print(f"Episode {ep} - Step {step} Status:")
                    print(f"Current filled ratio: {info['filled_ratio']:.3f}")
                    print("\nProducts remaining:")
                    for i, prod in enumerate(observation['products']):
                        if prod['quantity'] > 0:
                            print(f"Product {i}: size={prod['size']}, quantity={prod['quantity']}")
                    print("\nCurrent reward status:")
                    print(f"Episode reward so far: {episode_reward:.2f}")
                    print("="*50 + "\n")
                
                # Regular progress printing
                elif step % 100 == 0:
                    print(f"Episode {ep}, Step {step}, Total Steps {total_steps}")
                    print(f"Episode Reward (calculated): {episode_reward:.2f}")
                    print(f"Last Step Reward (calculated): {calculated_reward:.2f}")
                
                if terminated or truncated:
                    episode_rewards.append(episode_reward)
                    avg_reward = sum(episode_rewards[-10:]) / min(len(episode_rewards), 10)
                    
                    print("\n" + "="*50)
                    print(f"Episode {ep} FINISHED:")
                    print(f"Total steps: {step}")
                    print(f"Final filled ratio: {info['filled_ratio']:.3f}")
                    print("\nFinal products state:")
                    for i, prod in enumerate(observation['products']):
                        print(f"Product {i}: size={prod['size']}, quantity={prod['quantity']}")
                    print(f"\nEpisode reward: {episode_reward:.2f}")
                    print(f"Average reward (last 10): {avg_reward:.2f}")
                    print(f"Best reward so far: {best_reward:.2f}")
                    print("="*50 + "\n")
                    
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        # Save best model
                        policy.save_model("model_actor_critics_best")
                    
                    # Reset for next episode
                    observation, info = env.reset(seed=ep)
                    ep += 1
                    break

        # Save final model
        if should_exit:
            print("Saving model before exit...")
            policy.save_model(f"model_actor_critics_interrupted_step_{total_steps}")
            print("Model saved. Exiting...")
        else:
            policy.save_model("model_actor_critics_final")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Attempting to save model...")
        policy.save_model(f"model_actor_critics_error_step_{total_steps}")
        raise e

    # After training, set to evaluation mode
    print("\nTesting trained policy...")
    policy.training = False
    
    # Test the trained policy
    observation, info = env.reset(seed=1000)  # New seed for testing
    total_reward = 0
    
    for _ in range(200):  # Test episodes
        if should_exit:
            break
            
        action = policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Test step reward: {reward:.2f}, Total: {total_reward:.2f}")
        
        if terminated or truncated:
            print("Test episode finished")
            print("Final info:", info)
            observation, info = env.reset()

def ppo_policy(env, policy, num_episodes=100, logging_interval=300, save_interval=100):
    """
    Train and test a PPO policy
    
    Args:
        env: Gymnasium environment
        policy: Policy object implementing get_action() and update_policy()
        num_episodes: Number of training episodes
        logging_interval: Steps between detailed logging
        save_interval: Steps between model saves
    """
    # Reset environment
    observation, info = env.reset(seed=42)
    print("Initial info:", info)

    # Training variables
    ep = 0
    best_reward = float('-inf')
    episode_rewards = []
    total_steps = 0

    try:
        while ep < num_episodes and not should_exit:
            episode_reward = 0
            step = 0
            
            while True and not should_exit:
                # Get action from policy
                action = policy.get_action(observation, info)
                
                # Take action in environment
                next_observation, env_reward, terminated, truncated, info = env.step(action)

                # Calculate reward using your policy's method
                calculated_reward = policy.calculate_reward(action, observation, info)

                # Use calculated reward instead of environment reward
                episode_reward += calculated_reward

                # Update policy with calculated reward
                policy.update_policy(calculated_reward, terminated or truncated)
                
                # Move to next state
                observation = next_observation
                step += 1
                total_steps += 1
                
                # Save model every SAVE_INTERVAL steps
                if total_steps % save_interval == 0:
                    print(f"\nSaving model at step {total_steps}...")
                    policy.save_model(f"model_ppo_step_{total_steps}")
                
                # Detailed logging every LOGGING_INTERVAL steps
                if step % logging_interval == 0:
                    print("\n" + "="*50)
                    print(f"Episode {ep} - Step {step} Status:")
                    print(f"Current filled ratio: {info['filled_ratio']:.3f}")
                    print("\nProducts remaining:")
                    for i, prod in enumerate(observation['products']):
                        if prod['quantity'] > 0:
                            print(f"Product {i}: size={prod['size']}, quantity={prod['quantity']}")
                    print("\nCurrent reward status:")
                    print(f"Episode reward so far: {episode_reward:.2f}")
                    print("="*50 + "\n")
                
                # Regular progress printing
                elif step % 100 == 0:
                    print(f"Episode {ep}, Step {step}, Total Steps {total_steps}")
                    print(f"Episode Reward (from env): {episode_reward:.2f}")
                    print(f"Last Step Reward (calculated): {calculated_reward:.2f}")
                
                if terminated or truncated:
                    episode_rewards.append(episode_reward)  # Store the total episode reward
                    avg_reward = sum(episode_rewards[-10:]) / min(len(episode_rewards), 10)
                    
                    print("\n" + "="*50)
                    print(f"Episode {ep} FINISHED:")
                    print(f"Total steps: {step}")
                    print(f"Final filled ratio: {info['filled_ratio']:.3f}")
                    print("\nFinal products state:")
                    for i, prod in enumerate(observation['products']):
                        print(f"Product {i}: size={prod['size']}, quantity={prod['quantity']}")
                    print(f"\nEpisode reward: {episode_reward:.2f}")  # Print total episode reward
                    print(f"Average reward (last 10): {avg_reward:.2f}")
                    print(f"Best reward so far: {best_reward:.2f}")
                    print("="*50 + "\n")
                    
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        # Save best model
                        policy.save_model("model_ppo_best")
                    
                    # Reset for next episode
                    observation, info = env.reset(seed=ep)
                    ep += 1
                    break

        # Save final model
        if should_exit:
            print("Saving model before exit...")
            policy.save_model(f"model_ppo_interrupted_step_{total_steps}")
            print("Model saved. Exiting...")
        else:
            policy.save_model("model_ppo_final")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Attempting to save model...")
        policy.save_model(f"model_ppo_error_step_{total_steps}")
        raise e

    # After training, set to evaluation mode
    print("\nTesting trained policy...")
    policy.training = False
    
    # Test the trained policy
    observation, info = env.reset(seed=1000)  # New seed for testing
    total_reward = 0
    test_episode_reward = 0  # Add test episode reward tracking
    
    for _ in range(200):  # Test episodes
        if should_exit:
            break
            
        action = policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        test_episode_reward += reward  # Accumulate test episode reward
        print(f"Test step reward: {reward:.2f}, Episode Total: {test_episode_reward:.2f}, Overall Total: {total_reward:.2f}")
        
        if terminated or truncated:
            print("Test episode finished")
            print(f"Test episode reward: {test_episode_reward:.2f}")
            print("Final info:", info)
            observation, info = env.reset()
            test_episode_reward = 0  # Reset test episode reward

def dqn_policy(env, policy, num_episodes=100, logging_interval=300, save_interval=100):
    """
    Train and test a DQN policy
    
    Args:
        env: Gymnasium environment
        policy: Policy object implementing get_action() and update_policy()
        num_episodes: Number of training episodes
        logging_interval: Steps between detailed logging
        save_interval: Steps between model saves
    """
    # Reset environment
    observation, info = env.reset(seed=42)
    print("Initial info:", info)

    # Training variables
    ep = 0
    best_reward = float('-inf')
    episode_rewards = []
    total_steps = 0

    try:
        while ep < num_episodes and not should_exit:
            episode_reward = 0
            step = 0
            
            while True and not should_exit:
                # Get action from policy
                action = policy.get_action(observation, info)
                
                # Take action in environment
                next_observation, env_reward, terminated, truncated, info = env.step(action)
                
                # Calculate reward using policy's method
                calculated_reward = policy.calculate_reward(action, observation, info)
                episode_reward += calculated_reward
                
                # Update policy với calculated reward
                policy.update_policy(calculated_reward, terminated or truncated)
                
                # Move to next state
                observation = next_observation
                step += 1
                total_steps += 1
                
                # Save model every SAVE_INTERVAL steps
                if total_steps % save_interval == 0:
                    print(f"\nSaving model at step {total_steps}...")
                    policy.save_model(f"model_dqn_step_{total_steps}")
                
                # Detailed logging every LOGGING_INTERVAL steps
                if step % logging_interval == 0:
                    print("\n" + "="*50)
                    print(f"Episode {ep} - Step {step} Status:")
                    print(f"Current filled ratio: {info['filled_ratio']:.3f}")
                    print("\nProducts remaining:")
                    for i, prod in enumerate(observation['products']):
                        if prod['quantity'] > 0:
                            print(f"Product {i}: size={prod['size']}, quantity={prod['quantity']}")
                    print("\nCurrent reward status:")
                    print(f"Episode reward so far: {episode_reward:.2f}")
                    print("="*50 + "\n")
                
                # Regular progress printing
                elif step % 100 == 0:
                    print(f"Episode {ep}, Step {step}, Total Steps {total_steps}")
                    print(f"Episode Reward (calculated): {episode_reward:.2f}")
                    print(f"Last Step Reward (calculated): {calculated_reward:.2f}")
                
                if terminated or truncated:
                    episode_rewards.append(episode_reward)
                    avg_reward = sum(episode_rewards[-10:]) / min(len(episode_rewards), 10)
                    
                    print("\n" + "="*50)
                    print(f"Episode {ep} FINISHED:")
                    print(f"Total steps: {step}")
                    print(f"Final filled ratio: {info['filled_ratio']:.3f}")
                    print("\nFinal products state:")
                    for i, prod in enumerate(observation['products']):
                        print(f"Product {i}: size={prod['size']}, quantity={prod['quantity']}")
                    print(f"\nEpisode reward: {episode_reward:.2f}")
                    print(f"Average reward (last 10): {avg_reward:.2f}")
                    print(f"Best reward so far: {best_reward:.2f}")
                    print("="*50 + "\n")
                    
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        # Save best model
                        policy.save_model("model_dqn_best")
                    
                    # Reset for next episode
                    observation, info = env.reset(seed=ep)
                    ep += 1
                    break

        # Save final model
        if should_exit:
            print("Saving model before exit...")
            policy.save_model(f"model_dqn_interrupted_step_{total_steps}")
            print("Model saved. Exiting...")
        else:
            policy.save_model("model_dqn_final")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Attempting to save model...")
        policy.save_model(f"model_dqn_error_step_{total_steps}")
        raise e

    # After training, set to evaluation mode
    print("\nTesting trained policy...")
    policy.training = False
    
    # Test the trained policy
    observation, info = env.reset(seed=1000)  # New seed for testing
    total_reward = 0
    
    for _ in range(200):  # Test episodes
        if should_exit:
            break
            
        action = policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Test step reward: {reward:.2f}, Total: {total_reward:.2f}")
        
        if terminated or truncated:
            print("Test episode finished")
            print("Final info:", info)
            observation, info = env.reset()
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

if __name__ == "__main__":
    # Constants
    NUM_EPISODES = 100
    LOGGING_INTERVAL = 300  # Log every 300 steps
    SAVE_INTERVAL = 100    # Save model every 100 steps

    # Create environment
    env = gym.make(
        "gym_cutting_stock/CuttingStock-v0",
        render_mode="human",  # Comment this line to disable rendering
    )
    
    # Choose which policy to run
    POLICY_TYPE = "ppo"  # Options: "actor_critic", "ppo", "dqn", "random", "greedy"
    
    if POLICY_TYPE == "actor_critic":
        policy = ActorCriticPolicy()
        train_func = actor_critics_policy
        # Run training and testing
        train_func(
            env=env,
            policy=policy,
            num_episodes=NUM_EPISODES,
            logging_interval=LOGGING_INTERVAL,
            save_interval=SAVE_INTERVAL
        )
    elif POLICY_TYPE == "ppo":
        policy = ProximalPolicyOptimization()
        train_func = ppo_policy
        # Run training and testing
        train_func(
            env=env,
            policy=policy,
            num_episodes=NUM_EPISODES,
            logging_interval=LOGGING_INTERVAL,
            save_interval=SAVE_INTERVAL
        )
    elif POLICY_TYPE == "random":
        policy = RandomPolicy()
        simple_policy_run(env=env, policy=policy)
    elif POLICY_TYPE == "greedy":
        policy = GreedyPolicy()
        simple_policy_run(env=env, policy=policy)
    else:  # dqn
        policy = DeepQNetworkPolicy()
        train_func = dqn_policy
        # Run training and testing
        train_func(
            env=env,
            policy=policy,
            num_episodes=NUM_EPISODES,
            logging_interval=LOGGING_INTERVAL,
            save_interval=SAVE_INTERVAL
        )
    
    env.close()


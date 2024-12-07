import gym_cutting_stock
import gymnasium as gym
from student_submissions.s2210xxx.ProximalPolicyOptimization import ProximalPolicyOptimization
import numpy as np
import time
import multiprocessing as mp
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import tqdm
from tqdm.contrib.concurrent import process_map  # For parallel processing with progress bar


class BenchmarkMetrics:
    def __init__(self):
        self.metrics = defaultdict(list)
        
    def add_result(self, policy_name, episode_metrics):
        """Add results for a policy"""
        for key, value in episode_metrics.items():
            self.metrics[f"{policy_name}_{key}"].append(value)
    
    def get_summary(self, policy_name):
        """Get statistical summary for a policy"""
        summary = f"\n{'='*20} {policy_name} Performance Summary {'='*20}\n"
        
        # Performance Metrics
        filled_ratios = self.metrics[f"{policy_name}_filled_ratio"]
        solution_times = self.metrics[f"{policy_name}_solution_time"]
        stocks_used = self.metrics[f"{policy_name}_stocks_used"]
        waste_ratios = self.metrics[f"{policy_name}_waste_ratio"]
        
        # Calculate statistics
        summary += "\nFill Ratio Statistics:\n"
        summary += f"Mean: {np.mean(filled_ratios)*100:.2f}%\n"
        summary += f"Std Dev: {np.std(filled_ratios)*100:.2f}%\n"
        summary += f"Min: {np.min(filled_ratios)*100:.2f}%\n"
        summary += f"Max: {np.max(filled_ratios)*100:.2f}%\n"
        
        summary += "\nSolution Time Statistics (seconds):\n"
        summary += f"Mean: {np.mean(solution_times):.3f}\n"
        summary += f"Std Dev: {np.std(solution_times):.3f}\n"
        summary += f"Min: {np.min(solution_times):.3f}\n"
        summary += f"Max: {np.max(solution_times):.3f}\n"
        
        summary += "\nStock Usage Statistics:\n"
        summary += f"Mean: {np.mean(stocks_used):.1f}\n"
        summary += f"Std Dev: {np.std(stocks_used):.1f}\n"
        summary += f"Min: {np.min(stocks_used)}\n"
        summary += f"Max: {np.max(stocks_used)}\n"
        
        summary += "\nWaste Ratio Statistics:\n"
        summary += f"Mean: {np.mean(waste_ratios)*100:.2f}%\n"
        summary += f"Std Dev: {np.std(waste_ratios)*100:.2f}%\n"
        summary += f"Min: {np.min(waste_ratios)*100:.2f}%\n"
        summary += f"Max: {np.max(waste_ratios)*100:.2f}%\n"
        
        return summary
    
    def plot_comparison(self, policies, save_path="benchmark_results"):
        """Plot comparison graphs for all policies"""
        os.makedirs(save_path, exist_ok=True)
        
        metrics_to_plot = {
            'filled_ratio': ('Fill Ratio', '%'),
            'solution_time': ('Solution Time', 'seconds'),
            'stocks_used': ('Stocks Used', 'count'),
            'waste_ratio': ('Waste Ratio', '%')
        }
        
        for metric, (title, unit) in metrics_to_plot.items():
            plt.figure(figsize=(10, 6))
            
            for policy in policies:
                policy_name = policy.__name__
                values = self.metrics[f"{policy_name}_{metric}"]
                
                if 'ratio' in metric:
                    values = [v * 100 for v in values]  # Convert to percentage
                
                plt.boxplot(values, positions=[policies.index(policy)], 
                           labels=[policy_name])
            
            plt.title(f'{title} Comparison')
            plt.ylabel(f'{title} ({unit})')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_path, f'{metric}_comparison.png'))
            plt.close()

def initialize_policy(policy_class):
    """Initialize policy and load model if needed"""
    if policy_class == ProximalPolicyOptimization:
        # Create policy without any file handles
        policy = policy_class()
        # Close any potential file handles before returning
        if hasattr(policy, 'log_file') and policy.log_file is not None:
            policy.log_file.close()
            policy.log_file = None
        # Load model
        model_loaded = policy.load_model("saved_models/model_ppo_best.pt")
        if not model_loaded:
            print("Failed to load PPO model")
        policy.training = False  # Set to evaluation mode
    else:
        policy = policy_class()
    return policy

def run_episode(args):
    """Run a single episode and return metrics"""
    policy, seed = args
    env = gym.make("gym_cutting_stock/CuttingStock-v0")
    
    # Set random seed
    random.seed(seed)
    env.reset(seed=seed)
    
    # Initialize metrics
    start_time = time.time()
    observation, info = env.reset()
    terminated = False
    
    while not terminated:
        action = policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
    
    end_time = time.time()
    
    # Calculate metrics
    solution_time = end_time - start_time
    stocks_used = sum(1 for stock in observation['stocks'] if np.any(stock > 0))
    waste_ratio = 1 - info['filled_ratio']
    
    metrics = {
        'filled_ratio': info['filled_ratio'],
        'solution_time': solution_time,
        'stocks_used': stocks_used,
        'waste_ratio': waste_ratio
    }
    
    env.close()
    return metrics

def benchmark_policy(policy_class, num_episodes=100, num_processes=4):
    """Benchmark a policy using parallel processing"""
    print(f"\nBenchmarking {policy_class.__name__}...")
    
    # Create argument list for parallel processing
    seeds = [random.randint(1000, 10000) for _ in range(num_episodes)]
    args = [(initialize_policy(policy_class), seed) for seed in seeds]
    
    # Run episodes in parallel with progress bar
    results = process_map(
        run_episode, 
        args,
        max_workers=num_processes,
        desc=f"Running {policy_class.__name__}",
        total=num_episodes
    )
    
    # Collect metrics
    metrics = BenchmarkMetrics()
    for result in results:
        metrics.add_result(policy_class.__name__, result)
    
    return metrics

def compare_policies(policies, num_episodes=100, num_processes=4):
    """Compare multiple policies"""
    print(f"\nStarting benchmark comparison of {len(policies)} policies:")
    for idx, policy in enumerate(policies, 1):
        print(f"{idx}. {policy.__name__}")
    print(f"Total episodes to run: {num_episodes * len(policies)}\n")
    
    all_metrics = {}
    
    # Add overall progress bar for all policies
    with tqdm.tqdm(policies, desc="Policies", position=0) as policy_pbar:
        for policy in policy_pbar:
            policy_pbar.set_description(f"Testing {policy.__name__}")
            metrics = benchmark_policy(policy, num_episodes, num_processes)
            all_metrics[policy.__name__] = metrics
            print(metrics.get_summary(policy.__name__))
    
    # Create combined metrics for plotting
    combined_metrics = BenchmarkMetrics()
    for policy_name, metrics in all_metrics.items():
        for key, values in metrics.metrics.items():
            combined_metrics.metrics[key] = values
    
    # Plot comparisons
    combined_metrics.plot_comparison(policies)
    
    # Print comparison
    print("\n" + "="*20 + " Policy Comparison " + "="*20)
    for policy in policies:
        name = policy.__name__
        metrics = all_metrics[name]
        filled_ratios = metrics.metrics[f"{name}_filled_ratio"]
        solution_times = metrics.metrics[f"{name}_solution_time"]
        
        print(f"\n{name}:")
        print(f"Average Fill Ratio: {np.mean(filled_ratios)*100:.2f}%")
        print(f"Average Solution Time: {np.mean(solution_times):.3f}s")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define policies to benchmark
    policies = [
        ProximalPolicyOptimization
    ]
    
    # Run benchmark
    compare_policies(
        policies=policies,
        num_episodes=100,  # Number of episodes per policy
        num_processes=4    # Number of parallel processes
    )